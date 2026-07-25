#!/usr/bin/env python3
"""PN119 v2 — probe refit loop from the self-training sink (BUILD-PACK §v2).

Generalizes ~/shared/needfit/train_pn119_probe.py: instead of the fixed
100-item GPQA capture, trains on rows collected by the live router's sink
(features written at prefill-finalize, labels = observed thinking SPEND at
request finish), plus (by default) the original 100-item seed set for
continuity. CPU-only, zero GPU. Runs from a systemd timer; writes the new
probe via ATOMIC swap (pn119_atomic) so the engine's hot-reload can never
observe a torn artifact.

CENSORING GUARDS (§v2 item 3 — the loop must never train on its own
routing decisions and drift). A sink row is ELIGIBLE for training only if
its observed spend is an UNCENSORED measurement of true need:

  G1 thinking-enabled only (§v2 item 4): thinking-off traffic has no spend
     signal. Rows are kept only when the router recorded thinking=true
     (prompt tail ends in an OPEN <think>). Rows with thinking=false or
     unknown are dropped. Legacy rows (written before the router recorded
     the flag) are dropped unless --legacy-thinking-ok is passed — only
     defensible while the sink is known to be 100% shadow bench traffic.
  G6 no-generation rows are not evidence (added 2026-07-25, P1). A request
     that generated (almost) nothing MEASURED NOTHING: `</think>` was never
     emitted, so `_label_fields` reports rtok=0 AND cap_hit=True, and G3
     then promotes it to a DEEP positive. Diagnostic tooling
     (pn119_tap_capture.py, pn119_b3_numerics.py) sends max_tokens=1 by
     design, so every one of its rows is a synthetic deep positive. Measured
     on 2026-07-25: 130 of 151 eligible rows were exactly this traffic, all
     y=1, training prior 0.733. Rows with generated < --min-generated are
     dropped outright, and G3 is narrowed to match. Runs immediately after
     G1 — before G2, so the "lean cap-hit is positive evidence" exception
     can never resurrect a no-generation row.
  G2 no self-censored labels: under PN119_MODE=enforce a lean-routed
     request runs with the lean budget, so its observed spend is a FLOOR,
     not a label. Such rows (mode=enforce, route=lean, explore=false) are
     EXCLUDED — with one exception from the pack: a cap-hit on a
     lean-routed request IS positive (deep) evidence, so those rows are
     kept with y=1 (rtok censored, label known). Kept unconditionally:
     mode=shadow rows (router acts on nothing — fully uncensored),
     explore=true rows (PN119_EXPLORE gave them generous caps precisely so
     their labels stay honest), and deep-routed rows (they received the
     full budget).
  G3 cap-hit => y=1 regardless of measured rtok (spend was truncated by the
     cap, true need >= cap) — ONLY for rows that generated >= --min-generated.
     A cap-hit at zero generated tokens is a max_tokens artefact, not spend.
  G4 dedup by req_id (preemption/retry can double-log), last finish wins.
  G5 structural: a row needs BOTH a score line (features exist in the
     .bin) and a finish line (label exists); orphans are dropped.
  G7 tag/marker exclusion: --exclude-tag <glob> drops whole sink windows,
     and any `.synthetic-*.json` marker a capture tool drops in the sink
     dir excludes its own req_ids (see pn119_tap_capture.py).

Missing "mode" on legacy lines is treated as "shadow": the sink has only
ever been written under PN119_MODE=shadow (compose tcbench8021.yml, live
since 2026-07-25 15:23Z) and the enforce-era router always records mode.

QUALITY GATES. Two of the original three were structurally incapable of
catching a bad candidate:
  * AUC was IN-SAMPLE (fit X_train, score X_train). Measured on the seed:
    in-sample 0.9657, honest LOO 0.9411, shuffled-label noise floor 0.6796
    — the gate sat below the noise floor's neighbourhood by construction.
    It is now a TEMPORAL split: fit on ts < t_cut, gate on the most recent
    --holdout-frac of rows. LOO (exact ridge LOO at fixed PCA) is reported,
    never gated.
  * rho vs the frozen v1 LOO scores was anti-learning: >= 0.85 agreement
    with the 07-23 GPQA seed forbids the loop from ever learning anything
    that seed disagrees with. Split in two: a STABILITY gate =
    spearman(candidate, INCUMBENT) on a common recent sample, and the v1
    anchor demoted to a reported line with an alert threshold.
  * Both surviving gates are RANK-based and cannot see a calibration break
    (a monotone rescale is rank-identical and would swap the live probe
    with every score shifted past PN119_TDEEP). A CALIBRATION gate now
    bounds the deep fraction at the live threshold and the median shift
    against the incumbent on the reservoir.
  * The seed set must share the sink's CAPTURE SOURCE. Sink features come
    from the in-engine tap; lens-features-*.safetensors are HF-offline
    captures whose same-item cosine against the tap is ~0.968. Training on
    a mixture is training on two different feature spaces — refused.

CURSOR. n_new used to be `len(rows) - eligible_at_last_swap` "because the
sink is append-only". The first prune makes that negative forever: the
refit skips, exits 0, and the timer stays green while the loop is dead.
The cursor is now content-addressed — {last_ts_seen, per-file byte
offsets} — so pruning a window simply removes rows, and n_new stays a
count of rows this process has never processed.

MEMORY. The .bin files are memory-mapped and only rows that actually enter
the reservoir are materialised, so resident memory is bounded by the
reservoir cap (2000 x 30720 x fp16 = 123 MB) instead of growing with the
sink (a 30-day sink at the observed rate projected to ~13 GB).

Exit codes: 0 = swapped (or dry-run that would swap); 2 = candidate
REJECTED by a quality gate or REFUSED on a capture-source mismatch
(candidate npz preserved in the state dir for autopsy; the live probe is
untouched); 3 = SKIPPED, nothing was wrong but there was not enough new
data. 3 is deliberately distinct from 0 so a silent permanent skip cannot
masquerade as a healthy refit — the systemd unit needs
`SuccessExitStatus=3` for a skip not to show as a unit failure.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pn119_atomic import atomic_write_npz  # noqa: E402

LAYERS = (42, 47, 51)
D_MODEL = 5120
FEAT_DIM = len(LAYERS) * 2 * D_MODEL  # 30720
ROW_BYTES = FEAT_DIM * 2  # bf16
NEEDFIT = os.path.expanduser("~/shared/needfit")
RESULTS = os.path.expanduser("~/shared/folderX/qbench45/results")

# Sink rows are produced by the in-engine tap, always. Anything trained
# alongside them must come from the same capture path (see load_seed).
SINK_CAPTURE_SOURCE = "tap"

EXIT_OK, EXIT_REJECT, EXIT_SKIP = 0, 2, 3


@dataclass
class Row:
    req_id: str
    tag: str               # sink window tag (feats-<tag>.bin / meta-<tag>.jsonl)
    feat_path: str
    row_idx: int           # row index inside feats-<tag>.bin
    rtok: int              # measured thinking spend (censored if cap_hit)
    generated: int         # total tokens generated (G6 / narrowed G3)
    cap_hit: bool
    mode: str
    route: str
    explore: bool
    thinking: object       # True / False / None(unknown) / "legacy"
    ts: float
    end_off: int = 0       # byte offset just past this row's last meta line
    x: np.ndarray | None = field(default=None, repr=False)


# ── feature I/O: memmap + exact bf16→f32 widen (no torch, no full read) ────
def bf16_rows(path: str, idx) -> np.ndarray:
    """Materialise ONLY the requested row indices from a bf16 .bin.

    bf16 -> f32 is an exact 16-bit left shift (same exponent/mantissa
    layout), so numpy does it without torch and without touching the rest
    of the file: the memmap only faults in the pages we index.
    """
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    n_rows = mm.size // FEAT_DIM
    out = np.empty((len(idx), FEAT_DIM), dtype=np.float32)
    for k, i in enumerate(idx):
        if i >= n_rows:
            raise IndexError(f"{path}: row {i} beyond {n_rows}")
        raw = np.asarray(mm[i * FEAT_DIM:(i + 1) * FEAT_DIM], dtype=np.uint32)
        out[k] = (raw << np.uint32(16)).view(np.float32)
    del mm
    return out


def bin_row_count(path: str) -> int:
    return os.path.getsize(path) // ROW_BYTES


def materialise(rows: list[Row]) -> None:
    """Fill .x on the given rows, one memmap per sink window."""
    by_file: dict[str, list[Row]] = {}
    for r in rows:
        if r.x is None:
            by_file.setdefault(r.feat_path, []).append(r)
    for path, rs in by_file.items():
        X = bf16_rows(path, [r.row_idx for r in rs])
        for k, r in enumerate(rs):
            r.x = X[k]


def _iter_meta_lines(path: str):
    """Yield (parsed-or-None, text, end_byte_offset) for each meta line."""
    off = 0
    with open(path, "rb") as f:
        for raw in f:
            off += len(raw)
            s = raw.strip()
            if not s:
                continue
            yield s.decode("utf-8", "replace"), off


def load_markers(sink_dir: str) -> tuple[set[str], set[str]]:
    """`.synthetic-*.json` markers: {"tag": ..., "req_ids": [...]}.

    Capture tools that must hit the LIVE endpoint (and therefore the live
    sink) drop one of these naming the rows they caused, so their traffic
    is excluded even if the operator forgets --exclude-tag.
    """
    req_ids: set[str] = set()
    tags: set[str] = set()
    for p in sorted(glob.glob(os.path.join(sink_dir, ".synthetic-*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        req_ids.update(str(r) for r in m.get("req_ids", []))
        for t in m.get("tags", []) or ([m["tag"]] if m.get("tag") else []):
            tags.add(str(t))
    return req_ids, tags


def load_sink(sink_dir: str, counts: dict, exclude_tags=(),
              marker_req_ids=frozenset()) -> tuple[list[Row], dict]:
    """Parse the sink. Returns (rows without features, {tag: meta bytes}).

    Globs only the TOP level: files moved into .quarantine/ stay on disk as
    reference data but leave the training set.
    """
    rows: list[Row] = []
    sizes: dict[str, int] = {}
    tags = sorted(
        f[len("meta-"):-len(".jsonl")]
        for f in os.listdir(sink_dir)
        if f.startswith("meta-") and f.endswith(".jsonl")
    )
    for tag in tags:
        meta_p = os.path.join(sink_dir, f"meta-{tag}.jsonl")
        feat_p = os.path.join(sink_dir, f"feats-{tag}.bin")
        sizes[tag] = os.path.getsize(meta_p)
        if not os.path.isfile(feat_p) or os.path.getsize(feat_p) == 0:
            continue
        if any(fnmatch.fnmatch(tag, pat) for pat in exclude_tags):
            counts["g7_excluded_tag"] = counts.get("g7_excluded_tag", 0) + 1
            continue
        n_feat_rows = bin_row_count(feat_p)
        score_lines: dict[str, tuple[dict, int]] = {}
        finish_lines: dict[str, tuple[dict, int]] = {}
        for text, off in _iter_meta_lines(meta_p):
            try:
                m = json.loads(text)
            except json.JSONDecodeError:
                counts["bad_json"] = counts.get("bad_json", 0) + 1
                continue
            if m.get("finish"):
                finish_lines[m["req_id"]] = (m, off)   # G4: last finish wins
            elif "row" in m:
                score_lines[m["req_id"]] = (m, off)
        for req_id, (sm, s_off) in score_lines.items():
            counts["scored"] = counts.get("scored", 0) + 1
            if req_id in marker_req_ids:
                counts["g7_marker_excluded"] = counts.get("g7_marker_excluded", 0) + 1
                continue
            got = finish_lines.get(req_id)
            if got is None:
                counts["g5_no_finish"] = counts.get("g5_no_finish", 0) + 1
                continue
            fm, f_off = got
            ridx = int(sm["row"])
            if ridx >= n_feat_rows:
                counts["g5_no_feature_row"] = counts.get("g5_no_feature_row", 0) + 1
                continue
            generated = int(fm.get("generated", 0) or 0)
            rtok = fm.get("rtok")
            rtok = generated if rtok is None else int(rtok)
            rows.append(Row(
                req_id=req_id,
                tag=tag,
                feat_path=feat_p,
                row_idx=ridx,
                rtok=rtok,
                generated=generated,
                cap_hit=bool(fm.get("cap_hit", False)),
                mode=str(sm.get("mode", "shadow")),
                route=str(sm.get("route", "")),
                explore=bool(sm.get("explore", False)),
                thinking=fm["thinking"] if "thinking" in fm else "legacy",
                ts=float(fm.get("ts", 0.0)),
                end_off=max(s_off, f_off),
            ))
    return rows, sizes


def apply_guards(rows: list[Row], legacy_thinking_ok: bool, min_generated: int,
                 counts: dict) -> list[Row]:
    out = []
    for r in rows:
        # G1 — thinking-enabled only
        if r.thinking is True:
            pass
        elif r.thinking == "legacy" and legacy_thinking_ok:
            counts["g1_legacy_accepted"] = counts.get("g1_legacy_accepted", 0) + 1
        else:
            key = "g1_thinking_off" if r.thinking is False else (
                "g1_legacy_dropped" if r.thinking == "legacy" else "g1_thinking_unknown")
            counts[key] = counts.get(key, 0) + 1
            continue
        # G6 — a request that generated (almost) nothing measured no spend,
        # so it is not a label of ANY kind (least of all a deep positive).
        if r.generated < min_generated:
            counts["g6_no_generation"] = counts.get("g6_no_generation", 0) + 1
            continue
        # G2 — self-censoring
        if r.mode == "enforce" and r.route == "lean" and not r.explore:
            if r.cap_hit:
                counts["g2_lean_caphit_pos"] = counts.get("g2_lean_caphit_pos", 0) + 1
                out.append(r)  # y=1 via G3
            else:
                counts["g2_censored_dropped"] = counts.get("g2_censored_dropped", 0) + 1
            continue
        out.append(r)
    counts["eligible"] = len(out)
    return out


def label_for(rtok, cap_hit, generated, deep_thresh: int, min_generated: int) -> float:
    """G3, narrowed: a cap-hit is deep evidence only if it generated enough
    to have measured anything. `generated is None` = pre-G6 reservoir meta."""
    if cap_hit and (generated is None or int(generated) >= min_generated):
        return 1.0
    return 1.0 if int(rtok) >= deep_thresh else 0.0


# ── cursor: content-addressed, prune-proof ─────────────────────────────────
def load_cursor(path: str) -> dict:
    cur = {"schema": 2, "last_ts_seen": 0.0, "files": {}}
    if not os.path.isfile(path):
        return cur
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return cur
    if int(raw.get("schema", 1)) < 2:
        # v1 cursor held only {"eligible_at_last_swap": N} — a count, which
        # is exactly the thing that breaks on a prune. Nothing in it can be
        # translated into offsets, so treat the sink as unseen (the refit
        # re-offers rows the reservoir already holds; update_reservoir's
        # seen_req_ids absorbs that).
        cur["last_ts_seen"] = float(raw.get("ts", 0.0) or 0.0)
        return cur
    cur["last_ts_seen"] = float(raw.get("last_ts_seen", 0.0) or 0.0)
    cur["files"] = {str(k): int(v) for k, v in (raw.get("files") or {}).items()}
    return cur


def is_new(row: Row, cursor: dict) -> bool:
    off = cursor["files"].get(row.tag)
    if off is None:
        # Window the cursor has never recorded: new unless it predates the
        # last swap (a resurrected/renamed old file).
        return row.ts > cursor["last_ts_seen"]
    return row.end_off > off


def fit_probe(X: np.ndarray, y: np.ndarray, lam: float, pcs: int):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
    Vt = Vt[:pcs]
    P = Xs @ Vt.T
    A = np.hstack([P, np.ones((len(P), 1))])
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    scores = A @ w
    return mu, sd, Vt, w, scores


def score_with(mu, sd, Vt, w, X: np.ndarray) -> np.ndarray:
    Xs = (X - mu) / sd
    P = Xs @ Vt.T
    return np.hstack([P, np.ones((len(P), 1))]) @ w


def loo_scores_fixed_pca(X: np.ndarray, y: np.ndarray, lam: float,
                         mu, sd, Vt):
    """Exact leave-one-out predictions for the RIDGE at a fixed PCA basis.

    yhat_i^(-i) = y_i - e_i / (1 - h_ii) with h = A(A'A+lam I)^-1 A'. Takes
    the CALLER's already-fitted (mu, sd, Vt) — a second SVD of a
    2000 x 30720 reservoir costs ~100 s and ~1.5 GB for a number that is
    only ever printed. The PCA basis is fit on all rows, so this is
    optimistic — which is why it is REPORTED and never gated.
    """
    A = np.hstack([((X - mu) / sd) @ Vt.T, np.ones((len(X), 1))])
    G = np.linalg.inv(A.T @ A + lam * np.eye(A.shape[1]))
    H = A @ G @ A.T
    w = G @ A.T @ y
    resid = y - A @ w
    h = np.clip(np.diag(H), 0.0, 1.0 - 1e-9)
    return y - resid / (1.0 - h)


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    sa = a[sorter]
    obs = np.r_[True, sa[1:] != sa[:-1]]
    dense = obs.cumsum()[inv]
    counts = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (counts[dense] + counts[dense - 1] + 1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


# ── gates ──────────────────────────────────────────────────────────────────
def temporal_split(ts: np.ndarray, holdout_frac: float):
    """Indices (fit, holdout) with the most recent holdout_frac by TIME.

    The cut is a TIMESTAMP, never a row count: rows sharing a timestamp (a
    burst, or the whole seed set at ts=0) must not straddle the split, or
    the "out-of-sample" holdout contains rows the fit already saw. Among
    the cuts that leave the fit side non-empty, take the one whose holdout
    is closest to the nominal fraction — so the realised holdout can differ
    from it, and the caller gates on the size it actually got.
    """
    ts = np.asarray(ts, dtype=float)
    n = len(ts)
    if n == 0:
        return np.array([], int), np.array([], int)
    k = max(1, int(round(n * holdout_frac)))
    ts_sorted = np.sort(ts, kind="mergesort")
    uniq = np.unique(ts_sorted)
    starts = np.searchsorted(ts_sorted, uniq, side="left")
    valid = np.nonzero(starts > 0)[0]          # cut at uniq[0] empties the fit
    if len(valid) == 0:                        # every row shares one timestamp
        return np.array([], int), np.arange(n)
    n_hold = n - starts[valid]
    t_cut = uniq[valid[int(np.argmin(np.abs(n_hold - k)))]]
    return np.nonzero(ts < t_cut)[0], np.nonzero(ts >= t_cut)[0]


def gate_out_of_sample(X, y, ts, lam, pcs, holdout_frac, min_hold, min_auc,
                       report: dict):
    """Fit on the past, score the future. Returns list of failure strings."""
    fit_i, hold_i = temporal_split(ts, holdout_frac)
    report["oos_n_fit"] = int(len(fit_i))
    report["oos_n_hold"] = int(len(hold_i))
    if len(hold_i) < min_hold or len(fit_i) < pcs + 2:
        report["oos_auc"] = None
        return [f"out-of-sample gate not evaluable (fit={len(fit_i)} "
                f"hold={len(hold_i)}, need fit>{pcs + 1} hold>={min_hold})"]
    y_h = y[hold_i]
    if y_h.min() == y_h.max():
        report["oos_auc"] = None
        return [f"out-of-sample holdout is single-class (n={len(hold_i)}, "
                f"prior={float(y_h.mean()):.3f}) — AUC undefined"]
    mu, sd, Vt, w, _ = fit_probe(X[fit_i], y[fit_i], lam, pcs)
    a = auc(score_with(mu, sd, Vt, w, X[hold_i]), y_h)
    report["oos_auc"] = round(float(a), 4)
    return [] if a >= min_auc else [f"out-of-sample auc {a:.4f} < {min_auc}"]


def gate_stability(cand_scores, inc_scores, min_rho, min_n, report: dict):
    """spearman(candidate, incumbent) on a common recent sample."""
    n = 0 if cand_scores is None else len(cand_scores)
    report["stability_n"] = int(n)
    if inc_scores is None or n < min_n:
        report["stability_rho"] = None
        report["stability_note"] = ("no incumbent probe" if inc_scores is None
                                    else f"only {n} common recent rows (< {min_n})")
        return []          # nothing to be stable against — not a failure
    rho = spearman(np.asarray(cand_scores), np.asarray(inc_scores))
    report["stability_rho"] = round(float(rho), 4)
    return [] if rho >= min_rho else [
        f"stability rho vs incumbent {rho:.4f} < {min_rho}"]


def gate_calibration(train_scores, cand_res, inc_res, tdeep, min_deep_frac,
                     max_deep_frac, max_median_shift, report: dict):
    """The gate the rank-based ones structurally cannot be: absolute level.

    A monotone rescale of the candidate is rank-identical (AUC and every
    spearman unchanged) and still moves every score across PN119_TDEEP.
    """
    fails = []
    frac = float((np.asarray(train_scores) >= tdeep).mean())
    report["deep_frac_at_live_tdeep"] = round(frac, 4)
    report["live_tdeep"] = tdeep
    if not (min_deep_frac <= frac <= max_deep_frac):
        fails.append(f"deep fraction at PN119_TDEEP={tdeep} is {frac:.4f}, "
                     f"outside [{min_deep_frac}, {max_deep_frac}]")
    if cand_res is None or inc_res is None or len(cand_res) == 0:
        report["median_shift"] = None
        report["calibration_note"] = "no incumbent/reservoir — median shift unchecked"
        return fails
    shift = float(abs(np.median(cand_res) - np.median(inc_res)))
    report["median_shift"] = round(shift, 4)
    report["median_new"] = round(float(np.median(cand_res)), 4)
    report["median_old"] = round(float(np.median(inc_res)), 4)
    if not (shift < max_median_shift):
        fails.append(f"median score shift on the reservoir {shift:.4f} "
                     f">= {max_median_shift}")
    return fails


# ── seed ───────────────────────────────────────────────────────────────────
def seed_capture_source(path: str) -> str:
    """Classify a seed capture: 'tap' (live in-engine tap, same features the
    sink holds) vs 'offline-hf' (lens_pilot's HF/GPTQModel host capture).

    The two are NOT interchangeable: same-item cosine between them is ~0.968
    while the between-item max among the seed is 0.9869, i.e. the gap is of
    the same order as "a different question".
    """
    side = os.path.splitext(path)[0] + ".json"
    if os.path.isfile(side):
        try:
            with open(side, encoding="utf-8") as f:
                meta = (json.load(f) or {}).get("_meta", {})
        except (OSError, json.JSONDecodeError):
            meta = {}
        src = str(meta.get("source", ""))
        if "tap" in src.lower():
            return "tap"
        if src:
            return "offline-hf"
        if meta.get("model_dir"):
            return "offline-hf"
    base = os.path.basename(path).lower()
    if base.startswith("tap-features"):
        return "tap"
    if base.startswith("lens-features"):
        return "offline-hf"
    return "unknown"


def load_seed(deep_thresh: int, features: str, champion: str):
    """Seed set (v1 continuity anchor), captured through the SAME tap the
    sink is written from — see seed_capture_source."""
    import torch                                # noqa: PLC0415 — bf16 needs it
    from safetensors.torch import load_file     # noqa: PLC0415
    st = load_file(features)                    # bf16 — numpy cannot hold it
    champ = {}
    with open(champion, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            champ[rec["item_id"]] = rec.get("reasoning_tokens") or 0
    ids = [i for i in sorted(champ) if i in st]
    X = np.stack([st[i].to(torch.float32).flatten().numpy() for i in ids])
    y = np.array([1.0 if champ[i] >= deep_thresh else 0.0 for i in ids])
    return ids, X, y


# ── reservoir ──────────────────────────────────────────────────────────────
def update_reservoir(state_dir: str, rows: list[Row], cap: int,
                     rng: np.random.Generator, counts: dict,
                     persist: bool = True, seen_cap: int = 200_000):
    """Bounded raw-feature reservoir (pack: ~2000 requests) so old sink files
    can be pruned without losing PCA-refresh data. Classic reservoir
    sampling over the stream of eligible rows, persisted atomically.

    Two properties this function must hold:
      * seen_req_ids is PERSISTED. Without it every run re-offers rows a
        previous run's coin flip rejected, which both distorts the sampling
        (repeated draws) and re-materialises features for rows that will be
        thrown away again.
      * features are materialised ONLY for rows that win a slot. The
        sampling decision needs no feature data, so the .bin pages for
        losers are never faulted in.
    """
    res_path = os.path.join(state_dir, "reservoir.npz")
    if os.path.isfile(res_path):
        z = np.load(res_path, allow_pickle=True)
        R_X = list(z["X"])
        R_meta = [json.loads(s) for s in z["meta"]]
        seen_total = int(z["seen_total"])
        seen_ids = [str(s) for s in z["seen_req_ids"]] if "seen_req_ids" in z else []
    else:
        R_X, R_meta, seen_total, seen_ids = [], [], 0, []
    seen = set(seen_ids)
    seen.update(m["req_id"] for m in R_meta)
    new = [r for r in rows if r.req_id not in seen]
    counts["reservoir_offered_new"] = len(new)
    counts["reservoir_skipped_seen"] = len(rows) - len(new)

    plan: list[tuple[Row, int]] = []
    for r in new:
        seen_total += 1
        if len(R_X) < cap:
            slot = len(R_X)
            R_X.append(None)
            R_meta.append(None)
        else:
            j = int(rng.integers(0, seen_total))
            slot = j if j < cap else -1
        if slot >= 0:
            plan.append((r, slot))
    materialise([r for r, _s in plan])
    for r, slot in plan:
        R_X[slot] = r.x.astype(np.float16)
        R_meta[slot] = {"req_id": r.req_id, "rtok": r.rtok, "cap_hit": r.cap_hit,
                        "generated": r.generated, "mode": r.mode, "route": r.route,
                        "explore": r.explore, "ts": r.ts, "tag": r.tag}
    # A row can win a slot that a not-yet-filled append reserved only in the
    # append branch, which fills it immediately; nothing may stay None.
    keep = [k for k in range(len(R_X)) if R_X[k] is not None and R_meta[k] is not None]
    R_X = [R_X[k] for k in keep]
    R_meta = [R_meta[k] for k in keep]
    seen.update(r.req_id for r in new)
    seen_out = list(seen)[-seen_cap:] if len(seen) > seen_cap else list(seen)
    counts["reservoir_size"] = len(R_X)
    counts["reservoir_seen_total"] = seen_total
    counts["reservoir_seen_ids"] = len(seen_out)
    if persist:
        atomic_write_npz(res_path, {
            "X": np.stack(R_X) if R_X else np.zeros((0, FEAT_DIM), dtype=np.float16),
            "meta": np.array([json.dumps(m) for m in R_meta]),
            "seen_total": np.array(seen_total),
            "seen_req_ids": np.array(seen_out),
        })
    return R_X, R_meta


def _nan(v):
    return float("nan") if v is None else float(v)


def load_incumbent(path: str):
    if not os.path.isfile(path):
        return None
    try:
        z = np.load(path, allow_pickle=True)
        return (np.asarray(z["mu"]), np.asarray(z["sd"]),
                np.asarray(z["Vt10"]), np.asarray(z["w"]))
    except (OSError, KeyError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sink", default=f"{NEEDFIT}/pn119-sink")
    ap.add_argument("--out", default=f"{NEEDFIT}/pn119-live/probe.npz")
    ap.add_argument("--state", default=f"{NEEDFIT}/pn119-refit-state")
    ap.add_argument("--deep-thresh", type=int, default=2000,
                    help="rtok >= this => deep label (pack: champion band)")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--pcs", type=int, default=10)
    ap.add_argument("--min-new", type=int, default=500,
                    help="skip refit unless this many NEW eligible rows since last swap")
    ap.add_argument("--min-pos", type=int, default=10)
    ap.add_argument("--min-neg", type=int, default=10)
    ap.add_argument("--min-generated", type=int, default=32,
                    help="G6: rows that generated fewer tokens than this measured "
                         "no spend and are not evidence (max_tokens=1 diagnostics)")
    ap.add_argument("--exclude-tag", action="append", default=[],
                    help="G7: sink window tag (glob ok) to exclude; repeatable")
    ap.add_argument("--min-auc", type=float, default=0.80,
                    help="reject candidate below this OUT-OF-SAMPLE (temporal "
                         "holdout) AUC")
    ap.add_argument("--holdout-frac", type=float, default=0.20,
                    help="most-recent fraction by time held out of the gate fit")
    ap.add_argument("--min-holdout", type=int, default=8,
                    help="smallest holdout the out-of-sample gate will trust")
    ap.add_argument("--min-stability-rho", type=float, default=0.90,
                    help="reject candidate that rank-disagrees with the INCUMBENT "
                         "probe on a common recent sample")
    ap.add_argument("--stability-n", type=int, default=200,
                    help="size of the recent common sample for the stability gate")
    ap.add_argument("--min-stability-n", type=int, default=20,
                    help="below this many common rows the stability gate abstains")
    ap.add_argument("--anchor-rho-alert", type=float, default=0.60,
                    help="ALERT (not a gate) when rho vs router_loo_scores.json "
                         "falls below this — the v1 anchor may not veto learning")
    ap.add_argument("--live-tdeep", type=float,
                    default=float(os.environ.get("PN119_TDEEP", "0.495") or 0.495),
                    help="the live routing threshold the calibration gate uses")
    ap.add_argument("--min-deep-frac", type=float, default=0.20)
    ap.add_argument("--max-deep-frac", type=float, default=0.40)
    ap.add_argument("--max-median-shift", type=float, default=0.05,
                    help="max |median_new - median_old| on the reservoir")
    ap.add_argument("--reservoir", type=int, default=2000)
    ap.add_argument("--seed-features",
                    default=f"{NEEDFIT}/tap-features-20260725.safetensors",
                    help="seed capture; MUST share the sink's capture source (tap)")
    ap.add_argument("--seed-champion",
                    default=f"{RESULTS}/aibox-20260723-capt10full__gpqa_auto__"
                            f"thinkingcap_auto_t10.jsonl")
    ap.add_argument("--no-seed", action="store_true",
                    help="train on sink rows only (seed still used for the anchor line)")
    ap.add_argument("--allow-capture-mismatch", action="store_true",
                    help="train across capture sources anyway (you need a reason)")
    ap.add_argument("--legacy-thinking-ok", action="store_true",
                    help="accept pre-flag sink rows as thinking-enabled (G1); only "
                         "valid while the sink is known 100%% shadow thinking traffic")
    ap.add_argument("--force", action="store_true", help="ignore --min-new")
    ap.add_argument("--dry-run", action="store_true",
                    help="fit + gates + report, but do NOT swap or update state")
    args = ap.parse_args()

    os.makedirs(args.state, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    counts: dict = {}
    t0 = time.time()

    # Prune stale swap temps (a SIGKILL mid-swap orphans the mkstemp file;
    # it is a hidden dotfile and never the target name, so merely cosmetic).
    for d in {os.path.dirname(os.path.abspath(args.out)), args.state}:
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if f.startswith(".pn119-swap-") and t0 - os.path.getmtime(p) > 86400:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    marker_ids, marker_tags = load_markers(args.sink)
    exclude_tags = list(args.exclude_tag) + sorted(marker_tags)
    t_parse = time.time()
    raw, file_sizes = load_sink(args.sink, counts, exclude_tags, marker_ids)
    counts["parse_s"] = round(time.time() - t_parse, 2)
    rows = apply_guards(raw, args.legacy_thinking_ok, args.min_generated, counts)

    # min-new accounting — content-addressed cursor, so a prune cannot make
    # this negative (which used to skip the refit forever while exiting 0).
    cursor_p = os.path.join(args.state, "cursor.json")
    cursor = load_cursor(cursor_p)
    new_rows = [r for r in rows if is_new(r, cursor)]
    n_new = len(new_rows)
    counts["new_since_last_swap"] = n_new
    if not args.force and n_new < args.min_new:
        print(f"[refit] SKIP: {n_new} new eligible rows < --min-new {args.min_new} "
              f"(counts={json.dumps(counts)})")
        return EXIT_SKIP

    # Capture-source check BEFORE the reservoir is touched: a mismatch means
    # nothing downstream is meaningful, so nothing downstream should run.
    seed_src = seed_capture_source(args.seed_features)
    counts["seed_capture_source"] = seed_src
    counts["sink_capture_source"] = SINK_CAPTURE_SOURCE
    if (not args.no_seed and seed_src != SINK_CAPTURE_SOURCE
            and not args.allow_capture_mismatch):
        print(f"[refit] REFUSED: seed capture source {seed_src!r} "
              f"({os.path.basename(args.seed_features)}) != sink capture source "
              f"{SINK_CAPTURE_SOURCE!r}. Mixing an HF-offline capture with tap "
              f"features trains on two feature spaces (same-item cosine ~0.968). "
              f"Pass --seed-features <tap capture>, --no-seed, or "
              f"--allow-capture-mismatch if you have a reason.")
        return EXIT_REJECT

    rng = np.random.default_rng(int(t0))
    R_X, R_meta = update_reservoir(args.state, rows, args.reservoir, rng, counts,
                                   persist=not args.dry_run)
    X_res = (np.stack(R_X).astype(np.float32) if R_X
             else np.zeros((0, FEAT_DIM), np.float32))
    y_res = np.array([label_for(m["rtok"], m["cap_hit"], m.get("generated"),
                                args.deep_thresh, args.min_generated)
                      for m in R_meta])
    ts_res = np.array([float(m.get("ts", 0.0)) for m in R_meta])

    seed_ids, X_seed, y_seed = load_seed(args.deep_thresh, args.seed_features,
                                         args.seed_champion)

    if args.no_seed:
        X_train, y_train, ts_train = X_res, y_res, ts_res
        train_ids = [m["req_id"] for m in R_meta]
    else:
        X_train = np.vstack([X_seed, X_res]) if len(X_res) else X_seed
        y_train = np.concatenate([y_seed, y_res])
        # The seed predates every sink row: ts=0 pins it to the FIT side of
        # the temporal split, so the holdout is always live traffic.
        ts_train = np.concatenate([np.zeros(len(y_seed)), ts_res])
        train_ids = seed_ids + [m["req_id"] for m in R_meta]

    n_pos, n_neg = int(y_train.sum()), int((1 - y_train).sum())
    counts.update(train_rows=len(y_train), train_pos=n_pos, train_neg=n_neg,
                  train_prior=round(float(y_train.mean()), 4) if len(y_train) else None)
    if n_pos < args.min_pos or n_neg < args.min_neg:
        print(f"[refit] SKIP: class balance pos={n_pos} neg={n_neg} below minimum "
              f"({args.min_pos}/{args.min_neg}) (counts={json.dumps(counts)})")
        return EXIT_SKIP

    mu, sd, Vt, w, train_scores = fit_probe(X_train, y_train, args.lam, args.pcs)
    counts["auc_in_sample"] = round(auc(train_scores, y_train), 4)
    counts["auc_loo_reported"] = round(
        auc(loo_scores_fixed_pca(X_train, y_train, args.lam, mu, sd, Vt), y_train), 4)

    # v1 anchor: REPORTED, never a gate — gating on it forbids the loop from
    # ever learning anything the frozen 07-23 GPQA seed disagrees with.
    with open(f"{NEEDFIT}/router_loo_scores.json", encoding="utf-8") as f:
        loo = json.load(f)
    loo_map = dict(zip(loo["ids"], loo["scores"]))
    overlap = [i for i in seed_ids if i in loo_map]
    seed_scores = score_with(mu, sd, Vt, w, X_seed)
    seed_idx = {i: k for k, i in enumerate(seed_ids)}
    rho_anchor = spearman(np.array([seed_scores[seed_idx[i]] for i in overlap]),
                          np.array([loo_map[i] for i in overlap]))
    counts["rho_vs_loo_anchor"] = round(rho_anchor, 4)
    counts["rho_overlap_n"] = len(overlap)
    anchor_alert = rho_anchor < args.anchor_rho_alert

    prior = float(y_train.mean())
    suggested_tdeep = float(np.quantile(train_scores, 1.0 - prior))
    counts["suggested_tdeep"] = round(suggested_tdeep, 4)

    # ── gates ──────────────────────────────────────────────────────────────
    gates: dict = {}
    gate_fail = gate_out_of_sample(X_train, y_train, ts_train, args.lam, args.pcs,
                                   args.holdout_frac, args.min_holdout,
                                   args.min_auc, gates)
    inc = load_incumbent(args.out)
    cand_res = inc_res = None
    if len(X_res):
        recent = np.argsort(ts_res, kind="mergesort")[-args.stability_n:]
        cand_res = score_with(mu, sd, Vt, w, X_res[recent])
        if inc is not None:
            inc_res = score_with(*inc, X_res[recent])
    gate_fail += gate_stability(cand_res, inc_res, args.min_stability_rho,
                                args.min_stability_n, gates)
    gate_fail += gate_calibration(train_scores, cand_res, inc_res, args.live_tdeep,
                                  args.min_deep_frac, args.max_deep_frac,
                                  args.max_median_shift, gates)
    counts["gates"] = gates

    payload = {
        "mu": mu.astype(np.float64), "sd": sd.astype(np.float64),
        "Vt10": Vt.astype(np.float64), "w": w.astype(np.float64),
        "lam": args.lam, "pcs": args.pcs, "variant": "lens_only",
        "label": f"rtok_ge_{args.deep_thresh}_spend_selftrain",
        "feature_spec": "layers[42,47,51] x pools[last,mean] d5120 concat",
        "train_ids": np.array(train_ids),
        "refit_ts": t0, "refit_n_sink": len(X_res),
        "refit_n_seed": 0 if args.no_seed else len(X_seed),
        "refit_capture_source": SINK_CAPTURE_SOURCE,
        "refit_seed_features": os.path.basename(args.seed_features),
        # nan, never None: None round-trips as a pickled object array and the
        # engine's np.load of the probe should never need that.
        "refit_auc_oos": _nan(gates.get("oos_auc")),
        "refit_auc_in_sample": counts["auc_in_sample"],
        "refit_auc_loo": counts["auc_loo_reported"],
        "refit_stability_rho": _nan(gates.get("stability_rho")),
        "refit_rho_vs_loo_anchor": rho_anchor,
        "refit_deep_frac": _nan(gates.get("deep_frac_at_live_tdeep")),
        "refit_train_prior": prior,
        "refit_suggested_tdeep": suggested_tdeep,
    }

    counts["peak_rss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    report = {"ts": t0, "elapsed_s": round(time.time() - t0, 2), "counts": counts,
              "gates_failed": gate_fail, "anchor_alert": bool(anchor_alert),
              "out": args.out, "dry_run": args.dry_run,
              "args": {k: v for k, v in vars(args).items()}}
    if not args.dry_run:
        with open(os.path.join(args.state, "refit-report.json"), "w",
                  encoding="utf-8") as f:
            json.dump(report, f, indent=1)

    if anchor_alert:
        print(f"[refit] ALERT anchor rho vs router_loo_scores.json "
              f"{rho_anchor:.4f} < {args.anchor_rho_alert} — the candidate has "
              f"walked away from the v1 seed ordering. Not a gate; look at it.")

    if gate_fail:
        rej = os.path.join(args.state, f"rejected-{time.strftime('%Y%m%d-%H%M%S')}.npz")
        saved = ""
        if not args.dry_run:
            atomic_write_npz(rej, payload)
            saved = f"; candidate saved to {rej}"
        print(f"[refit] REJECTED ({'; '.join(gate_fail)}) — live probe untouched"
              f"{saved}\n[refit] {json.dumps(report)}")
        return EXIT_REJECT

    if args.dry_run:
        print(f"[refit] DRY-RUN ok — would swap {args.out}\n[refit] {json.dumps(report)}")
        return EXIT_OK

    atomic_write_npz(args.out, payload)
    with open(args.out, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:16]
    with open(cursor_p, "w", encoding="utf-8") as f:
        json.dump({"schema": 2,
                   "last_ts_seen": max([r.ts for r in rows], default=cursor["last_ts_seen"]),
                   "files": file_sizes, "ts": t0, "sha256_16": sha,
                   "n_new_consumed": n_new}, f)
    print(f"[refit] SWAPPED {args.out} sha256[:16]={sha} "
          f"auc_oos={gates.get('oos_auc')} stability_rho={gates.get('stability_rho')} "
          f"deep_frac={gates.get('deep_frac_at_live_tdeep')} "
          f"n_train={len(y_train)} (pos={n_pos}) prior={prior:.4f} "
          f"suggested_tdeep={suggested_tdeep:.4f}\n[refit] {json.dumps(report)}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
