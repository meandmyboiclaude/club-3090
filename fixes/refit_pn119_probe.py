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
  G2 no self-censored labels — INTERVAL CENSORING, not deletion (BUG-139,
     2026-07-25). Deleting censored rows is not neutral: measured on the
     live sink it kept 8 of 31 rows and 100% of the survivors were
     deep-routed, i.e. pure positive selection. Every row is now labelled
     through interval_label():
         not censored                       -> y = 1[rtok >= theta]
         censored, (budget - SLACK) >= theta -> y = 1
         censored otherwise                  -> y = None, weight 0,
                                                counted g3_interval_unresolved
     This is what closes the ABSORBING STATE. A lean-routed row truncated at
     b < theta used to be indistinguishable from a row that genuinely needed
     little (y=0 under shadow, y=1 under the old lean-cap-hit exception —
     wrong in both directions), so the loop reinforced the decision that
     created the row. It now contributes NOTHING instead of a fabricated
     label.
  G3 censoring detection. `cap_hit` sees only max_tokens, so a request
     stopped by its THINKING budget logs cap_hit=False and looks like a
     natural stop: measured 43 of 79 thinking rows sit at exactly
     (a PN100 100-grid grant) - 5, and only 4 log cap_hit=True. The router
     is adding `censored` / `budget_grant` / `budget_source` to the sink;
     this reads them when present and DERIVES them when absent (older
     rows), so the existing corpus is usable today:
       * explicit `censored` from the sink wins;
       * else `budget_grant` present  -> rtok >= budget_grant - SLACK;
       * else cap_hit (and generated >= --min-generated) -> censored, and
         the lower bound is rtok itself;
       * else the grant is derived as the smallest plausible PN100 grant
         >= rtok (the 100-grid from _continuous_budget, plus the tier
         budgets and the floor) and the same rule applies.
     SLACK = len(think_end_ids) + 8 = 13; the measured offset is exactly 5.
     Grid derivation is deliberately conservative: a genuine natural stop
     that happens to land within SLACK of a grid point is called censored
     and (below theta) becomes UNRESOLVED — it loses a row, it never
     invents a label.
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

DUAL-PROBE SHADOW PROMOTION (--promote-mode shadow, the default). A refit
does not swap the live probe. It STAGES the candidate
(<state>/candidate.npz + candidate.json) and stops; the engine picks it up
via PN119_PROBE_CANDIDATE and scores EVERY request with it while routing on
the incumbent alone. A later refit promotes only on evidence collected
AFTER the candidate went live-shadow (candidate.json:first_seen_ts):

  * a PAIRED DeLong test (fast Sun & Xu covariance) of candidate vs
    incumbent AUC on those rows' resolved labels — one-sided p < --delong-alpha
    AND auc_cand > auc_inc. Paired, because both probes scored the same
    requests; unpaired would ignore the covariance that makes the test sharp;
  * a DECISION-FLIP bound: |{rows where 1[cand>=T] != 1[inc>=T]}| / n must be
    <= --max-flip-frac (0.20). More than that is not a better probe, it is a
    DIFFERENT ROUTER, and it goes to a human;
  * the calibration gate, again, on the shadow scores.
The live probe is copied to probe.prev.npz before every swap and --rollback
puts it back, so a promotion is reversible without a refit.
--promote-mode direct is the pre-shadow behaviour (fit -> gates -> swap) and
is only defensible while no candidate-scoring router exists.

DRIFT REJECTION (gate_drift). Any of these rejects the candidate:
PSI > 0.25 on any principal component; a KS test with p < 0.01 AND D > 0.15
on any PC; a training prior outside [0.15, 0.55] (this one alone catches the
2026-07-25 poisoning — prior 0.733); an unresolved-censoring rate > 0.4 (the
corpus is mostly lower bounds and nothing can be learned from it); a stale
or failing b3 numerics report; or an explore-integrity violation.

EXPLORATION — implemented, NOT enabled. PN119_EXPLORE is 0.0 and must stay
0.0 until the CONSUMER honours an explore budget. Turning it on first is
strictly worse than leaving it off: an explore row would be an ordinary
censored row wearing a trusted-label badge. explore_decision() implements
the stratified epsilon (0.25 inside |score - T| < 0.10, 0.01 outside;
effective ~0.065 on the live score distribution, ~200 boundary rows in two
days against 17 for a uniform 0.03), the logged propensity p_explore, IPS
weights clipped at 50, a token-bucket ceiling that makes the cost a hard
number, and a 2% holdout on a DIFFERENT hash salt that is always
theta-sufficient and is excluded from training with no override flag.

ACCURACY MONITOR (--accuracy-monitor). `extra.resp_id` in the graded run
JSONLs joins the sink's `req_id` with its last `-` segment stripped
(measured: 339/339 across 12 runs). It alerts on deep-bucket accuracy
regression. `correct` is NEVER a training target: AUC(score -> correct) is
0.26-0.40, correctly BELOW 0.5, because the lens finds HARD items. The
training target is uplift, not correctness.

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
import math
import os
import resource
import shutil
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pn119_atomic import atomic_write_npz  # noqa: E402

LAYERS = (42, 47, 51)
D_MODEL = 5120
# Two eras of sink rows exist. The 30720-dim one is [last, mean] per layer;
# the LAST-ONLY router writes 15360. Nothing in a feats-*.bin says which, and
# a reader that guesses wrong does not fail — it glues two rows into one and
# joins them to the wrong labels. So: the window's `pn119_header` line is
# authoritative, byte-count inference is the fallback, and this constant is
# only the last resort (it is the era every window on disk today was written
# in — verified by size/row-count).
FEAT_DIM = len(LAYERS) * 2 * D_MODEL  # 30720
KNOWN_FEAT_DIMS = (len(LAYERS) * D_MODEL, FEAT_DIM)  # 15360, 30720
ROW_BYTES = FEAT_DIM * 2  # bf16
NEEDFIT = os.path.expanduser("~/shared/needfit")
RESULTS = os.path.expanduser("~/shared/folderX/qbench45/results")

# Sink rows are produced by the in-engine tap, always. Anything trained
# alongside them must come from the same capture path (see load_seed).
SINK_CAPTURE_SOURCE = "tap"

EXIT_OK, EXIT_REJECT, EXIT_SKIP = 0, 2, 3

# ── censoring (BUG-139) ───────────────────────────────────────────────────
# SLACK = len(think_end_ids) + 8. The LIVE engine's `</think>` is ONE token
# (248069), so the live value is 9 — and the running router publishes it in
# each window's pn119_header (censor_slack), which is authoritative and wins
# over this default. The measured budget-stop offset is exactly 5, comfortably
# inside either value.
THINK_END_IDS_N = 1
CENSOR_SLACK = THINK_END_IDS_N + 8           # 9
# PN100 grant shapes. _continuous_budget rounds to 100 and clamps to
# [GENESIS_PN100_BUDGET_FLOOR, GENESIS_PN100_BUDGET_CEIL]; the tiered path
# grants one of GENESIS_PN100_TIER_BUDGETS, which are NOT on the 100-grid.
BUDGET_GRID = 100
BUDGET_OFF_GRID_GRANTS = (128, 1024, 4096, 10240)

# The accuracy monitor never feeds training. Asserted by the guard suite.
ACCURACY_IS_MONITOR_ONLY = True

# ── exploration (designed, DISABLED — see the module docstring) ────────────
EXPLORE_BOUNDARY_HALFWIDTH = 0.10   # |score - T| < this == "at the boundary"
EXPLORE_EPS_BOUNDARY = 0.25
EXPLORE_EPS_TAIL = 0.01
EXPLORE_IPS_CLIP = 50.0
EXPLORE_SALT = "pn119-explore"
HOLDOUT_SALT = "pn119-holdout"      # DIFFERENT salt: the holdout must not be
HOLDOUT_FRAC = 0.02                 # a subset of (or disjoint-by-luck from)
                                    # the explore stream.


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
    # ── BUG-139 sink schema (router-written when present, derived when not)
    feat_dim: int = FEAT_DIM       # this WINDOW's feature width (two eras)
    slack: int = CENSOR_SLACK      # this window's censor slack (header-borne)
    censored: object = None        # True / False / None(unknown -> derive)
    budget_grant: int | None = None
    budget_source: str | None = None
    # ── dual-probe shadow scoring (router-written; see the docstring)
    cand_score: float | None = None
    cand_sha: str | None = None
    # ── exploration bookkeeping (all None while PN119_EXPLORE=0.0)
    p_explore: float | None = None
    holdout: bool = False
    score: float | None = None     # the INCUMBENT's live score for this row
    x: np.ndarray | None = field(default=None, repr=False)


# ── feature I/O: memmap + exact bf16→f32 widen (no torch, no full read) ────
def bf16_rows(path: str, idx, feat_dim: int = FEAT_DIM) -> np.ndarray:
    """Materialise ONLY the requested row indices from a bf16 .bin.

    bf16 -> f32 is an exact 16-bit left shift (same exponent/mantissa
    layout), so numpy does it without torch and without touching the rest
    of the file: the memmap only faults in the pages we index.
    """
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    n_rows = mm.size // feat_dim
    out = np.empty((len(idx), feat_dim), dtype=np.float32)
    for k, i in enumerate(idx):
        if i >= n_rows:
            raise IndexError(f"{path}: row {i} beyond {n_rows}")
        raw = np.asarray(mm[i * feat_dim:(i + 1) * feat_dim], dtype=np.uint32)
        out[k] = (raw << np.uint32(16)).view(np.float32)
    del mm
    return out


def bin_row_count(path: str, feat_dim: int = FEAT_DIM) -> int:
    return os.path.getsize(path) // (feat_dim * 2)


def window_feat_dim(bin_path: str, need_rows: int, header: dict | None):
    """(feat_dim, how) for one sink window — header, then bytes, then None.

    `need_rows` is max(row index) + 1, not a count of meta lines: a meta line
    can outlive its feature write (a crash between the two appends), and a
    dedup drops lines without dropping .bin rows.

    A file written at 30720 is also evenly divisible at 15360, so "divides
    evenly" is not enough — the candidate whose row count EQUALS need_rows is
    the real one. A window consistent with neither is not guessed at: the
    caller drops it. Reinterpreting a feature file is the one failure mode
    that turns garbage into plausible numbers.
    """
    if header and header.get("feat_dim"):
        return int(header["feat_dim"]), "header"
    size = os.path.getsize(bin_path)
    divides = [d for d in KNOWN_FEAT_DIMS if size % (d * 2) == 0]
    exact = [d for d in divides if size // (d * 2) == need_rows]
    if len(exact) == 1:
        return exact[0], "bytes"
    fits = [d for d in divides if size // (d * 2) >= need_rows]
    if fits:
        # Prefer the WIDEST that still holds every referenced row: a genuinely
        # 15360-wide file cannot hold need_rows at 30720, so this can only pick
        # the narrow one when the wide one is impossible.
        return max(fits), "bytes"
    return None, "ambiguous"


# ── interval censoring (BUG-139) ──────────────────────────────────────────
def derive_budget_grant(rtok: int, grid: int = BUDGET_GRID,
                        off_grid=BUDGET_OFF_GRID_GRANTS) -> int:
    """Smallest PN100 grant that could have produced this rtok.

    PN100 hands out either `round(raw/100)*100` (the continuous path) or one
    of the tier budgets, so the plausible grants above an observed spend are
    the next 100-grid point and any tier budget >= rtok. The smallest of
    those is the only one that could have TRUNCATED this row — a larger
    grant would have left the row free to stop where it did.
    """
    rtok = max(int(rtok), 0)
    cands = [((rtok + grid - 1) // grid) * grid]
    cands += [int(g) for g in off_grid if g >= rtok]
    return min(cands)


def censoring_of(row_like, slack: int = CENSOR_SLACK, min_generated: int = 32):
    """(censored, lower_bound, provenance) for one row.

    `row_like` is a Row or a reservoir meta dict — anything with rtok /
    cap_hit / generated / censored / budget_grant.

    lower_bound is what the row PROVES about true need:
      * uncensored          -> rtok (an exact measurement)
      * censored w/ budget  -> budget - slack   (the pack's rule)
      * censored w/o budget -> rtok             (a max_tokens cap-hit: all we
                                                 know is that it wanted more)
    """
    def _get(k, default=None):
        if isinstance(row_like, dict):
            return row_like.get(k, default)
        return getattr(row_like, k, default)

    rtok = int(_get("rtok", 0) or 0)
    generated = _get("generated")
    cap_hit = bool(_get("cap_hit", False))
    censored = _get("censored")
    budget = _get("budget_grant")
    budget = None if budget in (None, "", 0) else int(budget)

    # max_tokens truncation is censoring too, and `censored` does not cover it:
    # the router's flag means "the THINKING BUDGET stopped it". Both must be
    # consulted or a completion-capped row reads as a natural stop.
    capped = cap_hit and (generated is None or int(generated) >= min_generated)

    if censored is not None:                       # the router said so
        if censored:
            lb = (budget - slack) if budget is not None else rtok
            return True, int(lb), "sink"
        return (True, rtok, "cap_hit") if capped else (False, rtok, "sink")
    if budget is not None:                         # budget known, flag isn't
        if rtok >= budget - slack:
            return True, budget - slack, "budget"
        return (True, rtok, "cap_hit") if capped else (False, rtok, "budget")
    if capped:
        return True, rtok, "cap_hit"               # max_tokens truncation
    grant = derive_budget_grant(rtok)              # older rows: derive it
    if rtok >= grant - slack:
        return True, grant - slack, "grid"
    return False, rtok, "uncensored"


def interval_label(row_like, deep_thresh: int, slack: int = CENSOR_SLACK,
                   min_generated: int = 32):
    """(y, weight, bucket). y is None for an unresolved interval.

    This is the whole BUG-139 fix. A censored row is a LOWER BOUND: it can
    only ever resolve to y=1, and only when the bound already clears theta.
    Below theta it resolves to nothing at all — never to y=0 (which is the
    absorbing state: "lean routed it, it stopped early, therefore lean was
    right") and never to y=1 (the old cap-hit rule, wrong the other way).
    """
    cens, lb, prov = censoring_of(row_like, slack, min_generated)
    if not cens:
        y = 1.0 if lb >= deep_thresh else 0.0
        return y, 1.0, ("resolved_pos" if y else "resolved_neg"), prov
    if lb >= deep_thresh:
        return 1.0, 1.0, "censored_pos", prov
    return None, 0.0, "interval_unresolved", prov


def label_rows(row_likes, deep_thresh: int, counts: dict | None = None,
               slack: int = CENSOR_SLACK, min_generated: int = 32):
    """Vectorised interval labelling. Returns (y, w, keep_mask) as arrays.

    y is 0.0 where unresolved; w is 0.0 there, and callers must select on w.
    """
    ys, ws, buckets, provs = [], [], [], []
    for r in row_likes:
        # The window's own header-declared slack wins: it is derived from the
        # think_end_ids the engine actually ran with.
        s = getattr(r, "slack", None) if not isinstance(r, dict) else r.get("slack")
        y, w, b, p = interval_label(r, deep_thresh, int(s or slack), min_generated)
        ys.append(0.0 if y is None else y)
        ws.append(w)
        buckets.append(b)
        provs.append(p)
    y = np.asarray(ys, dtype=float)
    w = np.asarray(ws, dtype=float)
    if counts is not None:
        for b in ("resolved_pos", "resolved_neg", "censored_pos",
                  "interval_unresolved"):
            counts[f"g3_{b}"] = int(sum(1 for x in buckets if x == b))
        for p in ("sink", "budget", "cap_hit", "grid", "uncensored"):
            n = int(sum(1 for x in provs if x == p))
            if n:
                counts[f"censor_src_{p}"] = n
        n = len(buckets)
        counts["g3_unresolved_rate"] = round(
            counts["g3_interval_unresolved"] / n, 4) if n else 0.0
        counts["censored_rate"] = round(
            sum(1 for x in provs if x != "uncensored") / n, 4) if n else 0.0
    return y, w, np.asarray(buckets), np.asarray(provs)


def materialise(rows: list[Row]) -> None:
    """Fill .x on the given rows, one memmap per sink window."""
    by_file: dict[tuple, list[Row]] = {}
    for r in rows:
        if r.x is None:
            by_file.setdefault((r.feat_path, r.feat_dim), []).append(r)
    for (path, dim), rs in by_file.items():
        X = bf16_rows(path, [r.row_idx for r in rs], dim)
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
    """`.synthetic-*.json` markers: {"tags": [...], "req_ids": [...]}.

    Capture tools that must hit the LIVE endpoint (and therefore the live
    sink) drop one of these naming the rows they caused, so their traffic
    is excluded even if the operator forgets --exclude-tag.

    Exclusion is BY REQ_ID ONLY. The marker's `tags` are provenance — the
    windows the rows landed in — and must never become tag exclusions: a
    live sink window holds the diagnostic's rows and the genuine traffic
    that was flowing at the same moment, and dropping the window throws the
    genuine rows away too (measured 2026-07-25: 5 real rows lost to one
    3-row marker). Whole-window exclusion stays an explicit operator
    decision, i.e. --exclude-tag.
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
        score_lines: dict[str, tuple[dict, int]] = {}
        finish_lines: dict[str, tuple[dict, int]] = {}
        header: dict | None = None
        for text, off in _iter_meta_lines(meta_p):
            try:
                m = json.loads(text)
            except json.JSONDecodeError:
                counts["bad_json"] = counts.get("bad_json", 0) + 1
                continue
            if m.get("pn119_header"):
                header = m
                continue
            if m.get("finish"):
                finish_lines[m["req_id"]] = (m, off)   # G4: last finish wins
            elif "row" in m:
                score_lines[m["req_id"]] = (m, off)
        need_rows = 1 + max((int(sm["row"]) for sm, _o in score_lines.values()),
                            default=-1)
        dim, how = window_feat_dim(feat_p, need_rows, header)
        if dim is None:
            counts["g9_feat_dim_ambiguous"] = counts.get("g9_feat_dim_ambiguous", 0) + 1
            continue
        counts[f"feat_dim_{dim}_via_{how}"] = counts.get(
            f"feat_dim_{dim}_via_{how}", 0) + 1
        w_slack = int((header or {}).get(
            "censor_slack",
            len((header or {}).get("think_end_ids") or []) + 8
            if (header or {}).get("think_end_ids") else CENSOR_SLACK))
        n_feat_rows = bin_row_count(feat_p, dim)
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
            # BUG-139 schema: the router writes these on the finish line; the
            # budget grant is also stamped on the score line (it is known at
            # prefill), so accept it from either.
            bg = fm.get("budget_grant", sm.get("budget_grant"))
            cens = fm.get("censored", sm.get("censored"))
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
                feat_dim=dim, slack=w_slack,
                censored=None if cens is None else bool(cens),
                budget_grant=None if bg in (None, "") else int(bg),
                budget_source=(str(fm.get("budget_source",
                                          sm.get("budget_source", "")))
                               or None),
                cand_score=(None if sm.get("cand_score") is None
                            else float(sm["cand_score"])),
                cand_sha=(str(sm["cand_probe_sha"])
                          if sm.get("cand_probe_sha") else None),
                p_explore=(None if sm.get("p_explore") is None
                           else float(sm["p_explore"])),
                holdout=bool(sm.get("holdout", False)),
                score=(None if sm.get("score") is None
                       else float(sm["score"])),
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
        # G8 — the exploration holdout never trains, and there is no flag to
        # override that. A holdout you can switch off is a holdout you will
        # switch off the first time a gate is inconvenient.
        if r.holdout:
            counts["g8_holdout_excluded"] = counts.get("g8_holdout_excluded", 0) + 1
            continue
        # G2 — self-censoring is now INTERVAL CENSORING (BUG-139), applied at
        # label time by interval_label(), not by deletion here. Deleting the
        # censored rows kept 8 of 31 live rows and every survivor was
        # deep-routed; the loop was training on its own routing decisions
        # through the selection, not through the labels.
        if r.mode == "enforce" and r.route == "lean" and not r.explore:
            counts["g2_selfcensored_kept"] = counts.get("g2_selfcensored_kept", 0) + 1
        out.append(r)
    counts["eligible"] = len(out)
    return out


def select_feat_dim(rows: list[Row], counts: dict, want: int | None = None):
    """(dim, rows_in_that_dim). Two feature ERAS cannot share a training set.

    Same argument as the tap-vs-HF capture refusal: 15360 last-only and 30720
    last+mean are different spaces, and a probe fitted across both is fitted on
    neither. When the router changes era mid-sink the older windows drop out —
    counted and printed, never silently averaged in.
    """
    if not rows:
        return (want or FEAT_DIM), rows
    dim = want or max(rows, key=lambda r: r.ts).feat_dim
    keep = [r for r in rows if r.feat_dim == dim]
    dropped = len(rows) - len(keep)
    counts["feat_dim"] = int(dim)
    if dropped:
        counts["g9_feat_dim_mismatch"] = dropped
        counts["g9_feat_dims_seen"] = sorted({int(r.feat_dim) for r in rows})
    return dim, keep


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


# ── statistics the gates need (no scipy in lens-venv) ─────────────────────
def norm_sf(z: float) -> float:
    """P(Z > z) for a standard normal, via erfc — no scipy."""
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def _midrank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (the T of Sun & Xu's fast DeLong)."""
    J = np.argsort(x, kind="mergesort")
    Z = np.asarray(x, dtype=float)[J]
    n = len(x)
    T = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j + 1)
        i = j
    out = np.empty(n, dtype=float)
    out[J] = T
    return out


def delong_test(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray):
    """PAIRED AUC comparison (fast DeLong). Returns (auc_a, auc_b, z, p_one).

    Paired matters: both probes scored the SAME requests, so their AUC
    estimates are strongly correlated. An unpaired comparison throws that
    covariance away and needs several times the sample to see the same
    difference. p_one is one-sided for H1: auc_a > auc_b.
    """
    y = np.asarray(y, dtype=float)
    S = np.vstack([np.asarray(s_a, float), np.asarray(s_b, float)])
    pos, neg = S[:, y == 1], S[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    tx = np.vstack([_midrank(pos[k]) for k in range(2)])
    ty = np.vstack([_midrank(neg[k]) for k in range(2)])
    tz = np.vstack([_midrank(np.concatenate([pos[k], neg[k]]))
                    for k in range(2)])
    aucs = (tz[:, :m].sum(1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01) if m > 1 else np.zeros((2, 2))
    s10 = np.cov(v10) if n > 1 else np.zeros((2, 2))
    cov = np.asarray(s01, float) / m + np.asarray(s10, float) / n
    L = np.array([1.0, -1.0])
    var = float(L @ cov @ L)
    d = float(aucs[0] - aucs[1])
    if var <= 0:
        # Identical scores (or a degenerate sample): no evidence either way.
        return float(aucs[0]), float(aucs[1]), 0.0, 0.5
    z = d / math.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(z), float(norm_sf(z))


def psi(ref: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of `new` against `ref` on ref's deciles."""
    ref = np.asarray(ref, float)
    new = np.asarray(new, float)
    if len(ref) < bins or len(new) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, bins=edges)[0] / float(len(ref))
    q = np.histogram(new, bins=edges)[0] / float(len(new))
    eps = 1e-4
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def ks_2samp(a: np.ndarray, b: np.ndarray):
    """Two-sample Kolmogorov-Smirnov (D, p) — asymptotic p, no scipy."""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    allv = np.concatenate([a, b])
    cdf1 = np.searchsorted(a, allv, side="right") / n1
    cdf2 = np.searchsorted(b, allv, side="right") / n2
    d = float(np.max(np.abs(cdf1 - cdf2)))
    en = math.sqrt(n1 * n2 / float(n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2.0 * sum((-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam)
                  for j in range(1, 101))
    return d, float(min(1.0, max(0.0, p)))


# ── exploration policy (DESIGNED, DISABLED — PN119_EXPLORE stays 0.0) ─────
def _u01(req_id: str, salt: str) -> float:
    """Deterministic uniform(0,1) from a req_id under a named salt."""
    h = hashlib.sha256(f"{salt}:{req_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def explore_eps(score: float, tdeep: float,
                halfwidth: float = EXPLORE_BOUNDARY_HALFWIDTH,
                eps_boundary: float = EXPLORE_EPS_BOUNDARY,
                eps_tail: float = EXPLORE_EPS_TAIL) -> float:
    """Stratified epsilon: spend the exploration where the decision is."""
    return eps_boundary if abs(float(score) - float(tdeep)) < halfwidth else eps_tail


def explore_decision(req_id: str, score: float, tdeep: float, bucket=None,
                     **kw):
    """(explore, p_explore). Deterministic per req_id — the sink row and the
    consumer can never disagree about whether a request was an explore row.

    p_explore is the PROPENSITY and must be logged: without it the explore
    rows are a biased sample with no way to unbias them (see ips_weight).
    A token bucket, when supplied, caps the cost: a request that would have
    explored but finds the bucket empty is NOT an explore row, and its
    propensity is 0 (it must then be dropped from the IPS estimate, not
    reweighted).
    """
    eps = explore_eps(score, tdeep, **kw)
    fires = _u01(req_id, EXPLORE_SALT) < eps
    if fires and bucket is not None and not bucket.take():
        return False, 0.0
    return bool(fires), float(eps)


def explore_effective_rate(scores, tdeep: float,
                           halfwidth: float = EXPLORE_BOUNDARY_HALFWIDTH,
                           eps_boundary: float = EXPLORE_EPS_BOUNDARY,
                           eps_tail: float = EXPLORE_EPS_TAIL) -> float:
    """What the stratified policy actually costs on a given score sample."""
    s = np.asarray(scores, float)
    if not len(s):
        return float("nan")
    near = float((np.abs(s - float(tdeep)) < halfwidth).mean())
    return near * eps_boundary + (1.0 - near) * eps_tail


def explore_halfwidth_for_rate(scores, tdeep: float, target: float,
                               eps_boundary: float = EXPLORE_EPS_BOUNDARY,
                               eps_tail: float = EXPLORE_EPS_TAIL) -> float:
    """The boundary halfwidth that hits a target effective rate HERE.

    The pack's 0.10 was sized against an assumed score spread. The live probe's
    scores are far tighter than that (measured 2026-07-25: sd 0.117, and 62% of
    all scored requests sit within 0.10 of PN119_TDEEP), so 0.10 is not a
    boundary band, it is most of the traffic. Whoever turns exploration on must
    size the halfwidth against the CURRENT score distribution, not a constant.
    """
    s = np.asarray(scores, float)
    if not len(s) or eps_boundary <= eps_tail:
        return float("nan")
    near = (float(target) - eps_tail) / (eps_boundary - eps_tail)
    if not (0.0 < near < 1.0):
        return float("nan")
    return float(np.quantile(np.abs(s - float(tdeep)), near))


def explore_budget_for(deep_thresh: int, routed_budget: int | None,
                       margin: int = 400) -> int:
    """PN119_EXPLORE_BUDGET = max(theta + 400, routed_budget).

    An explore row exists to produce an UNCENSORED label, which it can only
    do if its budget can carry it past theta with room to stop naturally.
    Never smaller than what the row would have been granted anyway.
    """
    return int(max(deep_thresh + margin, routed_budget or 0))


def is_holdout(req_id: str, frac: float = HOLDOUT_FRAC) -> bool:
    """2% holdout on a DIFFERENT salt from the explore stream, so the two are
    independent rather than nested. Excluded from training by G8; there is
    deliberately no flag to include it."""
    return _u01(req_id, HOLDOUT_SALT) < frac


def ips_weight(p_explore: float, clip: float = EXPLORE_IPS_CLIP) -> float:
    """1/p, clipped. Unclipped IPS on eps_tail=0.01 gives single rows a
    weight of 100 — one lucky draw would own the fit."""
    p = float(p_explore or 0.0)
    if p <= 0.0:
        return 0.0
    return float(min(1.0 / p, clip))


class TokenBucket:
    """Hard ceiling on explore spend: `rate` tokens/hour, `capacity` burst."""

    def __init__(self, rate_per_hour: float, capacity: float, now: float = 0.0):
        self.rate = float(rate_per_hour) / 3600.0
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.t = float(now)

    def take(self, now: float | None = None, n: float = 1.0) -> bool:
        now = self.t if now is None else float(now)
        self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


def explore_enabled(env=None) -> bool:
    """Exploration is OFF unless BOTH the rate is positive AND the consumer
    has declared that it honours PN119_EXPLORE_BUDGET.

    Without the second half an explore row is an ordinary censored row with a
    trusted-label badge on it — strictly worse than no exploration at all,
    because the refit believes it.
    """
    env = os.environ if env is None else env
    try:
        rate = float(env.get("PN119_EXPLORE", "0") or 0.0)
    except ValueError:
        return False
    honoured = str(env.get("PN119_EXPLORE_BUDGET_HONOURED", "0")).strip() in (
        "1", "true", "True", "yes")
    return rate > 0.0 and honoured


def explore_integrity(rows, deep_thresh: int, slack: int = CENSOR_SLACK):
    """Violations that make explore rows untrustworthy. Empty == clean.

    An explore row must (a) carry its logged propensity and (b) have been
    granted a theta-sufficient budget. A censored explore row below theta is
    the exact failure mode the whole design exists to prevent.
    """
    bad_prop = bad_budget = censored_short = 0
    n = 0
    for r in rows:
        if not getattr(r, "explore", False):
            continue
        n += 1
        p = getattr(r, "p_explore", None)
        if p is None or not (0.0 < float(p) <= 1.0):
            bad_prop += 1
        bg = getattr(r, "budget_grant", None)
        if bg is not None and int(bg) < explore_budget_for(deep_thresh, None):
            bad_budget += 1
        cens, lb, _p = censoring_of(r, slack)
        if cens and lb < deep_thresh:
            censored_short += 1
    out = {}
    if n:
        out["explore_rows"] = n
    if bad_prop:
        out["explore_missing_propensity"] = bad_prop
    if bad_budget:
        out["explore_budget_below_theta"] = bad_budget
    if censored_short:
        out["explore_censored_below_theta"] = censored_short
    return out


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


def b3_report_state(path: str, max_age_h: float, now: float):
    """(ok, note) for the b3 numerics report — the probe's arithmetic check.

    A refit that runs while b3 is failing (or has not run since the last
    engine change) is a refit whose feature pipeline is unverified. Missing
    counts as failing: "no evidence" is not "no problem".
    """
    if not os.path.isfile(path):
        return False, f"b3 report missing ({path})"
    try:
        with open(path, encoding="utf-8") as f:
            rep = json.load(f) or {}
    except (OSError, json.JSONDecodeError) as e:
        return False, f"b3 report unreadable ({e})"
    age_h = (now - float(rep.get("ts", 0.0) or 0.0)) / 3600.0
    if not bool(rep.get("pass", False)):
        return False, f"b3 report says pass=false (age {age_h:.1f}h)"
    if age_h > max_age_h:
        return False, f"b3 report is stale ({age_h:.1f}h > {max_age_h}h)"
    return True, f"b3 pass, age {age_h:.1f}h"


def gate_drift(P_ref, P_new, prior, unresolved_rate, b3_ok, b3_note,
               explore_bad: dict, report: dict, max_psi=0.25, ks_p=0.01,
               ks_d=0.15, prior_lo=0.15, prior_hi=0.55, max_unresolved=0.40,
               min_side=50):
    """Reject a candidate whose INPUTS have moved, not just its outputs.

    P_ref / P_new are PC projections (rows x pcs) of the reference (older)
    and recent halves of the training set. Every other gate here scores the
    candidate on the same distribution it was fitted on, so a corpus that has
    shifted underneath the loop passes them all — which is precisely what the
    2026-07-25 poisoning did.
    """
    fails = []
    P_ref = np.asarray(P_ref, float)
    P_new = np.asarray(P_new, float)
    # Below min_side, PSI is measuring empty bins: with 10 reference deciles
    # and 8 new rows most bins are empty and the clipped log-ratio invents a
    # PSI of several units. A drift gate that fires on sample size is a drift
    # gate that gets switched off.
    if (P_ref.ndim == 2 and P_new.ndim == 2
            and min(len(P_ref), len(P_new)) >= min_side):
        psis, ks_stats = [], []
        for c in range(P_ref.shape[1]):
            psis.append(psi(P_ref[:, c], P_new[:, c]))
            ks_stats.append(ks_2samp(P_ref[:, c], P_new[:, c]))
        report["psi_max"] = round(float(np.max(psis)), 4)
        report["psi_argmax_pc"] = int(np.argmax(psis))
        worst = int(np.argmax([d for d, _p in ks_stats]))
        report["ks_d_max"] = round(float(ks_stats[worst][0]), 4)
        report["ks_p_at_dmax"] = float(f"{ks_stats[worst][1]:.3e}")
        if report["psi_max"] > max_psi:
            fails.append(f"feature drift: PSI {report['psi_max']:.3f} > {max_psi} "
                         f"on PC{report['psi_argmax_pc']}")
        for c, (d, p) in enumerate(ks_stats):
            if p < ks_p and d > ks_d:
                fails.append(f"feature drift: KS on PC{c} D={d:.3f} p={p:.2e} "
                             f"(reject at p<{ks_p} & D>{ks_d})")
                break
    else:
        report["psi_max"] = None
        report["drift_note"] = (
            f"feature drift not evaluable (fit={len(P_ref)} recent={len(P_new)}, "
            f"need >={min_side} a side)")
    report["train_prior"] = None if prior is None else round(float(prior), 4)
    if prior is not None and not (prior_lo <= float(prior) <= prior_hi):
        fails.append(f"training prior {float(prior):.4f} outside "
                     f"[{prior_lo}, {prior_hi}] — the label distribution is not "
                     f"the traffic's")
    report["unresolved_rate"] = (None if unresolved_rate is None
                                 else round(float(unresolved_rate), 4))
    if unresolved_rate is not None and float(unresolved_rate) > max_unresolved:
        fails.append(f"interval-unresolved rate {float(unresolved_rate):.4f} > "
                     f"{max_unresolved} — most of the corpus is a lower bound")
    report["b3_ok"] = bool(b3_ok)
    report["b3_note"] = b3_note
    if not b3_ok:
        fails.append(f"b3 numerics gate: {b3_note}")
    if explore_bad:
        report["explore_integrity"] = explore_bad
        bad = {k: v for k, v in explore_bad.items() if k != "explore_rows"}
        if bad:
            fails.append(f"explore integrity violated: {json.dumps(bad)}")
    return fails


def gate_shadow_promotion(y, w, s_cand, s_inc, tdeep, alpha, max_flip_frac,
                          min_n, report: dict):
    """Promote the candidate only if it BEAT the incumbent out of sample.

    Every argument is measured on rows the candidate scored in shadow AFTER
    it went live — the candidate never routed any of them, so this is a true
    out-of-sample comparison on the traffic that matters, not a re-scoring of
    the corpus it was fitted on.
    """
    keep = np.asarray(w, float) > 0
    y = np.asarray(y, float)[keep]
    a = np.asarray(s_cand, float)[keep]
    b = np.asarray(s_inc, float)[keep]
    report["shadow_n"] = int(len(y))
    if len(y) < min_n:
        report["shadow_note"] = f"only {len(y)} resolved shadow rows (< {min_n})"
        return [f"not enough post-shadow evidence: {len(y)} resolved rows < {min_n}"]
    if y.min() == y.max():
        report["shadow_note"] = f"single-class shadow window (prior {y.mean():.3f})"
        return ["shadow window is single-class — AUC undefined"]
    auc_a, auc_b, z, p = delong_test(y, a, b)
    report.update(shadow_auc_cand=round(auc_a, 4), shadow_auc_inc=round(auc_b, 4),
                  delong_z=round(z, 4), delong_p_one_sided=float(f"{p:.3e}"))
    flip = float((((a >= tdeep).astype(int) != (b >= tdeep).astype(int))).mean())
    report["decision_flip_frac"] = round(flip, 4)
    fails = []
    if flip > max_flip_frac:
        fails.append(f"decision-flip fraction {flip:.4f} > {max_flip_frac}: this "
                     f"is a DIFFERENT router, not a better probe — needs a human")
    if not (auc_a > auc_b):
        fails.append(f"candidate AUC {auc_a:.4f} does not beat incumbent {auc_b:.4f}")
    elif p >= alpha:
        fails.append(f"candidate AUC {auc_a:.4f} vs {auc_b:.4f} is not significant "
                     f"(paired DeLong one-sided p={p:.3g} >= {alpha})")
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


class FeatureSpaceMismatch(RuntimeError):
    """The seed and the sink are not in the same feature space."""


def load_seed(deep_thresh: int, features: str, champion: str,
              slack: int = CENSOR_SLACK, counts: dict | None = None,
              feat_dim: int | None = None):
    """Seed set (v1 continuity anchor), captured through the SAME tap the
    sink is written from — see seed_capture_source.

    The seed gets the SAME interval labels as the sink. Its `budget` column
    is the CLIENT's (None on the auto arm), so PN100's grant is unknown and
    the grid derivation is the only censoring signal available — which is
    exactly the sink's situation for pre-schema rows. Labelling the seed by
    a different rule than the traffic would put a systematic offset between
    the fit side and the holdout side of every gate.
    """
    import torch                                # noqa: PLC0415 — bf16 needs it
    from safetensors.torch import load_file     # noqa: PLC0415
    st = load_file(features)                    # bf16 — numpy cannot hold it
    champ = {}
    with open(champion, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            champ[rec["item_id"]] = {
                "rtok": rec.get("reasoning_tokens") or 0,
                "generated": rec.get("completion_tokens"),
                "cap_hit": not bool(rec.get("think_closed", True)),
                "budget_grant": rec.get("budget"),
                "censored": None,
            }
    ids = [i for i in sorted(champ) if i in st]
    X = np.stack([st[i].to(torch.float32).flatten().numpy() for i in ids])
    if feat_dim is not None and X.shape[1] != int(feat_dim):
        raise FeatureSpaceMismatch(
            f"seed capture {os.path.basename(features)} is {X.shape[1]}-dim "
            f"({st[ids[0]].shape[0]} blocks) but the sink's live windows are "
            f"{feat_dim}-dim. These are different feature spaces (last+mean vs "
            f"last-only), not different scalings — re-capture the seed through "
            f"the current tap, or pass --no-seed.")
    y, w, _b, _p = label_rows([champ[i] for i in ids], deep_thresh,
                              counts if counts is not None else None, slack)
    return ids, X, y, w


# ── reservoir ──────────────────────────────────────────────────────────────
def update_reservoir(state_dir: str, rows: list[Row], cap: int,
                     rng: np.random.Generator, counts: dict,
                     persist: bool = True, seen_cap: int = 200_000,
                     feat_dim: int = FEAT_DIM):
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
    R_X, R_meta, seen_total, seen_ids = [], [], 0, []
    if os.path.isfile(res_path):
        z = np.load(res_path, allow_pickle=True)
        X0 = np.asarray(z["X"])
        held_dim = (int(z["feat_dim"]) if "feat_dim" in z
                    else (int(X0.shape[1]) if X0.ndim == 2 and X0.shape[1] else feat_dim))
        if held_dim != feat_dim:
            # The stored features are from the other feature ERA. They cannot
            # be stacked with the new ones and they cannot be converted, so the
            # reservoir starts over rather than pretending.
            counts["reservoir_reset_feat_dim"] = held_dim
        else:
            R_X = list(X0)
            R_meta = [json.loads(s) for s in z["meta"]]
            seen_total = int(z["seen_total"])
            seen_ids = ([str(s) for s in z["seen_req_ids"]]
                        if "seen_req_ids" in z else [])
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
                        "explore": r.explore, "ts": r.ts, "tag": r.tag,
                        # BUG-139: the reservoir outlives the sink windows, so
                        # the censoring evidence has to travel WITH the row.
                        # A meta dict written before this carries none, and
                        # censoring_of() falls back to the grid derivation.
                        "censored": r.censored, "budget_grant": r.budget_grant,
                        "budget_source": r.budget_source,
                        "score": r.score, "cand_score": r.cand_score,
                        "cand_sha": r.cand_sha, "p_explore": r.p_explore}
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
            "X": np.stack(R_X) if R_X else np.zeros((0, feat_dim), dtype=np.float16),
            "meta": np.array([json.dumps(m) for m in R_meta]),
            "seen_total": np.array(seen_total),
            "seen_req_ids": np.array(seen_out),
            "feat_dim": np.array(feat_dim),
        })
    return R_X, R_meta


# ── accuracy monitor (join-only; NEVER a training target) ─────────────────
def join_key(req_id: str) -> str:
    """Sink req_id -> graded-run extra.resp_id (drop the last `-` segment)."""
    return str(req_id).rsplit("-", 1)[0]


def load_graded(results_dir: str, pattern: str = "*.jsonl") -> dict:
    """{resp_id: {"correct", "item_id", "run_id", "reasoning_tokens"}}."""
    out: dict = {}
    for p in sorted(glob.glob(os.path.join(results_dir, pattern))):
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        m = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rid = (m.get("extra") or {}).get("resp_id")
                    if not rid:
                        continue
                    out[str(rid)] = {
                        "correct": bool(m.get("correct", False)),
                        "item_id": m.get("item_id"),
                        "run_id": m.get("run_id"),
                        "reasoning_tokens": m.get("reasoning_tokens"),
                    }
        except OSError:
            continue
    return out


def accuracy_monitor(rows, results_dir: str, tdeep: float, state_dir: str,
                     report: dict, alert_drop: float = 0.10, min_n: int = 20,
                     persist: bool = True) -> list[str]:
    """Deep-bucket accuracy regression ALERTS. Returns alert strings.

    This is free: the graded runs already exist and `extra.resp_id` joins the
    sink's req_id with its last `-` segment stripped (measured 339/339 over
    12 runs). It is a MONITOR, never a gate on training and never a label:
    AUC(score -> correct) sits at 0.26-0.40 — correctly BELOW 0.5, because
    the lens finds HARD items, not wrong ones. Training on `correct` would
    teach the probe to route the easy questions deep.
    """
    graded = load_graded(results_dir)
    pairs = [(r, graded[join_key(r.req_id)]) for r in rows
             if join_key(r.req_id) in graded and r.score is not None]
    report["acc_join_n"] = len(pairs)
    report["acc_graded_n"] = len(graded)
    if not pairs:
        report["acc_note"] = "no sink row joins a graded run"
        return []
    sc = np.array([float(r.score) for r, _g in pairs])
    ok = np.array([1.0 if g["correct"] else 0.0 for _r, g in pairs])
    deep = sc >= float(tdeep)
    report["acc_auc_score_vs_correct"] = round(auc(sc, ok), 4)
    report["acc_overall"] = round(float(ok.mean()), 4)
    report["acc_deep_n"] = int(deep.sum())
    report["acc_lean_n"] = int((~deep).sum())
    report["acc_deep"] = (round(float(ok[deep].mean()), 4)
                          if deep.any() else None)
    report["acc_lean"] = (round(float(ok[~deep].mean()), 4)
                          if (~deep).any() else None)
    base_p = os.path.join(state_dir, "accuracy-baseline.json")
    base = {}
    if os.path.isfile(base_p):
        try:
            with open(base_p, encoding="utf-8") as f:
                base = json.load(f) or {}
        except (OSError, json.JSONDecodeError):
            base = {}
    alerts = []
    prev = base.get("acc_deep")
    if (prev is not None and report["acc_deep"] is not None
            and report["acc_deep_n"] >= min_n
            and report["acc_deep"] < float(prev) - alert_drop):
        alerts.append(f"deep-bucket accuracy {report['acc_deep']:.3f} is "
                      f"{float(prev) - report['acc_deep']:.3f} below the "
                      f"baseline {float(prev):.3f} (n={report['acc_deep_n']})")
    report["acc_baseline"] = prev
    if persist and report["acc_deep"] is not None and report["acc_deep_n"] >= min_n:
        if prev is None or report["acc_deep"] > float(prev):
            with open(base_p, "w", encoding="utf-8") as f:
                json.dump({"acc_deep": report["acc_deep"],
                           "acc_deep_n": report["acc_deep_n"],
                           "ts": time.time()}, f)
    return alerts


# ── dual-probe candidate state ────────────────────────────────────────────
def candidate_path(state_dir: str) -> str:
    return os.path.join(state_dir, "candidate.npz")


def load_candidate(state_dir: str) -> dict | None:
    p = os.path.join(state_dir, "candidate.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def npz_sha16(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def stage_candidate(state_dir: str, payload: dict, note: str = "") -> dict:
    """Write candidate.npz + candidate.json. The engine is expected to load
    it via PN119_PROBE_CANDIDATE and SCORE with it — never route on it."""
    cp = candidate_path(state_dir)
    atomic_write_npz(cp, payload)
    rec = {"sha256_16": npz_sha16(cp), "path": cp, "staged_ts": time.time(),
           "first_seen_ts": None, "note": note}
    with open(os.path.join(state_dir, "candidate.json"), "w",
              encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return rec


def note_candidate_live(state_dir: str, rec: dict, rows) -> dict:
    """Stamp first_seen_ts the first time the sink shows the router actually
    scoring with this candidate. Until that happens there is no shadow
    evidence and promotion cannot even be evaluated."""
    if rec.get("first_seen_ts"):
        return rec
    ts = [r.ts for r in rows
          if r.cand_score is not None and r.cand_sha == rec.get("sha256_16")]
    if not ts:
        return rec
    rec["first_seen_ts"] = float(min(ts))
    with open(os.path.join(state_dir, "candidate.json"), "w",
              encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return rec


def promote_candidate(out: str, cand_npz: str, prev: str) -> str:
    """live -> probe.prev.npz, candidate -> live. Both atomic; the previous
    probe survives so a promotion can be undone without a refit."""
    if os.path.isfile(out):
        shutil.copy2(out, prev + ".tmp")
        os.replace(prev + ".tmp", prev)
    z = np.load(cand_npz, allow_pickle=True)
    atomic_write_npz(out, {k: z[k] for k in z.files})
    return npz_sha16(out)


def rollback_probe(out: str, prev: str) -> str:
    """Swap BACK. The current live probe becomes the new .prev, so a rollback
    is itself reversible (a bad rollback is as likely as a bad promotion)."""
    if not os.path.isfile(prev):
        raise FileNotFoundError(prev)
    z = np.load(prev, allow_pickle=True)
    payload = {k: z[k] for k in z.files}
    if os.path.isfile(out):
        shutil.copy2(out, prev + ".tmp")
    atomic_write_npz(out, payload)
    if os.path.isfile(prev + ".tmp"):
        os.replace(prev + ".tmp", prev)
    return npz_sha16(out)


def _nan(v):
    return float("nan") if v is None else float(v)


def incumbent_feat_dim(path: str) -> int | None:
    if not os.path.isfile(path):
        return None
    try:
        z = np.load(path, allow_pickle=True)
        return int(np.asarray(z["mu"]).shape[0])
    except (OSError, KeyError, ValueError, IndexError):
        return None


def load_incumbent(path: str, feat_dim: int | None = None):
    """(probe, note). A probe in another feature space is NOT the incumbent.

    The live probe is a folded (mu, sd, Vt, w) in a fixed feature width. Scoring
    30720-dim sink rows with a 15360-dim probe is not a degraded comparison, it
    is a shape error — and until this returned None it was an uncaught
    ValueError in the middle of the gate block.
    """
    if not os.path.isfile(path):
        return None, "no incumbent probe on disk"
    try:
        z = np.load(path, allow_pickle=True)
        mu = np.asarray(z["mu"])
        if feat_dim is not None and mu.shape[0] != int(feat_dim):
            return None, (f"incumbent probe is {mu.shape[0]}-dim but the sink is "
                          f"{feat_dim}-dim — different feature spaces, so the "
                          f"stability and calibration comparisons ABSTAIN")
        return (mu, np.asarray(z["sd"]), np.asarray(z["Vt10"]),
                np.asarray(z["w"])), "ok"
    except (OSError, KeyError, ValueError) as e:
        return None, f"incumbent probe unreadable ({e})"


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
    # ── BUG-139 interval censoring
    ap.add_argument("--allow-feat-dim-change", action="store_true",
                    help="permit swapping in a probe whose feature width "
                         "differs from the live one (a real migration)")
    ap.add_argument("--feat-dim", type=int, default=None,
                    help="pin the sink feature width (15360 last-only / 30720 "
                         "last+mean); default = the newest window's")
    ap.add_argument("--censor-slack", type=int, default=CENSOR_SLACK,
                    help="rtok >= budget - SLACK => the row was budget-truncated "
                         "(len(think_end_ids)+8; the measured offset is 5)")
    ap.add_argument("--max-unresolved-frac", type=float, default=0.40,
                    help="drift gate: reject when more than this fraction of the "
                         "corpus is an unresolved censored interval")
    # ── dual-probe shadow promotion
    ap.add_argument("--promote-mode", choices=("shadow", "direct"),
                    default="shadow",
                    help="shadow (default): stage the candidate for "
                         "PN119_PROBE_CANDIDATE and promote only on paired "
                         "out-of-sample evidence. direct: fit->gates->swap "
                         "(pre-shadow behaviour; needs an explicit reason)")
    ap.add_argument("--delong-alpha", type=float, default=0.05,
                    help="one-sided paired DeLong significance for promotion")
    ap.add_argument("--max-flip-frac", type=float, default=0.20,
                    help="more than this fraction of routing decisions flipped "
                         "makes it a different router — promotion needs a human")
    ap.add_argument("--min-shadow-n", type=int, default=200,
                    help="resolved post-shadow rows required before the "
                         "promotion test is even attempted")
    ap.add_argument("--rollback", action="store_true",
                    help="restore probe.prev.npz over the live probe and exit")
    # ── drift gates
    ap.add_argument("--max-psi", type=float, default=0.25)
    ap.add_argument("--ks-p", type=float, default=0.01)
    ap.add_argument("--ks-d", type=float, default=0.15)
    ap.add_argument("--prior-lo", type=float, default=0.15)
    ap.add_argument("--prior-hi", type=float, default=0.55)
    ap.add_argument("--b3-report", default=f"{NEEDFIT}/pn119-b3-numerics-report.json")
    ap.add_argument("--b3-max-age-h", type=float, default=24.0)
    ap.add_argument("--skip-drift-gate", action="store_true",
                    help="report drift but do not reject on it (bootstrap only)")
    # ── accuracy monitor (never a training target)
    ap.add_argument("--results-dir", default=RESULTS,
                    help="graded run JSONLs for the accuracy monitor join")
    ap.add_argument("--acc-alert-drop", type=float, default=0.10)
    ap.add_argument("--accuracy-monitor", action="store_true",
                    help="run ONLY the accuracy monitor join and exit")
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

    prev_p = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                          "probe.prev.npz")
    if args.rollback:
        try:
            sha = rollback_probe(args.out, prev_p)
        except FileNotFoundError:
            print(f"[refit] ROLLBACK IMPOSSIBLE: no {prev_p} — the live probe "
                  f"has never been promoted over another one.")
            return EXIT_REJECT
        print(f"[refit] ROLLED BACK {args.out} <- {prev_p} sha256[:16]={sha} "
              f"(the probe it replaced is now the new .prev, so this is "
              f"itself reversible)")
        return EXIT_OK

    marker_ids, marker_tags = load_markers(args.sink)
    counts["marker_req_ids"] = len(marker_ids)
    counts["marker_tags_seen"] = len(marker_tags)   # provenance, NOT exclusions
    exclude_tags = list(args.exclude_tag)
    t_parse = time.time()
    raw, file_sizes = load_sink(args.sink, counts, exclude_tags, marker_ids)
    counts["parse_s"] = round(time.time() - t_parse, 2)

    if args.accuracy_monitor:
        # Monitor-only mode: the join is free and needs none of the training
        # machinery. `correct` never leaves this branch.
        acc: dict = {}
        alerts = accuracy_monitor(raw, args.results_dir, args.live_tdeep,
                                  args.state, acc, args.acc_alert_drop,
                                  persist=not args.dry_run)
        for a in alerts:
            print(f"[refit] ACCURACY ALERT {a}")
        print(f"[refit] accuracy-monitor {json.dumps(acc)}")
        return EXIT_REJECT if alerts else EXIT_OK

    rows = apply_guards(raw, args.legacy_thinking_ok, args.min_generated, counts)
    feat_dim, rows = select_feat_dim(rows, counts, args.feat_dim)
    if counts.get("g9_feat_dim_mismatch"):
        print(f"[refit] WARN: the sink spans {counts['g9_feat_dims_seen']} "
              f"feature widths; training on the newest ({feat_dim}) and "
              f"dropping {counts['g9_feat_dim_mismatch']} older rows. A probe "
              f"fitted across both eras is fitted on neither.")
    _live_dim = incumbent_feat_dim(args.out)
    if _live_dim is not None and _live_dim != feat_dim:
        print(f"[refit] WARN: live probe is {_live_dim}-dim, the sink's newest "
              f"windows are {feat_dim}-dim. Gates that compare against the "
              f"incumbent will ABSTAIN and no swap is possible until the two "
              f"agree (--allow-feat-dim-change to force).")

    # ── dual-probe promotion. A registered candidate OWNS the loop until it
    # is promoted or rejected: fitting a second candidate while the first is
    # still collecting shadow evidence would throw that evidence away.
    cand_rec = load_candidate(args.state)
    if cand_rec and args.promote_mode == "shadow":
        cand_rec = note_candidate_live(args.state, cand_rec, rows)
        first = cand_rec.get("first_seen_ts")
        shadow = [r for r in rows
                  if r.cand_score is not None
                  and r.cand_sha == cand_rec.get("sha256_16")
                  and r.score is not None
                  and (first is None or r.ts >= float(first))]
        pg: dict = {"candidate_sha": cand_rec.get("sha256_16"),
                    "candidate_staged_ts": cand_rec.get("staged_ts"),
                    "candidate_first_seen_ts": first}
        y_s, w_s, _b, _p = label_rows(shadow, args.deep_thresh, pg,
                                      args.censor_slack, args.min_generated)
        fails = gate_shadow_promotion(
            y_s, w_s, [r.cand_score for r in shadow], [r.score for r in shadow],
            args.live_tdeep, args.delong_alpha, args.max_flip_frac,
            args.min_shadow_n, pg)
        if fails and pg.get("shadow_n", 0) < args.min_shadow_n:
            print(f"[refit] HOLD: candidate {cand_rec.get('sha256_16')} is in "
                  f"shadow, {pg.get('shadow_n', 0)} resolved rows so far "
                  f"(need {args.min_shadow_n}). {json.dumps(pg)}")
            return EXIT_SKIP
        if fails:
            print(f"[refit] PROMOTION REFUSED ({'; '.join(fails)}) — live probe "
                  f"untouched, candidate left in shadow.\n[refit] {json.dumps(pg)}")
            return EXIT_REJECT
        if args.dry_run:
            print(f"[refit] DRY-RUN: would promote {cand_rec.get('sha256_16')} "
                  f"{json.dumps(pg)}")
            return EXIT_OK
        sha = promote_candidate(args.out, cand_rec["path"], prev_p)
        os.replace(os.path.join(args.state, "candidate.json"),
                   os.path.join(args.state,
                                f"promoted-{time.strftime('%Y%m%d-%H%M%S')}.json"))
        print(f"[refit] PROMOTED {args.out} sha256[:16]={sha} "
              f"auc {pg['shadow_auc_cand']} vs {pg['shadow_auc_inc']} "
              f"(paired DeLong p={pg['delong_p_one_sided']}, n={pg['shadow_n']}, "
              f"flips={pg['decision_flip_frac']}); previous probe kept at "
              f"{prev_p}\n[refit] {json.dumps(pg)}")
        return EXIT_OK

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
                                   persist=not args.dry_run, feat_dim=feat_dim)
    X_res = (np.stack(R_X).astype(np.float32) if R_X
             else np.zeros((0, feat_dim), np.float32))
    # BUG-139: interval labels. Weight 0 == "this row proves nothing at
    # theta", and those rows are DROPPED from the fit rather than being
    # given the label the router's own decision implies.
    y_res, w_res, _b_res, _p_res = label_rows(R_meta, args.deep_thresh, counts,
                                              args.censor_slack,
                                              args.min_generated)
    ts_res = np.array([float(m.get("ts", 0.0)) for m in R_meta])

    seed_counts: dict = {}
    try:
        seed_ids, X_seed, y_seed, w_seed = load_seed(
            args.deep_thresh, args.seed_features, args.seed_champion,
            args.censor_slack, seed_counts, feat_dim)
    except FeatureSpaceMismatch as e:
        if not args.no_seed:
            print(f"[refit] REFUSED: {e}")
            return EXIT_REJECT
        # --no-seed already said "do not train on it". The seed is still
        # loaded for the v1 anchor LINE, and that line is worth less than the
        # run: drop the anchor, keep going, say so.
        print(f"[refit] WARN: {e}\n[refit] WARN: --no-seed given, so the run "
              f"continues WITHOUT the v1 anchor line.")
        counts["seed_dropped_feat_dim"] = True
        seed_ids, y_seed, w_seed = [], np.zeros(0), np.zeros(0)
        X_seed = np.zeros((0, feat_dim), np.float32)
    counts["seed_unresolved"] = seed_counts.get("g3_interval_unresolved", 0)

    if args.no_seed:
        X_train, y_train, w_train, ts_train = X_res, y_res, w_res, ts_res
        train_ids = [m["req_id"] for m in R_meta]
    else:
        X_train = np.vstack([X_seed, X_res]) if len(X_res) else X_seed
        y_train = np.concatenate([y_seed, y_res])
        w_train = np.concatenate([w_seed, w_res])
        # The seed predates every sink row: ts=0 pins it to the FIT side of
        # the temporal split, so the holdout is always live traffic.
        ts_train = np.concatenate([np.zeros(len(y_seed)), ts_res])
        train_ids = seed_ids + [m["req_id"] for m in R_meta]

    resolved = np.asarray(w_train, float) > 0
    counts["train_rows_offered"] = int(len(y_train))
    counts["train_rows_unresolved"] = int((~resolved).sum())
    unresolved_rate = (float((~resolved).mean()) if len(y_train) else 0.0)
    counts["train_unresolved_rate"] = round(unresolved_rate, 4)
    X_train = X_train[resolved]
    y_train = y_train[resolved]
    ts_train = ts_train[resolved]
    train_ids = [i for i, k in zip(train_ids, resolved) if k]

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
    if overlap:
        seed_scores = score_with(mu, sd, Vt, w, X_seed)
        seed_idx = {i: k for k, i in enumerate(seed_ids)}
        rho_anchor = spearman(
            np.array([seed_scores[seed_idx[i]] for i in overlap]),
            np.array([loo_map[i] for i in overlap]))
    else:
        rho_anchor = float("nan")     # no seed in this feature space
    counts["rho_vs_loo_anchor"] = round(rho_anchor, 4)
    counts["rho_overlap_n"] = len(overlap)
    anchor_alert = bool(overlap) and rho_anchor < args.anchor_rho_alert

    prior = float(y_train.mean())
    suggested_tdeep = float(np.quantile(train_scores, 1.0 - prior))
    counts["suggested_tdeep"] = round(suggested_tdeep, 4)

    # ── gates ──────────────────────────────────────────────────────────────
    gates: dict = {}
    gate_fail = gate_out_of_sample(X_train, y_train, ts_train, args.lam, args.pcs,
                                   args.holdout_frac, args.min_holdout,
                                   args.min_auc, gates)
    inc, inc_note = load_incumbent(args.out, feat_dim)
    gates["incumbent"] = inc_note
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

    # ── drift: are the INPUTS still the traffic this probe is for? Compared
    # across the same temporal cut the AUC gate uses, in the candidate's own
    # PC basis (drift that the basis cannot see cannot affect the score).
    fit_i, hold_i = temporal_split(ts_train, args.holdout_frac)
    P_all = ((X_train - mu) / sd) @ Vt.T
    b3_ok, b3_note = b3_report_state(args.b3_report, args.b3_max_age_h, t0)
    drift_fail = gate_drift(
        P_all[fit_i], P_all[hold_i], prior, unresolved_rate, b3_ok, b3_note,
        explore_integrity(rows, args.deep_thresh, args.censor_slack), gates,
        args.max_psi, args.ks_p, args.ks_d, args.prior_lo, args.prior_hi,
        args.max_unresolved_frac)
    if args.skip_drift_gate:
        gates["drift_failed_but_ignored"] = drift_fail
    else:
        gate_fail += drift_fail

    # Accuracy monitor: ALERTS only, and computed from `correct`, which is
    # never a label. Runs on the raw sink so thinking-off rows still count.
    acc_alerts = accuracy_monitor(raw, args.results_dir, args.live_tdeep,
                                  args.state, gates, args.acc_alert_drop,
                                  persist=not args.dry_run)
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
        "refit_unresolved_rate": unresolved_rate,
        "refit_censor_slack": args.censor_slack,
        "refit_psi_max": _nan(gates.get("psi_max")),
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
    for a in acc_alerts:
        print(f"[refit] ACCURACY ALERT {a} — `correct` is a MONITOR, never a "
              f"label (AUC(score->correct) is below 0.5 by design).")

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
        verb = "stage as candidate" if args.promote_mode == "shadow" else "swap"
        print(f"[refit] DRY-RUN ok — would {verb} {args.out}\n"
              f"[refit] {json.dumps(report)}")
        return EXIT_OK

    # A candidate the ENGINE cannot load is not a candidate. The router folds
    # the probe against its own FEAT_DIM, so a width change has to be a
    # deliberate, announced migration — not a side effect of which sink windows
    # happened to be on disk this run.
    live_dim = incumbent_feat_dim(args.out)
    if (live_dim is not None and live_dim != feat_dim
            and not args.allow_feat_dim_change):
        print(f"[refit] REFUSED: the live probe is {live_dim}-dim and this "
              f"candidate is {feat_dim}-dim (the sink's newest windows). "
              f"Swapping it in would hand the router a probe in a feature "
              f"space it does not compute. Re-capture the seed and let the "
              f"sink refill in the new width, or pass "
              f"--allow-feat-dim-change if this IS the migration.\n"
              f"[refit] {json.dumps(report)}")
        return EXIT_REJECT

    if args.promote_mode == "shadow":
        # The candidate NEVER routes. It is staged for PN119_PROBE_CANDIDATE,
        # scores every request alongside the incumbent, and is promoted by a
        # later run on evidence it could not have influenced.
        rec = stage_candidate(args.state, payload,
                              note=f"auc_oos={gates.get('oos_auc')}")
        with open(cursor_p, "w", encoding="utf-8") as f:
            json.dump({"schema": 2,
                       "last_ts_seen": max([r.ts for r in rows],
                                           default=cursor["last_ts_seen"]),
                       "files": file_sizes, "ts": t0,
                       "sha256_16": rec["sha256_16"],
                       "n_new_consumed": n_new}, f)
        print(f"[refit] STAGED candidate sha256[:16]={rec['sha256_16']} at "
              f"{candidate_path(args.state)} — set PN119_PROBE_CANDIDATE to it; "
              f"the LIVE probe is untouched and promotion needs "
              f">={args.min_shadow_n} resolved shadow rows plus a paired DeLong "
              f"win.\n[refit] {json.dumps(report)}")
        return EXIT_OK

    if os.path.isfile(args.out):
        shutil.copy2(args.out, prev_p)
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
