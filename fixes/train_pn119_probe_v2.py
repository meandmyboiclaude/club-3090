#!/usr/bin/env python3
"""PN119 probe v2 — train + HONESTLY evaluate a candidate lens-router probe.

Tests one claim: that dropping the MEAN pools, regressing log(rtok+1)
instead of classifying rtok>=2000, and widening to 20 PCs produces a
materially better probe than the shipped one, with the router's scoring
contract unchanged in shape.

    train_pn119_probe_v2.py --blocks last --target log_rtok --pcs 20 \
        --lam 10 --extra-scalar log_prompt_tok

WHAT THIS SCRIPT IS FOR. ~/shared/needfit/train_pn119_probe.py fits the
seed and prints an in-sample AUC; fixes/refit_pn119_probe.py fits the sink
and gates on a temporal split of the same pool. Neither answers "does this
probe transfer to traffic it was not trained on", which is the only
question that matters for a router. Here the seed (100 GPQA items, tap
capture) is the ONLY training data and the live sink is a held-out,
never-trained-on, different-distribution test set. Every headline number
comes with its own shuffled-label noise floor computed under the identical
protocol, because a metric without its floor is not a result.

FEATURES. The tap emits [6, 5120] row-major per request:
    L42-last, L42-mean, L47-last, L47-mean, L51-last, L51-mean
--blocks last keeps rows 0/2/4 (15360 dims), --blocks both keeps all six
(30720, the shipped layout). Optional --extra-scalar values are appended
AFTER the lens block, standardised with everything else.

SERVE CONTRACT. The npz keeps the incumbent's keys so the router's folded
scorer needs no new code path:
    xs = (x - mu) / sd ; score = concat(xs @ Vt10.T, [1]) @ w
Scalars must BYPASS the PCA (one unit-variance column among 15360 is
invisible to a variance-ranked basis, so folding it into the SVD input
silently discards it). That is expressed inside Vt10 rather than in new
router code: rows [0:pcs] are the PCA basis, zero-padded over the scalar
columns; the trailing rows are one-hot selectors on the scalar columns.
The product is then exactly hstack([lens PCs, standardised scalars]) —
train_pn119_probe.py's lens_plus_scalars convention — while remaining a
single [k, FEAT_DIM] matmul that folds to one dot product.

DATA HYGIENE. The sink is poisoned by diagnostic traffic: max_tokens=1
probes generate nothing, `</think>` never fires, so rtok=0 AND cap_hit=True
and the shipped label rule promotes each to a DEEP positive. Rows that
generated < --min-generated are dropped and counted, whether or not they
have already been moved to the sink's .quarantine/ (this script globs the
top level only, like the refit).

LABEL CAVEAT this script measures and prints rather than hides: sink rtok
is the position of `</think>` in the output, so a request whose thinking
budget forced `</think>` at the grant records a rtok EQUAL TO THE GRANT
with cap_hit=False. Those labels are right-censored and invisible to the
censoring guards. The report counts them so the ceiling they impose
on rho and on AUC@2000 is visible next to the numbers.

Exit codes: 0 = candidate written and every headline metric cleared its own
noise floor; 1 = usage/data failure; 2 = a headline metric did not clear its
noise floor (the npz is still written, for autopsy).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

LAYERS = (42, 47, 51)
D_MODEL = 5120
POOLS = ("last", "mean")
BLOCK_NAMES = tuple(f"L{l}-{p}" for l in LAYERS for p in POOLS)
FULL_FEAT_DIM = len(BLOCK_NAMES) * D_MODEL  # 30720 — the shipped tap layout
ROW_BYTES = FULL_FEAT_DIM * 2  # bf16 on disk

NEEDFIT = os.path.expanduser("~/shared/needfit")
RESULTS = os.path.expanduser("~/shared/folderX/qbench45/results")

EXIT_OK, EXIT_FAIL, EXIT_NOISE = 0, 1, 2

# The two sink windows that existed when the v2 claim was measured; kept so
# the claimed numbers can be reproduced on exactly their rows, not just on
# whatever the sink has grown to since.
REPLICATION_TAGS = ("20260725-184327", "20260725-192541")


# ── feature plumbing ───────────────────────────────────────────────────────
def block_slice(names: tuple[str, ...]) -> np.ndarray:
    """Column indices of the named [1, 5120] pool blocks in the flat vector."""
    idx = []
    for n in names:
        k = BLOCK_NAMES.index(n)
        idx.append(np.arange(k * D_MODEL, (k + 1) * D_MODEL))
    return np.concatenate(idx)


def selected_blocks(which: str) -> tuple[str, ...]:
    if which == "both":
        return BLOCK_NAMES
    return tuple(n for n in BLOCK_NAMES if n.endswith("-" + which))


def bf16_rows(path: str, idx) -> np.ndarray:
    """Materialise only the requested rows of a raw bf16 .bin.

    bf16 -> f32 is an exact 16-bit left shift, so numpy does it without
    torch and the memmap faults in nothing else.
    """
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    n_rows = mm.size // FULL_FEAT_DIM
    out = np.empty((len(idx), FULL_FEAT_DIM), dtype=np.float32)
    for k, i in enumerate(idx):
        if i >= n_rows:
            raise IndexError(f"{path}: row {i} beyond {n_rows}")
        raw = np.asarray(mm[i * FULL_FEAT_DIM:(i + 1) * FULL_FEAT_DIM],
                         dtype=np.uint32)
        out[k] = (raw << np.uint32(16)).view(np.float32)
    del mm
    return out


# ── data ───────────────────────────────────────────────────────────────────
def seed_capture_source(path: str) -> str:
    """'tap' (live in-engine capture, the space the sink and the router live
    in) vs 'offline-hf' (lens_pilot's HF/GPTQModel host capture).

    Not interchangeable, and the difference does not announce itself: the two
    captures of the SAME 100 items have mean same-item cosine 0.971, yet a
    probe trained on the wrong one still ranks the sink at spearman 0.988
    against the right one. Only an explicit provenance check catches it --
    which is why this is a guard and not a comment. lens-features-20260725 is
    byte-identical to lens-features-20260723: that "re-capture" was a no-op.
    """
    side = os.path.splitext(path)[0] + ".json"
    if os.path.isfile(side):
        try:
            with open(side, encoding="utf-8") as f:
                meta = (json.load(f) or {}).get("_meta", {})
        except (OSError, json.JSONDecodeError):
            meta = {}
        if "tap" in str(meta.get("source", "")).lower():
            return "tap"
        if meta.get("source") or meta.get("model_dir"):
            return "offline-hf"
    base = os.path.basename(path).lower()
    if base.startswith("tap-features"):
        return "tap"
    if base.startswith("lens-features"):
        return "offline-hf"
    return "unknown"


def load_seed(features: str, champion: str):
    """The 100-item GPQA capture: X [n, 30720], rtok, prompt_tok, ids.

    Must be a TAP capture. lens-features-20260723 is an HF-offline capture
    whose same-item cosine against the serving tap is ~0.968 — a different
    feature space, and the sink rows this is evaluated on come from the tap.
    """
    import torch                                # noqa: PLC0415 — bf16 needs it
    from safetensors.torch import load_file     # noqa: PLC0415
    st = load_file(features)
    champ = {}
    with open(champion, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            champ[r["item_id"]] = r
    ids = [i for i in sorted(champ) if i in st]
    if not ids:
        raise SystemExit(f"[v2] no overlap between {features} and {champion}")
    X = np.stack([st[i].to(torch.float32).flatten().numpy() for i in ids])
    rtok = np.array([float(champ[i].get("reasoning_tokens") or 0) for i in ids])
    ptok = np.array([float(champ[i].get("prompt_tokens") or 0) for i in ids])
    return ids, X, rtok, ptok


def load_sink(sink_dir: str, min_generated: int, counts: dict):
    """Parse the sink into labelled rows. Conventions from refit_pn119_probe.

    Meta is two line kinds keyed by req_id: a score line (carries `row`, the
    index into feats-<tag>.bin, plus prompt_tok/route/mode) and a finish line
    (carries generated/thinking/rtok/cap_hit). A row needs both.
    """
    rows = []
    tags = sorted(f[len("meta-"):-len(".jsonl")]
                  for f in os.listdir(sink_dir)
                  if f.startswith("meta-") and f.endswith(".jsonl"))
    for tag in tags:
        meta_p = os.path.join(sink_dir, f"meta-{tag}.jsonl")
        feat_p = os.path.join(sink_dir, f"feats-{tag}.bin")
        if not os.path.isfile(feat_p) or os.path.getsize(feat_p) == 0:
            continue
        n_feat_rows = os.path.getsize(feat_p) // ROW_BYTES
        score_lines, finish_lines = {}, {}
        with open(meta_p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    counts["bad_json"] = counts.get("bad_json", 0) + 1
                    continue
                if m.get("finish"):
                    finish_lines[m["req_id"]] = m      # last finish wins
                elif "row" in m:
                    score_lines[m["req_id"]] = m
        for req_id, sm in score_lines.items():
            counts["scored"] = counts.get("scored", 0) + 1
            fm = finish_lines.get(req_id)
            if fm is None:
                counts["drop_no_finish"] = counts.get("drop_no_finish", 0) + 1
                continue
            ridx = int(sm["row"])
            if ridx >= n_feat_rows:
                counts["drop_no_feature_row"] = counts.get("drop_no_feature_row", 0) + 1
                continue
            thinking = fm.get("thinking", "legacy")
            if thinking is not True:
                key = ("drop_thinking_off" if thinking is False else
                       "drop_thinking_legacy" if thinking == "legacy" else
                       "drop_thinking_unknown")
                counts[key] = counts.get(key, 0) + 1
                continue
            generated = int(fm.get("generated", 0) or 0)
            if generated < min_generated:
                # A request that generated nothing MEASURED nothing: `</think>`
                # never fired, so rtok=0 and cap_hit=True and the shipped label
                # rule calls it a deep positive. This is the sink poison.
                counts["drop_no_generation"] = counts.get("drop_no_generation", 0) + 1
                continue
            rtok = fm.get("rtok")
            rtok = generated if rtok is None else int(rtok)
            rows.append({
                "req_id": req_id, "tag": tag, "feat_path": feat_p,
                "row_idx": ridx, "rtok": rtok, "generated": generated,
                "cap_hit": bool(fm.get("cap_hit", False)),
                "prompt_tok": int(sm.get("prompt_tok", 0) or 0),
                "route": str(sm.get("route", "")),
                "mode": str(sm.get("mode", "shadow")),
                "explore": bool(sm.get("explore", False)),
                "ts": float(fm.get("ts", 0.0)),
            })
    counts["eligible"] = len(rows)
    return rows


def materialise(rows: list[dict]) -> np.ndarray:
    X = np.empty((len(rows), FULL_FEAT_DIM), dtype=np.float32)
    by_file: dict[str, list[int]] = {}
    for k, r in enumerate(rows):
        by_file.setdefault(r["feat_path"], []).append(k)
    for path, ks in by_file.items():
        X[ks] = bf16_rows(path, [rows[k]["row_idx"] for k in ks])
    return X


# ── fit ────────────────────────────────────────────────────────────────────
def build_x(X_lens: np.ndarray, scalars: np.ndarray | None,
            cols: np.ndarray) -> np.ndarray:
    x = X_lens[:, cols]
    if scalars is not None and scalars.size:
        x = np.hstack([x, scalars])
    return x


def make_basis(X: np.ndarray, pcs: int, n_scalars: int):
    """(mu, sd, M) — the UNSUPERVISED half of the probe, so it is computed
    once per (data, pcs) and reused across every lambda and every label
    permutation. The SVD is the expensive step; nothing about it sees y.

    M is [pcs + n_scalars, FEAT_DIM]: PCA rows zero over the scalar columns,
    then one-hot rows selecting each scalar. So the router's `xs @ Vt10.T`
    yields the PCA-then-bypass layout with no router change.
    """
    n_lens = X.shape[1] - n_scalars
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Xs[:, :n_lens], full_matrices=False)
    M = np.zeros((pcs + n_scalars, X.shape[1]))
    M[:pcs, :n_lens] = Vt[:pcs]
    for k in range(n_scalars):
        M[pcs + k, n_lens + k] = 1.0
    return mu, sd, M


def fit_probe(X, y, lam: float, pcs: int, n_scalars: int, basis=None):
    """Ridge on [PCA(lens) | standardised scalars], returned in serve form."""
    mu, sd, M = basis if basis is not None else make_basis(X, pcs, n_scalars)
    w = readout_only(mu, sd, M, X, y, lam)
    return mu, sd, M, w


def readout_only(mu, sd, M, X, y, lam):
    """Refit just the ridge readout on a fixed (mu, sd, M). The basis is
    unsupervised, so holding it while permuting y is the right null."""
    A = np.hstack([((X - mu) / sd) @ M.T, np.ones((len(X), 1))])
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)


def score_with(mu, sd, M, w, X: np.ndarray) -> np.ndarray:
    A = np.hstack([((X - mu) / sd) @ M.T, np.ones((len(X), 1))])
    return A @ w


def loo_scores(X, y, lam, pcs, n_scalars, basis=None):
    """Exact ridge LOO at a fixed basis: yhat_i^(-i) = y_i - e_i/(1 - h_ii).

    The basis (mu/sd/PCA) is still fit on all rows, so this is optimistic —
    which is why the transfer test to the sink, not this, is the headline.
    """
    mu, sd, M = basis if basis is not None else make_basis(X, pcs, n_scalars)
    A = np.hstack([((X - mu) / sd) @ M.T, np.ones((len(X), 1))])
    G = np.linalg.inv(A.T @ A + lam * np.eye(A.shape[1]))
    w = G @ A.T @ y
    h = np.clip(np.diag(A @ G @ A.T), 0.0, 1.0 - 1e-9)
    return y - (y - A @ w) / (1.0 - h)


# ── metrics ────────────────────────────────────────────────────────────────
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


def spearman(a, b) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def metrics(scores: np.ndarray, rtok: np.ndarray, thresholds) -> dict:
    out = {"rho": round(spearman(scores, rtok), 4)}
    for t in thresholds:
        yb = (rtok >= t).astype(float)
        out[f"auc@{t}"] = (round(auc(scores, yb), 4)
                           if 0 < yb.sum() < len(yb) else None)
    return out


def floor_summary(vals: list[dict], keys, pct=95) -> dict:
    out = {}
    for k in keys:
        v = np.array([d[k] for d in vals if d.get(k) is not None], dtype=float)
        v = v[np.isfinite(v)]
        if not len(v):
            out[k] = None
            continue
        out[k] = {"mean": round(float(v.mean()), 4),
                  f"p{pct}": round(float(np.percentile(v, pct)), 4),
                  "max": round(float(v.max()), 4)}
    return out


def load_npz_probe(path: str):
    z = np.load(path, allow_pickle=True)
    return (np.asarray(z["mu"], dtype=np.float64),
            np.asarray(z["sd"], dtype=np.float64),
            np.asarray(z["Vt10"], dtype=np.float64),
            np.asarray(z["w"], dtype=np.float64))


# ── report helpers ─────────────────────────────────────────────────────────
def targets_for(rtok: np.ndarray, target: str, deep_thresh: int) -> np.ndarray:
    if target == "binary":
        return (rtok >= deep_thresh).astype(float)
    if target == "log_rtok":
        return np.log(rtok + 1.0)
    raise SystemExit(f"[v2] unknown --target {target}")


def scalars_for(names, ptok: np.ndarray) -> np.ndarray:
    cols = []
    for n in names:
        if n == "log_prompt_tok":
            cols.append(np.log(ptok + 1.0))
        else:
            raise SystemExit(f"[v2] unknown --extra-scalar {n}")
    return (np.stack(cols, axis=1) if cols
            else np.zeros((len(ptok), 0), dtype=np.float64))


def censoring_report(rows: list[dict]) -> dict:
    """Count rtok values that repeat often enough to be budget grants.

    rtok is the index of `</think>` in the output. A thinking-budget forcer
    that injects `</think>` at the grant therefore records rtok == grant with
    cap_hit=False, so the censoring guards cannot see it.
    """
    rt = np.array([r["rtok"] for r in rows])
    vals, cnt = np.unique(rt, return_counts=True)
    modal = [(int(v), int(c)) for v, c in zip(vals, cnt) if c >= 3]
    modal.sort(key=lambda t: -t[1])
    n_modal = sum(c for _v, c in modal)
    return {
        "n": len(rt), "n_distinct_rtok": int(len(vals)),
        "modal_values_ge3": modal[:8],
        "rows_at_modal_values": n_modal,
        "frac_at_modal_values": round(n_modal / len(rt), 4) if len(rt) else None,
        "cap_hit_flagged": int(sum(1 for r in rows if r["cap_hit"])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", choices=("both", "last", "mean"), default="last",
                    help="which pool halves to keep (both = the shipped 30720)")
    ap.add_argument("--target", choices=("binary", "log_rtok"), default="log_rtok")
    ap.add_argument("--pcs", type=int, default=20)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--extra-scalar", action="append", default=None,
                    choices=("log_prompt_tok",),
                    help="scalar appended after the lens block; repeatable")
    ap.add_argument("--no-extra-scalar", action="store_true",
                    help="force zero scalars (overrides --extra-scalar)")
    ap.add_argument("--deep-thresh", type=int, default=2000)
    ap.add_argument("--auc-thresholds", default="2000,2500")
    ap.add_argument("--seed-features",
                    default=f"{NEEDFIT}/tap-features-20260725.safetensors",
                    help="seed capture; MUST be a TAP capture (see seed_capture_source)")
    ap.add_argument("--allow-capture-mismatch", action="store_true",
                    help="train on an HF-offline capture anyway (you need a reason)")
    ap.add_argument("--seed-champion",
                    default=f"{RESULTS}/aibox-20260723-capt10full__gpqa_auto__"
                            f"thinkingcap_auto_t10.jsonl")
    ap.add_argument("--sink", default=f"{NEEDFIT}/pn119-sink")
    ap.add_argument("--min-generated", type=int, default=32)
    ap.add_argument("--incumbent", default=f"{NEEDFIT}/pn119-live/probe.npz")
    ap.add_argument("--out", default=f"{NEEDFIT}/probe-v2/pn119-probe-v2.npz")
    ap.add_argument("--json-out", default=f"{NEEDFIT}/probe-v2/pn119-probe-v2-report.json")
    ap.add_argument("--noise-trials", type=int, default=400)
    ap.add_argument("--rng-seed", type=int, default=119)
    ap.add_argument("--lambda-curve", default="0.03,0.1,0.3,1,3,10,30,100")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="do not write the npz")
    args = ap.parse_args()

    scal_names = [] if args.no_extra_scalar else list(args.extra_scalar or [])
    thresholds = [int(t) for t in args.auc_thresholds.split(",") if t.strip()]
    t0 = time.time()
    rng = np.random.default_rng(args.rng_seed)
    rep: dict = {"ts": t0, "args": {k: v for k, v in vars(args).items()},
                 "scalars": scal_names, "thresholds": thresholds}

    print("=" * 78)
    print("PN119 probe v2 — train on the seed, TEST on the live sink")
    print("=" * 78)

    # ── data ───────────────────────────────────────────────────────────────
    seed_src = seed_capture_source(args.seed_features)
    if seed_src != "tap" and not args.allow_capture_mismatch:
        print(f"[v2] FAIL: seed {os.path.basename(args.seed_features)} is a "
              f"{seed_src!r} capture. The sink rows this is TESTED on come from "
              f"the tap, and the seed is the TRAINING set, so an HF-offline seed "
              f"contaminates every number in this report -- not just the "
              f"in-domain ones. Pass a tap capture (tap-features-*.safetensors) "
              f"or --allow-capture-mismatch.")
        return EXIT_FAIL
    seed_ids, X_seed_full, rtok_seed, ptok_seed = load_seed(
        args.seed_features, args.seed_champion)
    counts: dict = {}
    rows = load_sink(args.sink, args.min_generated, counts)
    if not rows:
        print(f"[v2] FAIL: no eligible sink rows (counts={json.dumps(counts)})")
        return EXIT_FAIL
    X_sink_full = materialise(rows)
    rtok_sink = np.array([r["rtok"] for r in rows], dtype=float)
    ptok_sink = np.array([r["prompt_tok"] for r in rows], dtype=float)
    tags_sink = np.array([r["tag"] for r in rows])
    rep["counts"] = counts
    rep["seed"] = {"n": len(seed_ids), "features": os.path.basename(args.seed_features),
                   "capture_source": seed_src,
                   "pos@2000": int((rtok_seed >= 2000).sum()),
                   "pos@2500": int((rtok_seed >= 2500).sum())}

    print(f"\nseed  : n={len(seed_ids)} from {os.path.basename(args.seed_features)} "
          f"[{seed_src}] (pos@2000={rep['seed']['pos@2000']})")
    print(f"sink  : {json.dumps(counts)}")
    print(f"        eval n={len(rows)}  "
          f"pos@2000={int((rtok_sink >= 2000).sum())} "
          f"pos@2500={int((rtok_sink >= 2500).sum())}")
    drop = counts.get("drop_no_generation", 0)
    print(f"        dropped for generated < {args.min_generated}: {drop} "
          f"(top-level sink only; already-quarantined windows are not counted)")
    cens = censoring_report(rows)
    rep["censoring"] = cens
    print(f"        LABEL CENSORING: {cens['rows_at_modal_values']}/{cens['n']} rows "
          f"({cens['frac_at_modal_values']}) sit on {len(cens['modal_values_ge3'])} "
          f"repeated rtok values {cens['modal_values_ge3'][:5]}")
    print(f"        cap_hit flags only {cens['cap_hit_flagged']} of them — the "
          f"budget forcer emits </think>, so cap_hit cannot see this censoring.")

    # ── configurations ─────────────────────────────────────────────────────
    def make_cfg(name, blocks, target, pcs, lam, scalars):
        cols = block_slice(selected_blocks(blocks))
        return {"name": name, "blocks": blocks, "target": target, "pcs": pcs,
                "lam": lam, "scalars": list(scalars), "cols": cols,
                "feat_dim": len(cols) + len(scalars)}

    v2 = make_cfg("v2", args.blocks, args.target, args.pcs, args.lam, scal_names)
    inc_method = make_cfg("incumbent-method", "both", "binary", 10, 10.0, [])
    cfgs = [inc_method, v2]
    if not args.no_ablations:
        cfgs += [
            make_cfg("abl:v2 minus scalar", args.blocks, args.target, args.pcs,
                     args.lam, []),
            make_cfg("abl:v2 with both-pools", "both", args.target, args.pcs,
                     args.lam, scal_names),
            make_cfg("abl:v2 with binary target", args.blocks, "binary",
                     args.pcs, args.lam, scal_names),
            make_cfg("abl:last-only, else incumbent", "last", "binary", 10, 10.0, []),
        ]

    _prep_cache: dict = {}

    def prep(cfg):
        """(X_train, X_test, y_train, basis). The basis is unsupervised and
        lambda-independent, so it is built once and reused by every label
        permutation and every point on the lambda curve — otherwise the
        noise floors alone cost hundreds of 100x15360 SVDs."""
        key = (cfg["blocks"], tuple(cfg["scalars"]), cfg["target"], cfg["pcs"])
        if key not in _prep_cache:
            Xtr = build_x(X_seed_full, scalars_for(cfg["scalars"], ptok_seed),
                          cfg["cols"])
            Xte = build_x(X_sink_full, scalars_for(cfg["scalars"], ptok_sink),
                          cfg["cols"])
            ytr = targets_for(rtok_seed, cfg["target"], args.deep_thresh)
            basis = make_basis(Xtr, cfg["pcs"], len(cfg["scalars"]))
            _prep_cache[key] = (Xtr, Xte, ytr, basis)
        return _prep_cache[key]

    # ── headline: transfer to the sink, with a floor for every number ──────
    print("\n" + "-" * 78)
    print("OUT-OF-DOMAIN TRANSFER — fit on the 100-item seed, score the live sink")
    print("-" * 78)

    inc = None
    if os.path.isfile(args.incumbent):
        inc = load_npz_probe(args.incumbent)
        if inc[0].size != FULL_FEAT_DIM:
            print(f"[v2] incumbent {args.incumbent} has dim {inc[0].size}, "
                  f"expected {FULL_FEAT_DIM} — skipping it")
            inc = None
    ood: dict = {}
    if inc is not None:
        s = score_with(inc[0], inc[1], inc[2], inc[3], X_sink_full)
        ood["incumbent (live npz)"] = metrics(s, rtok_sink, thresholds)
        rep["incumbent_path"] = args.incumbent

    floors: dict = {}
    fitted: dict = {}
    for cfg in cfgs:
        Xtr, Xte, ytr, basis = prep(cfg)
        mu, sd, M, w = fit_probe(Xtr, ytr, cfg["lam"], cfg["pcs"],
                                 len(cfg["scalars"]), basis=basis)
        s = score_with(mu, sd, M, w, Xte)
        ood[cfg["name"]] = metrics(s, rtok_sink, thresholds)
        fitted[cfg["name"]] = (mu, sd, M, w, s)
        if cfg["name"] in ("v2", "incumbent-method"):
            trials = []
            for _ in range(args.noise_trials):
                wp = readout_only(mu, sd, M, Xtr, rng.permutation(ytr), cfg["lam"])
                trials.append(metrics(score_with(mu, sd, M, wp, Xte),
                                      rtok_sink, thresholds))
            floors[cfg["name"]] = floor_summary(
                trials, ["rho"] + [f"auc@{t}" for t in thresholds])

    keys = ["rho"] + [f"auc@{t}" for t in thresholds]
    hdr = f"{'config':<28}" + "".join(f"{k:>12}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    for name, m in ood.items():
        print(f"{name:<28}" + "".join(
            f"{(m[k] if m[k] is not None else float('nan')):>12.4f}" for k in keys))
    for name, fl in floors.items():
        print(f"{('  floor(shuffled) ' + name):<28}" + "".join(
            f"{fl[k]['mean']:>12.4f}" if fl[k] else f"{'--':>12}" for k in keys))
        print(f"{('  floor p95 ' + name):<28}" + "".join(
            f"{fl[k]['p95']:>12.4f}" if fl[k] else f"{'--':>12}" for k in keys))
    rep["ood_all"] = {"n": len(rows), "metrics": ood, "floors": floors}

    # replication subset: the rows the v2 claim was measured on
    sub = np.isin(tags_sink, REPLICATION_TAGS)
    if sub.sum() >= 10:
        sub_ood = {}
        if inc is not None:
            sub_ood["incumbent (live npz)"] = metrics(
                score_with(inc[0], inc[1], inc[2], inc[3], X_sink_full[sub]),
                rtok_sink[sub], thresholds)
        for name, (mu, sd, M, w, s) in fitted.items():
            sub_ood[name] = metrics(s[sub], rtok_sink[sub], thresholds)
        print(f"\nreplication subset (windows {', '.join(REPLICATION_TAGS)}, "
              f"n={int(sub.sum())}) — the rows the claim was measured on")
        print(hdr)
        print("-" * len(hdr))
        for name, m in sub_ood.items():
            print(f"{name:<28}" + "".join(
                f"{(m[k] if m[k] is not None else float('nan')):>12.4f}" for k in keys))
        rep["ood_replication"] = {"n": int(sub.sum()), "tags": list(REPLICATION_TAGS),
                                  "metrics": sub_ood}

    # ── LOO on the seed (in-domain, honest within that domain) ─────────────
    print("\n" + "-" * 78)
    print("LEAVE-ONE-OUT ON THE SEED (in-domain) — AUC vs rtok >= "
          f"{args.deep_thresh}")
    print("-" * 78)
    y_bin_seed = (rtok_seed >= args.deep_thresh).astype(float)
    loo_rep = {}
    for cfg in cfgs:
        Xtr, _Xte, ytr, basis = prep(cfg)
        a = auc(loo_scores(Xtr, ytr, cfg["lam"], cfg["pcs"], len(cfg["scalars"]),
                           basis=basis), y_bin_seed)
        loo_rep[cfg["name"]] = round(float(a), 4)
        print(f"{cfg['name']:<28}{a:>12.4f}")
    # floor: permute the training labels, keep the basis
    Xtr, _Xte, ytr, basis = prep(v2)
    f_trials = []
    for _ in range(args.noise_trials):
        yp = rng.permutation(ytr)
        f_trials.append({"auc": auc(loo_scores(Xtr, yp, v2["lam"], v2["pcs"],
                                               len(v2["scalars"]), basis=basis),
                                    y_bin_seed)})
    loo_floor = floor_summary(f_trials, ["auc"])
    print(f"{'  floor(shuffled) v2':<28}{loo_floor['auc']['mean']:>12.4f}"
          f"   p95={loo_floor['auc']['p95']:.4f}")
    rep["loo_seed"] = {"auc": loo_rep, "floor": loo_floor,
                       "n": len(seed_ids), "pos": int(y_bin_seed.sum())}

    # ── lambda curve ───────────────────────────────────────────────────────
    lams = [float(x) for x in args.lambda_curve.split(",") if x.strip()]
    print("\n" + "-" * 78)
    print("LAMBDA CURVE (v2 config) — is the claim of a flat lambda true?")
    print("-" * 78)
    print(f"{'lam':>8}{'LOO auc(seed)':>16}{'OOD rho':>12}"
          + "".join(f"{'OOD auc@' + str(t):>14}" for t in thresholds))
    Xtr, Xte, ytr, basis = prep(v2)
    curve = []
    for lam in lams:
        a_loo = auc(loo_scores(Xtr, ytr, lam, v2["pcs"], len(v2["scalars"]),
                               basis=basis), y_bin_seed)
        mu, sd, M, w = fit_probe(Xtr, ytr, lam, v2["pcs"], len(v2["scalars"]),
                                 basis=basis)
        m = metrics(score_with(mu, sd, M, w, Xte), rtok_sink, thresholds)
        curve.append({"lam": lam, "loo_auc_seed": round(float(a_loo), 4), **m})
        print(f"{lam:>8g}{a_loo:>16.4f}{m['rho']:>12.4f}"
              + "".join(f"{m['auc@' + str(t)]:>14.4f}" for t in thresholds))
    sp = [c["loo_auc_seed"] for c in curve]
    print(f"  LOO AUC span over lambda [{min(lams):g}, {max(lams):g}]: "
          f"{max(sp) - min(sp):.4f}  (flat if <= 0.02)")
    rep["lambda_curve"] = curve
    rep["lambda_flat"] = bool(max(sp) - min(sp) <= 0.02)

    # ── verdict against the floors ─────────────────────────────────────────
    v2m, v2f = ood["v2"], floors["v2"]
    fails = []
    for k in keys:
        if v2m[k] is None or v2f[k] is None:
            continue
        if v2m[k] <= v2f[k]["p95"]:
            fails.append(f"v2 {k}={v2m[k]:.4f} does not clear its shuffled-label "
                         f"p95 floor {v2f[k]['p95']:.4f}")

    # ── write the candidate ────────────────────────────────────────────────
    mu, sd, M, w, s_sink = fitted["v2"]
    prior = float((rtok_seed >= args.deep_thresh).mean())
    train_scores = score_with(mu, sd, M, w, prep(v2)[0])
    suggested_tdeep = float(np.quantile(train_scores, 1.0 - prior))
    payload = {
        "mu": mu.astype(np.float64), "sd": sd.astype(np.float64),
        "Vt10": M.astype(np.float64), "w": w.astype(np.float64),
        "lam": v2["lam"], "pcs": v2["pcs"],
        "variant": f"v2_{args.blocks}_{args.target}"
                   + ("_" + "+".join(scal_names) if scal_names else ""),
        "label": (f"log1p_rtok" if args.target == "log_rtok"
                  else f"rtok_ge_{args.deep_thresh}"),
        "feature_spec": ("layers[42,47,51] x pools["
                         + ",".join(sorted({b.split('-')[1]
                                            for b in selected_blocks(args.blocks)}))
                         + f"] d{D_MODEL} concat"
                         + (" + " + "+".join(scal_names) if scal_names else "")),
        "feat_dim": v2["feat_dim"],
        "blocks": np.array(selected_blocks(args.blocks)),
        "scalars": np.array(scal_names),
        "train_ids": np.array(seed_ids),
        "train_source": os.path.basename(args.seed_features),
        "v2_ood_n": len(rows),
        "v2_ood_rho": v2m["rho"],
        "v2_ood_auc2000": v2m.get("auc@2000"),
        "v2_ood_auc2500": v2m.get("auc@2500"),
        "v2_loo_auc_seed": loo_rep["v2"],
        "v2_noise_floor_rho_p95": v2f["rho"]["p95"] if v2f["rho"] else None,
        "suggested_tdeep": suggested_tdeep,
        "built_ts": t0,
        "NOT_PROMOTED": "candidate only — router FEAT_DIM and PN119_TDEEP both "
                        "change with this probe; see the report",
    }
    print("\n" + "-" * 78)
    print("CANDIDATE")
    print("-" * 78)
    print(f"feat_dim {FULL_FEAT_DIM} -> {v2['feat_dim']}  "
          f"(sink bytes/request {FULL_FEAT_DIM * 2} -> {v2['feat_dim'] * 2})")
    print(f"score scale is {'log(rtok+1)' if args.target == 'log_rtok' else '0..1'}"
          f"; PN119_TDEEP must be recalibrated -> suggested {suggested_tdeep:.4f} "
          f"(train prior {prior:.3f})")
    if not args.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez(args.out, **payload)
        print(f"wrote {args.out}")
        if args.json_out:
            rep["verdict_fails"] = fails
            rep["elapsed_s"] = round(time.time() - t0, 2)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=1, default=str)
            print(f"wrote {args.json_out}")
    else:
        print("(dry run — nothing written)")

    print("\n" + "=" * 78)
    if fails:
        for f_ in fails:
            print(f"[v2] NOISE FLOOR NOT CLEARED: {f_}")
        return EXIT_NOISE
    print("[v2] every headline metric cleared its own shuffled-label p95 floor.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
