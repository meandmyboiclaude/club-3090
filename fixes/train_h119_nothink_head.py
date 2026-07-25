#!/usr/bin/env python3
"""Train and honestly evaluate a tier-0 "no-think-safe" head for H119.

Target: `label_clean` from ~/shared/needfit/tier0/mine_tier0.py -- content_ok
under no-think in 3/3 replicates, restricted to the 110 prod_mixed_v2 items
whose three no-think replicates all completed (the other 85 were lost to
BUG-127 allocator aborts, which say nothing about whether the model needed to
think).  `label_rep3` -- content_ok on the single clean-boot replicate -- is
carried as the full-coverage secondary target.

Candidate heads, in increasing cost:

  const     always-no-think                              (the null)
  lane      one number per prompt-template hash; the hash is sha256 of the
            first 160 chars of the system message, which the ENGINE can
            compute in the prefill it already does. On this suite the 8
            distinct hashes map 1:1 onto the 8 caller categories, so this is
            the free lane policy with no caller cooperation required.
  ptok      prompt token count only (already on the sink meta line)
  lane+ptok
  lens      the H119 prefill lens itself: layers 42/47/51, mean+last pooled,
            FEAT_DIM 30720, standardise -> PCA -> ridge, exactly the shape
            fixes/pn119_router.py folds into one dot product at load
  lens+ptok

Honesty gates, applied to every head:
  * no in-sample number is ever quoted as performance -- 200-item heads use a
    TEMPORAL split (fit on the older 60% by request timestamp, score the newer
    40%), which on this suite is also a provenance shift (prod-* -> prod2-*);
    the 30-item lens head uses leave-one-ITEM-out with all 7 of that item's
    captures held out together
  * every headline metric is reported next to a SHUFFLED-LABEL floor: the same
    pipeline, same split, labels permuted, N times, mean and p95
  * the ORACLE (perfect head) saving is printed beside every realised saving
  * the precision/recall curve is printed with the measured accuracy cost per
    false positive, so an operating point is chosen, not defaulted into

Inputs, both produced offline and both outside this repo (~/shared is not a
git working tree, so they cannot be committed alongside this file):
  ~/shared/needfit/tier0/mine_tier0.py        -> tier0-labels.json
  ~/shared/needfit/tier0/pair_lens_features.py -> tier0-features.npz

CPU-only, numpy-only, reads banked artefacts. Touches no service and no GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

TIER0 = os.path.expanduser("~/shared/needfit/tier0")
QBENCH = os.path.expanduser("~/shared/folderX/qbench45")
LABELS = os.path.join(TIER0, "tier0-labels.json")
FEATS = os.path.join(TIER0, "tier0-features.npz")
DATASET = os.path.join(QBENCH, "data", "prod_mixed_v2.jsonl")

FEAT_DIM = 30720          # 3 layers x {mean,last} x 5120, per pn119_router.py
PREFIX_CHARS = 160        # system-prompt prefix that defines a lane


# ── model: standardise -> PCA -> ridge on a 0/1 target ──────────────────────
def fit(X, y, lam, pcs):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    pcs = min(pcs, min(Xs.shape))
    _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
    Vt = Vt[:pcs]
    A = np.hstack([Xs @ Vt.T, np.ones((len(Xs), 1))])
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return mu, sd, Vt, w


def score(m, X):
    mu, sd, Vt, w = m
    Xs = (X - mu) / sd
    return np.hstack([Xs @ Vt.T, np.ones((len(Xs), 1))]) @ w


def auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, c - h), min(1.0, c + h)


# ── data ────────────────────────────────────────────────────────────────────
def lane_hash(item: dict) -> str:
    msgs = item.get("messages") or []
    sysm = next((m["content"] for m in msgs if m.get("role") == "system"),
                msgs[0]["content"] if msgs else "")
    return hashlib.sha256((sysm or "")[:PREFIX_CHARS].encode()).hexdigest()[:8]


def load():
    with open(LABELS, encoding="utf-8") as f:
        pack = json.load(f)["items"]
    with open(DATASET, encoding="utf-8") as f:
        ds = {str(json.loads(l)["id"]): json.loads(l) for l in f if l.strip()}
    for i, r in pack.items():
        r["lane"] = lane_hash(ds[i])
        r["label_rep3"] = bool(r["nt_content_ok"][2])
        # what a perfect route saves on this row: forcing </think> at the first
        # generated token removes exactly the reasoning tokens, and
        # latency == completion_tokens / tok_per_s holds to 0.04 s on this arm.
        r["saving_s"] = (r["prod_rtok"] / r["prod_tps"]) if r["prod_tps"] else 0.0
    return pack


def design(rows, kind, Xlens=None):
    """Feature matrix for a head. `lane` is one-hot over training lanes; unseen
    lanes at score time fall to the all-zero row, i.e. the intercept."""
    cols = []
    if "lane" in kind:
        lanes = sorted({r["lane"] for r in rows})
        cols.append(np.array([[1.0 if r["lane"] == L else 0.0 for L in lanes]
                              for r in rows]))
    if "ptok" in kind:
        p = np.array([[r["prod_prompt_tok"]] for r in rows], dtype=np.float64)
        cols.append(np.hstack([p / 1000.0, np.log1p(p),
                               np.array([[r["n_messages"]] for r in rows],
                                        dtype=np.float64)]))
    if "lens" in kind:
        cols.append(Xlens)
    return np.hstack(cols) if cols else np.zeros((len(rows), 1))


# ── evaluation harness ──────────────────────────────────────────────────────
def temporal_eval(rows, kind, target, lam, pcs, Xmap=None, holdout=0.4):
    order = np.argsort([r["prod_ts"] for r in rows])
    rows = [rows[i] for i in order]
    n_hold = max(2, int(round(holdout * len(rows))))
    tr, te = rows[:-n_hold], rows[-n_hold:]
    Xtr = design(tr, kind, None if Xmap is None
                 else np.stack([Xmap[r["item_id"]] for r in tr]))
    lanes = sorted({r["lane"] for r in tr})

    def des(rs):
        cols = []
        if "lane" in kind:
            cols.append(np.array([[1.0 if r["lane"] == L else 0.0 for L in lanes]
                                  for r in rs]))
        if "ptok" in kind:
            p = np.array([[r["prod_prompt_tok"]] for r in rs], dtype=np.float64)
            cols.append(np.hstack([p / 1000.0, np.log1p(p),
                                   np.array([[r["n_messages"]] for r in rs],
                                            dtype=np.float64)]))
        if "lens" in kind:
            cols.append(np.stack([Xmap[r["item_id"]] for r in rs]))
        return np.hstack(cols) if cols else np.zeros((len(rs), 1))

    Xtr, Xte = des(tr), des(te)
    ytr = np.array([float(r[target]) for r in tr])
    yte = np.array([float(r[target]) for r in te])
    m = fit(Xtr, ytr, lam, pcs)
    return score(m, Xte), yte, te


def gram(Xall, extra=None):
    """Standardise once, then the [m, m] Gram. d = 30720 >> m = 210, so every
    fold below is run in the DUAL: the PCA basis of a fold is the eigenbasis
    of its Gram submatrix, and a held-out row projects through its Gram column
    alone -- no [d, pcs] basis is ever formed, which turns a 40 s SVD per fold
    into free. mu/sd are global, which is unsupervised (never touches y) and
    is the only thing here that is not strictly per-fold."""
    X = Xall.astype(np.float64)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    if extra is not None:
        e = np.asarray(extra, dtype=np.float64)
        e = (e - e.mean(0)) / (e.std(0) + 1e-6)
        Xs = np.hstack([Xs, e])
    return Xs @ Xs.T


def loo_item_eval_dual(rows, target, lam, pcs, G, cap_item):
    """Leave-one-ITEM-out on a precomputed Gram. All 7 captures of the held-out
    item leave with it, so an item never appears on both sides of the split."""
    ids = [r["item_id"] for r in rows]
    pos = {iid: k for k, iid in enumerate(ids)}
    y = np.array([float(r[target]) for r in rows])
    ycap = np.array([y[pos[c]] for c in cap_item])
    out = np.empty(len(rows))
    for k, iid in enumerate(ids):
        tr = cap_item != iid
        Gtr = G[np.ix_(tr, tr)]
        ev, U = np.linalg.eigh(Gtr)
        idx = np.argsort(ev)[::-1][:pcs]
        ev, U = np.clip(ev[idx], 1e-9, None), U[:, idx]
        s = np.sqrt(ev)
        P = U * s                                    # train PCA scores
        Pte = G[np.ix_(~tr, tr)] @ U / s             # held-out projections
        A = np.hstack([P, np.ones((len(P), 1))])
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ycap[tr])
        Ate = np.hstack([Pte, np.ones((len(Pte), 1))])
        out[k] = float((Ate @ w).mean())             # captures of one item vote
    return out, y


def shuffled_floor(fn, rows, target, n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([float(r[target]) for r in rows])
    vals = []
    for _ in range(n):
        perm = rng.permutation(len(rows))
        shuf = [dict(r) for r in rows]
        for j, r in enumerate(shuf):
            r[target] = bool(y[perm[j]])
        s, yy, *_ = fn(shuf)
        a = auc(s, yy)
        if not np.isnan(a):
            vals.append(a)
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 95)), len(v)


def pr_table(s, y, rows, total_wall, oracle_wall, cost_per_fp,
             suite_oracle=None):
    """Precision/recall + realised saving + accuracy cost, per threshold.

    `saved%` counts the wall removed from EVERY routed row, true positives and
    false ones alike -- a false positive still saves its reasoning tokens, it
    just does so on a request that needed them. That is why `%oracle` can
    exceed 100%: over-routing buys wall with accuracy, and the `ok lost`
    column is the price. Read the two together, never `%oracle` alone.
    `suite%` rescales the realised saving onto the whole 200-item prod wall so
    it is directly comparable to the headline oracle."""
    hdr = (f"    {'thr':>7} {'routed':>7} {'prec':>7} {'rec':>7} "
           f"{'saved%':>8} {'%oracle':>8} {'ok lost':>8}")
    print(hdr + (f" {'suite%':>7}" if suite_oracle else ""))
    seen = set()
    npos = max(1, int((y == 1).sum()))
    for q in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        thr = float(np.quantile(s, q))
        sel = s >= thr
        k = int(sel.sum())
        if k in seen or k == 0:
            continue
        seen.add(k)
        tp = int(((y == 1) & sel).sum())
        fp = k - tp
        saved = sum(rows[i]["saving_s"] for i in range(len(rows)) if sel[i])
        frac = saved / oracle_wall if oracle_wall else 0.0
        line = (f"    {thr:>7.3f} {k:>7d} {tp/k:>7.1%} {tp/npos:>7.1%} "
                f"{saved/total_wall:>8.2%} {frac:>8.0%} {fp*cost_per_fp:>8.2f}")
        if suite_oracle:
            line += f" {frac*suite_oracle:>7.2%}"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="label_clean",
                    choices=("label_clean", "label_rep3"))
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--pcs", type=int, default=8)
    ap.add_argument("--shuffles", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(TIER0, "tier0-head-report.json"))
    args = ap.parse_args()

    pack = load()
    target = args.target
    # A router only ever decides on requests the engine WOULD have thought on.
    eng = [r for r in pack.values() if r["prod_thinking_engaged"]]
    pool = [r for r in eng if r["clean"]] if target == "label_clean" else eng
    pool.sort(key=lambda r: r["prod_ts"])
    y_all = np.array([float(r[target]) for r in pool])
    print(f"target {target}: {int(y_all.sum())}/{len(pool)} positive "
          f"({y_all.mean():.1%} base rate)")

    total_wall = sum(r["prod_latency"] for r in pack.values())
    oracle_wall = sum(r["saving_s"] for r in pool if r[target])
    print(f"prod wall {total_wall:.0f}s; ORACLE (perfect head) saves "
          f"{oracle_wall:.0f}s = {oracle_wall/total_wall:.2%}")

    # measured accuracy cost of one false positive, from the banked labels
    unsafe = [r for r in pool if not r[target]]
    cost_per_fp = float(np.mean([r["prod_content_ok"] -
                                 (r["nt_ok_rate_valid"] or 0.0) for r in unsafe]))
    print(f"measured cost of ONE false positive: {cost_per_fp:.3f} content_ok "
          f"rows lost (n={len(unsafe)} negatives)\n")

    report = {"target": target, "n": len(pool), "base_rate": float(y_all.mean()),
              "oracle_frac": oracle_wall / total_wall,
              "cost_per_fp": cost_per_fp, "heads": {}}

    # ── heads on all engaged items, temporal split ─────────────────────────
    print("── request-shape heads, TEMPORAL split (fit oldest 60%, score "
          "newest 40%) ──")
    print("   the newest 40% is entirely prod2-* capture-window traffic, so "
          "this is a\n   provenance shift, not a random split.")
    for kind in ("const", "lane", "ptok", "lane+ptok"):
        def run(rs, kind=kind):
            if kind == "const":
                order = np.argsort([r["prod_ts"] for r in rs])
                rs = [rs[i] for i in order]
                n_h = max(2, int(round(0.4 * len(rs))))
                te = rs[-n_h:]
                return (np.zeros(len(te)),
                        np.array([float(r[target]) for r in te]), te)
            return temporal_eval(rs, kind, target, args.lam, args.pcs)
        s, yte, te = run(pool)
        a = auc(s, yte)
        fm, f95, _ = shuffled_floor(run, pool, target, n=args.shuffles)
        print(f"  {kind:<10} n_test={len(te):3d}  AUC {a:.3f}   "
              f"shuffled floor mean {fm:.3f} p95 {f95:.3f}   "
              f"{'BEATS' if a > f95 else 'AT/BELOW'} the floor")
        report["heads"][kind] = {"auc": None if np.isnan(a) else a,
                                 "floor_mean": fm, "floor_p95": f95,
                                 "n_test": len(te)}
        if kind != "const" and not np.isnan(a):
            ow = sum(r["saving_s"] for r in te if r[target])
            tw = sum(r["prod_latency"] for r in te)
            print(f"    held-out oracle on these rows: {ow/tw:.2%} of their "
                  f"wall (whole-suite oracle {oracle_wall/total_wall:.2%})")
            pr_table(s, yte, te, tw, ow, cost_per_fp,
                     suite_oracle=oracle_wall / total_wall)
            report["heads"][kind]["oracle_test_frac"] = ow / tw

    # ── adversarial: does anything survive an UNSEEN lane? ─────────────────
    print("\n── leave-one-LANE-out (the deployment risk: a new hindsight "
          "prompt template) ──")
    lanes = sorted({r["lane"] for r in pool})

    def lolo(rs, kind, tgt=None):
        tgt = tgt or target
        s_all, y_all_l, rows_all = [], [], []
        for L in lanes:
            tr = [r for r in rs if r["lane"] != L]
            te = [r for r in rs if r["lane"] == L]
            if not te or len({r[tgt] for r in tr}) < 2:
                continue
            seen_lanes = sorted({r["lane"] for r in tr})

            def des(q_rows, kind=kind, seen_lanes=seen_lanes):
                cols = []
                if "lane" in kind:
                    cols.append(np.array([[1.0 if r["lane"] == q else 0.0
                                           for q in seen_lanes]
                                          for r in q_rows]))
                if "ptok" in kind:
                    p = np.array([[r["prod_prompt_tok"]] for r in q_rows],
                                 float)
                    cols.append(np.hstack([p / 1000.0, np.log1p(p)]))
                return np.hstack(cols)
            m = fit(des(tr), np.array([float(r[tgt]) for r in tr]),
                    args.lam, args.pcs)
            s_all.append(score(m, des(te)))
            y_all_l.append(np.array([float(r[tgt]) for r in te]))
            rows_all += te
        if not s_all:
            return np.zeros(0), np.zeros(0), []
        return np.concatenate(s_all), np.concatenate(y_all_l), rows_all

    for kind in ("lane", "ptok", "lane+ptok"):
        s2, y2, rows_all = lolo(pool, kind)
        if not len(s2):
            continue
        a2 = auc(s2, y2)
        fm, f95, _ = shuffled_floor(lambda rs, k=kind: lolo(rs, k), pool,
                                    target, n=args.shuffles)
        print(f"  {kind:<10} pooled-over-folds AUC {a2:.3f}  shuffled floor "
              f"mean {fm:.3f} p95 {f95:.3f}   "
              f"{'BEATS' if a2 > f95 else 'AT/BELOW'} the floor"
              + ("   (lane one-hot on an unseen lane collapses to the "
                 "intercept, by construction)" if kind == "lane" else ""))
        report["heads"].setdefault(kind, {})["loo_lane"] = {
            "auc": None if np.isnan(a2) else a2,
            "floor_mean": fm, "floor_p95": f95, "n": len(rows_all)}

    # ── lens head ──────────────────────────────────────────────────────────
    print("\n── lens head (real PN119 features, sink-paired) ──")
    if not os.path.exists(FEATS):
        print("  no tier0-features.npz -- run pair_lens_features.py first")
        return 1
    z = np.load(FEATS, allow_pickle=True)
    ids = [str(x) for x in z["item_ids"]]
    Xall, cap_item = z["X_all"], np.array([str(x) for x in z["cap_item"]])
    have = [r for r in pool if r["item_id"] in set(ids)]
    have.sort(key=lambda r: r["item_id"])
    hid = {r["item_id"] for r in have}
    keep = np.array([c in hid for c in cap_item])
    Xall, cap_item = Xall[keep], cap_item[keep]
    yv = np.array([float(r[target]) for r in have])
    print(f"  paired items with a {target} label: {len(have)}/{len(pool)}  "
          f"({int(yv.sum())} positive)  feature rows {Xall.shape}")
    lanes = {r["lane"] for r in have}
    print(f"  ALL of them are one lane ({len(lanes)} distinct prompt-template "
          f"hash), so this head can only be judged\n  on WITHIN-LANE "
          f"discrimination -- which is exactly the residual the lane rule "
          f"leaves.")
    n_neg = int((yv == 0).sum())
    report["lens_pairing"] = {"n_items": len(have), "n_neg": n_neg,
                              "n_capture_rows": int(Xall.shape[0]),
                              "n_lanes": len(lanes)}
    if n_neg < 5:
        print(f"  NOT EVALUABLE: only {n_neg} negative(s) in the paired set. "
              f"An AUC over {len(have)} items with {n_neg} negative is a\n"
              f"  coin flip dressed as a metric -- it is computed below for "
              f"completeness and must NOT be quoted\n  as head performance. "
              f"The pairing is real; the LABEL COVERAGE on it is not. "
              f"See the shadow-pass note.")
    if len(set(yv)) < 2:
        print("  degenerate: one class only. No head is learnable here.")
    else:
        pmap = {r["item_id"]: r["prod_prompt_tok"] for r in have}
        grams = {
            "lens": gram(Xall),
            "lens+ptok": gram(Xall, [[pmap[c]] for c in cap_item]),
        }
        for kind in ("lens", "lens+ptok"):
            def run(rs, kind=kind):
                return loo_item_eval_dual(rs, target, args.lam,
                                          min(args.pcs, 6), grams[kind],
                                          cap_item) + (rs,)
            s, yy, _ = run(have)
            a = auc(s, yy)
            fm, f95, _ = shuffled_floor(run, have, target,
                                        n=min(args.shuffles, 60))
            print(f"  {kind:<10} LOO-by-item  AUC {a:.3f}   shuffled floor "
                  f"mean {fm:.3f} p95 {f95:.3f}   "
                  f"{'BEATS' if a > f95 else 'AT/BELOW'} the floor")
            report["heads"][kind] = {"auc": None if np.isnan(a) else a,
                                     "floor_mean": fm, "floor_p95": f95,
                                     "n": len(have)}
            ow = sum(r["saving_s"] for r in have if r[target])
            tw = sum(r["prod_latency"] for r in have)
            pr_table(s, yy, have, tw, ow, cost_per_fp)

    print("\n── the one shadow pass that closes the feature gap ──")
    print("  The sink pairs to item ids through extra.resp_id and it works "
          "TODAY -- the join above is\n  measured, not assumed. What is "
          "missing is coverage: every prod_mixed_v2 run banked since the\n"
          "  sink started has been a 30-item screen, all of one lane.")
    print("  Needed: ONE replay of all 200 prod_mixed_v2 prompts through "
          ":8021 with PN119_MODE=shadow\n  and the v3fam arm "
          "(thinkingcap_auto_t10, enable_thinking=true), banked as a normal "
          "run.\n  Shadow scores and sinks but enforces no budget, so it "
          "changes no output and risks no accuracy;\n  pair_lens_features.py "
          "then covers 200/200 with no new benchmark and no new label.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
