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
  G3 cap-hit anywhere => y=1 regardless of measured rtok (spend was
     truncated by the cap, true need >= cap).
  G4 dedup by req_id (preemption/retry can double-log), last finish wins.
  G5 structural: a row needs BOTH a score line (features exist in the
     .bin) and a finish line (label exists); orphans are dropped.

Missing "mode" on legacy lines is treated as "shadow": the sink has only
ever been written under PN119_MODE=shadow (compose tcbench8021.yml, live
since 2026-07-25 15:23Z) and the enforce-era router always records mode.

Exit codes: 0 = swapped, or legitimately skipped (not enough new data);
2 = candidate REJECTED by a quality gate (candidate npz preserved in the
state dir for autopsy; the live probe is untouched; systemd surfaces the
failure).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pn119_atomic import atomic_write_npz  # noqa: E402

LAYERS = (42, 47, 51)
D_MODEL = 5120
FEAT_DIM = len(LAYERS) * 2 * D_MODEL  # 30720
ROW_BYTES = FEAT_DIM * 2  # bf16
NEEDFIT = os.path.expanduser("~/shared/needfit")
RESULTS = os.path.expanduser("~/shared/folderX/qbench45/results")


@dataclass
class Row:
    req_id: str
    x: np.ndarray          # [FEAT_DIM] float32
    rtok: int              # measured thinking spend (censored if cap_hit)
    cap_hit: bool
    mode: str
    route: str
    explore: bool
    thinking: object       # True / False / None(unknown) / "legacy"
    ts: float


def bf16_bin_to_f32(path: str) -> np.ndarray:
    import torch
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size % FEAT_DIM:
        raise ValueError(f"{path}: {raw.size*2} bytes not a multiple of row size {ROW_BYTES}")
    return torch.from_numpy(raw.reshape(-1, FEAT_DIM)).view(torch.bfloat16).float().numpy()


def load_sink(sink_dir: str, counts: dict) -> list[Row]:
    rows: list[Row] = []
    tags = sorted(
        f[len("meta-"):-len(".jsonl")]
        for f in os.listdir(sink_dir)
        if f.startswith("meta-") and f.endswith(".jsonl")
    )
    for tag in tags:
        meta_p = os.path.join(sink_dir, f"meta-{tag}.jsonl")
        feat_p = os.path.join(sink_dir, f"feats-{tag}.bin")
        if not os.path.isfile(feat_p) or os.path.getsize(feat_p) == 0:
            continue
        feats = bf16_bin_to_f32(feat_p)
        score_lines: dict[str, dict] = {}
        finish_lines: dict[str, dict] = {}
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
                    finish_lines[m["req_id"]] = m       # G4: last finish wins
                elif "row" in m:
                    score_lines[m["req_id"]] = m
        for req_id, sm in score_lines.items():
            counts["scored"] = counts.get("scored", 0) + 1
            fm = finish_lines.get(req_id)
            if fm is None:
                counts["g5_no_finish"] = counts.get("g5_no_finish", 0) + 1
                continue
            ridx = sm["row"]
            if ridx >= len(feats):
                counts["g5_no_feature_row"] = counts.get("g5_no_feature_row", 0) + 1
                continue
            generated = int(fm.get("generated", 0) or 0)
            rtok = fm.get("rtok")
            rtok = generated if rtok is None else int(rtok)
            rows.append(Row(
                req_id=req_id,
                x=feats[ridx],
                rtok=rtok,
                cap_hit=bool(fm.get("cap_hit", False)),
                mode=str(sm.get("mode", "shadow")),
                route=str(sm.get("route", "")),
                explore=bool(sm.get("explore", False)),
                thinking=fm["thinking"] if "thinking" in fm else "legacy",
                ts=float(fm.get("ts", 0.0)),
            ))
    return rows


def apply_guards(rows: list[Row], legacy_thinking_ok: bool, counts: dict) -> list[Row]:
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


def load_seed(deep_thresh: int):
    """Original 100-item GPQA capture (v1 training set) — continuity anchor."""
    import torch
    from safetensors.torch import load_file
    st = load_file(f"{NEEDFIT}/lens-features-20260723.safetensors")  # bf16
    champ = {}
    with open(f"{RESULTS}/aibox-20260723-capt10full__gpqa_auto__thinkingcap_auto_t10.jsonl",
              encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            champ[rec["item_id"]] = rec.get("reasoning_tokens") or 0
    ids = [i for i in sorted(champ) if i in st]
    X = np.stack([st[i].to(torch.float32).flatten().numpy() for i in ids])
    y = np.array([1.0 if champ[i] >= deep_thresh else 0.0 for i in ids])
    return ids, X, y


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


def update_reservoir(state_dir: str, rows: list[Row], cap: int, rng: np.random.Generator,
                     counts: dict):
    """Bounded raw-feature reservoir (pack: ~2000 requests) so old sink files
    can be pruned without losing PCA-refresh data. Classic reservoir
    sampling over the stream of eligible rows, persisted atomically."""
    res_path = os.path.join(state_dir, "reservoir.npz")
    if os.path.isfile(res_path):
        z = np.load(res_path, allow_pickle=True)
        R_X = list(z["X"])
        R_meta = [json.loads(s) for s in z["meta"]]
        seen_total = int(z["seen_total"])
    else:
        R_X, R_meta, seen_total = [], [], 0
    known = {m["req_id"] for m in R_meta}
    new = [r for r in rows if r.req_id not in known]
    for r in new:
        seen_total += 1
        item_meta = {"req_id": r.req_id, "rtok": r.rtok, "cap_hit": r.cap_hit,
                     "mode": r.mode, "route": r.route, "explore": r.explore,
                     "ts": r.ts}
        if len(R_X) < cap:
            R_X.append(r.x.astype(np.float16))
            R_meta.append(item_meta)
        else:
            j = int(rng.integers(0, seen_total))
            if j < cap:
                R_X[j] = r.x.astype(np.float16)
                R_meta[j] = item_meta
    counts["reservoir_size"] = len(R_X)
    counts["reservoir_seen_total"] = seen_total
    atomic_write_npz(res_path, {
        "X": np.stack(R_X) if R_X else np.zeros((0, FEAT_DIM), dtype=np.float16),
        "meta": np.array([json.dumps(m) for m in R_meta]),
        "seen_total": np.array(seen_total),
    })
    return R_X, R_meta


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
    ap.add_argument("--min-auc", type=float, default=0.80,
                    help="reject candidate below this in-sample AUC")
    ap.add_argument("--min-rho", type=float, default=0.85,
                    help="reject candidate whose seed-set scores rank-correlate "
                         "below this vs router_loo_scores.json")
    ap.add_argument("--reservoir", type=int, default=2000)
    ap.add_argument("--no-seed", action="store_true",
                    help="train on sink rows only (seed still used for the rho gate)")
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

    raw = load_sink(args.sink, counts)
    rows = apply_guards(raw, args.legacy_thinking_ok, counts)

    # min-new accounting (sink is append-only => eligible count is monotonic)
    cursor_p = os.path.join(args.state, "cursor.json")
    last_eligible = 0
    if os.path.isfile(cursor_p):
        try:
            last_eligible = int(json.load(open(cursor_p, encoding="utf-8"))
                                .get("eligible_at_last_swap", 0))
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    n_new = len(rows) - last_eligible
    counts["new_since_last_swap"] = n_new
    if not args.force and n_new < args.min_new:
        print(f"[refit] skip: {n_new} new eligible rows < --min-new {args.min_new} "
              f"(counts={json.dumps(counts)})")
        return 0

    rng = np.random.default_rng(int(t0))
    if args.dry_run:
        res_meta = None
        X_res = np.stack([r.x for r in rows]) if rows else np.zeros((0, FEAT_DIM), np.float32)
        y_res = np.array([1.0 if (r.cap_hit or r.rtok >= args.deep_thresh) else 0.0
                          for r in rows])
    else:
        R_X, R_meta = update_reservoir(args.state, rows, args.reservoir, rng, counts)
        res_meta = R_meta
        X_res = (np.stack(R_X).astype(np.float32) if R_X
                 else np.zeros((0, FEAT_DIM), np.float32))
        y_res = np.array([1.0 if (m["cap_hit"] or m["rtok"] >= args.deep_thresh) else 0.0
                          for m in R_meta])

    seed_ids, X_seed, y_seed = load_seed(args.deep_thresh)
    if args.no_seed:
        X_train, y_train = X_res, y_res
        train_ids = [r.req_id for r in rows] if res_meta is None else \
            [m["req_id"] for m in res_meta]
    else:
        X_train = np.vstack([X_seed, X_res]) if len(X_res) else X_seed
        y_train = np.concatenate([y_seed, y_res])
        train_ids = seed_ids + ([r.req_id for r in rows] if res_meta is None else
                                [m["req_id"] for m in res_meta])

    n_pos, n_neg = int(y_train.sum()), int((1 - y_train).sum())
    counts.update(train_rows=len(y_train), train_pos=n_pos, train_neg=n_neg)
    if n_pos < args.min_pos or n_neg < args.min_neg:
        print(f"[refit] skip: class balance pos={n_pos} neg={n_neg} below minimum "
              f"({args.min_pos}/{args.min_neg})")
        return 0

    mu, sd, Vt, w, train_scores = fit_probe(X_train, y_train, args.lam, args.pcs)
    cand_auc = auc(train_scores, y_train)

    # rho gate vs the frozen leakage-free reference (always runs, seed or not)
    loo = json.load(open(f"{NEEDFIT}/router_loo_scores.json", encoding="utf-8"))
    loo_map = dict(zip(loo["ids"], loo["scores"]))
    overlap = [i for i in seed_ids if i in loo_map]
    seed_scores = score_with(mu, sd, Vt, w, X_seed)
    seed_idx = {i: k for k, i in enumerate(seed_ids)}
    rho = spearman(np.array([seed_scores[seed_idx[i]] for i in overlap]),
                   np.array([loo_map[i] for i in overlap]))
    prior = float(y_train.mean())
    suggested_tdeep = float(np.quantile(train_scores, 1.0 - prior))
    counts.update(auc=round(cand_auc, 4), rho_vs_loo=round(rho, 4),
                  rho_overlap_n=len(overlap),
                  suggested_tdeep=round(suggested_tdeep, 4))

    payload = {
        "mu": mu.astype(np.float64), "sd": sd.astype(np.float64),
        "Vt10": Vt.astype(np.float64), "w": w.astype(np.float64),
        "lam": args.lam, "pcs": args.pcs, "variant": "lens_only",
        "label": f"rtok_ge_{args.deep_thresh}_spend_selftrain",
        "feature_spec": "layers[42,47,51] x pools[last,mean] d5120 concat",
        "train_ids": np.array(train_ids),
        "refit_ts": t0, "refit_n_sink": len(X_res),
        "refit_n_seed": 0 if args.no_seed else len(X_seed),
        "refit_auc": cand_auc, "refit_rho_vs_loo": rho,
        "refit_suggested_tdeep": suggested_tdeep,
    }

    gate_fail = []
    if not (cand_auc >= args.min_auc):
        gate_fail.append(f"auc {cand_auc:.4f} < {args.min_auc}")
    if not (rho >= args.min_rho):
        gate_fail.append(f"rho {rho:.4f} < {args.min_rho}")

    report = {"ts": t0, "elapsed_s": round(time.time() - t0, 2), "counts": counts,
              "gates_failed": gate_fail, "out": args.out, "dry_run": args.dry_run,
              "args": {k: v for k, v in vars(args).items()}}
    if not args.dry_run:
        with open(os.path.join(args.state, "refit-report.json"), "w",
                  encoding="utf-8") as f:
            json.dump(report, f, indent=1)

    if gate_fail:
        rej = os.path.join(args.state, f"rejected-{time.strftime('%Y%m%d-%H%M%S')}.npz")
        saved = ""
        if not args.dry_run:
            atomic_write_npz(rej, payload)
            saved = f"; candidate saved to {rej}"
        print(f"[refit] REJECTED ({'; '.join(gate_fail)}) — live probe untouched"
              f"{saved}\n[refit] {json.dumps(report)}")
        return 2

    if args.dry_run:
        print(f"[refit] DRY-RUN ok — would swap {args.out}\n[refit] {json.dumps(report)}")
        return 0

    atomic_write_npz(args.out, payload)
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()[:16]
    with open(cursor_p, "w", encoding="utf-8") as f:
        json.dump({"eligible_at_last_swap": len(rows), "ts": t0, "sha256_16": sha}, f)
    print(f"[refit] SWAPPED {args.out} sha256[:16]={sha} auc={cand_auc:.4f} "
          f"rho={rho:.4f} n_train={len(y_train)} (pos={n_pos}) "
          f"suggested_tdeep={suggested_tdeep:.4f}\n[refit] {json.dumps(report)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
