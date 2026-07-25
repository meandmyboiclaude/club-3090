#!/usr/bin/env python3
"""PN119 v2 refit — the guards and gates that stop the loop poisoning itself.

Run: ~/shared/needfit/lens-venv/bin/python fixes/test_refit_guards.py
     (no boot, no GPU, no container — CPU numpy + a synthetic sink in tmp)

WHY THIS FILE EXISTS
--------------------
2026-07-25: diagnostic tooling (pn119_tap_capture.py, pn119_b3_numerics.py)
sends max_tokens=1. Such a request emits no `</think>`, so the router's finish
line reads rtok=0 + cap_hit=True, and guard G3 turned every one of them into a
DEEP positive. Measured on the live sink that day: 130 of 151 eligible rows
were this synthetic traffic, all y=1, training prior 0.733 — and
`refit_pn119_probe.py --dry-run --force` PASSED EVERY GATE with it in.

It passed because both gates were structurally incapable of failing:
  * AUC was IN-SAMPLE (fit X_train, score X_train);
  * rho was measured against the frozen v1 LOO scores, which forbids learning
    rather than detecting breakage;
  * and both are RANK statistics, so a monotone rescale — every score pushed
    across the live PN119_TDEEP — is invisible to them (case 5, the one that
    would have shipped a probe that routes everything deep).

Each case below is one of those failure modes, plus the cursor/scaling bugs
that would have made the loop silently dead (case 6) or unboundedly resident
(case 8).
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "fixes"))
import refit_pn119_probe as R  # noqa: E402

NEEDFIT = pathlib.Path(os.path.expanduser("~/shared/needfit"))
PY = str(NEEDFIT / "lens-venv/bin/python")

_fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


# ---------------------------------------------------------------- fakes
def mkrow(req_id="r", generated=500, rtok=500, cap_hit=False, mode="shadow",
          route="deep", explore=False, thinking=True, ts=100.0, tag="t"):
    return R.Row(req_id=req_id, tag=tag, feat_path="", row_idx=0, rtok=rtok,
                 generated=generated, cap_hit=cap_hit, mode=mode, route=route,
                 explore=explore, thinking=thinking, ts=ts)


def write_sink(dirpath, tag, rows, feat_rows=None):
    """Write one sink window: feats-<tag>.bin + meta-<tag>.jsonl.

    `rows` = list of dicts with req_id/generated/rtok/cap_hit/mode/route/
    explore/thinking/ts; one feature row per entry.
    """
    os.makedirs(dirpath, exist_ok=True)
    n = len(rows)
    X = (np.random.default_rng(abs(hash(tag)) % 2**31)
         .standard_normal((n, R.FEAT_DIM)).astype(np.float32)
         if feat_rows is None else feat_rows)
    bf = (X.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    bf.tofile(os.path.join(dirpath, f"feats-{tag}.bin"))
    with open(os.path.join(dirpath, f"meta-{tag}.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"req_id": r["req_id"], "row": i,
                                "score": 0.5, "route": r.get("route", "deep"),
                                "prompt_tok": 100, "ts": r.get("ts", 100.0),
                                "mode": r.get("mode", "shadow"),
                                "explore": r.get("explore", False)}) + "\n")
            f.write(json.dumps({"req_id": r["req_id"], "finish": True, "score": 0.5,
                                "generated": r.get("generated", 500),
                                "ts": r.get("ts", 100.0),
                                "thinking": r.get("thinking", True),
                                "rtok": r.get("rtok", 500),
                                "cap_hit": r.get("cap_hit", False),
                                "explore": r.get("explore", False),
                                "mode": r.get("mode", "shadow")}) + "\n")
    return X


def synthetic_rows(n, **kw):
    return [dict({"req_id": f"{kw.get('prefix','q')}-{i}"}, **
                 {k: v for k, v in kw.items() if k != "prefix"}) for i in range(n)]


def learnable_feats(rows, deep_thresh=2000, seed=11):
    """Features a probe can actually fit: one direction carries the label."""
    rng = np.random.default_rng(seed)
    y = np.array([1.0 if r["rtok"] >= deep_thresh else 0.0 for r in rows])
    d = rng.standard_normal(R.FEAT_DIM).astype(np.float32)
    d /= np.linalg.norm(d)
    return (rng.standard_normal((len(rows), R.FEAT_DIM)).astype(np.float32) * 0.05
            + y[:, None] * d[None, :] * 8.0).astype(np.float32)


# ---------------------------------------------------------------- cases
def case1_g6_drops_no_generation() -> None:
    """The P1 bug: max_tokens=1 rows are cap_hit=True, rtok=0 => y=1."""
    print("case 1 — G6: a request that generated nothing is not evidence")
    rows = ([mkrow(req_id=f"syn-{i}", generated=0, rtok=0, cap_hit=True)
             for i in range(130)]
            + [mkrow(req_id=f"real-{i}", generated=900, rtok=900) for i in range(8)])
    counts: dict = {}
    kept = R.apply_guards(rows, False, 32, counts)
    check("130 no-generation rows dropped", counts.get("g6_no_generation") == 130,
          f"g6_no_generation={counts.get('g6_no_generation')}")
    check("8 genuine rows survive", len(kept) == 8, f"eligible={len(kept)}")
    check("--min-generated 0 disables G6 (escape hatch intact)",
          len(R.apply_guards(rows, False, 0, {})) == 138)
    prior = np.mean([R.label_for(r.rtok, r.cap_hit, r.generated, 2000, 32)
                     for r in rows])
    prior_kept = np.mean([R.label_for(r.rtok, r.cap_hit, r.generated, 2000, 32)
                          for r in kept])
    check("narrowed G3: synthetic cap-hits are no longer deep positives",
          prior < 0.10, f"prior over ALL rows {prior:.3f} (was 1.0 under old G3)")
    check("kept rows keep their own labels", prior_kept == 0.0,
          f"prior over kept {prior_kept:.3f}")


def case2_g6_runs_before_g2() -> None:
    """G2 keeps a lean cap-hit as positive evidence. A zero-generation row
    must never reach that exception."""
    print("case 2 — G6 precedes the G2 lean-cap-hit exception")
    row = mkrow(req_id="lean0", generated=0, rtok=0, cap_hit=True,
                mode="enforce", route="lean")
    counts: dict = {}
    kept = R.apply_guards([row], False, 32, counts)
    check("dropped by G6, not resurrected by G2", len(kept) == 0
          and counts.get("g6_no_generation") == 1
          and "g2_lean_caphit_pos" not in counts, json.dumps(counts))
    counts2: dict = {}
    kept2 = R.apply_guards([mkrow(req_id="lean1", generated=800, rtok=800,
                                  cap_hit=True, mode="enforce", route="lean")],
                           False, 32, counts2)
    check("a real lean cap-hit is still kept as y=1",
          len(kept2) == 1 and counts2.get("g2_lean_caphit_pos") == 1
          and R.label_for(kept2[0].rtok, True, kept2[0].generated, 2000, 32) == 1.0)


def case3_tag_and_marker_exclusion(tmp) -> None:
    print("case 3 — G7: --exclude-tag and .synthetic-*.json markers")
    sink = os.path.join(tmp, "sink3")
    write_sink(sink, "20260725-183616", synthetic_rows(10, prefix="cap", generated=0,
                                                       rtok=0, cap_hit=True))
    write_sink(sink, "20260725-184327", synthetic_rows(6, prefix="live", generated=900,
                                                       rtok=900))
    rows, _sizes = R.load_sink(sink, {})
    check("both windows load", len(rows) == 16, f"rows={len(rows)}")
    rows, _ = R.load_sink(sink, {}, exclude_tags=["*-183616"])
    check("--exclude-tag drops the whole window", len(rows) == 6, f"rows={len(rows)}")
    with open(os.path.join(sink, ".synthetic-tap-capture-x.json"), "w",
              encoding="utf-8") as f:
        json.dump({"tool": "tap-capture", "req_ids": ["live-0", "live-1"]}, f)
    ids, tags = R.load_markers(sink)
    rows, _ = R.load_sink(sink, {}, marker_req_ids=ids)
    check("marker req_ids excluded by id", len(rows) == 14 and not tags,
          f"rows={len(rows)} marker_ids={len(ids)}")


def case4_temporal_split_is_out_of_sample() -> None:
    print("case 4 — the AUC gate now fits the past and scores the future")
    ts = np.concatenate([np.zeros(100), np.arange(1000.0, 1021.0)])  # seed + sink
    fit, hold = R.temporal_split(ts, 0.20)
    check("no row is on both sides", len(set(fit) & set(hold)) == 0)
    check("every seed row (ts=0) is on the fit side", set(range(100)) <= set(fit),
          f"fit={len(fit)} hold={len(hold)}")
    check("holdout is strictly newer than the fit",
          ts[hold].min() > ts[fit].max(), f"cut={ts[hold].min()}")
    fit2, hold2 = R.temporal_split(np.full(50, 7.0), 0.20)
    check("a single-timestamp burst cannot be split", len(fit2) == 0 and len(hold2) == 50)

    # A candidate fit on noise: in-sample AUC is high, out-of-sample is a coin
    # flip. The old in-sample gate passed exactly this.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 40))
    y = (rng.random(60) < 0.4).astype(float)
    ts_n = np.arange(60.0)
    _mu, _sd, _Vt, _w, s_in = R.fit_probe(X, y, 10.0, 10)
    rep: dict = {}
    fails = R.gate_out_of_sample(X, y, ts_n, 10.0, 10, 0.20, 8, 0.80, rep)
    check("noise candidate: in-sample AUC looks fine",
          R.auc(s_in, y) >= 0.70, f"in-sample={R.auc(s_in, y):.4f}")
    check("noise candidate REJECTED out-of-sample", bool(fails),
          f"oos_auc={rep.get('oos_auc')}")


def case5_monotone_rescale_is_rejected() -> None:
    """THE case. Rank-identical, median +0.10 — invisible to AUC and to every
    spearman, fatal at a fixed PN119_TDEEP."""
    print("case 5 — a monotone rescale (rank-identical, median +0.10) is rejected")
    rng = np.random.default_rng(7)
    inc = np.sort(rng.normal(0.45, 0.25, 300))
    cand = inc + 0.10                      # strictly monotone, ranks untouched
    y = (inc > np.median(inc)).astype(float)
    check("rank statistics are BLIND to it: spearman == 1.0",
          R.spearman(cand, inc) == 1.0, f"rho={R.spearman(cand, inc):.4f}")
    check("...and AUC is identical", R.auc(cand, y) == R.auc(inc, y),
          f"auc {R.auc(cand, y):.4f} == {R.auc(inc, y):.4f}")
    rep: dict = {}
    check("the stability gate passes it (by design — it is a RANK gate)",
          R.gate_stability(cand, inc, 0.90, 20, rep) == [], f"rho={rep['stability_rho']}")
    rep = {}
    fails = R.gate_calibration(cand, cand, inc, 0.495, 0.20, 0.40, 0.05, rep)
    check("the CALIBRATION gate rejects it", bool(fails), "; ".join(fails))
    check("...naming the median shift",
          any("median" in f for f in fails), f"median_shift={rep['median_shift']}")
    rep = {}
    check("an unshifted candidate is NOT rejected",
          R.gate_calibration(inc, inc, inc, 0.495, 0.20, 0.60, 0.05, rep) == [],
          f"deep_frac={rep['deep_frac_at_live_tdeep']} shift={rep['median_shift']}")
    rep = {}
    fails = R.gate_calibration(inc + 5.0, inc, inc, 0.495, 0.20, 0.40, 0.05, rep)
    check("a candidate that routes EVERYTHING deep is rejected",
          any("deep fraction" in f for f in fails), "; ".join(fails))


def case6_cursor_survives_a_prune(tmp) -> None:
    """`n_new = len(rows) - eligible_at_last_swap` goes negative on the first
    prune and the refit then skips forever while exiting 0."""
    print("case 6 — content-addressed cursor: pruning a window keeps n_new sane")
    sink = os.path.join(tmp, "sink6")
    write_sink(sink, "20260725-100000", synthetic_rows(10, prefix="a", ts=100.0))
    write_sink(sink, "20260725-110000", synthetic_rows(10, prefix="b", ts=200.0))
    rows, sizes = R.load_sink(sink, {})
    rows = R.apply_guards(rows, False, 32, {})
    cur0 = R.load_cursor(os.path.join(tmp, "nonexistent.json"))
    check("first run: every row is new", sum(R.is_new(r, cur0) for r in rows) == 20)
    cursor = {"schema": 2, "last_ts_seen": 200.0, "files": dict(sizes)}
    check("second run, nothing appended: n_new == 0",
          sum(R.is_new(r, cursor) for r in rows) == 0)

    write_sink(sink, "20260725-120000", synthetic_rows(5, prefix="c", ts=300.0))
    rows2 = R.apply_guards(R.load_sink(sink, {})[0], False, 32, {})
    check("a new window counts exactly its rows",
          sum(R.is_new(r, cursor) for r in rows2) == 5)

    os.unlink(os.path.join(sink, "feats-20260725-100000.bin"))
    os.unlink(os.path.join(sink, "meta-20260725-100000.jsonl"))
    rows3 = R.apply_guards(R.load_sink(sink, {})[0], False, 32, {})
    n_new = sum(R.is_new(r, cursor) for r in rows3)
    old_style = len(rows3) - 20        # the shipped formula, for contrast
    check("after PRUNING a window n_new is still 5", n_new == 5,
          f"n_new={n_new} (old formula would say {old_style})")
    check("the old formula would have gone negative (skip forever, exit 0)",
          old_style < 0, f"{old_style}")

    # append to an existing window: only the appended rows are new
    with open(os.path.join(sink, "meta-20260725-110000.jsonl"), "a",
              encoding="utf-8") as f:
        f.write(json.dumps({"req_id": "b-new", "row": 0, "score": 0.5,
                            "route": "deep", "prompt_tok": 10, "ts": 400.0,
                            "mode": "shadow", "explore": False}) + "\n")
        f.write(json.dumps({"req_id": "b-new", "finish": True, "score": 0.5,
                            "generated": 900, "ts": 400.0, "thinking": True,
                            "rtok": 900, "cap_hit": False, "explore": False,
                            "mode": "shadow"}) + "\n")
    rows4 = R.apply_guards(R.load_sink(sink, {})[0], False, 32, {})
    check("an APPEND to a known window counts 1 (byte offsets, not counts)",
          sum(R.is_new(r, cursor) for r in rows4) == 6,
          f"n_new={sum(R.is_new(r, cursor) for r in rows4)}")


def case7_seen_req_ids_persist(tmp) -> None:
    print("case 7 — update_reservoir stops re-offering already-rejected rows")
    sink = os.path.join(tmp, "sink7")
    state = os.path.join(tmp, "state7")
    os.makedirs(state, exist_ok=True)
    write_sink(sink, "20260725-130000", synthetic_rows(40, prefix="s", ts=100.0))
    rows = R.apply_guards(R.load_sink(sink, {})[0], False, 32, {})
    rng = np.random.default_rng(1)
    c1: dict = {}
    R_X, R_meta = R.update_reservoir(state, rows, 10, rng, c1)
    check("reservoir honours its cap", len(R_X) == 10, f"size={len(R_X)}")
    check("all 40 rows were offered once", c1["reservoir_offered_new"] == 40)
    c2: dict = {}
    R.update_reservoir(state, rows, 10, rng, c2)
    check("second run re-offers NONE of them", c2["reservoir_offered_new"] == 0,
          f"offered={c2['reservoir_offered_new']} skipped={c2['reservoir_skipped_seen']}")
    z = np.load(os.path.join(state, "reservoir.npz"), allow_pickle=True)
    check("seen_req_ids persisted", "seen_req_ids" in z and len(z["seen_req_ids"]) == 40,
          f"n={len(z['seen_req_ids']) if 'seen_req_ids' in z else 'absent'}")
    check("reservoir meta carries `generated` (G3 needs it)",
          all("generated" in json.loads(s) for s in z["meta"]))


def case8_memmap_only_materialises_winners(tmp) -> None:
    print("case 8 — features are memmapped; only reservoir winners materialise")
    sink = os.path.join(tmp, "sink8")
    X = write_sink(sink, "20260725-140000", synthetic_rows(20, prefix="m", ts=100.0))
    rows = R.apply_guards(R.load_sink(sink, {})[0], False, 32, {})
    check("load_sink returns rows with NO feature data attached",
          all(r.x is None for r in rows))
    want = [3, 11]
    got = R.bf16_rows(os.path.join(sink, "feats-20260725-140000.bin"), want)
    ref = (X.view(np.uint32) >> np.uint32(16) << np.uint32(16)).view(np.float32)
    check("bf16 widen is exact (numpy shift == torch .view(bfloat16).float())",
          np.array_equal(got, ref[want]), f"max|diff|={np.abs(got - ref[want]).max()}")
    R.materialise([rows[3]])
    check("materialise fills only what it was asked for",
          rows[3].x is not None and all(rows[i].x is None for i in (0, 1, 2, 4)))
    big = os.path.join(tmp, "sink8b")
    write_sink(big, "20260725-141000", synthetic_rows(200, prefix="n", ts=100.0))
    state = os.path.join(tmp, "state8")
    os.makedirs(state, exist_ok=True)
    rows = R.apply_guards(R.load_sink(big, {})[0], False, 32, {})
    R.update_reservoir(state, rows, 5, np.random.default_rng(3), {})
    mat = sum(r.x is not None for r in rows)
    # reservoir sampling materialises cap + ~cap*ln(n/cap) winners, never n:
    # the losers' .bin pages are never faulted in.
    check("with cap=5 only the winners of 200 rows materialise", mat <= 40,
          f"materialised={mat}/200 (a full read would be 200)")


def case9_capture_source_refusal() -> None:
    print("case 9 — the seed must share the sink's capture source")
    tap = NEEDFIT / "tap-features-20260725.safetensors"
    off = NEEDFIT / "lens-features-20260725.safetensors"
    check("tap capture classified 'tap'", R.seed_capture_source(str(tap)) == "tap",
          R.seed_capture_source(str(tap)))
    check("HF offline capture classified 'offline-hf'",
          R.seed_capture_source(str(off)) == "offline-hf",
          R.seed_capture_source(str(off)))
    check("the sink's own source is the tap", R.SINK_CAPTURE_SOURCE == "tap")
    out = subprocess.run(
        [PY, str(REPO / "fixes/refit_pn119_probe.py"), "--dry-run", "--force",
         "--seed-features", str(off)],
        capture_output=True, text=True, timeout=600)
    check("refit REFUSES an offline-hf seed against a tap sink (exit 2)",
          out.returncode == 2 and "REFUSED" in out.stdout,
          out.stdout.strip().splitlines()[0][:120] if out.stdout else out.stderr[-200:])


def case10_cli_exit_codes(tmp) -> None:
    print("case 10 — skip has its own exit code (3), swap has 0, reject has 2")
    sink = os.path.join(tmp, "sink10")
    state = os.path.join(tmp, "state10")
    out = os.path.join(tmp, "live10", "probe.npz")
    meta = [dict(req_id=f"e-{i}", generated=900, ts=100.0 + i,
                 rtok=(3000 if i % 3 else 100)) for i in range(40)]
    write_sink(sink, "20260725-150000", meta, feat_rows=learnable_feats(meta))
    base = [PY, str(REPO / "fixes/refit_pn119_probe.py"), "--sink", sink,
            "--state", state, "--out", out, "--no-seed"]
    r1 = subprocess.run(base + ["--min-new", "500"], capture_output=True,
                        text=True, timeout=900)
    check("not enough new rows => exit 3, not 0", r1.returncode == 3,
          f"rc={r1.returncode} {r1.stdout.strip()[:100]}")
    r2 = subprocess.run(base + ["--min-new", "5", "--max-deep-frac", "0.9"],
                        capture_output=True, text=True, timeout=900)
    check("a passing candidate swaps and exits 0", r2.returncode == 0
          and os.path.isfile(out), f"rc={r2.returncode} {r2.stdout.strip()[:160]}")
    cur = json.load(open(os.path.join(state, "cursor.json"), encoding="utf-8"))
    check("cursor is content-addressed (per-file byte offsets)",
          cur.get("schema") == 2 and cur.get("files")
          and "eligible_at_last_swap" not in cur, json.dumps(cur)[:160])
    r3 = subprocess.run(base + ["--min-new", "5", "--max-deep-frac", "0.9"],
                        capture_output=True, text=True, timeout=900)
    check("immediately re-run: nothing new => exit 3", r3.returncode == 3,
          f"rc={r3.returncode} {r3.stdout.strip()[:100]}")


def case11_real_sink_is_clean() -> None:
    """The live sink after the 2026-07-25 quarantine. Numbers, not vibes."""
    print("case 11 — the LIVE sink parses clean (quarantine landed)")
    sink = str(NEEDFIT / "pn119-sink")
    if not os.path.isdir(sink):
        check("live sink present", False, "no sink dir — skipped")
        return
    counts: dict = {}
    rows = R.apply_guards(R.load_sink(sink, counts)[0], False, 32, counts)
    check("no no-generation rows left in the training set",
          counts.get("g6_no_generation", 0) == 0,
          f"g6={counts.get('g6_no_generation', 0)} eligible={len(rows)}")
    q = NEEDFIT / "pn119-sink/.quarantine"
    check("the poison is quarantined, not deleted",
          q.is_dir() and len(list(q.glob("meta-*.jsonl"))) == 4,
          f"{len(list(q.glob('meta-*.jsonl'))) if q.is_dir() else 0} meta files")
    counts_q: dict = {}
    rows_q = R.apply_guards(R.load_sink(str(q), counts_q)[0], False, 32, counts_q)
    check("quarantined rows are still readable (b3 reference data)",
          counts_q.get("scored") == 130 and len(rows_q) == 0,
          f"scored={counts_q.get('scored')} g6={counts_q.get('g6_no_generation')}")
    if rows:
        y = np.array([R.label_for(r.rtok, r.cap_hit, r.generated, 2000, 32)
                      for r in rows])
        print(f"  [info] live sink: {len(rows)} eligible, sink-only prior "
              f"{y.mean():.3f}")


def main() -> int:
    print("PN119 v2 refit guards + gates\n")
    tmp = tempfile.mkdtemp(prefix="pn119-refit-test-")
    try:
        case1_g6_drops_no_generation()
        case2_g6_runs_before_g2()
        case3_tag_and_marker_exclusion(tmp)
        case4_temporal_split_is_out_of_sample()
        case5_monotone_rescale_is_rejected()
        case6_cursor_survives_a_prune(tmp)
        case7_seen_req_ids_persist(tmp)
        case8_memmap_only_materialises_winners(tmp)
        case9_capture_source_refusal()
        case10_cli_exit_codes(tmp)
        case11_real_sink_is_clean()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if _fails:
        print(f"FAILED: {len(_fails)} — {', '.join(_fails)}")
        return 1
    print("ALL PASS")
    print("VERDICT: no-generation rows cannot be labels; a rank-identical "
          "rescale cannot pass; the cursor survives a prune; a skip is exit 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
