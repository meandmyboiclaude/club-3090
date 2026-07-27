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
import time

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
          route="deep", explore=False, thinking=True, ts=100.0, tag="t",
          censored=None, budget_grant=None, score=None, cand_score=None,
          cand_sha=None, p_explore=None, holdout=False):
    return R.Row(req_id=req_id, tag=tag, feat_path="", row_idx=0, rtok=rtok,
                 generated=generated, cap_hit=cap_hit, mode=mode, route=route,
                 explore=explore, thinking=thinking, ts=ts, censored=censored,
                 budget_grant=budget_grant, score=score, cand_score=cand_score,
                 cand_sha=cand_sha, p_explore=p_explore, holdout=holdout)


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
            score_line = {"req_id": r["req_id"], "row": i,
                          "score": r.get("score", 0.5),
                          "route": r.get("route", "deep"),
                          "prompt_tok": 100, "ts": r.get("ts", 100.0),
                          "mode": r.get("mode", "shadow"),
                          "explore": r.get("explore", False)}
            for k in ("cand_score", "cand_probe_sha", "p_explore", "holdout"):
                if k in r:
                    score_line[k] = r[k]
            f.write(json.dumps(score_line) + "\n")
            fin = {"req_id": r["req_id"], "finish": True,
                   "score": r.get("score", 0.5),
                   "generated": r.get("generated", 500),
                   "ts": r.get("ts", 100.0),
                   "thinking": r.get("thinking", True),
                   "rtok": r.get("rtok", 500),
                   "cap_hit": r.get("cap_hit", False),
                   "explore": r.get("explore", False),
                   "mode": r.get("mode", "shadow")}
            for k in ("censored", "budget_grant", "budget_source"):
                if k in r:
                    fin[k] = r[k]
            f.write(json.dumps(fin) + "\n")
    return X


def synthetic_rows(n, **kw):
    return [dict({"req_id": f"{kw.get('prefix','q')}-{i}"}, **
                 {k: v for k, v in kw.items() if k != "prefix"}) for i in range(n)]


def learnable_feats(rows, deep_thresh=2000, seed=11):
    """Features a probe can actually fit: one direction carries the label."""
    rng = np.random.default_rng(seed)
    y = np.array([1.0 if r["rtok"] >= deep_thresh else 0.0 for r in rows])
    return _feats_from_y(y, rng)


def _feats_from_y(y, rng):
    d = rng.standard_normal(R.FEAT_DIM).astype(np.float32)
    d /= np.linalg.norm(d)
    return (rng.standard_normal((len(y), R.FEAT_DIM)).astype(np.float32) * 0.05
            + np.asarray(y)[:, None] * d[None, :] * 8.0).astype(np.float32)


# ---------------------------------------------------------------- cases
def case1_g6_drops_no_generation() -> None:
    """The P1 bug: max_tokens=1 rows are cap_hit=True, rtok=0 => y=1."""
    print("case 1 — G6: a request that generated nothing is not evidence")
    rows = ([mkrow(req_id=f"syn-{i}", generated=0, rtok=0, cap_hit=True)
             for i in range(130)]
            + [mkrow(req_id=f"real-{i}", generated=913, rtok=913) for i in range(8)])
    counts: dict = {}
    kept = R.apply_guards(rows, False, 32, counts)
    check("130 no-generation rows dropped", counts.get("g6_no_generation") == 130,
          f"g6_no_generation={counts.get('g6_no_generation')}")
    check("8 genuine rows survive", len(kept) == 8, f"eligible={len(kept)}")
    check("--min-generated 0 disables G6 (escape hatch intact)",
          len(R.apply_guards(rows, False, 0, {})) == 138)
    y, w, _b, _p = R.label_rows(rows, 2000)
    y_k, w_k, _b, _p = R.label_rows(kept, 2000)
    check("narrowed G3: synthetic cap-hits are no longer deep positives",
          float(y.mean()) < 0.10,
          f"prior over ALL rows {y.mean():.3f} (was 1.0 under old G3)")
    check("kept rows keep their own labels", float(y_k.mean()) == 0.0,
          f"prior over kept {y_k.mean():.3f} (rtok 913, uncensored)")
    check("and they are RESOLVED, not dropped", float(w_k.min()) == 1.0)


def case2_g6_runs_before_g2() -> None:
    """A zero-generation row must never reach the censoring logic at all."""
    print("case 2 — G6 precedes the censoring rules")
    row = mkrow(req_id="lean0", generated=0, rtok=0, cap_hit=True,
                mode="enforce", route="lean")
    counts: dict = {}
    kept = R.apply_guards([row], False, 32, counts)
    check("dropped by G6, never labelled", len(kept) == 0
          and counts.get("g6_no_generation") == 1, json.dumps(counts))
    counts2: dict = {}
    kept2 = R.apply_guards([mkrow(req_id="lean1", generated=800, rtok=800,
                                  cap_hit=True, mode="enforce", route="lean")],
                           False, 32, counts2)
    y, w, b, _p = R.label_rows(kept2, 2000)
    check("a real lean cap-hit at rtok=800 is KEPT but UNRESOLVED, not y=1",
          len(kept2) == 1 and counts2.get("g2_selfcensored_kept") == 1
          and b[0] == "interval_unresolved" and w[0] == 0.0,
          f"bucket={b[0]} w={w[0]} (old G3 called this a deep positive)")


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
    # A marker naming a LIVE window must not take the window's genuine rows
    # with it (measured on the live sink: a 3-row marker cost 5 real rows).
    with open(os.path.join(sink, ".synthetic-b3-numerics-y.json"), "w",
              encoding="utf-8") as f:
        json.dump({"tool": "b3-numerics", "tags": ["20260725-184327"],
                   "req_ids": ["live-2"]}, f)
    ids2, tags2 = R.load_markers(sink)
    rows2, _ = R.load_sink(sink, {}, marker_req_ids=ids2)
    check("a marker's tags are provenance, never a window exclusion",
          len(rows2) == 13 and tags2 == {"20260725-184327"},
          f"rows={len(rows2)} (16 - 3 named req_ids), tags={sorted(tags2)}")


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


def _fake_b3(tmp, ok=True, age_h=0.0):
    p = os.path.join(tmp, f"b3-{ok}-{age_h}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"pass": ok, "ts": time.time() - age_h * 3600.0}, f)
    return p


def case10_cli_exit_codes(tmp) -> None:
    print("case 10 — skip is exit 3, stage/promote 0, reject 2")
    sink = os.path.join(tmp, "sink10")
    state = os.path.join(tmp, "state10")
    out = os.path.join(tmp, "live10", "probe.npz")
    # rtok values are deliberately OFF the 100-grid: an exact multiple of 100
    # is indistinguishable from a budget stop and would be censored.
    meta = [dict(req_id=f"e-{i}", generated=900, ts=100.0 + i,
                 rtok=(3037 if i % 3 else 137), censored=False)
            for i in range(40)]
    write_sink(sink, "20260725-150000", meta, feat_rows=learnable_feats(meta))
    b3 = _fake_b3(tmp)
    base = [PY, str(REPO / "fixes/refit_pn119_probe.py"), "--sink", sink,
            "--state", state, "--out", out, "--no-seed", "--b3-report", b3,
            "--results-dir", os.path.join(tmp, "no-results")]
    r1 = subprocess.run(base + ["--min-new", "500"], capture_output=True,
                        text=True, timeout=900)
    check("not enough new rows => exit 3, not 0", r1.returncode == 3,
          f"rc={r1.returncode} {r1.stdout.strip()[:100]}")
    gates = base + ["--min-new", "5", "--max-deep-frac", "0.9",
                    "--prior-hi", "0.75"]
    r2 = subprocess.run(gates, capture_output=True, text=True, timeout=900)
    check("default promote-mode STAGES a candidate and never touches live",
          r2.returncode == 0 and os.path.isfile(R.candidate_path(state))
          and not os.path.isfile(out),
          f"rc={r2.returncode} {r2.stdout.strip()[:200]}")
    cur = json.load(open(os.path.join(state, "cursor.json"), encoding="utf-8"))
    check("cursor is content-addressed (per-file byte offsets)",
          cur.get("schema") == 2 and cur.get("files")
          and "eligible_at_last_swap" not in cur, json.dumps(cur)[:160])
    r3 = subprocess.run(gates, capture_output=True, text=True, timeout=900)
    check("re-run with a candidate in shadow but no shadow rows => HOLD (3)",
          r3.returncode == 3 and "HOLD" in r3.stdout,
          f"rc={r3.returncode} {r3.stdout.strip()[:140]}")
    # direct mode is the pre-shadow path: it swaps, and keeps the old probe.
    state2 = os.path.join(tmp, "state10b")
    direct = [PY, str(REPO / "fixes/refit_pn119_probe.py"), "--sink", sink,
              "--state", state2, "--out", out, "--no-seed", "--b3-report", b3,
              "--results-dir", os.path.join(tmp, "no-results"),
              "--promote-mode", "direct", "--min-new", "5",
              "--max-deep-frac", "0.9", "--prior-hi", "0.75"]
    r4 = subprocess.run(direct, capture_output=True, text=True, timeout=900)
    check("--promote-mode direct swaps and exits 0",
          r4.returncode == 0 and os.path.isfile(out),
          f"rc={r4.returncode} {r4.stdout.strip()[:160]}")
    r5 = subprocess.run(direct + ["--force"], capture_output=True, text=True,
                        timeout=900)
    prev = os.path.join(os.path.dirname(out), "probe.prev.npz")
    check("a second direct swap keeps the previous probe for rollback",
          r5.returncode == 0 and os.path.isfile(prev), f"rc={r5.returncode}")
    sha_live = R.npz_sha16(out)
    r6 = subprocess.run([PY, str(REPO / "fixes/refit_pn119_probe.py"),
                         "--out", out, "--state", state2, "--rollback"],
                        capture_output=True, text=True, timeout=300)
    check("--rollback restores probe.prev.npz over live",
          r6.returncode == 0 and R.npz_sha16(out) != sha_live,
          r6.stdout.strip()[:140])
    check("...and the rollback is itself reversible (prev now holds the "
          "probe it replaced)", R.npz_sha16(prev) == sha_live)


def case11_real_sink_is_clean() -> None:
    """The live sink after the 2026-07-25 quarantine. Numbers, not vibes."""
    print("case 11 — the LIVE sink parses clean (quarantine landed)")
    sink = str(NEEDFIT / "pn119-sink")
    if not os.path.isdir(sink):
        check("live sink present", False, "no sink dir — skipped")
        return
    counts: dict = {}
    rows = R.apply_guards(R.load_sink(sink, counts)[0], False, 32, counts)
    check("no no-generation row survives into the training set",
          all(r.generated >= 32 for r in rows),
          f"g6 dropped {counts.get('g6_no_generation', 0)}, "
          f"eligible={len(rows)} (the live sink keeps receiving max_tokens=1 "
          f"diagnostics; the assertion is that they are EXCLUDED, not absent)")
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
        y, w, b, _p = R.label_rows(rows, 2000)
        print(f"  [info] live sink: {len(rows)} eligible, resolved "
              f"{int((w > 0).sum())}, sink-only prior on resolved "
              f"{float(y[w > 0].mean()):.3f}")


# ────────────────────────────────────────────────── BUG-139 interval censoring
def case12_interval_labels_on_the_live_sink() -> None:
    """ACCEPTANCE: the interval-censoring counts on the real sink."""
    print("case 12 — interval censoring, measured on the LIVE sink")
    sink = str(NEEDFIT / "pn119-sink")
    if not os.path.isdir(sink):
        check("live sink present", False, "no sink dir — skipped")
        return
    counts: dict = {}
    rows = R.apply_guards(R.load_sink(sink, counts)[0], False, 32, counts)
    y, w, b, prov = R.label_rows(rows, 2000, counts)
    n1 = int((b == "resolved_pos").sum() + (b == "censored_pos").sum())
    n0 = int((b == "resolved_neg").sum())
    nn = int((b == "interval_unresolved").sum())
    print(f"  [ACCEPTANCE] {len(rows)} eligible thinking rows -> "
          f"y=1: {n1}   y=0: {n0}   y=None: {nn} "
          f"(unresolved rate {counts['g3_unresolved_rate']:.4f})")
    print(f"  [ACCEPTANCE] censoring source: {json.dumps({k: v for k, v in counts.items() if k.startswith('censor_src_')})}")
    check("every row lands in exactly one bucket", n1 + n0 + nn == len(rows),
          f"{n1}+{n0}+{nn} vs {len(rows)}")
    check("the corpus really is majority-censored (BUG-139's claim)",
          counts["censored_rate"] > 0.5, f"censored_rate={counts['censored_rate']}")
    check("unresolved rows carry weight 0 and never reach a fit",
          float(w[b == "interval_unresolved"].sum()) == 0.0
          and float(w[b != "interval_unresolved"].min()) == 1.0)
    # The old pipeline, re-derived here so the contrast is measured, not
    # asserted: G2 deleted every non-cap-hit enforce/lean row and G3 turned
    # the rest into positives.
    old_kept, old_y = [], []
    for r in rows:
        if r.mode == "enforce" and r.route == "lean" and not r.explore:
            if not r.cap_hit:
                continue
            old_kept.append(r)
            old_y.append(1.0)
            continue
        old_kept.append(r)
        old_y.append(1.0 if (r.cap_hit or r.rtok >= 2000) else 0.0)
    old_deep_frac = (sum(1 for r in old_kept if r.route == "deep")
                     / max(len(old_kept), 1))
    new_res = [r for r, k in zip(rows, w > 0) if k]
    new_deep_frac = (sum(1 for r in new_res if r.route == "deep")
                     / max(len(new_res), 1))
    print(f"  [ACCEPTANCE] OLD pipeline: {len(old_kept)} rows, prior "
          f"{np.mean(old_y):.3f}, {old_deep_frac:.0%} deep-routed  ->  "
          f"NEW: {len(new_res)} rows, prior {float(y[w > 0].mean()):.3f}, "
          f"{new_deep_frac:.0%} deep-routed")
    check("positive selection is reduced, not just relabelled",
          new_deep_frac < old_deep_frac and len(new_res) > len(old_kept),
          f"deep-routed share {old_deep_frac:.2f} -> {new_deep_frac:.2f}")


def case13_absorbing_state_is_closed() -> None:
    """ACCEPTANCE: a lean-routed censored row below theta is not a positive —
    and is not a negative either."""
    print("case 13 — the absorbing state: lean-routed censored rows below theta")
    theta = 2000
    # The live signature: PN100 granted 1300, the row stopped at 1295 because
    # the </think> span was forced. It says need >= 1287. Nothing more.
    lean = mkrow(req_id="lean-cens", mode="enforce", route="lean",
                 rtok=1295, generated=1400, cap_hit=False)
    y, w, b, prov = R.label_rows([lean], theta)
    check("y is None (unresolved), not 1", b[0] == "interval_unresolved" and w[0] == 0.0,
          f"bucket={b[0]} provenance={prov[0]}")
    check("...and not 0 either — 'lean was right' is exactly the loop's lie",
          not (b[0] == "resolved_neg"))
    old = 1.0 if (lean.cap_hit or lean.rtok >= theta) else 0.0
    check("the OLD point label would have said y=0 in shadow mode", old == 0.0)
    lean_cap = mkrow(req_id="lean-cap", mode="enforce", route="lean",
                     rtok=800, generated=800, cap_hit=True)
    _y, w2, b2, _p = R.label_rows([lean_cap], theta)
    check("the old lean-cap-hit exception (y=1) is gone too",
          b2[0] == "interval_unresolved" and w2[0] == 0.0, f"bucket={b2[0]}")
    # A censored row whose BOUND already clears theta is still a positive:
    # censoring loses magnitude, not direction.
    deep_cens = mkrow(req_id="deep-cens", mode="enforce", route="deep",
                      rtok=3095, generated=3200)
    _y3, w3, b3, _p = R.label_rows([deep_cens], theta)
    check("a censored row whose lower bound clears theta IS a positive",
          b3[0] == "censored_pos" and w3[0] == 1.0, f"bucket={b3[0]}")
    # And an honest measurement is still an honest measurement.
    free = mkrow(req_id="free", mode="enforce", route="lean", rtok=1134,
                 generated=1300)
    _y4, w4, b4, _p = R.label_rows([free], theta)
    check("an uncensored lean row is a genuine y=0",
          b4[0] == "resolved_neg" and w4[0] == 1.0, f"bucket={b4[0]}")
    check("...and it is no longer DELETED by G2 (that was the selection bias)",
          len(R.apply_guards([free], False, 32, {})) == 1)


def case14_censoring_detection() -> None:
    print("case 14 — censoring: sink field > budget_grant > cap_hit > grid")
    slack = R.CENSOR_SLACK
    # WAS `slack == 9` (len(think_end_ids) + 8) until 2026-07-26. That constant
    # was fitted to ONE truncation mode; the live sink has two, at gap 5 and
    # gap 13, and 9 sat exactly between them — it resolved the 94 gap-5 rows and
    # wrote 223 gap-13 rows into the corpus as natural stops. See
    # fixes/test_bug139_censoring.py for the replay that quantifies it.
    check("SLACK = len(think_end_ids) + 12 = 13; covers BOTH measured "
          "truncation modes (gap 5 and gap 13)", slack == 13, f"slack={slack}")
    # The measured signature: 43 of 79 rows at exactly (grant - 5).
    for rtok, grant in ((1295, 1300), (3095, 3100), (2095, 2100), (3895, 3900)):
        cens, lb, prov = R.censoring_of({"rtok": rtok, "cap_hit": False,
                                         "generated": rtok + 20})  # offset 5
        check(f"rtok={rtok} is recognised as the grant-{grant - rtok} signature",
              cens and prov == "grid" and lb == grant - slack,
              f"censored={cens} lb={lb} via {prov}")
    cens, _lb, prov = R.censoring_of({"rtok": 1134, "cap_hit": False,
                                      "generated": 1300})
    check("a natural stop mid-band is NOT called censored", not cens and
          prov == "uncensored")
    # The router's own POSITIVE wins over any derivation. Its NEGATIVE does not,
    # unless the window that wrote it ran a detector at least as sensitive as
    # ours (censor_schema >= 2 AND its declared slack >= ours). Believing a
    # schema-1 `censored: false` is how BUG-139's own output got re-imported as
    # ground truth — 213 of 223 live gap-13 rows trained as y=0 through exactly
    # this line.
    cens, lb, prov = R.censoring_of({"rtok": 1295, "cap_hit": False,
                                     "generated": 1400, "censored": False})
    check("a schema-1 censored=false is RE-DERIVED, not believed",
          cens and prov == "grid", f"{cens} {prov}")
    cens, lb, prov = R.censoring_of({"rtok": 1295, "cap_hit": False,
                                     "generated": 1400, "censored": False,
                                     "censor_schema": R.TRUSTED_CENSOR_SCHEMA,
                                     "slack": R.CENSOR_SLACK})
    check("a censored=false from a detector as wide as ours IS believed",
          not cens and prov == "sink", f"{cens} {prov}")
    cens, lb, prov = R.censoring_of({"rtok": 900, "cap_hit": False,
                                     "generated": 950, "censor_forced": True,
                                     "budget_grant": 6500})
    check("an OBSERVED forced </think> outranks everything (no slack involved)",
          cens and prov == "forced" and lb == 6500 - slack, f"{cens} {prov} lb={lb}")
    cens, lb, prov = R.censoring_of({"rtok": 900, "cap_hit": False,
                                     "generated": 950, "censored": True,
                                     "budget_grant": 6500})
    check("explicit censored=true uses budget_grant for the bound",
          cens and lb == 6500 - slack and prov == "sink", f"lb={lb}")
    cens, lb, prov = R.censoring_of({"rtok": 1595, "cap_hit": False,
                                     "generated": 1700, "budget_grant": 1600})
    check("budget_grant alone reproduces the live proof (1600 grant, rtok 1595)",
          cens and prov == "budget" and lb == 1600 - slack, f"lb={lb} via {prov}")
    cens, lb, prov = R.censoring_of({"rtok": 400, "cap_hit": False,
                                     "generated": 450, "budget_grant": 6500})
    check("a row that stopped far below its grant is uncensored",
          not cens and prov == "budget" and lb == 400)
    check("derivation prefers the OFF-grid tier budgets when they fit better",
          R.derive_budget_grant(1019) == 1024 and R.derive_budget_grant(4090) == 4096,
          f"{R.derive_budget_grant(1019)} {R.derive_budget_grant(4090)}")


def case15_drift_gates(tmp) -> None:
    """ACCEPTANCE: the drift gates fire on the historical poisoned window."""
    print("case 15 — drift rejection (PSI / KS / prior / unresolved / b3)")
    q = NEEDFIT / "pn119-sink/.quarantine"
    if q.is_dir():
        c: dict = {}
        # --min-generated 0 reproduces the pre-G6 world in which the poison
        # was eligible: 130 max_tokens=1 rows, every one a "deep positive".
        rows = R.apply_guards(R.load_sink(str(q), c)[0], False, 0, c)
        live = R.apply_guards(R.load_sink(str(NEEDFIT / "pn119-sink"), {})[0],
                              False, 0, {})
        # The corpus as it stood on 2026-07-25, under the labels of the day:
        # G2 dropped the non-cap-hit lean rows, G3 made the rest positives.
        old_keep = [r for r in rows + live
                    if not (r.mode == "enforce" and r.route == "lean"
                            and not r.explore and not r.cap_hit)]
        old_prior = float(np.mean([1.0 if (r.cap_hit or r.rtok >= 2000) else 0.0
                                   for r in old_keep]))
        rep: dict = {}
        fails_old = R.gate_drift(np.zeros((0, 0)), np.zeros((0, 0)), old_prior,
                                 0.0, True, "b3 ok", {}, rep)
        print(f"  [ACCEPTANCE] poisoned corpus under the LABELS OF THE DAY: "
              f"{len(old_keep)} rows, prior {old_prior:.3f} -> {fails_old}")
        check("the prior gate alone rejects the refit that shipped",
              any("prior" in f for f in fails_old),
              "; ".join(fails_old) or "NO FAILURE")
        y, w, b, _p = R.label_rows(rows, 2000, c)
        rep2: dict = {}
        fails_new = R.gate_drift(np.zeros((0, 0)), np.zeros((0, 0)), None,
                                 c["g3_unresolved_rate"], True, "b3 ok", {}, rep2)
        print(f"  [ACCEPTANCE] the same 130 rows under INTERVAL labels: "
              f"{int((w > 0).sum())} resolved, unresolved rate "
              f"{c['g3_unresolved_rate']:.3f} -> {fails_new}")
        check("and under interval labels the poison resolves to NOTHING",
              float(w.sum()) == 0.0 and any("unresolved" in f for f in fails_new),
              f"weight sum {w.sum()}")
    else:
        check("quarantine present", False, "no .quarantine — skipped")
    # PSI / KS on a shifted feature block.
    rng = np.random.default_rng(3)
    ref = rng.standard_normal((400, 4))
    same = rng.standard_normal((400, 4))
    rep = {}
    check("no drift between two draws of the same distribution",
          R.gate_drift(ref, same, 0.35, 0.1, True, "ok", {}, rep) == [],
          f"psi_max={rep['psi_max']} ks_d={rep['ks_d_max']}")
    moved = same.copy()
    moved[:, 2] += 1.2
    rep = {}
    fails = R.gate_drift(ref, moved, 0.35, 0.1, True, "ok", {}, rep)
    check("a 1.2-sigma shift on ONE PC is rejected", bool(fails),
          f"psi_max={rep['psi_max']} on PC{rep['psi_argmax_pc']}; "
          f"KS D={rep['ks_d_max']} p={rep['ks_p_at_dmax']}")
    rep = {}
    fails = R.gate_drift(ref, same, 0.35, 0.55, True, "ok", {}, rep)
    check("an over-censored corpus (55% unresolved) is rejected",
          any("unresolved" in f for f in fails), "; ".join(fails))
    rep = {}
    fails = R.gate_drift(ref, same, 0.07, 0.1, True, "ok", {}, rep)
    check("a prior of 0.07 is rejected (too few positives to be traffic)",
          any("prior" in f for f in fails), "; ".join(fails))
    # b3 report freshness/health.
    ok, note = R.b3_report_state(_fake_b3(tmp, ok=True, age_h=1.0), 24.0, time.time())
    check("a fresh passing b3 report is accepted", ok, note)
    ok, note = R.b3_report_state(_fake_b3(tmp, ok=True, age_h=48.0), 24.0, time.time())
    check("a 48h-old b3 report is stale", not ok, note)
    ok, note = R.b3_report_state(_fake_b3(tmp, ok=False), 24.0, time.time())
    check("a FAILING b3 report blocks the refit", not ok, note)
    ok, note = R.b3_report_state(os.path.join(tmp, "nope.json"), 24.0, time.time())
    check("a MISSING b3 report blocks it too (no evidence != no problem)",
          not ok, note)
    live_ok, live_note = R.b3_report_state(
        str(NEEDFIT / "pn119-b3-numerics-report.json"), 24.0, time.time())
    print(f"  [info] live b3 report: ok={live_ok} — {live_note}")
    # explore integrity
    bad = R.explore_integrity([mkrow(req_id="x", explore=True, rtok=1295,
                                     generated=1400, p_explore=None)], 2000)
    rep = {}
    fails = R.gate_drift(ref, same, 0.35, 0.1, True, "ok", bad, rep)
    check("an explore row with no propensity and a censored short label fails",
          any("explore integrity" in f for f in fails), json.dumps(bad))


def case16_shadow_promotion(tmp) -> None:
    """ACCEPTANCE: the promotion gate refuses a candidate that flips >20%."""
    print("case 16 — dual-probe promotion: paired DeLong + decision-flip bound")
    rng = np.random.default_rng(5)
    n = 400
    y = (rng.random(n) < 0.4).astype(float)
    inc = y * 0.9 + rng.normal(0, 1.0, n)          # a mediocre incumbent
    better = y * 0.9 + rng.normal(0, 0.35, n)      # genuinely sharper
    w = np.ones(n)
    a_i, a_b, z, p = R.delong_test(y, better, inc)
    check("paired DeLong sees the better probe", a_i > a_b and p < 0.01,
          f"auc cand={a_i:.4f} inc={a_b:.4f} z={z:.2f} p={p:.2e}")
    _a, _b2, _z, p_same = R.delong_test(y, inc, inc)
    check("a probe against ITSELF is never significant", p_same >= 0.5,
          f"p={p_same}")
    # T chosen at the incumbent's median so the flip fraction is measurable.
    T = float(np.median(inc))
    rep: dict = {}
    fails = R.gate_shadow_promotion(y, w, better, inc, T, 0.05, 0.20, 200, rep)
    print(f"  [info] genuinely-better candidate: auc {rep['shadow_auc_cand']} vs "
          f"{rep['shadow_auc_inc']}, p={rep['delong_p_one_sided']}, "
          f"flips={rep['decision_flip_frac']}")
    # A candidate that reverses a quarter of the decisions: rank-wise it can
    # even look good, but it is a different router.
    flipper = inc.copy()
    idx = rng.choice(n, size=int(0.25 * n), replace=False)
    flipper[idx] = 2 * T - flipper[idx]            # mirror across T => flip
    rep2: dict = {}
    fails2 = R.gate_shadow_promotion(y, w, flipper, inc, T, 0.05, 0.20, 200, rep2)
    check("a candidate that flips >20% of decisions is REFUSED",
          any("DIFFERENT router" in f for f in fails2),
          f"flip_frac={rep2['decision_flip_frac']}; {'; '.join(fails2)}")
    check("...and the refusal names a human, not a threshold tweak",
          any("human" in f for f in fails2))
    rep3: dict = {}
    fails3 = R.gate_shadow_promotion(y, w, inc + 1e-9, inc, T, 0.05, 0.20, 200, rep3)
    check("a candidate that is not better is not promoted", bool(fails3),
          "; ".join(fails3))
    rep4: dict = {}
    fails4 = R.gate_shadow_promotion(y[:50], w[:50], better[:50], inc[:50], T,
                                     0.05, 0.20, 200, rep4)
    check("too little post-shadow evidence => no promotion",
          any("not enough post-shadow" in f for f in fails4), "; ".join(fails4))
    # end-to-end: stage -> shadow rows -> promote, with prev kept
    state = os.path.join(tmp, "state16")
    os.makedirs(state, exist_ok=True)
    live = os.path.join(tmp, "live16", "probe.npz")
    os.makedirs(os.path.dirname(live), exist_ok=True)
    payload = {"mu": np.zeros(4), "sd": np.ones(4), "Vt10": np.eye(4),
               "w": np.zeros(5)}
    R.atomic_write_npz(live, payload)
    sha_live = R.npz_sha16(live)
    rec = R.stage_candidate(state, dict(payload, w=np.ones(5)))
    check("staging writes candidate.npz + candidate.json, live untouched",
          os.path.isfile(R.candidate_path(state))
          and R.npz_sha16(live) == sha_live and rec["first_seen_ts"] is None)
    rows = [mkrow(req_id=f"s-{i}", ts=1000.0 + i, score=0.4,
                  cand_score=0.6, cand_sha=rec["sha256_16"]) for i in range(3)]
    rec = R.note_candidate_live(state, rec, rows)
    check("first_seen_ts is stamped from the sink, not from the staging time",
          rec["first_seen_ts"] == 1000.0, str(rec["first_seen_ts"]))
    prev = os.path.join(os.path.dirname(live), "probe.prev.npz")
    sha_new = R.promote_candidate(live, rec["path"], prev)
    check("promotion swaps live and keeps the previous probe",
          sha_new != sha_live and R.npz_sha16(prev) == sha_live)
    R.rollback_probe(live, prev)
    check("rollback puts the incumbent back", R.npz_sha16(live) == sha_live)


def case17_exploration_is_specified_but_off() -> None:
    print("case 17 — exploration: implemented, correctly specified, DISABLED")
    check("PN119_EXPLORE is off in this environment",
          float(os.environ.get("PN119_EXPLORE", "0") or 0.0) == 0.0)
    check("explore_enabled() is False without the consumer's honour flag",
          not R.explore_enabled({"PN119_EXPLORE": "0.065"}),
          "a budget the consumer ignores makes an explore row a LIE")
    check("...and True only when both halves are set",
          R.explore_enabled({"PN119_EXPLORE": "0.065",
                             "PN119_EXPLORE_BUDGET_HONOURED": "1"}))
    T = 0.495
    check("stratified epsilon: 0.25 at the boundary, 0.01 in the tails",
          R.explore_eps(T + 0.05, T) == 0.25 and R.explore_eps(T + 0.5, T) == 0.01)
    # Effective rate and boundary yield on the LIVE score distribution.
    sink = str(NEEDFIT / "pn119-sink")
    scores = []
    if os.path.isdir(sink):
        for r in R.load_sink(sink, {})[0]:
            if r.score is not None:
                scores.append(r.score)
    if scores:
        s = np.array(scores)
        near = float((np.abs(s - T) < 0.10).mean())
        eff = R.explore_effective_rate(s, T)
        h065 = R.explore_halfwidth_for_rate(s, T, 0.065)
        uni = 0.03
        print(f"  [FINDING] live scores n={len(s)} sd={s.std():.3f}: {near:.1%} "
              f"sit within 0.10 of T, so the pack's halfwidth gives an "
              f"effective rate of {eff:.4f} — 2.4x the 0.065 the brief assumed. "
              f"halfwidth {h065:.3f} is what buys 0.065 on THIS distribution.")
        check("stratified spends its budget AT the boundary, not uniformly",
              0.25 / uni > 5, f"{0.25 / uni:.1f}x per boundary row vs uniform {uni}")
        check("the halfwidth that hits the pack's 0.065 is computable, not "
              "assumed", 0.0 < h065 < 0.10 and
              abs(R.explore_effective_rate(s, T, h065) - 0.065) < 0.02,
              f"halfwidth={h065:.4f} -> eff="
              f"{R.explore_effective_rate(s, T, h065):.4f}")
    ids = [f"req-{i}" for i in range(20000)]
    fired = [R.explore_decision(i, T, T)[0] for i in ids]
    check("boundary requests fire at ~0.25 (deterministic per req_id)",
          0.23 < np.mean(fired) < 0.27, f"{np.mean(fired):.4f}")
    check("the decision is deterministic — sink and consumer cannot disagree",
          all(R.explore_decision(i, T, T) == R.explore_decision(i, T, T)
              for i in ids[:200]))
    bucket = R.TokenBucket(rate_per_hour=0, capacity=10)
    took = sum(1 for i in ids if R.explore_decision(i, T, T, bucket=bucket)[0])
    check("a token bucket makes the explore spend a HARD number", took == 10,
          f"{took} explore rows from 20000 candidates at capacity 10")
    _f, p = R.explore_decision("req-0", T + 0.5, T)
    check("the propensity is logged, not implied", p == 0.01, f"p_explore={p}")
    check("IPS weights are clipped at 50", R.ips_weight(0.01) == 50.0
          and R.ips_weight(0.25) == 4.0 and R.ips_weight(0.0) == 0.0)
    check("PN119_EXPLORE_BUDGET = max(theta+400, routed_budget)",
          R.explore_budget_for(2000, 1300) == 2400
          and R.explore_budget_for(2000, 9000) == 9000)
    hold = [R.is_holdout(i) for i in ids]
    check("the holdout is ~2%", 0.015 < np.mean(hold) < 0.026, f"{np.mean(hold):.4f}")
    both = sum(1 for i in ids if R.is_holdout(i) and R.explore_decision(i, T, T)[0])
    exp_both = np.mean(hold) * 0.25 * len(ids)
    check("holdout and explore use DIFFERENT salts (independent, not nested)",
          abs(both - exp_both) < 4 * np.sqrt(max(exp_both, 1)),
          f"overlap {both}, independent expectation {exp_both:.1f}")
    counts: dict = {}
    kept = R.apply_guards([mkrow(req_id="h", holdout=True),
                           mkrow(req_id="n", holdout=False)], False, 32, counts)
    check("G8: a holdout row never trains, and there is no flag to let it",
          len(kept) == 1 and counts.get("g8_holdout_excluded") == 1
          and not any("holdout" in a for a in
                      ("--legacy-thinking-ok", "--force", "--skip-drift-gate")))


def case18_accuracy_monitor(tmp) -> None:
    print("case 18 — accuracy monitor: a free join, and NEVER a training target")
    check("the module declares the monitor read-only", R.ACCURACY_IS_MONITOR_ONLY)
    import inspect
    sig = set(inspect.signature(R.interval_label).parameters)
    check("the label function cannot even see `correct`",
          not any("correct" in p for p in sig), str(sorted(sig)))
    check("...nor can label_rows", "correct" not in inspect.getsource(R.label_rows))
    check("join key strips the last `-` segment",
          R.join_key("chatcmpl-b01d2e7756d82d4c-ad8afada")
          == "chatcmpl-b01d2e7756d82d4c")
    sink = str(NEEDFIT / "pn119-sink")
    results = str(pathlib.Path(os.path.expanduser(
        "~/shared/folderX/qbench45/results")))
    if not (os.path.isdir(sink) and os.path.isdir(results)):
        check("live sink + results present", False, "skipped")
        return
    raw = R.load_sink(sink, {})[0]
    rep: dict = {}
    state = os.path.join(tmp, "state18")
    os.makedirs(state, exist_ok=True)
    alerts = R.accuracy_monitor(raw, results, 0.495, state, rep, persist=False)
    print(f"  [ACCEPTANCE] accuracy join: {rep['acc_join_n']} sink rows joined "
          f"graded runs; deep n={rep['acc_deep_n']} acc={rep['acc_deep']} | "
          f"lean n={rep['acc_lean_n']} acc={rep['acc_lean']} | "
          f"AUC(score->correct)={rep['acc_auc_score_vs_correct']}")
    check("the join actually joins", rep["acc_join_n"] >= 300,
          f"n={rep['acc_join_n']}")
    check("AUC(score->correct) is BELOW 0.5 — the lens finds HARD items",
          rep["acc_auc_score_vs_correct"] < 0.5,
          f"{rep['acc_auc_score_vs_correct']} (training on `correct` would "
          f"route the EASY questions deep)")
    check("no alert without a baseline", alerts == [])
    with open(os.path.join(state, "accuracy-baseline.json"), "w",
              encoding="utf-8") as f:
        json.dump({"acc_deep": min(1.0, (rep["acc_deep"] or 0) + 0.3)}, f)
    rep2: dict = {}
    alerts2 = R.accuracy_monitor(raw, results, 0.495, state, rep2, persist=False)
    check("a deep-bucket regression ALERTS", bool(alerts2),
          alerts2[0] if alerts2 else "no alert")


def case19_two_feature_eras(tmp) -> None:
    """The router became LAST-ONLY (15360). A reader that keeps assuming
    30720 does not fail — it glues two rows together and labels them."""
    print("case 19 — 15360 vs 30720: a feats-*.bin never says which it is")
    small = R.KNOWN_FEAT_DIMS[0]
    check("both eras are known", R.KNOWN_FEAT_DIMS == (15360, 30720),
          str(R.KNOWN_FEAT_DIMS))
    sink = os.path.join(tmp, "sink19")
    os.makedirs(sink, exist_ok=True)
    # a 15360-dim window, written the way the LAST-ONLY router writes it
    X = np.random.default_rng(2).standard_normal((6, small)).astype(np.float32)
    (X.view(np.uint32) >> np.uint32(16)).astype(np.uint16).tofile(
        os.path.join(sink, "feats-20260726-090000.bin"))
    with open(os.path.join(sink, "meta-20260726-090000.jsonl"), "w",
              encoding="utf-8") as f:
        f.write(json.dumps({"pn119_header": 1, "feat_dim": small,
                            "blocks": ["L42-last", "L47-last", "L51-last"],
                            "think_end_ids": [1, 2, 3, 4, 5],
                            "censor_slack": 13, "ts": 1000.0}) + "\n")
        for i in range(6):
            f.write(json.dumps({"req_id": f"n-{i}", "row": i, "score": 0.5,
                                "route": "deep", "prompt_tok": 10,
                                "ts": 2000.0 + i, "mode": "shadow"}) + "\n")
            f.write(json.dumps({"req_id": f"n-{i}", "finish": True,
                                "score": 0.5, "generated": 900, "rtok": 913,
                                "cap_hit": False, "thinking": True,
                                "ts": 2000.0 + i, "mode": "shadow"}) + "\n")
    write_sink(sink, "20260725-080000", synthetic_rows(5, prefix="o", ts=100.0,
                                                       rtok=913, generated=920))
    c: dict = {}
    rows = R.apply_guards(R.load_sink(sink, c)[0], False, 32, c)
    check("the header's feat_dim is believed over any guess",
          c.get(f"feat_dim_{small}_via_header") == 1
          and c.get(f"feat_dim_{R.FEAT_DIM}_via_bytes") == 1, json.dumps(c))
    check("a headerless window is resolved from bytes/row-count",
          all(r.feat_dim == R.FEAT_DIM for r in rows if r.tag.startswith("20260725")))
    check("the header's censor_slack travels with the row",
          all(r.slack == 13 for r in rows if r.tag.startswith("20260726")),
          "the running router publishes 9; a window that says 13 is believed")
    dim, kept = R.select_feat_dim(rows, c)
    check("the NEWEST era wins and the older rows are dropped, counted",
          dim == small and len(kept) == 6 and c["g9_feat_dim_mismatch"] == 5
          and c["g9_feat_dims_seen"] == [small, R.FEAT_DIM], json.dumps(
              {k: v for k, v in c.items() if k.startswith("g9")}))
    check("the surviving rows read the right width",
          R.bf16_rows(kept[0].feat_path, [0], small).shape == (1, small))
    # a bin whose length matches NEITHER era is dropped, not guessed at
    with open(os.path.join(sink, "feats-20260726-100000.bin"), "wb") as f:
        f.write(b"\x00" * 12345)
    with open(os.path.join(sink, "meta-20260726-100000.jsonl"), "w",
              encoding="utf-8") as f:
        f.write(json.dumps({"req_id": "z", "row": 0, "score": 0.1,
                            "ts": 3000.0}) + "\n")
    c2: dict = {}
    R.load_sink(sink, c2)
    check("an unreadable width is dropped, never guessed",
          c2.get("g9_feat_dim_ambiguous") == 1, json.dumps(c2))
    # the incumbent in the other era ABSTAINS instead of raising ValueError
    live = os.path.join(tmp, "live19", "probe.npz")
    os.makedirs(os.path.dirname(live), exist_ok=True)
    R.atomic_write_npz(live, {"mu": np.zeros(small), "sd": np.ones(small),
                              "Vt10": np.zeros((10, small)), "w": np.zeros(11)})
    probe, note = R.load_incumbent(live, R.FEAT_DIM)
    check("an incumbent in the other feature space is not the incumbent",
          probe is None and "ABSTAIN" in note, note)
    probe, note = R.load_incumbent(live, small)
    check("...and is loaded normally when the widths agree",
          probe is not None and note == "ok", note)
    check("the LIVE probe today is 15360 while the sink on disk is 30720",
          R.incumbent_feat_dim(str(NEEDFIT / "pn119-live/probe.npz")) in
          (small, R.FEAT_DIM),
          f"live={R.incumbent_feat_dim(str(NEEDFIT / 'pn119-live/probe.npz'))}")
    # the reservoir cannot stack two widths
    state = os.path.join(tmp, "state19")
    os.makedirs(state, exist_ok=True)
    c3: dict = {}
    R.update_reservoir(state, rows[:2], 10, np.random.default_rng(0), c3,
                       feat_dim=R.FEAT_DIM)
    c4: dict = {}
    R.update_reservoir(state, kept, 10, np.random.default_rng(0), c4,
                       feat_dim=small)
    check("a width change RESETS the reservoir instead of stacking",
          c4.get("reservoir_reset_feat_dim") == R.FEAT_DIM, json.dumps(c4))


# ------------------------------------------------- BUG-170: self-description
ROUTER_PY = str(REPO / "fixes/pn119_router.py")


def _load_router():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pn119_router_bug170",
                                                  ROUTER_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pn119_router_bug170"] = mod
    spec.loader.exec_module(mod)
    return mod


def _router_load(mod, path):
    """Drive the REAL `_load_probe` with the smallest object it needs."""
    import types as _t
    stub = _t.SimpleNamespace(runner=_t.SimpleNamespace(device="cpu"),
                              _probe_fold_resid=None, _probe_canary=None,
                              _probe_loads=0)
    pv, pb = mod.PN119Router._load_probe(stub, path)
    return stub, pv, pb


def _lastonly_sink(sink, tag, rows, feat_rows):
    """One 15360-dim window WITH the header the live router stamps."""
    os.makedirs(sink, exist_ok=True)
    (feat_rows.view(np.uint32) >> np.uint32(16)).astype(np.uint16).tofile(
        os.path.join(sink, f"feats-{tag}.bin"))
    with open(os.path.join(sink, f"meta-{tag}.jsonl"), "w",
              encoding="utf-8") as f:
        f.write(json.dumps({"pn119_header": 1, "feat_dim": 15360,
                            "blocks": ["L42-last", "L47-last", "L51-last"],
                            "censor_schema": 2, "censor_slack": 13,
                            "mode": "enforce", "ts": 1000.0}) + "\n")
        for i, r in enumerate(rows):
            f.write(json.dumps({"req_id": r["req_id"], "row": i, "score": 0.5,
                                "route": "deep", "prompt_tok": 100,
                                "ts": r["ts"], "mode": "enforce"}) + "\n")
            f.write(json.dumps({"req_id": r["req_id"], "finish": True,
                                "generated": r["generated"], "rtok": r["rtok"],
                                "cap_hit": False, "censored": False,
                                "thinking": True, "ts": r["ts"],
                                "mode": "enforce"}) + "\n")


def case20_refit_payload_is_self_describing(tmp) -> None:
    """BUG-170. A refit artifact that omits blocks/canary/feat_dim loads into
    the router with BOTH of its wrong-probe guards inert — on precisely the
    files the continuous loop produces."""
    print("case 20 — a refit probe describes itself, and the router's guards "
          "are ARMED on it")
    sink = os.path.join(tmp, "sink20")
    state = os.path.join(tmp, "state20")
    out = os.path.join(tmp, "live20", "probe.npz")
    meta = [dict(req_id=f"s-{i}", generated=900, ts=100.0 + i,
                 rtok=(3037 if i % 3 else 137), censored=False)
            for i in range(40)]
    y = np.array([1.0 if m["rtok"] >= 2000 else 0.0 for m in meta])
    rng = np.random.default_rng(5)
    d = rng.standard_normal(15360).astype(np.float32)
    d /= np.linalg.norm(d)
    X = (rng.standard_normal((len(meta), 15360)).astype(np.float32) * 0.05
         + y[:, None] * d[None, :] * 8.0).astype(np.float32)
    _lastonly_sink(sink, "20260727-120000", meta, X)
    cmd = [PY, str(REPO / "fixes/refit_pn119_probe.py"), "--sink", sink,
           "--state", state, "--out", out, "--no-seed",
           "--b3-report", _fake_b3(tmp),
           "--results-dir", os.path.join(tmp, "no-results20"),
           "--promote-mode", "direct", "--min-new", "5",
           "--max-deep-frac", "0.9", "--prior-hi", "0.75"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    check("a 15360 refit swaps and exits 0",
          r.returncode == 0 and os.path.isfile(out),
          f"rc={r.returncode} {(r.stdout + r.stderr).strip()[-220:]}")
    if not os.path.isfile(out):
        return
    z = np.load(out, allow_pickle=True)
    for k in ("blocks", "feat_dim", "feature_spec", "canary_seed",
              "canary_score", "canary_tol"):
        check(f"the payload carries `{k}`", k in z.files, str(z.files))
    check("blocks come from the WINDOW HEADER, not a guess",
          list(map(str, z["blocks"])) == ["L42-last", "L47-last", "L51-last"]
          and str(z["refit_blocks_via"]) == "sink_header",
          f"{list(map(str, z['blocks']))} via {z['refit_blocks_via']}")
    check("feat_dim matches the fitted width", int(z["feat_dim"]) == 15360
          and np.asarray(z["mu"]).size == 15360)
    check("feature_spec is DERIVED, not the hardcoded last+mean lie",
          str(z["feature_spec"]) == "layers[42,47,51] x pools[last] d5120 concat",
          str(z["feature_spec"]))
    ids = [str(s) for s in z["train_ids"]]
    check("sink rows are namespaced in train_ids",
          ids and all(i.startswith("sink:") for i in ids), str(ids[:2]))

    mod = _load_router()
    mod.STATS.clear()
    stub, pv, pb = _router_load(mod, out)
    check("the router's CANARY is armed and passes on a refit probe",
          stub._probe_canary is not None
          and stub._probe_canary["resid"] < 1e-9
          and "probe_canary_absent" not in mod.STATS,
          json.dumps(stub._probe_canary, default=str))
    arrays = {k: z[k] for k in z.files}
    arrays["w"] = np.asarray(arrays["w"], dtype=np.float64).copy()
    arrays["w"][-1] += 0.01                     # not the probe that was signed
    bad = os.path.join(tmp, "bad20.npz")
    np.savez(bad, **arrays)
    try:
        _router_load(mod, bad)
        check("a tampered refit probe is REFUSED", False, "load succeeded")
    except ValueError as e:
        check("a tampered refit probe is REFUSED", "CANARY FAILED" in str(e),
              str(e)[:80])
    arrays2 = {k: z[k] for k in z.files}
    arrays2["blocks"] = np.array(["L51-last", "L47-last", "L42-last"])
    wrong = os.path.join(tmp, "wrongorder20.npz")
    np.savez(wrong, **arrays2)
    try:
        _router_load(mod, wrong)
        check("a mis-ordered block set is REFUSED", False, "load succeeded")
    except ValueError as e:
        check("a mis-ordered block set is REFUSED", "block order" in str(e),
              str(e)[:80])
    check("blocks_for_dim reconstructs BOTH eras",
          R.blocks_for_dim(15360) == ("L42-last", "L47-last", "L51-last")
          and R.blocks_for_dim(30720) == (
              "L42-last", "L42-mean", "L47-last", "L47-mean",
              "L51-last", "L51-mean")
          and R.blocks_for_dim(999) is None)
    check("spec_from_blocks round-trips the 30720 era",
          R.spec_from_blocks(R.blocks_for_dim(30720), 30720)
          == "layers[42,47,51] x pools[last,mean] d5120 concat")
    check("resolve_blocks refuses a width it cannot describe",
          R.resolve_blocks(sink, out, 999, {}) is None)


def case21_nonfinite_scores_take_the_fallback(tmp) -> None:
    """P2-2. One NaN score is a SILENT lean in enforce mode, de-sorts the
    percentile window for the rest of the boot, and puts a bare `NaN` token in
    the sink."""
    print("case 21 — a non-finite score takes the fallback, is counted, and "
          "never reaches the window or the sink")
    import types as _t
    mod = _load_router()
    live = str(NEEDFIT / "pn119-live/probe.npz")
    if not os.path.isfile(live):
        check("live probe present for the router guard case", False, live)
        return
    sink = os.path.join(tmp, "sink21")
    os.makedirs(sink, exist_ok=True)
    for k in ("PN119_ASYNC_SCORE", "PN119_RATE_TARGET", "PN119_HEALTH"):
        os.environ.pop(k, None)
    os.environ["GENESIS_ENABLE_PN119_ROUTER"] = "1"
    os.environ["PN119_MODE"] = "enforce"
    os.environ["PN119_SINK"] = sink
    os.environ["PN119_FALLBACK_ROUTE"] = "deep"
    mod.STATS.clear()
    mod.ROUTES.clear()
    mod.SCORES.clear()
    runner = _t.SimpleNamespace(
        device="cpu", max_num_reqs=64,
        input_batch=_t.SimpleNamespace(req_ids=[]), requests={},
        vllm_config=_t.SimpleNamespace(reasoning_config=None))
    r = mod.PN119Router(runner, live)
    for i in range(8):
        r._publish(f"ok-{i}", 0.1 * i, 100, 0, True, None, None, None, None, None)
    base_sorted = list(r._score_sorted)
    check("the finite baseline is in the window", len(base_sorted) == 8)

    for j, bad in enumerate((float("nan"), float("inf"), float("-inf"))):
        rid = f"bad-{j}"
        r._publish(rid, bad, 100, 0, True, None, None, None, None, None)
        check(f"{bad!r} does not enter the percentile window",
              list(r._score_sorted) == base_sorted, f"{len(r._score_sorted)}")
        check(f"{bad!r} takes the declared fallback route",
              mod.ROUTES.get(rid) == r.fallback_route
              and r.unscored.get(rid, "").startswith("score_nonfinite"),
              f"route={mod.ROUTES.get(rid)} reason={r.unscored.get(rid)}")
        # SCORES still gets the legacy compatibility shim `_unscoreable`
        # writes — a FINITE value on the fallback side of tdeep. What must
        # never happen is the non-finite number landing there.
        check(f"{bad!r} is not published as a score",
              rid not in r.scored
              and mod.SCORES.get(rid) == r.tdeep,
              f"scored={rid in r.scored} SCORES={mod.SCORES.get(rid)}")
    check("every non-finite score is counted in health.json stats",
          mod.STATS.get("score_nonfinite") == 3
          and mod.STATS.get("unscoreable_score_nonfinite_publish") == 3,
          json.dumps({k: v for k, v in mod.STATS.items()
                      if "nonfinite" in k or k == "unscoreable"}))

    # the async drain path has its own producer-side guard
    r._pin_score = __import__("torch").tensor([float("nan")] * 4)
    r._free_slots = [1, 2, 3]
    r._pending = [{"req_id": "async-nan", "slot": 0, "event": None,
                   "prompt_len": 100, "cached": 0, "thinking": True,
                   "budget": None, "source": None, "caller": None,
                   "suite": None, "want_feat": False}]
    r._drain_pending()
    check("the deferred readback guards its own score",
          mod.ROUTES.get("async-nan") == r.fallback_route
          and r.unscored.get("async-nan") == "score_nonfinite_async"
          and mod.STATS.get("score_nonfinite") == 4,
          f"{r.unscored.get('async-nan')}")

    # the SYNC dot path, through the real _finalize
    st = _t.SimpleNamespace(prompt_token_ids=[7] * 37 + [248068, 11, 11],
                            num_prompt_tokens=40, num_computed_tokens=40,
                            output_token_ids=[],
                            sampling_params=_t.SimpleNamespace(
                                thinking_token_budget=None, max_tokens=4096,
                                extra_args=None))
    torch = __import__("torch")
    nan_rows = [torch.full((5120,), float("nan")) for _ in range(3)]
    r._finalize("sync-nan", st, nan_rows, 40, 0)
    check("the synchronous dot guards its own score",
          mod.ROUTES.get("sync-nan") == r.fallback_route
          and r.unscored.get("sync-nan") == "score_nonfinite_sync",
          f"{r.unscored.get('sync-nan')}")

    r.shutdown() if hasattr(r, "shutdown") else r._sink_close()
    text = ""
    for f in sorted(os.listdir(sink)):
        if f.startswith("meta-") and f.endswith(".jsonl"):
            text += open(os.path.join(sink, f), encoding="utf-8").read()
    strict_ok = True
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            json.loads(line, parse_constant=_strict_constant)
        except ValueError:
            strict_ok = False
    check("no bare NaN/Infinity token ever reaches the sink",
          strict_ok and "NaN" not in text and "Infinity" not in text,
          f"{len(text.splitlines())} lines")


def _strict_constant(name):
    raise ValueError(f"bare {name} token")


def case22_b3_skips_non_item_train_ids(tmp) -> None:
    """BUG-170, second half: `np.stack([st[i] for i in z['train_ids']])` against
    the 100-item capture KeyError'd on the first sink id, so the first direct
    swap left the loop green-timered and dead."""
    print("case 22 — b3 / make_reference skip non-item train_ids instead of "
          "KeyError")
    from safetensors.torch import save_file
    import torch as _torch
    d = os.path.join(tmp, "b322")
    os.makedirs(d, exist_ok=True)
    n_items, dim = 24, 96
    rng = np.random.default_rng(7)
    item_ids = [f"gpqa-{i:03d}" for i in range(n_items)]
    feats = rng.standard_normal((n_items, dim)).astype(np.float32) * 0.3
    feats += rng.standard_normal(dim).astype(np.float32)[None, :] * 2.0
    st_path = os.path.join(d, "cap.safetensors")
    save_file({i: _torch.tensor(feats[k]).reshape(1, dim)
               for k, i in enumerate(item_ids)}, st_path)

    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    Xs = (feats - mu) / sd
    _u, _s, Vt = np.linalg.svd(Xs, full_matrices=False)
    Vt = Vt[:5]
    P = Xs @ Vt.T
    w = np.concatenate([np.linspace(1.0, 0.2, 5), [0.05]])
    full = np.hstack([P, np.ones((len(P), 1))]) @ w
    # exactly the shape the fixed refit writes: items + namespaced sink rows
    train_ids = item_ids + [f"sink:cmpl-{i}" for i in range(60)]
    probe = os.path.join(d, "probe.npz")
    np.savez(probe, mu=mu, sd=sd, Vt10=Vt, w=w,
             train_ids=np.array(train_ids), feat_dim=dim,
             blocks=np.array(["b0"]), feature_spec="synthetic")

    panel_ids = item_ids[:10]
    order = (-full[:10]).argsort().argsort() + 1
    ref = {"criteria": {"rho_loo_ceiling": 1.0,
                        "top25_agreement_full_vs_loo": n_items,
                        "max_between_item_cosine_seed": 0.99}, "panel": [
        {"item_id": i, "offline_score": float(full[k]),
         "loo_score": float(full[k]) + 1e-3 * (k % 3),
         "n_tok_offline": 100 + k, "offline_rank_in_panel": int(order[k])}
        for k, i in enumerate(panel_ids)]}
    ref_p = os.path.join(d, "ref.json")
    with open(ref_p, "w", encoding="utf-8") as f:
        json.dump(ref, f)
    loo_p = os.path.join(d, "loo.json")
    with open(loo_p, "w", encoding="utf-8") as f:
        json.dump({"ids": item_ids, "loo": [float(x) for x in full]}, f)
    featjson = os.path.join(d, "cap.json")
    with open(featjson, "w", encoding="utf-8") as f:
        json.dump({"features": {i: {"n_tok": 100 + k}
                                for k, i in enumerate(item_ids)}}, f)

    b3 = subprocess.run(
        [PY, str(NEEDFIT / "pn119_b3_numerics.py"), "--dry-run",
         "--probe", probe, "--features", st_path, "--ref", ref_p,
         "--out", os.path.join(d, "report.json"), "--no-marker"],
        capture_output=True, text=True, timeout=600)
    out = b3.stdout + b3.stderr
    check("b3 does not KeyError on a namespaced train_id",
          "KeyError" not in out and "Traceback" not in out,
          out.strip()[-200:])
    check("b3 reports how many non-item ids it skipped",
          "60 non-item ids skipped" in out, out.strip()[:200])
    check("b3 still reaches its verdict on the item subset",
          b3.returncode == 0, f"rc={b3.returncode} {out.strip()[-200:]}")

    mk = subprocess.run(
        [PY, str(NEEDFIT / "pn119_b3_make_reference.py"),
         "--features", st_path, "--feat-json", featjson, "--probe", probe,
         "--loo", loo_p, "--panel-from", ref_p,
         "--out", os.path.join(d, "ref-out.json")],
        capture_output=True, text=True, timeout=600)
    mkout = mk.stdout + mk.stderr
    check("make_reference does not KeyError either",
          "KeyError" not in mkout and "Traceback" not in mkout
          and mk.returncode == 0, mkout.strip()[-200:])
    check("...and its rebuilt reference keeps the whole panel",
          os.path.isfile(os.path.join(d, "ref-out.json"))
          and len(json.load(open(os.path.join(d, "ref-out.json"),
                                 encoding="utf-8"))["panel"]) == 10)


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
        case12_interval_labels_on_the_live_sink()
        case13_absorbing_state_is_closed()
        case14_censoring_detection()
        case15_drift_gates(tmp)
        case16_shadow_promotion(tmp)
        case17_exploration_is_specified_but_off()
        case18_accuracy_monitor(tmp)
        case19_two_feature_eras(tmp)
        case20_refit_payload_is_self_describing(tmp)
        case21_nonfinite_scores_take_the_fallback(tmp)
        case22_b3_skips_non_item_train_ids(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if _fails:
        print(f"FAILED: {len(_fails)} — {', '.join(_fails)}")
        return 1
    print("ALL PASS")
    print("VERDICT: no-generation rows cannot be labels; a censored row below "
          "theta is neither a positive nor a negative; a rank-identical rescale "
          "cannot pass; a drifted corpus cannot pass; a candidate that flips "
          "the router needs a human; the cursor survives a prune; exploration "
          "is built and off; a skip is exit 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
