#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PN162 ledger updater — HOST-side tests (2026-07-27).

Stdlib only, no GPU, no vLLM, no network, no container.

    /usr/bin/python3 fixes/test_pn162_ledger_update.py       # standalone
    python -m pytest fixes/test_pn162_ledger_update.py -q    # where available

Covered
  * outcome classification against the BUG-139 schema-2 trust order, and the
    two drop classes that would otherwise poison k (cap_hit, no-generation)
  * the PN100 grant grid + its inverse, including PN162's own feedback trap
    (once k != 1, grant/260 no longer recovers the steps estimate)
  * the k update: bump / decay / no-op, both clamps, and the per-pass cap
  * the sink reader: pairing, byte cursor, IDEMPOTENCE (a second pass over
    unchanged files learns nothing), unfinished rows held for the next pass,
    truncation recovery, synthetic-marker exclusion
  * CONVERGENCE — repeated bound rows raise the grant until it stops binding,
    then it stabilises in a band instead of running to the clamp
  * THE USER'S 10x STORY as an explicit end-to-end simulation: the same 100
    requests run ten times back to back, driven through the real run_pass()
    against a real sink directory and a real ledger file
  * key-schema agreement with the in-engine consumer (the two derivations must
    stay byte-identical) and the composite-cell occupancy rule
  * telemetry / oracle / explore-report shape and their honesty about what the
    sink can and cannot support
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_MW = (_ROOT / "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis"
       / "middleware")


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


up = _load_by_path("_pn162_updater_under_test",
                   _HERE / "pn162_ledger_update.py")
cal = _load_by_path("_pn162_cal_for_keycheck", _MW / "pn162_budget_cal.py")


def _load_auto_budget():
    for n in ("vllm", "vllm._genesis", "vllm._genesis.middleware",
              "vllm._genesis.middleware.lazy_reasoner"):
        if n not in sys.modules:
            sys.modules[n] = types.ModuleType(n)
    sys.modules["vllm._genesis.middleware.lazy_reasoner"]._extract_text_from_message = (
        lambda m: "")
    sys.modules["vllm._genesis.middleware"].pn162_budget_cal = cal
    return _load_by_path("_ab_for_gridcheck", _MW / "auto_budget.py")


ab = _load_auto_budget()

CFG = dict(up.DEFAULTS)
_TMPDIRS = []


def tmpdir() -> str:
    d = tempfile.mkdtemp(prefix="pn162-upd-")
    _TMPDIRS.append(d)
    return d


# ─── sink row factory ────────────────────────────────────────────────────────
_SEQ = {"n": 0}


def sink_rows(grant, rtok, *, route="lean", score=0.1, generated=None,
              censored=False, censor_forced=False, cap_hit=False,
              thinking=True, prompt_tok=300, arm=None, req_id=None, ts=1.0):
    _SEQ["n"] += 1
    rid = req_id or f"req-{_SEQ['n']}"
    gen = (rtok + 120) if generated is None else generated
    score_line = {"req_id": rid, "row": _SEQ["n"] - 1, "score": score,
                  "route": route, "prompt_tok": prompt_tok, "ts": ts,
                  "budget_grant": grant, "budget_source": "pn100",
                  "caller": arm, "suite": None, "routable": True}
    finish_line = {"req_id": rid, "finish": True, "ts": ts + 0.1,
                   "rtok": rtok, "generated": gen, "cap_hit": cap_hit,
                   "censored": censored, "censor_forced": censor_forced,
                   "censor_src": "none", "thinking": thinking,
                   "budget_grant": grant, "budget_source": "pn100"}
    return score_line, finish_line


def write_sink(path, pairs, header=True):
    """Append (score, finish) pairs to a meta-*.jsonl exactly as the router
    interleaves them: score first, finish after."""
    os.makedirs(path, exist_ok=True)
    f = os.path.join(path, "meta-20260727-000000.jsonl")
    with open(f, "a", encoding="utf-8") as fh:
        if header and not os.path.getsize(f) if os.path.exists(f) else header:
            fh.write(json.dumps({"pn119_header": 1, "censor_schema": 2,
                                 "censor_slack": 13, "mode": "enforce"}) + "\n")
        for s, fin in pairs:
            fh.write(json.dumps(s) + "\n")
            if fin is not None:
                fh.write(json.dumps(fin) + "\n")
    return f


def row(grant, rtok, **kw):
    s, f = sink_rows(grant, rtok, **kw)
    return up._pair(s["req_id"], s, f)


# ─── classification ──────────────────────────────────────────────────────────
def test_censor_forced_outranks_everything():
    r = row(1300, 400, censored=False, censor_forced=True)
    assert up.classify(r, CFG) == "bound"


def test_router_censored_verdict_is_bound():
    assert up.classify(row(1300, 900, censored=True), CFG) == "bound"


def test_arithmetic_bind_within_slack():
    assert up.classify(row(1300, 1287, CFG and 0 or 0 or 0) if False
                       else row(1300, 1287), CFG) == "bound"
    assert up.classify(row(1300, 1300), CFG) == "bound"
    assert up.classify(row(1300, 1286), CFG) != "bound"   # 1300-13 = 1287


def test_slack_needs_more_than_40_percent():
    assert up.classify(row(1000, 500), CFG) == "slack"     # 500 > 400
    assert up.classify(row(1000, 700), CFG) == "ok"        # 300 <= 400
    assert up.classify(row(1000, 599), CFG) == "slack"
    assert up.classify(row(1000, 601), CFG) == "ok"


def test_drops():
    assert up.classify(row(None, 500), CFG) == "drop:no_grant"
    assert up.classify(row(0, 500), CFG) == "drop:no_grant"
    assert up.classify(row(1300, 100, thinking=False), CFG) == "drop:thinking_off"
    assert up.classify(row(1300, 1, generated=1), CFG) == "drop:no_generation"
    # a max_tokens truncation is NOT a thinking-budget bind: counting it would
    # let a client's small max_tokens inflate every bucket's k
    assert up.classify(row(4000, 500, cap_hit=True), CFG) == "drop:cap_hit"
    # ... unless the holder actually forced </think>, which outranks it
    assert up.classify(row(4000, 500, cap_hit=True, censor_forced=True),
                       CFG) == "bound"


# ─── the grant grid and its inverse ──────────────────────────────────────────
def test_grant_grid_matches_the_engine():
    """`pn100_grant` must reproduce auto_budget._continuous_budget exactly."""
    os.environ["GENESIS_PN100_CONTINUOUS"] = "1"
    os.environ["GENESIS_PN100_TOK_PER_STEP"] = "260"
    os.environ.pop("GENESIS_ENABLE_PN162_BUDGET_CAL", None)
    os.environ.pop("GENESIS_PN100_STEP_BUDGET_MAP", None)
    cal.reset_cache()
    for s in range(1, 60):
        assert up.pn100_grant(s, 1.0, CFG) == ab._continuous_budget(2, s), s


def test_inverse_is_exact_at_k1():
    for s in range(1, 40):
        g = up.pn100_grant(s, 1.0, CFG)
        assert up.invert_steps(g, {}, CFG) == up.bucket_of(s), (s, g)


def test_inverse_survives_pn162s_own_feedback():
    """THE trap: with k=1.5 on bucket 5 the grant is 2000, and the naive
    quotient 2000/260 = 7.7 -> 8 would credit the WRONG bucket, compounding
    every pass. Inverting through the live map recovers 5."""
    kmap = {"5": 1.5}
    g = up.pn100_grant(5, 1.5, CFG)
    assert g == 2000
    assert up.invert_steps(g, {}, CFG) == 8          # the naive answer
    assert up.invert_steps(g, kmap, CFG) == 5        # the right one


def test_inverse_falls_back_when_no_candidate_matches():
    assert up.invert_steps(1234, {}, CFG) >= 1       # off-grid -> no crash


# ─── the k update ────────────────────────────────────────────────────────────
def test_bump_decay_and_noop():
    k, _ = up.update_buckets([dict(_bucket=5, _outcome="bound", _keys=["5"])],
                             {}, CFG)
    assert abs(k["5"] - 1.15) < 1e-9
    k, _ = up.update_buckets([dict(_bucket=5, _outcome="slack", _keys=["5"])],
                             {}, CFG)
    assert abs(k["5"] - 0.97) < 1e-9
    k, _ = up.update_buckets([dict(_bucket=5, _outcome="ok", _keys=["5"])],
                             {"5": 1.3}, CFG)
    assert k["5"] == 1.3


def test_clamps_and_per_pass_cap():
    rows = [dict(_bucket=5, _outcome="bound", _keys=["5"])] * 40
    k, _ = up.update_buckets(rows, {"5": 1.0}, CFG)
    assert k["5"] == 1.5                              # max_step, not 3.0
    k2, _ = up.update_buckets(rows, {"5": 2.9}, CFG)
    assert k2["5"] == 3.0                             # k_max
    rows = [dict(_bucket=5, _outcome="slack", _keys=["5"])] * 200
    k3, _ = up.update_buckets(rows, {"5": 0.8}, CFG)
    assert k3["5"] == 0.7                             # k_min


# ─── the sink reader ─────────────────────────────────────────────────────────
def test_reader_pairs_and_cursors():
    d = tmpdir()
    write_sink(d, [sink_rows(1300, 400), sink_rows(1300, 1290)])
    counts = {}
    rows, offs = up.read_sink_since(d, {}, CFG, counts)
    assert len(rows) == 2
    rows2, offs2 = up.read_sink_since(d, offs, CFG, {})
    assert rows2 == [] and offs2 == offs           # IDEMPOTENT


def test_reader_holds_unfinished_rows_for_the_next_pass():
    d = tmpdir()
    s, f = sink_rows(1300, 400, req_id="pending")
    write_sink(d, [(s, None)])
    counts = {}
    rows, offs = up.read_sink_since(d, {}, CFG, counts)
    assert rows == [] and counts["unfinished"] == 1
    with open(os.path.join(d, "meta-20260727-000000.jsonl"), "a",
              encoding="utf-8") as fh:
        fh.write(json.dumps(f) + "\n")
    rows2, _ = up.read_sink_since(d, offs, CFG, {})
    assert len(rows2) == 1 and rows2[0]["req_id"] == "pending"


def test_reader_recovers_from_truncation():
    d = tmpdir()
    write_sink(d, [sink_rows(1300, 400)])
    _, offs = up.read_sink_since(d, {}, CFG, {})
    p = os.path.join(d, "meta-20260727-000000.jsonl")
    os.remove(p)
    write_sink(d, [sink_rows(900, 300)])
    counts = {}
    rows, _ = up.read_sink_since(d, offs, CFG, counts)
    assert counts.get("files_truncated") == 1 and len(rows) == 1


def test_reader_skips_synthetic_markers():
    d = tmpdir()
    s, f = sink_rows(1300, 400, req_id="synthetic-1")
    write_sink(d, [(s, f), sink_rows(1300, 400)])
    with open(os.path.join(d, ".synthetic-b3-numerics-x.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"req_ids": ["synthetic-1"]}, fh)
    counts = {}
    rows, _ = up.read_sink_since(d, {}, CFG, counts)
    assert counts["marker_excluded"] == 1 and len(rows) == 1


def test_reader_tolerates_garbage_lines():
    d = tmpdir()
    write_sink(d, [sink_rows(1300, 400)])
    with open(os.path.join(d, "meta-20260727-000000.jsonl"), "a",
              encoding="utf-8") as fh:
        fh.write("{not json\n\n")
    counts = {}
    rows, _ = up.read_sink_since(d, {}, CFG, counts)
    assert len(rows) == 1 and counts["bad_json"] == 1


# ─── ledger io ───────────────────────────────────────────────────────────────
def test_atomic_write_and_consumer_can_read_it():
    d = tmpdir()
    p = os.path.join(d, "pn162-ledger.json")
    up.atomic_write_json(p, {"schema": 1, "bucket": {"5": 1.4}, "exact": {}})
    assert not [x for x in os.listdir(d) if x.startswith(".")]   # no tmp left
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    os.environ["GENESIS_PN162_LEDGER"] = p
    os.environ["GENESIS_PN162_RESTAT_S"] = "0"
    cal.reset_cache()
    try:
        assert cal.budget_multiplier(5) == 1.4
    finally:
        for k in ("GENESIS_ENABLE_PN162_BUDGET_CAL", "GENESIS_PN162_LEDGER",
                  "GENESIS_PN162_RESTAT_S"):
            os.environ.pop(k, None)
        cal.reset_cache()


# ─── key schema: the two sides must agree ────────────────────────────────────
def test_key_derivation_matches_the_consumer():
    for schema in up.KEY_SCHEMAS:
        for steps in (0, 1, 4, 5, 12, 16, 17, 99, None):
            for ptok in (None, 0, 255, 256, 1023, 4096, 100000):
                assert up.bucket_keys(steps, ptok, schema) == \
                    cal.bucket_keys(steps, ptok, schema), (schema, steps, ptok)
    assert up.KEY_SCHEMAS == cal.KEY_SCHEMAS
    assert up.DEFAULT_PTOK_BANDS == cal.DEFAULT_PTOK_BANDS
    assert up.MAX_BUCKET == cal.MAX_BUCKET


def test_marginal_key_is_always_last():
    for schema in up.KEY_SCHEMAS:
        assert up.bucket_keys(5, 300, schema)[-1] == "5"


def test_thin_composite_cells_are_not_written():
    cfg = dict(CFG, key_schema="steps_ptok", min_cell=10)
    win = [dict(_bucket=5, _keys=["5|p1", "5"], _outcome="ok")] * 3
    win += [dict(_bucket=8, _keys=["8|p2", "8"], _outcome="ok")] * 25
    allowed = up.writable_keys(win, cfg)
    assert "5|p1" not in allowed and "5" in allowed
    assert "8|p2" in allowed and "8" in allowed


def test_composite_row_credits_cell_and_marginal():
    cfg = dict(CFG, key_schema="steps_ptok", min_cell=1)
    rows = [dict(_bucket=8, _keys=["8|p2", "8"], _outcome="bound")]
    k, per = up.update_buckets(rows, {}, cfg, {"8|p2", "8"})
    assert abs(k["8|p2"] - 1.15) < 1e-9
    assert abs(k["8"] - 1.15) < 1e-9        # the fallback stays representative
    assert per["8|p2"]["bound"] == per["8"]["bound"] == 1


# ─── convergence ─────────────────────────────────────────────────────────────
def simulate(true_need: int, grant: int, cfg: dict):
    """What the engine would record for an item needing `true_need` thinking
    tokens under `grant`. Force-closed at the cap, otherwise a natural stop."""
    if true_need >= grant - cfg["censor_slack"]:
        return dict(grant=grant, rtok=grant - cfg["censor_slack"],
                    censor_forced=True)
    return dict(grant=grant, rtok=true_need, censor_forced=False)


def test_repeated_binding_converges_upward_then_stabilises():
    """The property the whole design rests on: a bucket whose grant binds is
    raised until it stops binding, and then it does NOT keep climbing."""
    cfg = dict(CFG, max_step=10.0)      # per-row dynamics, unthrottled
    steps, need = 5, 2400               # 5 x 260 = 1300 -> badly under-granted
    kmap, grants = {}, []
    for _ in range(40):
        k = up.lookup_k(kmap, up.bucket_keys(steps), cfg)
        g = up.pn100_grant(steps, k, cfg)
        grants.append(g)
        sim = simulate(need, g, cfg)
        r = row(sim["grant"], sim["rtok"], censor_forced=sim["censor_forced"])
        r["_bucket"] = steps
        r["_keys"] = up.bucket_keys(steps)
        r["_outcome"] = up.classify(r, cfg)
        kmap, _ = up.update_buckets([r], kmap, cfg)
    assert grants[0] == 1300
    assert max(grants) > need, "never grew past the true need"
    tail = grants[-12:]
    assert min(tail) >= need, f"fell back under the need: {tail}"
    # and it settles in a band rather than running to the ceiling
    assert max(tail) <= need / (1 - cfg["slack_frac"]) + 200, tail
    assert max(tail) < cfg["budget_ceil"]


def test_a_loose_bucket_decays_and_stops_at_the_floor():
    cfg = dict(CFG, max_step=10.0)
    steps, need = 10, 200               # 10 x 260 = 2600 for a 200-tok item
    kmap = {}
    for _ in range(200):
        k = up.lookup_k(kmap, up.bucket_keys(steps), cfg)
        g = up.pn100_grant(steps, k, cfg)
        sim = simulate(need, g, cfg)
        r = row(sim["grant"], sim["rtok"], censor_forced=sim["censor_forced"])
        r["_bucket"], r["_keys"] = steps, up.bucket_keys(steps)
        r["_outcome"] = up.classify(r, cfg)
        kmap, _ = up.update_buckets([r], kmap, cfg)
    assert kmap["10"] == cfg["k_min"]
    assert up.pn100_grant(steps, kmap["10"], cfg) == 1800   # 2600 x 0.7


# ─── THE USER'S 10x STORY, end to end ────────────────────────────────────────
#: The workload for the 10x story. Bucket-COHERENT by construction, because
#: that is the failure mode a per-bucket multiplier can actually fix: the
#: estimator's bias is a property of a step value, not of one item. See
#: `test_bucket_k_cannot_rescue_one_item_inside_a_fitted_bucket` for the case
#: it cannot fix, and the ledger doc for the two remedies.
#:   bucket 5  x10  need = 2.2 x grant -> the user's "cap too low on 10 items"
#:   bucket 8  x30  need = 0.25 x grant -> wildly over-granted
#:   buckets 10-15 x60 need = 0.72 x grant -> already right
def _workload():
    items = []
    for _ in range(10):
        items.append((5, int(up.pn100_grant(5, 1.0, CFG) * 2.2)))
    for _ in range(30):
        items.append((8, int(up.pn100_grant(8, 1.0, CFG) * 0.25)))
    for i in range(60):
        s = 10 + (i % 6)
        items.append((s, int(up.pn100_grant(s, 1.0, CFG) * 0.72)))
    return items


def _tenx(rounds=10, items=None, cfg=None):
    """Run the SAME 100 requests `rounds` times back to back through the real
    run_pass(), against a real sink dir, ledger file and cursor file."""
    d = tmpdir()
    sink = os.path.join(d, "sink")
    ledger = os.path.join(d, "pn162-ledger.json")
    cursor = os.path.join(d, "pn162-cursor.json")
    os.makedirs(sink, exist_ok=True)
    cfg = cfg or dict(CFG, window=2000)     # shipped params otherwise
    items = items or _workload()

    history, grants = [], []
    for rnd in range(rounds):
        led = up.read_json(ledger, {}) or {}
        kmap = up.load_kmap(led, cfg)
        pairs, bound, gs = [], 0, {}
        for idx, (steps, need) in enumerate(items):
            k = up.lookup_k(kmap, up.bucket_keys(steps), cfg)
            g = up.pn100_grant(steps, k, cfg)
            gs[steps] = g
            sim = simulate(need, g, cfg)
            if sim["censor_forced"]:
                bound += 1
            pairs.append(sink_rows(sim["grant"], sim["rtok"],
                                   censor_forced=sim["censor_forced"],
                                   req_id=f"r{rnd}-i{idx}",
                                   ts=1000.0 + rnd * 100 + idx))
        write_sink(sink, pairs)
        up.run_pass(cfg, sink, ledger, cursor)
        history.append(bound)
        grants.append(gs)
    return history, up.read_json(ledger, {}), cfg, items, grants


def test_ten_rounds_of_the_same_hundred_calibrate():
    """THE user's story: the same 100 requests, ten times back to back. Run 1
    force-closes 10 items; by run 10 the caps fit and nothing is force-closed
    — with no correctness signal and no extra LLM call anywhere in the loop."""
    history, led, cfg, items, _ = _tenx()
    assert history[0] == 10, f"round 1 must bind the 10 under-granted: {history}"
    assert history[-1] == 0, f"round 10 must bind nothing: {history}"
    assert all(h == 0 for h in history[3:]), history
    kmap = up.load_kmap(led, cfg)
    assert kmap["5"] > 2.0, kmap          # raised, not flailing
    assert "10" not in kmap or kmap["10"] == 1.0   # the fitted buckets untouched


def test_ten_rounds_also_reclaim_the_over_granted():
    _, led, cfg, _, grants = _tenx()
    kmap = up.load_kmap(led, cfg)
    assert kmap["8"] == cfg["k_min"], kmap
    assert grants[-1][8] < grants[0][8], (grants[0][8], grants[-1][8])


def test_ten_rounds_leave_the_already_right_buckets_alone():
    """No-op on OK is the third arm of the loop and it has to hold: a bucket
    that was already right must not drift just because it is being observed."""
    _, led, cfg, _, grants = _tenx()
    kmap = up.load_kmap(led, cfg)
    for b in ("10", "11", "12", "13", "14", "15"):
        assert kmap.get(b, 1.0) == 1.0, (b, kmap.get(b))
        assert grants[0][int(b)] == grants[-1][int(b)]


def test_bucket_k_cannot_rescue_one_item_inside_a_fitted_bucket():
    """The documented LIMIT of a per-bucket control, asserted so it cannot be
    forgotten: one badly under-granted item sharing a bucket with 20 correctly
    granted ones is never rescued — every bump it earns is cancelled by the
    slack the others accrue at the raised grant. The remedies are a finer key
    schema (steps x shape band) or the per-prompt `exact` leg, not a bigger k.
    """
    base = up.pn100_grant(6, 1.0, CFG)
    items = [(6, int(base * 3.0))] + [(6, int(base * 0.72))] * 20
    history, led, cfg, _, _ = _tenx(rounds=10, items=items)
    assert history[0] == 1
    assert history[-1] == 1, f"the fixture stopped isolating the case: {history}"
    assert up.load_kmap(led, cfg).get("6", 1.0) < 1.6


def test_run_pass_is_idempotent_on_an_unchanged_sink():
    d = tmpdir()
    sink, ledger, cursor = (os.path.join(d, x) for x in
                            ("sink", "led.json", "cur.json"))
    os.makedirs(sink)
    write_sink(sink, [sink_rows(1300, 1290, censor_forced=True)])
    a = up.run_pass(CFG, sink, ledger, cursor)
    b = up.run_pass(CFG, sink, ledger, cursor)
    assert a["bucket"] == b["bucket"]
    assert b["window"]["new_scored"] == 0


def test_ledger_written_by_run_pass_is_consumer_readable():
    _, led_dict, cfg, _, _ = _tenx(rounds=3)
    d = tmpdir()
    p = os.path.join(d, "led.json")
    up.atomic_write_json(p, led_dict)
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    os.environ["GENESIS_PN162_LEDGER"] = p
    os.environ["GENESIS_PN162_RESTAT_S"] = "0"
    cal.reset_cache()
    try:
        for b, k in led_dict["bucket"].items():
            if "|" in b:
                continue
            assert abs(cal.budget_multiplier(int(b)) - k) < 1e-6, b
    finally:
        for k in ("GENESIS_ENABLE_PN162_BUDGET_CAL", "GENESIS_PN162_LEDGER",
                  "GENESIS_PN162_RESTAT_S"):
            os.environ.pop(k, None)
        cal.reset_cache()


# ─── telemetry / oracle / explore ────────────────────────────────────────────
def _win(rows):
    out = []
    for r in rows:
        r["_bucket"] = up.invert_steps(r["budget_grant"], {}, CFG,
                                       r.get("prompt_tok"))
        r["_keys"] = up.bucket_keys(r["_bucket"], r.get("prompt_tok"))
        r["_outcome"] = up.classify(r, CFG)
        out.append(r)
    return out


def test_telemetry_reports_realized_steps_not_just_k():
    win = _win([row(1300, 2080 // 2 + i) for i in range(20)])
    tel = up.telemetry(win, {"5": 1.2}, {}, {}, {"edges": [], "bands": {}}, CFG)
    t = tel["5"]
    assert t["n"] == 20 and t["k"] == 1.2
    assert t["rtok_med"] is not None and t["steps_real_med"] is not None
    # "estimated 5, realized ~4" must be visible without any other tool
    assert abs(t["steps_real_med"] - t["rtok_med"] / 260.0) < 0.01


def test_anchor_adherence_is_lean_only_and_bounded():
    on = [row(1300, 5 * 260, route="lean") for _ in range(10)]      # exactly N
    off = [row(1300, 1 * 260, route="lean") for _ in range(10)]     # nowhere near
    tel = up.telemetry(_win(on + off), {}, {}, {},
                       {"edges": [], "bands": {}}, CFG)
    assert tel["5"]["n_lean"] == 20
    assert abs(tel["5"]["anchor_adherence"] - 0.5) < 1e-6
    deep = _win([row(1300, 5 * 260, route="deep") for _ in range(5)])
    tel2 = up.telemetry(deep, {}, {}, {}, {"edges": [], "bands": {}}, CFG)
    assert tel2["5"]["anchor_adherence"] is None


def test_oracle_needs_both_lanes_and_says_so():
    lean_only = _win([row(1300, 600, route="lean", score=i / 100.0)
                      for i in range(60)])
    o = up.announcement_oracle(lean_only, CFG)
    for band in o["bands"].values():
        assert band["n_deep_free"] == 0
        assert band["announce_bias"] is None      # never invented


def test_oracle_measures_bias_when_deep_rows_exist():
    rows = []
    for i in range(60):
        rows.append(row(1300, 5 * 260, route="lean", score=0.5))    # N = 5
        rows.append(row(6000, 12 * 260, route="deep", score=0.5))   # free-run 12
    o = up.announcement_oracle(_win(rows), CFG)
    biases = [b["announce_bias"] for b in o["bands"].values()
              if b["announce_bias"] is not None]
    assert biases and all(x > 5 for x in biases), o


def test_oracle_excludes_bound_deep_rows():
    """A bound deep row measured its cap, not its need."""
    rows = []
    for _ in range(60):
        rows.append(row(1300, 5 * 260, route="lean", score=0.5))
        rows.append(row(1300, 1290, route="deep", score=0.5,
                        censor_forced=True))
    o = up.announcement_oracle(_win(rows), CFG)
    for b in o["bands"].values():
        assert b["n_deep_free"] == 0


def test_explore_report_is_inert_without_arms():
    rep = up.explore_report(_win([row(1300, 600) for _ in range(5)]), CFG)
    assert rep["armed"] is False


def test_explore_report_compares_derivable_proxies():
    ctrl = [row(1300, 1290, censor_forced=True, arm="pn162:c", generated=1295)
            for _ in range(10)]
    expl = [row(1800, 900, arm="pn162:e1", generated=1400) for _ in range(10)]
    rep = up.explore_report(_win(ctrl + expl), CFG)
    assert rep["armed"] is True
    cells = rep["buckets"]
    arms = next(iter(cells.values())) if len(cells) == 1 else None
    allarms = {}
    for v in cells.values():
        allarms.update(v)
    assert allarms["pn162:c"]["bound_rate"] == 1.0
    assert allarms["pn162:e1"]["bound_rate"] == 0.0
    # answer tokens = generated - rtok, the only answer-shape proxy the sink
    # can support (correctness / rescue fires / format failures cannot).
    assert allarms["pn162:c"]["answer_tok_med"] == 5.0
    assert allarms["pn162:e1"]["answer_tok_med"] == 500.0
    assert allarms["pn162:c"]["empty_answer_rate"] == 1.0
    assert arms is None or True


def test_exact_is_written_empty_never_faked():
    d = tmpdir()
    sink = os.path.join(d, "sink")
    os.makedirs(sink)
    write_sink(sink, [sink_rows(1300, 1290, censor_forced=True)])
    led = up.run_pass(CFG, sink, os.path.join(d, "l.json"),
                      os.path.join(d, "c.json"))
    assert led["exact"] == {}
    assert "no prompt hash" in led["exact_note"]


def test_ledger_carries_its_params_and_key_schema():
    d = tmpdir()
    sink = os.path.join(d, "sink")
    os.makedirs(sink)
    write_sink(sink, [sink_rows(1300, 600)])
    led = up.run_pass(CFG, sink, os.path.join(d, "l.json"),
                      os.path.join(d, "c.json"))
    assert led["key_schema"] == "steps"
    assert led["params"]["bump"] == 1.15 and led["params"]["decay"] == 0.97
    assert led["schema"] == up.LEDGER_SCHEMA
    assert "ptok" in led["key_bands"]


# ─── standalone runner ───────────────────────────────────────────────────────
def main() -> int:
    failures = []
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — this IS the reporter
            failures.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  PASS  {name}")
    for d in _TMPDIRS:
        shutil.rmtree(d, ignore_errors=True)
    print()
    if failures:
        print(f"FAILED {len(failures)}/{len(tests)}: {failures}")
        return 1
    print(f"ALL {len(tests)} PN162 UPDATER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
