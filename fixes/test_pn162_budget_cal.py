#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PN162 budget calibrator — CONSUMER tests (2026-07-27).

No GPU, no vLLM, no container, no network. `auto_budget` is loaded by path with
its one vllm import stubbed (same technique as test_pn100_total_ceiling.py) and
`pn162_budget_cal` is loaded as its real sibling, so this exercises the exact
code the engine runs.

    /usr/bin/python3 fixes/test_pn162_budget_cal.py       # standalone
    python -m pytest fixes/test_pn162_budget_cal.py -q    # where pytest exists

Covered
  * dark by default: flag off -> byte-identical PN100 grants even with a
    perfectly good ledger sitting on disk
  * the multiplier: grant' = round100(steps x TOK_PER_STEP x k[bucket])
  * bucket folding at 16, clamping to [K_MIN, K_MAX]
  * every failure mode is IDENTITY: no file, empty file, truncated JSON, wrong
    root type, garbage k values, an unreadable path
  * the STEP_BUDGET_MAP guard — refuses to multiply, loudly, and the map branch
    is unreachable-by-construction anyway
  * mtime cache: one re-parse per change, throttled re-stat, and a swapped
    ledger is picked up
  * exact leg: two flags, bound-only, floor-never-reduces, inert without both
  * explore arm: disarmed identity, N+delta never negative, control labelled,
    budget-follows toggle, and the arm stamp lands on vllm_xargs
  * the announced N is NEVER multiplied by k (only the token grant is)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

_MW = (Path(__file__).resolve().parents[1] / "models/qwen3.6-27b/vllm/patches"
       / "genesis/vllm/_genesis/middleware")
_AB = _MW / "auto_budget.py"
_CAL = _MW / "pn162_budget_cal.py"


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load():
    for n in ("vllm", "vllm._genesis", "vllm._genesis.middleware",
              "vllm._genesis.middleware.lazy_reasoner"):
        if n not in sys.modules:
            sys.modules[n] = types.ModuleType(n)
    sys.modules["vllm._genesis.middleware.lazy_reasoner"]._extract_text_from_message = (
        lambda m: "")
    cal = _load_by_path("_pn162_under_test", _CAL)
    # auto_budget's package import resolves against the stub package, so put
    # the real calibrator there — this is also what proves the guarded import
    # in auto_budget picks up a module rather than silently landing on None.
    sys.modules["vllm._genesis.middleware"].pn162_budget_cal = cal
    ab = _load_by_path("_ab_pn162_under_test", _AB)
    return ab, cal


ab, cal = _load()

_OWNED_ENV = (
    "GENESIS_ENABLE_PN162_BUDGET_CAL",
    "GENESIS_ENABLE_PN162_EXACT",
    "GENESIS_ENABLE_PN162_EXPLORE",
    "GENESIS_PN162_LEDGER",
    "GENESIS_PN162_RESTAT_S",
    "GENESIS_PN162_K_MIN",
    "GENESIS_PN162_K_MAX",
    "GENESIS_PN162_EXACT_MULT",
    "PN162_EXPLORE_EPS",
    "PN162_EXPLORE_DELTAS",
    "PN162_EXPLORE_BUDGET_FOLLOWS",
    "GENESIS_PN100_CONTINUOUS",
    "GENESIS_PN100_TOK_PER_STEP",
    "GENESIS_PN100_STEP_BUDGET_MAP",
    "GENESIS_PN100_HIGHSTEP_MULT",
    "GENESIS_PN100_BUDGET_FLOOR",
    "GENESIS_PN100_BUDGET_CEIL",
)

_TMP = tempfile.mkdtemp(prefix="pn162-test-")
_SEQ = {"n": 0}


def reset_env() -> None:
    for k in _OWNED_ENV:
        os.environ.pop(k, None)
    cal.reset_cache()


def base_env() -> None:
    """The live bench boot's PN100 settings (tcbench8021.yml:396-397)."""
    reset_env()
    os.environ["GENESIS_PN100_CONTINUOUS"] = "1"
    os.environ["GENESIS_PN100_TOK_PER_STEP"] = "260"
    os.environ["GENESIS_PN162_RESTAT_S"] = "0"   # tests never wait on a stat


def write_ledger(bucket=None, exact=None, raw=None) -> str:
    """A fresh ledger path per call — the consumer caches on (path, mtime,
    size), and two writes inside one filesystem mtime tick would otherwise
    look identical on a coarse-timestamp FS."""
    _SEQ["n"] += 1
    p = os.path.join(_TMP, f"ledger-{_SEQ['n']}.json")
    if raw is None:
        raw = {"schema": 1, "updated_ts": time.time(),
               "bucket": {str(k): v for k, v in (bucket or {}).items()},
               "exact": exact or {}}
    with open(p, "w", encoding="utf-8") as f:
        if isinstance(raw, str):
            f.write(raw)
        else:
            json.dump(raw, f)
    os.environ["GENESIS_PN162_LEDGER"] = p
    cal.reset_cache()
    return p


class Req:
    """Minimal stand-in for vLLM's ChatCompletionRequest."""

    def __init__(self):
        self.chat_template_kwargs = None
        self.thinking_token_budget = None
        self.vllm_xargs = None
        self.max_tokens = None
        self.max_completion_tokens = None


class FrozenReq(Req):
    """A pydantic-frozen-model stand-in: writes raise once frozen."""

    def freeze(self):
        object.__setattr__(self, "_frozen", True)
        return self

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("frozen")
        object.__setattr__(self, name, value)


# ─── dark by default ─────────────────────────────────────────────────────────
def test_flag_off_is_byte_identical():
    base_env()
    write_ledger({5: 3.0, 8: 0.7})
    assert cal.budget_multiplier(5) == 1.0
    assert ab._continuous_budget(2, 5) == 1300      # 5 x 260 = 1300
    assert ab._continuous_budget(2, 8) == 2100      # 8 x 260 = 2080 -> 2100


def test_flag_on_no_ledger_is_identity():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    os.environ["GENESIS_PN162_LEDGER"] = os.path.join(_TMP, "does-not-exist")
    cal.reset_cache()
    assert cal.budget_multiplier(5) == 1.0
    assert ab._continuous_budget(2, 5) == 1300


# ─── the multiplier ──────────────────────────────────────────────────────────
def test_multiplier_scales_the_grant_before_rounding():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    write_ledger({5: 1.5, 8: 0.8})
    assert cal.budget_multiplier(5) == 1.5
    # 5 x 260 x 1.5 = 1950 -> round100 -> 2000 (NOT round100(1300) x 1.5)
    assert ab._continuous_budget(2, 5) == 2000
    # 8 x 260 x 0.8 = 1664 -> 1700
    assert ab._continuous_budget(2, 8) == 1700


def test_unknown_bucket_is_identity():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    write_ledger({5: 2.0})
    assert cal.budget_multiplier(7) == 1.0
    assert ab._continuous_budget(2, 7) == 1800      # 1820 -> 1800


def test_bucket_folds_at_16():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    write_ledger({16: 1.2})
    assert cal.bucket_of(16) == cal.bucket_of(30) == cal.bucket_of(999) == 16
    assert cal.bucket_of(0) == cal.bucket_of(-4) == 1
    assert cal.budget_multiplier(30) == 1.2
    assert cal.budget_multiplier(16) == 1.2


def test_k_is_clamped_both_ways():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    write_ledger({3: 99.0, 4: 0.01})
    assert cal.budget_multiplier(3) == cal.K_MAX_DEFAULT
    assert cal.budget_multiplier(4) == cal.K_MIN_DEFAULT
    os.environ["GENESIS_PN162_K_MAX"] = "1.4"
    assert cal.budget_multiplier(3) == 1.4


def test_floor_and_ceiling_still_bind():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    write_ledger({1: 0.7, 16: 3.0})
    # PN100's own clamps come AFTER k, so k can never escape them.
    assert ab._continuous_budget(2, 1) >= 128
    assert ab._continuous_budget(2, 40) <= 10240


# ─── every failure mode is identity ──────────────────────────────────────────
def test_broken_ledgers_are_all_identity():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    for raw in ('{"bucket": {"5": 1.5}',            # truncated
                '',                                  # empty
                'null',                              # wrong root
                '[1,2,3]',                           # wrong root
                'not json at all'):
        write_ledger(raw=raw)
        assert cal.budget_multiplier(5) == 1.0, raw
        assert ab._continuous_budget(2, 5) == 1300, raw


def test_garbage_bucket_values_are_dropped_not_fatal():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    write_ledger(raw={"schema": 1, "bucket": {
        "5": 1.5, "6": "banana", "7": None, "eight": 2.0,
        "99": 2.0, "0": 2.0, "9": float("nan") if False else 1.9}})
    assert cal.budget_multiplier(5) == 1.5
    assert cal.budget_multiplier(6) == 1.0
    assert cal.budget_multiplier(7) == 1.0
    assert cal.budget_multiplier(9) == 1.9


def test_unreadable_ledger_is_identity():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    p = write_ledger({5: 2.0})
    assert cal.budget_multiplier(5) == 2.0
    os.chmod(p, 0o000)
    cal.reset_cache()
    try:
        # root can read a 000 file; only assert the no-raise contract there.
        assert isinstance(cal.budget_multiplier(5), float)
        if os.geteuid() != 0:
            assert cal.budget_multiplier(5) == 1.0
    finally:
        os.chmod(p, 0o644)


# ─── the STEP_BUDGET_MAP guard ───────────────────────────────────────────────
def test_step_budget_map_refuses_k():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    write_ledger({5: 2.0})
    assert cal.budget_multiplier(5) == 2.0
    os.environ["GENESIS_PN100_STEP_BUDGET_MAP"] = "3:800,10:2600"
    before = cal.get_stats()["map_refused"]
    assert cal.budget_multiplier(5) == 1.0
    assert cal.get_stats()["map_refused"] == before + 1
    # and the map branch returns its own number, untouched by k:
    # steps 5 interpolates 800..2600 over 3..10 -> 1314 -> round100 -> 1300.
    # With k=2.0 the un-mapped path would have said 2600.
    assert ab._continuous_budget(2, 5) == 1300


# ─── mtime cache ─────────────────────────────────────────────────────────────
def test_ledger_swap_is_picked_up():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    p = os.path.join(_TMP, "hot.json")
    os.environ["GENESIS_PN162_LEDGER"] = p
    cal.reset_cache()
    for k in (1.2, 1.8, 0.9):
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"schema": 1, "bucket": {"5": k}}, f)
        os.utime(p, None)
        assert cal.budget_multiplier(5) == k


def test_restat_throttle_holds_the_cached_value():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    p = os.path.join(_TMP, "throttled.json")
    os.environ["GENESIS_PN162_LEDGER"] = p
    os.environ["GENESIS_PN162_RESTAT_S"] = "60"
    cal.reset_cache()
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"schema": 1, "bucket": {"5": 1.2}}, f)
    assert cal.budget_multiplier(5) == 1.2
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"schema": 1, "bucket": {"5": 2.5}}, f)
    assert cal.budget_multiplier(5) == 1.2          # throttled, still cached
    os.environ["GENESIS_PN162_RESTAT_S"] = "0"
    cal._CACHE["next_stat"] = 0.0
    assert cal.budget_multiplier(5) == 2.5


def test_reparse_only_on_change():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    write_ledger({5: 1.3})
    n0 = cal.get_stats()["reloads"]
    for _ in range(20):
        cal.budget_multiplier(5)
    assert cal.get_stats()["reloads"] == n0 + 1


# ─── the exact leg ───────────────────────────────────────────────────────────
_EXACT = {"deadbeef" * 4: {"last_grant": 1600, "last_outcome": "bound",
                           "last_rtok": 1587, "n_seen": 3}}
_H = "deadbeef" * 4


def test_exact_needs_both_flags():
    base_env()
    write_ledger({}, exact=_EXACT)
    assert cal.exact_floor(_H, 1300) == 1300                 # both off
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    assert cal.exact_floor(_H, 1300) == 1300                 # EXACT still off
    os.environ["GENESIS_ENABLE_PN162_EXACT"] = "1"
    assert cal.exact_floor(_H, 1300) == 2000                 # 1600 x 1.25


def test_exact_only_fires_on_bound_and_never_lowers():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    os.environ["GENESIS_ENABLE_PN162_EXACT"] = "1"
    write_ledger({}, exact={
        _H: {"last_grant": 1600, "last_outcome": "slack"},
        "a" * 32: {"last_grant": 400, "last_outcome": "bound"},
    })
    assert cal.exact_floor(_H, 1300) == 1300                 # slack -> no-op
    assert cal.exact_floor("a" * 32, 3000) == 3000           # floor < grant
    assert cal.exact_floor("nosuchhash", 1300) == 1300
    assert cal.exact_floor(None, 1300) == 1300


def test_exact_middleware_leg_is_identity_by_default():
    base_env()
    r = Req()
    assert ab._pn162_exact(r, 1300, "f" * 64) == 1300
    assert r.vllm_xargs is None


def test_exact_middleware_leg_stamps_and_raises():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    os.environ["GENESIS_ENABLE_PN162_EXACT"] = "1"
    key = "b" * 64
    write_ledger({}, exact={key[:32]: {"last_grant": 1600,
                                       "last_outcome": "bound"}})
    r = Req()
    assert ab._pn162_exact(r, 1300, key) == 2000
    assert r.vllm_xargs["pn162_phash"] == key[:32]


# ─── the exploration arm ─────────────────────────────────────────────────────
def test_explore_disarmed_is_identity():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    write_ledger({5: 1.0})
    assert cal.explore_arm(5) == (5, 5, None)
    r = Req()
    assert ab._pn162_arm(r, 5) == (5, 5)
    assert r.vllm_xargs is None


def test_explore_eps_zero_labels_control_only():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    os.environ["GENESIS_ENABLE_PN162_EXPLORE"] = "1"
    write_ledger({5: 1.0})
    for _ in range(20):
        ann, size, arm = cal.explore_arm(5)
        assert (ann, size) == (5, 5)
        assert arm == "pn162:c"


def test_explore_bumps_and_never_goes_negative():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    os.environ["GENESIS_ENABLE_PN162_EXPLORE"] = "1"
    os.environ["PN162_EXPLORE_EPS"] = "1.0"
    os.environ["PN162_EXPLORE_DELTAS"] = "1,2,-3,0"
    write_ledger({5: 1.0})
    seen = set()
    for _ in range(200):
        ann, size, arm = cal.explore_arm(5)
        assert ann in (6, 7) and size == ann
        assert arm in ("pn162:e1", "pn162:e2")
        seen.add(ann)
    assert seen == {6, 7}


def test_explore_budget_follows_toggle():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    os.environ["GENESIS_ENABLE_PN162_EXPLORE"] = "1"
    os.environ["PN162_EXPLORE_EPS"] = "1.0"
    os.environ["PN162_EXPLORE_DELTAS"] = "2"
    os.environ["PN162_EXPLORE_BUDGET_FOLLOWS"] = "0"
    write_ledger({5: 1.0})
    assert cal.explore_arm(5) == (7, 5, "pn162:e2")          # anchor-only arm


def test_explore_arm_is_stamped_for_the_sink():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    os.environ["GENESIS_ENABLE_PN162_EXPLORE"] = "1"
    os.environ["PN162_EXPLORE_EPS"] = "1.0"
    os.environ["PN162_EXPLORE_DELTAS"] = "1"
    write_ledger({5: 1.0})
    r = Req()
    assert ab._pn162_arm(r, 5) == (6, 6)
    # x_caller (not caller) — a bench harness stamping `caller` still wins.
    assert r.vllm_xargs[cal.ARM_XARG] == "pn162:e1"


def test_stamp_is_fail_open_on_a_frozen_request():
    base_env()
    r = FrozenReq()
    r.vllm_xargs = {"h119_overridable": 1}
    r.freeze()
    cal.stamp_xargs(r, arm="pn162:e1")               # must not raise
    assert r.vllm_xargs == {"h119_overridable": 1}


def test_explore_arm_survives_none_steps():
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    os.environ["GENESIS_ENABLE_PN162_EXPLORE"] = "1"
    os.environ["PN162_EXPLORE_EPS"] = "1.0"
    assert cal.explore_arm(None) == (None, None, None)
    assert ab._pn162_arm(Req(), None) == (None, None)


# ─── k must never touch the announced N ──────────────────────────────────────
def test_k_scales_the_grant_but_not_the_announced_n():
    """The LEAN lane renders the announced N into the prompt
    (`_contract_v3_sized`), so a k-corrected N would be a prompt change.
    PN162 sizes only."""
    base_env()
    os.environ["GENESIS_ENABLE_PN162_BUDGET_CAL"] = "1"
    write_ledger({5: 2.0})
    r = Req()
    ann, size = ab._pn162_arm(r, 5)
    assert (ann, size) == (5, 5)                     # explore disarmed
    grant = ab._continuous_budget(2, size)
    assert grant == 2600                             # 5 x 260 x 2.0
    ab._stash_steps(r, ann)
    assert r.chat_template_kwargs["pn100_steps"] == 5   # NOT 10, NOT 2600/260


# ─── wiring / provenance ─────────────────────────────────────────────────────
def test_auto_budget_actually_bound_the_calibrator():
    assert ab._pn162 is not None
    assert ab._pn162.FLAG == "GENESIS_ENABLE_PN162_BUDGET_CAL"


def test_flag_strings_present_for_the_patch_id_linter():
    src = (_CAL).read_text(encoding="utf-8")
    assert "GENESIS_ENABLE_PN162_BUDGET_CAL" in src
    assert "GENESIS_ENABLE_PN162_EXACT" in src
    assert "GENESIS_ENABLE_PN162_EXPLORE" in src


def test_stats_surface_exists():
    s = cal.get_stats()
    for k in ("applied", "identity", "no_ledger", "map_refused", "exact_hits",
              "reloads", "read_errors", "explore_arm", "explore_control"):
        assert k in s


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
    reset_env()
    print()
    if failures:
        print(f"FAILED {len(failures)}/{len(tests)}: {failures}")
        return 1
    print(f"ALL {len(tests)} PN162 CONSUMER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
