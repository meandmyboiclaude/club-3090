"""BUG-121 — PN114 `_live_think_len()` returns the frozen `think_count` at close.

Symptom on the record (tcbench-pn114enf30-container.log, gpqa-151):
    PN114: CLOSE (stable) at think=5 grace=384
while the request's real think position was ~1210.

MECHANISM (proven by test_close_reports_frozen_think_count below)
    `_arm()` sets state["in_end"] = True to hand the row to the holder's
    end-forcer.  `_finish_probe()` -> `_close()` then runs with in_end STILL
    True, and pn108._think_token_slice() short-circuits to None on
    `if state.get("in_end"): return None`.  `_close()` has no ordering guard,
    so it takes its `state.get("think_count", 0)` fallback — the value
    BUG-120 established is frozen near its init value while a request is
    under budget.  `_resume_thinking()` avoids this by flipping in_end back
    to False *before* it reads the slice (it carries the explicit comment
    "computed after the in_end flip so the slice is valid"); `_close()`,
    `_finish_probe()`'s cooldown and `request_confirm()` did not get the same
    treatment.

    Consequence on the bare-cut branch: thinking_token_budget := 5 + grace
    instead of 1210 + grace.  The 07-23 run was harmless only because
    GENESIS_PN112_WRAPUP was ON, so `think` reached nothing but the log line.

REACHABILITY (see test_live_softland_seat_never_uses_frozen_fallback)
    On the 2026-07-26 shipping config the four fallback sites are all behind
    PN114 probe/confirm/wrapup env flags that are OFF.  The one live caller —
    `observe_state()`, reached because GENESIS_ENABLE_PN121_SOFTLAND=1 — has
    NO think_count fallback and treats None as "not mid-think".  That site is
    correct as written and these tests pin it that way.

NO GPU REQUIRED.  Pure dict/state logic — nothing here touches torch, the
model or the server.  Run it the house way, in a throwaway container off the
pinned image so the plateau sources under test are the bytes the next boot
would execute:

  sudo podman run --rm --entrypoint /bin/bash \\
    -v $REPO/fixes:/fixes:ro \\
    -v $REPO/models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis:\\
       /usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \\
    localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725 -lc \\
    'python3 -m pytest -q --noconftest /fixes/test_bug121_live_think_len.py'

It also runs on any host that has pytest but no vLLM: when `vllm.logger` is
not importable a stub package is installed and the plateau sources are read
straight out of the repo tree.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import types
from pathlib import Path

import pytest

GENESIS_VLLM = (Path(__file__).resolve().parents[1] / "models" / "qwen3.6-27b"
                / "vllm" / "patches" / "genesis" / "vllm")

THINK_POS = 1210      # the real think position in the BUG-121 trace
FROZEN_COUNT = 5      # what the holder's think_count reads there


def _install_vllm_stub() -> None:
    """Make `vllm._genesis.plateau.*` importable with or without vLLM.

    In-container the real package is used (that is the point of the house
    runner).  Off-box, `vllm` is faked and pointed at the repo's genesis tree.
    """
    try:
        importlib.import_module("vllm.logger")
        return
    except Exception:
        pass
    if "vllm" in sys.modules and getattr(sys.modules["vllm"], "_bug121_stub", False):
        return
    vllm = types.ModuleType("vllm")
    vllm.__path__ = [str(GENESIS_VLLM)]
    vllm._bug121_stub = True
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = logging.getLogger
    vllm.logger = logger_mod
    sys.modules["vllm"] = vllm
    sys.modules["vllm.logger"] = logger_mod


@pytest.fixture()
def plateau(monkeypatch):
    """Fresh pn114/pn108 with every PN114 env flag cleared."""
    _install_vllm_stub()
    for var in ("GENESIS_ENABLE_PN114_PROBE", "GENESIS_PN112_WRAPUP",
                "GENESIS_PN112_CONFIRM", "GENESIS_PN112_WRAPUP_AT_CAP",
                "GENESIS_ENABLE_PN121_SOFTLAND", "GENESIS_PN114_MODE",
                "GENESIS_PN114_GRACE", "GENESIS_PN112_GRACE"):
        monkeypatch.delenv(var, raising=False)
    pn108 = importlib.import_module("vllm._genesis.plateau.pn108")
    pn114 = importlib.import_module("vllm._genesis.plateau.pn114")
    importlib.reload(pn114)
    # the ids file is written at boot by fixes/pn114_boot_ids.py; supply the
    # live shape (from the 2026-07-26 boot log) so _ids() never touches /tmp.
    pn114._IDS = {"probe": [1, 2, 3], "close_paren": [4], "newline": [198],
                  "nl_end": [200], "wrapup_close": list(range(17)),
                  "softland_close": list(range(17)), "tool_call": [248058],
                  "ppen": [5]}
    return types.SimpleNamespace(pn108=pn108, pn114=pn114)


def _mid_think_state(think_pos: int = THINK_POS) -> dict:
    """A row mid-think on this deployment's prompt-opened-<think> template
    (continue_thinking=True — see pn108._think_token_slice BUG-107d note)."""
    return {
        "continue_thinking": True,
        "in_think": True,
        "in_end": False,
        "end_thinking": -1,
        "start_thinking": -1,
        "output_tok_ids": list(range(10_000, 10_000 + think_pos)),
        "think_count": FROZEN_COUNT,
        "thinking_token_budget": 4000,
        "check_count_down": 4000,
        "end_count": 0,
        "force_index": [],
    }


# --------------------------------------------------------------------------
# 1. the slice itself is right when it is allowed to run
# --------------------------------------------------------------------------

def test_slice_reports_the_real_depth_mid_think(plateau):
    state = _mid_think_state()
    st = plateau.pn114._st(state)
    st["tsl"] = 1
    assert plateau.pn114._live_think_len(state, st) == THINK_POS


def test_slice_returns_none_once_in_end_is_set(plateau):
    """The short-circuit that BUG-121 trips over — asserted directly so the
    mechanism is pinned even if _close() is later rewritten."""
    state = _mid_think_state()
    st = plateau.pn114._st(state)
    st["tsl"] = 1
    state["in_end"] = True
    assert plateau.pn108._think_token_slice(state, 1) is None
    assert plateau.pn114._live_think_len(state, st) is None


# --------------------------------------------------------------------------
# 2. the defect: _close() reads the slice after _arm() set in_end
# --------------------------------------------------------------------------

def test_arm_sets_in_end_before_close_runs(plateau):
    """_arm() is what makes the slice unavailable to _close()."""
    state = _mid_think_state()
    plateau.pn114._arm(state, [1, 2, 3], "probe_force", "req-bug121")
    assert state["in_end"] is True
    assert state["in_think"] is False


def test_close_uses_the_real_depth_not_the_frozen_count(plateau, monkeypatch):
    """THE BUG-121 REGRESSION TEST.

    Reproduces `CLOSE (stable) at think=5` on a row whose real depth is 1210:
    with wrapup OFF the bare-cut branch writes think + grace into
    thinking_token_budget, so the wrong depth is directly observable.

    Pre-fix this asserted 389 (= FROZEN_COUNT + 384) and failed.
    """
    monkeypatch.setenv("GENESIS_PN114_GRACE", "384")
    state = _mid_think_state()
    st = plateau.pn114._st(state)
    st["tsl"] = 1
    plateau.pn114._arm(state, [1, 2, 3], "probe_force", "req-bug121")

    plateau.pn114._close(state, st, "stable")

    budget = state["thinking_token_budget"]
    assert budget != FROZEN_COUNT + 384, (
        "BUG-121 regression: _close() took the frozen think_count fallback")
    assert budget == THINK_POS + 384, budget


def test_in_span_read_does_not_mutate_the_row(plateau):
    """The in_span read borrows in_end/in_think — it must hand them back
    exactly, or the holder's end-forcer loses the span mid-flight."""
    state = _mid_think_state()
    st = plateau.pn114._st(state)
    st["tsl"] = 1
    plateau.pn114._arm(state, [1, 2, 3], "probe_force", "req-bug121")
    before = (state["in_end"], state["in_think"])

    assert plateau.pn114._live_think_len(state, st, in_span=True) == THINK_POS

    assert (state["in_end"], state["in_think"]) == before == (True, False)


def test_in_span_default_is_off_for_non_span_callers(plateau):
    """Default must stay False: for observe_state / request_confirm a None is
    the real 'not mid-think' signal and must not be papered over."""
    state = _mid_think_state()
    st = plateau.pn114._st(state)
    st["tsl"] = 1
    state["in_end"] = True
    assert plateau.pn114._live_think_len(state, st) is None
    assert plateau.pn114._live_think_len(state, st, in_span=True) == THINK_POS


def test_in_span_still_none_when_the_block_really_ended(plateau):
    """in_span relaxes the in_end short-circuit only. A block that has
    genuinely closed (end_thinking set) must still read None."""
    state = _mid_think_state()
    st = plateau.pn114._st(state)
    st["tsl"] = 1
    state["in_end"] = True
    state["end_thinking"] = 4242
    assert plateau.pn114._live_think_len(state, st, in_span=True) is None


# --------------------------------------------------------------------------
# 3. reachability: the one LIVE call site is clean
# --------------------------------------------------------------------------

def test_live_softland_seat_never_uses_frozen_fallback(plateau, monkeypatch):
    """observe_state() is the only _live_think_len caller reachable on the
    2026-07-26 shipping config (GENESIS_ENABLE_PN121_SOFTLAND=1, every other
    PN114 flag OFF).  It must hand PN121 the real depth, and on a None slice
    it must release rather than substitute think_count."""
    monkeypatch.setenv("GENESIS_ENABLE_PN121_SOFTLAND", "1")
    importlib.reload(plateau.pn114)
    pn114 = plateau.pn114
    pn114._IDS = {"probe": [1, 2, 3], "newline": [198], "nl_end": [200],
                  "close_paren": [4], "wrapup_close": list(range(17)),
                  "softland_close": list(range(17)), "tool_call": [248058],
                  "ppen": [5]}
    assert pn114.softland_enabled() and pn114.any_enabled()

    seen: list = []
    fake = types.ModuleType("vllm._genesis.plateau.pn121_softland")
    fake.observe = lambda state, think, seq_idx, req_id: seen.append(("observe", think))
    fake.release = lambda state: seen.append(("release", None))
    monkeypatch.setitem(sys.modules, "vllm._genesis.plateau.pn121_softland", fake)
    # `from vllm._genesis.plateau import pn121_softland` resolves the PACKAGE
    # ATTRIBUTE once the real module has been imported by any earlier test, so
    # patching sys.modules alone silently misses.
    pkg = importlib.import_module("vllm._genesis.plateau")
    monkeypatch.setattr(pkg, "pn121_softland", fake, raising=False)

    state = _mid_think_state()
    pn114.observe_state(state, think_start_len=1, seq_idx=0, req_id="r")
    assert seen == [("observe", THINK_POS)], seen

    # slice unavailable -> release, and crucially NOT ("observe", 5)
    seen.clear()
    state2 = _mid_think_state()
    state2["in_end"] = True
    pn114.observe_state(state2, think_start_len=1, seq_idx=0, req_id="r")
    assert seen == [("release", None)], seen


def test_close_is_unreachable_with_only_softland_enabled(plateau, monkeypatch):
    """_close() is called only from _finish_probe(), which on_force_complete()
    only reaches for a probe/confirm phase.  With softland the sole enabled
    mechanism, no phase PN121 sets routes there."""
    monkeypatch.setenv("GENESIS_ENABLE_PN121_SOFTLAND", "1")
    importlib.reload(plateau.pn114)
    pn114 = plateau.pn114
    assert pn114.probes_enabled() is False
    assert pn114.confirm_enabled() is False
    assert pn114.wrapup_enabled() is False

    calls: list = []
    monkeypatch.setattr(pn114, "_close",
                        lambda *a, **k: calls.append(a), raising=True)
    # PN121's landing goes through arm_wrapup, which sets phase "wrapup";
    # on_force_complete must not route a "wrapup" phase into _finish_probe.
    state = _mid_think_state()
    st = pn114._st(state)
    st["phase"] = "wrapup"
    st["reason"] = None
    pn114.on_force_complete(state)
    assert calls == []


def test_source_has_exactly_the_known_fallback_sites(plateau):
    """Canary against silent growth of the frozen-count fallback pattern.
    If this fires, a new _live_think_len caller appeared — check whether it
    is on a live path before shipping it."""
    src = Path(plateau.pn114.__file__).read_text()
    plain = src.count("_live_think_len(state, st)")
    in_span = src.count("_live_think_len(state, st, in_span=True)")
    n_fallbacks = src.count('state.get("think_count", 0)')
    # plain: _resume_thinking (flips in_end itself), request_confirm and
    # observe_state (both guaranteed in_end=False).  in_span: _finish_probe's
    # cooldown and _close, the two BUG-121 sites.
    assert plain == 3, f"_live_think_len plain call sites moved: {plain}"
    assert in_span == 2, f"_live_think_len in-span call sites moved: {in_span}"
    assert n_fallbacks == 4, f"think_count fallbacks moved: {n_fallbacks}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
