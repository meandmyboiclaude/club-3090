"""PN108 plateau detector — unit tests (pure python, no vllm import)."""

import importlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _genesis.plateau import pn108  # noqa: E402


CFG = {
    "arm_after": 512, "window": 128, "floor": 0.20, "consec": 3,
    "repeat_min": 3, "grace": 0, "enforce": True,
}


def _fresh(monkeypatch, enabled=True):
    if enabled:
        monkeypatch.setenv("GENESIS_ENABLE_PN108_PLATEAU_CAP", "1")
    else:
        monkeypatch.delenv("GENESIS_ENABLE_PN108_PLATEAU_CAP", raising=False)
    pn108.reset_stats()


def _novel_stream(n, seed=7):
    rng = random.Random(seed)
    return [rng.randrange(0, 50000) for _ in range(n)]


def _loop_stream(n, period=12, seed=3):
    rng = random.Random(seed)
    base = [rng.randrange(0, 50000) for _ in range(period)]
    return [base[i % period] for i in range(n)]


def test_novel_stream_never_fires(monkeypatch):
    _fresh(monkeypatch)
    det = pn108.PlateauDetector(dict(CFG))
    assert det.observe(_novel_stream(8192)) is False
    assert det.low_streak == 0


def test_loop_stream_fires_after_arm_and_streak(monkeypatch):
    _fresh(monkeypatch)
    det = pn108.PlateauDetector(dict(CFG))
    # novel prefix past arming, then a tight loop
    assert det.observe(_novel_stream(512)) is False
    fired = det.observe(_loop_stream(128 * 3))
    assert fired is True
    # sticky
    assert det.observe(_novel_stream(256)) is True


def _dna_stream(n, seed=11):
    """Low-novelty but LEGITIMATE content: transcription over a 4-token
    alphabet (gpqa-127 class). Trigram novelty collapses, but 8-grams are
    mostly unique — the periodicity gate must spare it."""
    rng = random.Random(seed)
    return [rng.randrange(0, 4) for _ in range(n)]


def test_dna_transcription_is_spared(monkeypatch):
    _fresh(monkeypatch)
    det = pn108.PlateauDetector(dict(CFG))
    det.observe(_novel_stream(512))
    assert det.observe(_dna_stream(128 * 8)) is False  # novelty low, no fire


def test_pre_arm_loop_gives_no_verdict(monkeypatch):
    _fresh(monkeypatch)
    det = pn108.PlateauDetector(dict(CFG))
    assert det.observe(_loop_stream(CFG["arm_after"])) is False
    assert det.low_streak == 0  # windows scored but not judged


def test_mtp_chunked_feed_equivalent(monkeypatch):
    _fresh(monkeypatch)
    stream = _novel_stream(512) + _loop_stream(600)
    bulk = pn108.PlateauDetector(dict(CFG))
    bulk_fired = bulk.observe(stream)
    chunked = pn108.PlateauDetector(dict(CFG))
    i, rng, fired = 0, random.Random(1), False
    while i < len(stream):
        step = rng.randint(1, 4)  # MTP accepted-chunk arrival
        fired = chunked.observe(stream[i : i + step])
        i += step
    assert fired == bulk_fired is True
    assert chunked.scored_tokens == bulk.scored_tokens


def _mk_state(output, start_thinking=0, think_count=None, budget=10240):
    return {
        "in_think": True,
        "in_end": False,
        "check_count_down": budget,
        "think_count": think_count if think_count is not None else len(output),
        "end_count": 0,
        "prompt_tok_ids": None,
        "output_tok_ids": list(output),
        "thinking_token_budget": budget,
        "prev_output_length": 0,
        "spec_token_ids": [],
        "force_index": [],
        "start_thinking": start_thinking,
        "end_thinking": -1,
        "in_spec_mode": False,
        "bonus_token_forced": False,
        "continue_thinking": False,
        "scan_offset": 0,
    }


THINK_START_LEN = 2  # e.g. ["<think>", "\n"] — passed by the holder


def test_observe_state_enforce_fires_and_mutates_budget(monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.setenv("GENESIS_PN108_MODE", "enforce")
    monkeypatch.setenv("GENESIS_PN108_ARM_AFTER_TOKENS", "512")
    monkeypatch.setenv("GENESIS_PN108_WINDOW_TOKENS", "128")
    monkeypatch.setenv("GENESIS_PN108_CONSEC_WINDOWS", "3")
    tokens = [0] * THINK_START_LEN + _novel_stream(512) + _loop_stream(600)
    state = _mk_state(tokens, start_thinking=0, think_count=len(tokens) - THINK_START_LEN)
    pn108.observe_state(state, THINK_START_LEN)
    assert state["thinking_token_budget"] == state["think_count"]  # grace=0
    assert state["check_count_down"] == 0
    assert state.get("pn108_applied") is True
    assert pn108.get_stats()["fires"] == 1


def test_observe_state_shadow_default_logs_but_never_mutates(monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.delenv("GENESIS_PN108_MODE", raising=False)  # default = shadow
    monkeypatch.setenv("GENESIS_PN108_ARM_AFTER_TOKENS", "512")
    monkeypatch.setenv("GENESIS_PN108_WINDOW_TOKENS", "128")
    monkeypatch.setenv("GENESIS_PN108_CONSEC_WINDOWS", "3")
    tokens = [0] * THINK_START_LEN + _novel_stream(512) + _loop_stream(600)
    state = _mk_state(tokens, start_thinking=0, think_count=len(tokens) - THINK_START_LEN)
    budget_before = state["thinking_token_budget"]
    countdown_before = state["check_count_down"]
    pn108.observe_state(state, THINK_START_LEN)
    assert state["thinking_token_budget"] == budget_before
    assert state["check_count_down"] == countdown_before
    assert state.get("pn108_applied") is True  # verdict recorded once
    stats = pn108.get_stats()
    assert stats["shadow_fires"] == 1 and stats["fires"] == 0


def test_observe_state_incremental_no_double_feed(monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.setenv("GENESIS_PN108_ARM_AFTER_TOKENS", "512")
    monkeypatch.setenv("GENESIS_PN108_WINDOW_TOKENS", "128")
    tokens = [0] * THINK_START_LEN + _novel_stream(2048)
    state = _mk_state(tokens)
    pn108.observe_state(state, THINK_START_LEN)
    det = state["pn108"]
    scored_once = det.scored_tokens
    pn108.observe_state(state, THINK_START_LEN)  # same output again
    assert det.scored_tokens == scored_once  # nothing re-fed


def test_gate_off_is_total_noop(monkeypatch):
    _fresh(monkeypatch, enabled=False)
    tokens = [0] * THINK_START_LEN + _loop_stream(4096)
    state = _mk_state(tokens)
    before = dict(state)
    pn108.observe_state(state, THINK_START_LEN)
    assert state == before  # no detector key, no mutation


def test_not_in_think_is_noop(monkeypatch):
    _fresh(monkeypatch)
    state = _mk_state(_loop_stream(4096))
    state["end_thinking"] = 100  # think already closed
    pn108.observe_state(state, THINK_START_LEN)
    assert "pn108" not in state


def test_second_think_block_gets_fresh_detector(monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.setenv("GENESIS_PN108_ARM_AFTER_TOKENS", "512")
    monkeypatch.setenv("GENESIS_PN108_WINDOW_TOKENS", "128")
    # think #1 at basis 0
    t1 = [0] * THINK_START_LEN + _novel_stream(1024)
    state = _mk_state(t1, start_thinking=0)
    pn108.observe_state(state, THINK_START_LEN)
    det1 = state["pn108"]
    assert det1.scored_tokens > 0
    # think #1 closes, think #2 opens later at a new absolute index
    state["output_tok_ids"] = t1 + [9] * 300 + [0] * THINK_START_LEN + _novel_stream(256)
    state["start_thinking"] = len(t1) + 300
    state["end_thinking"] = -1
    pn108.observe_state(state, THINK_START_LEN)
    det2 = state["pn108"]
    assert det2 is not det1  # fresh detector, not the silenced old one
    assert det2.scored_tokens > 0  # and it actually consumed think #2


def test_prompt_side_think_is_observed_bug107d(monkeypatch):
    """BUG-107d regression: template opens <think> in the PROMPT, so
    start_thinking is prompt-space (~600) while output is short. The old
    slice returned None forever -> detector structurally inert (proven
    in-container by the 2026-07-19 shadow window)."""
    _fresh(monkeypatch)
    monkeypatch.setenv("GENESIS_PN108_ARM_AFTER_TOKENS", "512")
    monkeypatch.setenv("GENESIS_PN108_WINDOW_TOKENS", "128")
    state = _mk_state(_novel_stream(1024), start_thinking=600)
    state["prompt_tok_ids"] = list(range(700))  # prompt-space world
    state["continue_thinking"] = True
    state["in_think"] = True
    pn108.observe_state(state, THINK_START_LEN)
    assert "pn108" in state, "detector never created (BUG-107d shape)"
    assert state["pn108"].scored_tokens > 0
    assert pn108.get_stats()["observed_requests"] == 1


def test_prompt_side_think_closed_is_noop(monkeypatch):
    _fresh(monkeypatch)
    state = _mk_state(_novel_stream(1024), start_thinking=600)
    state["continue_thinking"] = True
    state["in_think"] = False  # block closed
    pn108.observe_state(state, THINK_START_LEN)
    assert "pn108" not in state


def test_module_reload_is_clean():
    mod = importlib.reload(pn108)
    assert mod.get_stats()["fires"] == 0 and mod._STATE_KEY == "pn108"
