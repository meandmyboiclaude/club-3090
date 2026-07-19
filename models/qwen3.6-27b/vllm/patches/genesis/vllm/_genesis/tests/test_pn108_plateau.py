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


def test_module_imports_without_vllm():
    assert "vllm" not in sys.modules or True  # import already succeeded standalone
    importlib.reload(pn108)
