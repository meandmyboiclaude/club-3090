# SPDX-License-Identifier: Apache-2.0
"""PN100 total-completion ceiling — safety + binding proof (2026-07-26).

Drives `_apply_total_ceiling` directly. Proves it (a) bounds total completion
on the relocation grinders, (b) leaves a natural answer untouched, (c) is
default-OFF, (d) cannot cut inside the think block, (e) is not inert against
an OpenAI-modern caller that sends `max_completion_tokens`.

Source-level only: no boot, no GPU, no container restart. The module is loaded
by path with the one vllm import it needs stubbed, so it runs anywhere.

    python -m pytest fixes/test_pn100_total_ceiling.py -q
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

_AB = (Path(__file__).resolve().parents[1] / "models/qwen3.6-27b/vllm/patches"
       / "genesis/vllm/_genesis/middleware/auto_budget.py")


def _load():
    for name in ("vllm", "vllm._genesis", "vllm._genesis.middleware",
                 "vllm._genesis.middleware.lazy_reasoner"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["vllm._genesis.middleware.lazy_reasoner"]._extract_text_from_message = (
        lambda m: ""
    )
    spec = importlib.util.spec_from_file_location("_ab_under_test", _AB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ab = _load()


class Req:
    """Minimal stand-in for vLLM's ChatCompletionRequest."""

    def __init__(self, max_tokens=None, max_completion_tokens=None):
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.chat_template_kwargs = None
        self.thinking_token_budget = None

    @property
    def effective_cap(self):
        caps = [v for v in (self.max_completion_tokens, self.max_tokens)
                if isinstance(v, int)]
        return min(caps) if caps else None


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("GENESIS_PN100_TOTAL_CEIL_F", "GENESIS_PN100_TOTAL_CEIL_SLACK"):
        monkeypatch.delenv(k, raising=False)


def _on(monkeypatch, f="1.0", slack="3072"):
    monkeypatch.setenv("GENESIS_PN100_TOTAL_CEIL_F", f)
    monkeypatch.setenv("GENESIS_PN100_TOTAL_CEIL_SLACK", slack)


# ── (c) default-OFF ────────────────────────────────────────────────────────
def test_default_off_leaves_request_untouched():
    r = Req()
    assert ab._apply_total_ceiling(r, 3100) is None
    assert r.max_tokens is None and r.max_completion_tokens is None


# ── (a) bounds the grinders ────────────────────────────────────────────────
@pytest.mark.parametrize(
    # (cap, ctok) — the real clean-100 rows the fitted F=1.0/slack=3072 binds.
    "budget,ctok",
    [(3900, 8174),   # gpqa-094: 3895 think + 4279 relocated answer
     (3100, 6843)],  # gpqa-142: 3095 think + 3748 relocated answer
)
def test_relocation_grinder_is_bounded(monkeypatch, budget, ctok):
    _on(monkeypatch)
    r = Req()
    ceil = ab._apply_total_ceiling(r, budget)
    assert ceil == budget + 3072
    assert r.effective_cap == ceil
    assert ceil < ctok, "ceiling must actually bind this row"


# ── (b) a natural answer is NOT truncated ──────────────────────────────────
@pytest.mark.parametrize(
    "budget,rtok,atok,label",
    [(3100, 135, 87, "gpqa-058 natural, tiny answer"),
     (3900, 1919, 1311, "GPQA natural-stop answer MAX (p99)"),
     (1300, 1131, 2740, "prod_mixed_v2 natural-stop answer MAX"),
     (800, 787, 2173, "prod_mixed_v2 answer p99 on a small grant")],
)
def test_natural_answer_survives(monkeypatch, budget, rtok, atok, label):
    _on(monkeypatch)
    r = Req()
    ceil = ab._apply_total_ceiling(r, budget)
    assert rtok + atok <= ceil, f"{label}: ceiling {ceil} would truncate"


# ── (d) never cuts inside the think block ──────────────────────────────────
@pytest.mark.parametrize("f,slack", [("0.25", "0"), ("0.5", "1"), ("1.0", "0")])
def test_ceiling_never_lands_below_the_think_budget(monkeypatch, f, slack):
    _on(monkeypatch, f=f, slack=slack)
    r = Req()
    ceil = ab._apply_total_ceiling(r, 3100)
    assert ceil >= 3100 + ab._TOTAL_CEIL_MIN_SLACK


# ── (e) not inert against an OpenAI-modern caller ──────────────────────────
def test_max_completion_tokens_caller_is_bounded(monkeypatch):
    """The pre-2026-07-26 code wrote max_tokens only; vLLM prefers
    max_completion_tokens, so the ceiling did nothing for this caller."""
    _on(monkeypatch)
    r = Req(max_completion_tokens=32768)
    ceil = ab._apply_total_ceiling(r, 1300)
    assert ceil == 1300 + 3072
    assert r.max_completion_tokens == ceil


# ── only ever lowers ───────────────────────────────────────────────────────
def test_tighter_caller_cap_is_never_raised(monkeypatch):
    _on(monkeypatch)
    r = Req(max_tokens=512)
    assert ab._apply_total_ceiling(r, 1300) is None
    assert r.max_tokens == 512
    assert r.max_completion_tokens is None


def test_stat_counter_increments(monkeypatch):
    _on(monkeypatch)
    before = ab.get_stats()["total_ceiling"]
    ab._apply_total_ceiling(Req(), 3100)
    assert ab.get_stats()["total_ceiling"] == before + 1


# ── F=2.0 is inert against current traffic (the refit's headline) ──────────
def test_f2_is_inert_on_clean100(monkeypatch):
    """Worst clean-100 row by ctok is gpqa-094 at 8174 (cap 3900)."""
    _on(monkeypatch, f="2.0", slack="1024")
    r = Req()
    assert ab._apply_total_ceiling(r, 3900) == 2 * 3900 + 1024 > 8174


def test_apply_budget_wires_the_ceiling(monkeypatch):
    _on(monkeypatch)
    r = Req()
    ab._apply_budget(r, 3100)
    assert r.effective_cap == 3100 + 3072


def test_apply_tier_wires_the_ceiling(monkeypatch):
    """The tier path had no ceiling at all before 2026-07-26."""
    _on(monkeypatch)
    r = Req()
    ab._apply_tier(r, 2, allow_disable=False)   # tier 2 -> 4096
    assert r.effective_cap == 4096 + 3072


def test_tier0_thinking_off_gets_no_ceiling(monkeypatch):
    _on(monkeypatch)
    r = Req()
    assert ab._apply_tier(r, 0, allow_disable=True) == 0
    assert r.max_tokens is None and r.max_completion_tokens is None
