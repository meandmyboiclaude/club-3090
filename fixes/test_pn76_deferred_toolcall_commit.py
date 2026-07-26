#!/usr/bin/env python3
"""PN76 — deferred tool-call commit: the two regressions it shipped with.

PN76 defers opening a tool call until a FUNC_PREFIX terminal confirms the
`<tool_call>` was real, so prose that merely MENTIONS the tag stops truncating
content and emitting a phantom call. That part works. It shipped with no test
of its own, and upstream's replay suite — which does cover it — was red:
85 failed / 178 errors against 3230 passed on a stock engine.

Both regressions are here, isolated by rebuilding the parser tree per arm:

  R1  THE HOLD COULD NEVER BE CONFIRMED FOR EIGHT PARSERS.
      FUNC_PREFIX is declared by qwen3 ALONE. deepseek_v4, deepseek_v32,
      gemma4, glm47_moe, inkling, kimi_k2, minimax_m2 and seed_oss all declare
      TOOL_START without it, so an unconditional hold aborted EVERY real tool
      call into content — upstream reported it as "Tool call count mismatch:
      expected 1, got 0" at collection time.

  R2  IT CHANGED THE CLOSED EMPTY BLOCK IT WAS NEVER MEANT TO TOUCH.
      Upstream #46091's (TOOL_PREAMBLE, TOOL_END) transition absorbs
      `<tool_call></tool_call>`. PN76's hold intercepted that terminal and
      re-emitted the literal, so content came back as
      '<tool_call></tool_call>Content after empty tools.' instead of
      'Content after empty tools.'. PN76's own docstring says #46091 already
      covers the closed case and PN76 targets the UNCLOSED one.

Run in-container, patch first (as with every /fixes suite):

    python3 /fixes/patch_pn76_engine_deferred_toolcall_commit.py && \
    python3 -m pytest --noconftest -q /fixes/test_pn76_deferred_toolcall_commit.py
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

PARSER_PKG = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/parser"
)


def _pn76_applied() -> bool:
    import inspect  # noqa: PLC0415

    from vllm.parser.engine import streaming_parser_engine as _spe  # noqa: PLC0415

    try:
        return "_tc_can_confirm" in inspect.getsource(_spe)
    except OSError:
        return False


PN76_APPLIED = _pn76_applied()
needs_patch = pytest.mark.skipif(
    not PN76_APPLIED,
    reason=(
        "PN76's confirmability guard is not present in this engine. Run "
        "'python3 /fixes/patch_pn76_engine_deferred_toolcall_commit.py' first."
    ),
)


def _engine_cls():
    from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine

    return StreamingParserEngine


def _stub(transition_terminals):
    """Duck-typed engine exposing only what _tc_can_confirm reads."""
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            transitions={("CONTENT", t): object() for t in transition_terminals}
        )
    )


# ── R1: the confirmability guard ──────────────────────────────────────────
@needs_patch
def test_config_with_func_prefix_defers():
    fn = _engine_cls()._tc_can_confirm
    assert fn(_stub(["TOOL_START", "FUNC_PREFIX", "TOOL_END"])) is True


@needs_patch
def test_config_without_func_prefix_does_not_defer():
    """The eight non-qwen3 parsers. Without this, every real tool call they
    parse is aborted into content, silently."""
    fn = _engine_cls()._tc_can_confirm
    assert fn(_stub(["TOOL_START", "TOOL_END"])) is False


@needs_patch
def test_confirmability_is_memoised():
    fn = _engine_cls()._tc_can_confirm
    obj = _stub(["TOOL_START", "FUNC_PREFIX"])
    assert fn(obj) is True
    # Emptying the config must not change the answer: it is computed once.
    obj.config.transitions = {}
    assert fn(obj) is True
    assert obj._tc_confirmable is True


def test_only_qwen3_declares_func_prefix():
    """The ecosystem fact the guard rests on. If a future parser gains
    FUNC_PREFIX this test tells you to re-check the guard rather than letting
    the assumption rot silently."""
    if not PARSER_PKG.is_dir():
        pytest.skip(f"parser package not at {PARSER_PKG}")
    declaring = sorted(
        p.name
        for p in PARSER_PKG.glob("*.py")
        if "FUNC_PREFIX" in p.read_text(encoding="utf-8", errors="replace")
    )
    assert declaring == ["qwen3.py"], (
        f"FUNC_PREFIX is no longer qwen3-only: {declaring}. PN76's hold is "
        "released by that terminal, so re-check _tc_can_confirm."
    )


def test_parsers_with_tool_start_but_no_func_prefix_are_the_known_eight():
    if not PARSER_PKG.is_dir():
        pytest.skip(f"parser package not at {PARSER_PKG}")
    affected = sorted(
        p.stem
        for p in PARSER_PKG.glob("*.py")
        if "TOOL_START" in (txt := p.read_text(encoding="utf-8", errors="replace"))
        and "FUNC_PREFIX" not in txt
    )
    assert affected == [
        "deepseek_v32", "deepseek_v4", "gemma4", "glm47_moe",
        "inkling", "kimi_k2", "minimax_m2", "seed_oss",
    ], f"the set PN76 must not defer for has changed: {affected}"


# ── R2: the closed empty block ────────────────────────────────────────────
@needs_patch
def test_closed_empty_tool_block_is_absorbed():
    """`<tool_call></tool_call>` is upstream #46091's case, not PN76's.
    Re-emitting the literal is what upstream's replay suite caught."""
    import inspect

    from vllm.parser.engine import streaming_parser_engine as spe

    src = inspect.getsource(spe)
    assert 'if terminal == "TOOL_END":' in src, (
        "the TOOL_END absorb branch is gone — a closed empty block will be "
        "re-emitted as a literal <tool_call> into client content again"
    )
    # The absorb must happen while holding, i.e. before the generic flush.
    hold_idx = src.index("if self._tc_hold:")
    absorb_idx = src.index('if terminal == "TOOL_END":')
    flush_idx = src.index("flushed = self._tc_flush_hold()")
    assert hold_idx < absorb_idx < flush_idx, (
        "TOOL_END absorb is not inside the hold branch ahead of the flush"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "--noconftest", "-p", "no:cacheprovider"]))
