# In-container pytest for patch_pn135_streaming_think_end.py
# (AUDIT-leak-paths-20260726.md class L2; upstream vllm#39697 class).
#
#   sudo podman run --rm \
#     -v /home/user/club-3090/fixes:/fixes:ro \
#     --entrypoint /bin/bash localhost/vllm-qwen36-endgame:<tag> -c \
#     'python3 /fixes/patch_pn135_streaming_think_end.py && \
#      python3 -m pytest --noconftest -q /fixes/test_pn135_streaming_think_end.py'
#
# Drop the patch invocation to see the pre-fix failures (T1/T2 red).
#
# T1/T2 are the leak; T3-T7 are the regression guards for the branch the
# strict-demotion was written for (unbacked <tool_call> TEXT stays content).
import pytest

from unittest.mock import MagicMock

from vllm.parser.qwen3 import Qwen3Parser

THINK_START_ID = 50
THINK_END_ID = 51
TOOL_CALL_ID = 60
TOOL_CALL_END_ID = 61

VOCAB = {
    "<think>": THINK_START_ID,
    "</think>": THINK_END_ID,
    "<tool_call>": TOOL_CALL_ID,
    "</tool_call>": TOOL_CALL_END_ID,
}

# Ordinary (non-special) BPE ids — the "spelled as text" case.
TEXT_IDS = (7001, 7002, 7003)


def _mock_tokenizer():
    id_to_text = {v: k for k, v in VOCAB.items()}
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.get_vocab.return_value = dict(VOCAB)
    tok.decode.side_effect = lambda ids: "".join(
        id_to_text.get(i, chr(i) if i < 128 else f"<{i}>") for i in ids
    )
    tok.all_special_tokens = list(VOCAB.keys())
    tok.all_special_ids = list(VOCAB.values())
    return tok


@pytest.fixture
def parser():
    return Qwen3Parser(_mock_tokenizer())


def stream(parser, chunks):
    """Drive extract_reasoning_streaming; returns (reasoning, content).

    Each chunk is (text, delta_token_ids). Passing non-empty ids is what
    streaming serving always does (chat_completion/serving.py), which is
    what flips the engine into strict token-id mode.
    """
    reasoning_parts, content_parts = [], []
    prev_text, prev_ids = "", []
    for text, ids in chunks:
        cur_text = prev_text + text
        cur_ids = prev_ids + list(ids)
        delta = parser.extract_reasoning_streaming(
            previous_text=prev_text,
            current_text=cur_text,
            delta_text=text,
            previous_token_ids=tuple(prev_ids),
            current_token_ids=tuple(cur_ids),
            delta_token_ids=tuple(ids),
        )
        if delta:
            if delta.reasoning:
                reasoning_parts.append(delta.reasoning)
            if delta.content:
                content_parts.append(delta.content)
        prev_text, prev_ids = cur_text, list(cur_ids)
    return "".join(reasoning_parts), "".join(content_parts)


# ── T1: the leak — text-spelled </think> must terminate reasoning ──────────
def test_text_spelled_think_end_terminates_reasoning(parser):
    reasoning, content = stream(
        parser,
        [
            ("Let me analyze.", TEXT_IDS),
            ("</think>", TEXT_IDS),  # no THINK_END_ID: spelled as BPE text
            ("The answer is 42.", TEXT_IDS),
        ],
    )
    assert "</think>" not in reasoning, "marker left inside reasoning_content"
    assert "</think>" not in content, "raw </think> leaked to client content"
    assert reasoning == "Let me analyze."
    # Pre-fix this is "" — the engine never leaves REASONING and the whole
    # answer is stranded in reasoning_content.
    assert content == "The answer is 42."


# ── T2: duplicate </think> in CONTENT must be absorbed, not emitted ────────
def test_text_spelled_duplicate_think_end_absorbed(parser):
    reasoning, content = stream(
        parser,
        [
            ("thinking", TEXT_IDS),
            ("</think>", (THINK_END_ID,)),  # real special token: ends reasoning
            ("Answer.", TEXT_IDS),
            ("</think>", TEXT_IDS),  # text-spelled duplicate while in CONTENT
            (" More.", TEXT_IDS),
        ],
    )
    assert reasoning == "thinking"
    assert "</think>" not in content, "raw </think> leaked to client content"
    assert content == "Answer. More."


# ── T3: regression guard — unbacked <tool_call> TEXT stays content ─────────
def test_text_spelled_tool_call_still_demoted_to_content(parser):
    _, content = stream(
        parser,
        [
            ("r", TEXT_IDS),
            ("</think>", (THINK_END_ID,)),
            ("prose about <tool_call> tags", TEXT_IDS),
        ],
    )
    assert "<tool_call>" in content, (
        "strict token-id demotion for tool tags was weakened — an unbacked "
        "<tool_call> text fragment must NOT open a tool call"
    )


# ── T4: regression guard — text-spelled <think> is not a terminal either ──
def test_text_spelled_think_start_still_demoted(parser):
    reasoning, content = stream(
        parser,
        [
            ("<think>", TEXT_IDS),
            ("body", TEXT_IDS),
            ("</think>", (THINK_END_ID,)),
            ("out", TEXT_IDS),
        ],
    )
    assert reasoning == "<think>body"
    assert content == "out"


# ── T5: regression guard — real token-id path unchanged ───────────────────
def test_token_id_think_end_unchanged(parser):
    reasoning, content = stream(
        parser,
        [
            ("<think>", (THINK_START_ID,)),
            ("reason", TEXT_IDS),
            ("</think>", (THINK_END_ID,)),
            ("answer", TEXT_IDS),
        ],
    )
    assert reasoning == "reason"
    assert content == "answer"


# ── T6: regression guard — non-streaming path unchanged ───────────────────
def test_non_streaming_unchanged(parser):
    reasoning, content = parser.extract_reasoning(
        "<think>Let me analyze.</think>The answer is 42.", None
    )
    assert reasoning == "Let me analyze."
    assert content == "The answer is 42."


# ── T7: regression guard — no </think> at all means all-reasoning ─────────
def test_streaming_without_think_end_is_all_reasoning(parser):
    reasoning, content = stream(
        parser,
        [("truncated output", TEXT_IDS)],
    )
    assert reasoning == "truncated output"
    assert content == ""
