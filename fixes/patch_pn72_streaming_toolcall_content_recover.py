"""PN72 — recover streaming content that contains literal `<tool_call>` markup.

Regression introduced by the vLLM Rust "Streaming Parser Engine" (#45413, present from
nightly b4c80ec0): the engine-based reasoning parser runs every streaming delta with
skip_tool_parsing=True and BUFFERS any tool-call-looking markup (`<tool_call>` …) it sees.
`AbstractParser._flush_engine_parsers` then DROPS that buffer whenever reasoning has ended,
to avoid leaking a stray `"` during a real tool call. But when there is NO real tool call —
e.g. the model's prose merely *mentions* `<tool_call>` (club-3090 #72) — the buffered chars
are legitimate content, and dropping them silences the stream from the tag onward.

PROVEN regression (old 1033ffac vs new b4c80ec0, identical prompt, streaming):
  old: full content (300/231 chars)   new: truncated at `<tool_call>` (25/5 chars)

Fix: narrow the skip condition. Only skip flushing the reasoning parser's buffer when a REAL
tool call occurred (`state.function_name_returned` — a function name was emitted). Otherwise
flush it, recovering the withheld content. Real tool calls keep the original no-leak behavior;
normal streams (no buffered markup) are unaffected (finish_streaming returns None).

Target: vllm/parser/abstract_parser.py  (AbstractParser._flush_engine_parsers).
Style: standalone commit-patch like the other /fixes; runs in the compose entrypoint after
apply_all. Idempotent; never exits non-zero (set -e safe); graceful no-op if the anchor is gone.
"""
import sys
import pathlib

LOG = "[pn72-stream-recover]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/parser/abstract_parser.py"
)
MARKER = "# PN72:"

OLD = (
    "            if parser is self._reasoning_parser and reasoning_ended:\n"
    "                continue\n"
)
NEW = (
    "            if (\n"
    "                parser is self._reasoning_parser\n"
    "                and reasoning_ended\n"
    "                and self._stream_state.function_name_returned\n"
    "            ):\n"
    "                # PN72: only skip the reasoning-buffer flush during a REAL\n"
    "                # tool call (a function name was emitted). When no tool call\n"
    "                # occurred, the buffered chars are legit content (e.g. prose\n"
    "                # mentioning '<tool_call>') the Rust engine withheld with\n"
    "                # skip_tool_parsing=True -> recover it instead of dropping it.\n"
    "                continue\n"
)


def main():
    if not TARGET.exists():
        print(f"{LOG} SKIP: {TARGET} not present on this vLLM build; no-op.", file=sys.stderr)
        return
    text = TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} already applied")
        return
    if OLD not in text:
        print(
            f"{LOG} SKIP: anchor not found in {TARGET.name} — _flush_engine_parsers shape "
            f"changed (vLLM bumped?); re-anchor before relying on this fix. No-op.",
            file=sys.stderr,
        )
        return
    TARGET.write_text(text.replace(OLD, NEW, 1))
    print(f"{LOG} applied: reasoning-buffer flush now recovers non-tool-call content")


main()
