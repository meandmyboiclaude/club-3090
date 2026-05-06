#!/usr/bin/env python3
"""
Local mitigation for the qwen3coder_tool_parser SSE-silence bug.

Bug: vllm/tool_parsers/qwen3coder_tool_parser.py extract_tool_calls_streaming
flips is_tool_call_started=True on either special-token-id match OR string
match against the literal `<tool_call>` characters. Both paths mis-fire when
the model emits `<tool_call>` in narrative output -- as the special token
when the BPE tokenizer recognizes the tag, and as the string when the tag
appears in markdown / prose contexts. The flip is sticky; subsequent
deltas all return None and the SSE wire goes silent until max_tokens.

Local fix: defer commit to is_tool_call_started=True until `<function=`
appears in the slack window after `<tool_call>`. Real tool calls in the
qwen3coder format have `<tool_call>\\n<function=...` adjacency; if no
`<function=` arrives within 64 chars past the tag, treat the `<tool_call>`
mention as benign content and continue streaming. Both trigger paths
(token-id and string) go through the same deferred check.

Run AFTER `python3 -m vllm._genesis.patches.apply_all` so this lands on top
of any Genesis patch that touches qwen3coder_tool_parser.py.

Filed:
  - club-3090 issue:  https://github.com/noonghunna/club-3090/issues/72
  - upstream filing:  https://github.com/vllm-project/vllm/issues  (filed 2026-05-07)

Originally proposed by @troymroberts (issue #72) as P61c. Re-named to
function-descriptive form (this is a club-3090 sidecar, not Genesis-blessed).

Idempotent. Aborts loudly if the target pattern isn't found.
"""
import sys
import pathlib

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/qwen3coder_tool_parser.py"
)
SENTINEL_V1 = "P61C-V1-TOKEN-ID-ONLY"
SENTINEL_V2 = "P61C-V2-DEFER-UNTIL-FUNCTION"
LOG_PREFIX = "[qwen3coder-tool-parser-defer]"

ORIG = """            if (
                self.tool_call_start_token_id in delta_token_ids
                or self.tool_call_start_token in delta_text
            ):
                self.is_tool_call_started = True
                # Return any content before the tool call
                if self.tool_call_start_token in delta_text:
                    content_before = delta_text[
                        : delta_text.index(self.tool_call_start_token)
                    ]
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None"""

# V1 patch (token-id only) that we may need to revert if present.
V1_PATCHED = """            # P61C-V1-TOKEN-ID-ONLY: removed string-match commit trigger to fix
            # SSE silence on prose containing literal `<tool_call>` text.
            # Genesis P61c proposed upstream; pending fix:
            #   - vllm-project/vllm  (upstream)
            #   - noonghunna/club-3090  (Genesis-side workaround)
            if self.tool_call_start_token_id in delta_token_ids:
                self.is_tool_call_started = True
                # Return any content before the tool call
                if self.tool_call_start_token in delta_text:
                    content_before = delta_text[
                        : delta_text.index(self.tool_call_start_token)
                    ]
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None"""

NEW = f"""            # {SENTINEL_V2}: defer commit until <function= header follows the
            # <tool_call> token within a 64-char slack window. Guards against
            # the model emitting <tool_call> (as special token OR string) in
            # narrative reasoning without an actual tool-call header -- a
            # case that flips is_tool_call_started=True permanently and
            # silently drops all subsequent content via the serving layer's
            # `if delta_message is None: continue` path. Filed at:
            #   - vllm-project/vllm  (upstream bug filing)
            #   - noonghunna/club-3090 issue #72  (this club-3090 sidecar)
            if (
                self.tool_call_start_token_id in delta_token_ids
                or self.tool_call_start_token in delta_text
            ):
                _tc_idx = current_text.find(self.tool_call_start_token)
                if _tc_idx == -1:
                    # Token id present but tag string not in accumulated text:
                    # tokenizer edge case. Conservative: emit delta as content,
                    # don't commit.
                    return DeltaMessage(content=delta_text or None)
                _slack_end = _tc_idx + len(self.tool_call_start_token) + 64
                if "<function=" in current_text[_tc_idx:_slack_end]:
                    # Real tool call confirmed. Original commit path.
                    self.is_tool_call_started = True
                    if self.tool_call_start_token in delta_text:
                        content_before = delta_text[
                            : delta_text.index(self.tool_call_start_token)
                        ]
                        if content_before:
                            return DeltaMessage(content=content_before)
                    return None
                # No <function= header yet. Whether slack has expired or not,
                # emit this delta as content -- never silently drop chunks
                # while uncertain. If <function= eventually arrives within
                # slack, we'll commit then; if not, we've correctly streamed
                # the prose.
                return DeltaMessage(content=delta_text or None)"""


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG_PREFIX} ERROR: {TARGET} not found", file=sys.stderr)
        return 1
    src = TARGET.read_text()
    if SENTINEL_V2 in src:
        print(f"{LOG_PREFIX} already applied (V2 sentinel present), skipping")
        return 0
    if V1_PATCHED in src:
        # Patched by V1; revert that block first so V2 can apply.
        src = src.replace(V1_PATCHED, ORIG, 1)
        print(f"{LOG_PREFIX} reverted V1 (token-id only) patch")
    if ORIG not in src:
        print(
            f"{LOG_PREFIX} ERROR: target pattern not found in {TARGET}.\n"
            f"  vLLM may have been bumped; review qwen3coder_tool_parser.py "
            f"before re-applying.",
            file=sys.stderr,
        )
        return 1
    new_src = src.replace(ORIG, NEW, 1)
    TARGET.write_text(new_src)
    print(
        f"{LOG_PREFIX} applied: deferred-commit guard in "
        f"{TARGET.name} (commit only when <function= follows <tool_call> "
        f"within 64 chars)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
