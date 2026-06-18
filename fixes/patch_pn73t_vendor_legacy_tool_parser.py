"""PN73T — vendor the legacy (non-engine) qwen3_coder TOOL parser onto the new base.

Companion to PN73 (reasoning parser). The Rust engine tool parser also buffers/drops
`<tool_call>` content during streaming (the `tools=on` half of the regression). Repoint
`--tool-call-parser qwen3_coder` at the OLD non-engine Python parser (engine_based_streaming
=False -> legacy streaming) with the club-3090 #72 deferred-commit fix baked in.

Registration maps "qwen3_coder" -> module `qwen3_engine_tool_parser`, class `Qwen3EngineToolParser`.
We overwrite that module with the vendored legacy parser (aliased Qwen3EngineToolParser =
Qwen3CoderToolParser). Source: /fixes/vendored_qwen3coder_tool_parser.py (1033ffac + deferred-commit).

Idempotent; never exits non-zero (set -e safe); no-op if source/target absent.
"""
import sys
import pathlib

LOG = "[pn73t-vendor-tool]"
SRC = pathlib.Path("/fixes/vendored_qwen3coder_tool_parser.py")
DST = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/qwen3_engine_tool_parser.py"
)
MARKER = "# [PN73T]"


def main():
    if not SRC.exists():
        print(f"{LOG} SKIP: vendored source {SRC} missing; no-op.", file=sys.stderr)
        return
    if not DST.parent.exists():
        print(f"{LOG} SKIP: target dir {DST.parent} absent; no-op.", file=sys.stderr)
        return
    if DST.exists() and MARKER in DST.read_text():
        print(f"{LOG} already applied")
        return
    DST.write_text(SRC.read_text())
    print(f"{LOG} applied: 'qwen3_coder' tool parser repointed to legacy non-engine parser "
          f"(+ deferred-commit fix; legacy streaming, no <tool_call> content-drop)")


main()
