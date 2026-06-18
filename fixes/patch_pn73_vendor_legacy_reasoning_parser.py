"""PN73 — vendor the legacy (non-engine) Qwen3 reasoning parser onto the new base.

The vLLM Rust "Streaming Parser Engine" (#45413, b4c80ec0) regressed streaming: the
engine-based reasoning parser buffers `<tool_call>` markup it sees in content (with
skip_tool_parsing=True) and the content is dropped — proven old=300 vs new=25 chars.
The Python flush layer can't recover it (loss is inside the Rust engine).

Fix (the TQ3-style way — patch, not rebuild): repoint `--reasoning-parser qwen3` at the
OLD non-engine Python parser, which sets engine_based_streaming=False -> the legacy
streaming path runs -> no Rust buffering -> `<tool_call>` content streams normally.
The old parser is interface-compatible with the new BaseThinkingReasoningParser (verified:
imports clean, all required methods present).

How: the lazy registration maps "qwen3" -> module `qwen3_engine_reasoning_parser`, class
`Qwen3ParserReasoningAdapter`. We overwrite that module with the vendored legacy parser
(which aliases Qwen3ParserReasoningAdapter = Qwen3ReasoningParser). Source of the vendored
file: /fixes/vendored_qwen3_reasoning_parser.py (extracted from the 1033ffac image).

Idempotent; never exits non-zero (set -e safe); no-op if the source/target is absent.
"""
import sys
import pathlib

LOG = "[pn73-vendor-reasoning]"
SRC = pathlib.Path("/fixes/vendored_qwen3_reasoning_parser.py")
DST = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/reasoning/qwen3_engine_reasoning_parser.py"
)
MARKER = "# [PN73]"


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
    print(f"{LOG} applied: 'qwen3' reasoning parser repointed to legacy non-engine parser "
          f"(engine_based_streaming=False -> legacy streaming, no <tool_call> content-drop)")


main()
