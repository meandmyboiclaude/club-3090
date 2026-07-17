#!/usr/bin/env python3
"""PN99 — restore prompt/completion text on trace spans (V0 `gen_ai.prompt` parity).

Upstream V1 tracing (vllm/v1/engine/output_processor.py::do_tracing) emits
token counts + latency only; the V0 engine used to attach the prompt text as
`gen_ai.prompt`. That content is what makes per-trace debugging in Phoenix
useful (aibox: otelcol maps gen_ai.prompt/completion -> OpenInference
input.value/output.value so the Phoenix UI shows Input/Output panes again).

Adds, gated by VLLM_TRACE_CONTENT_MAX_CHARS (default 8192, 0 = off):
  - gen_ai.prompt      — req_state.prompt, middle-truncated
  - gen_ai.completion  — req_state.detokenizer.output_text, middle-truncated

Runs only when --otlp-traces-endpoint is set (do_tracing is behind
tracing_enabled), so zero cost on untraced deployments. Self-retires if
upstream reintroduces gen_ai.prompt in this file.
"""
import pathlib
import sys

LOG = "[pn99-trace-content]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/output_processor.py"
)
MARKER = "# PN99:"

IMPORT_OLD = "import asyncio\n"
IMPORT_NEW = "import asyncio\nimport os  # PN99: trace-content env knob\n"

HELPER_ANCHOR = (
    "# shared empty CPU tensor used as a placeholder pooling output\n"
    'EMPTY_CPU_TENSOR = torch.empty(0, device="cpu")\n'
)
HELPER_NEW = HELPER_ANCHOR + (
    "\n"
    "# PN99: max chars of prompt/completion text attached to trace spans;\n"
    "# 0 disables (= upstream behavior).\n"
    '_TRACE_CONTENT_MAX_CHARS = int(os.getenv("VLLM_TRACE_CONTENT_MAX_CHARS", "8192"))\n'
    "\n"
    "\n"
    "def _pn99_truncate_middle(text: str, max_chars: int) -> str:\n"
    "    if len(text) <= max_chars:\n"
    "        return text\n"
    "    half = max_chars // 2\n"
    "    return (\n"
    '        f"{text[:half]}\\n"\n'
    '        f"...[{len(text) - max_chars} chars truncated]...\\n"\n'
    '        f"{text[-half:]}"\n'
    "    )\n"
)

SITE_OLD = (
    "        if req_state.n:\n"
    "            attributes[SpanAttributes.GEN_AI_REQUEST_N] = req_state.n\n"
    "\n"
    "        instrument_manual(\n"
)
SITE_NEW = (
    "        if req_state.n:\n"
    "            attributes[SpanAttributes.GEN_AI_REQUEST_N] = req_state.n\n"
    "\n"
    "        # PN99: V0-parity prompt/completion capture (see /fixes header).\n"
    "        if _TRACE_CONTENT_MAX_CHARS > 0:\n"
    "            if req_state.prompt:\n"
    '                attributes["gen_ai.prompt"] = _pn99_truncate_middle(\n'
    "                    req_state.prompt, _TRACE_CONTENT_MAX_CHARS\n"
    "                )\n"
    "            if (\n"
    "                req_state.detokenizer is not None\n"
    "                and req_state.detokenizer.output_text\n"
    "            ):\n"
    '                attributes["gen_ai.completion"] = _pn99_truncate_middle(\n'
    "                    req_state.detokenizer.output_text, _TRACE_CONTENT_MAX_CHARS\n"
    "                )\n"
    "\n"
    "        instrument_manual(\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"{LOG} already applied — skipping")
        return 0
    if "gen_ai.prompt" in src:
        print(f"{LOG} upstream now sets gen_ai.prompt — patch self-retired, drop PN99")
        return 0
    for name, old in (("import", IMPORT_OLD), ("helper", HELPER_ANCHOR), ("site", SITE_OLD)):
        n = src.count(old)
        if n == 0:
            print(
                f"{LOG} FATAL: anchor-not-found ({name}) — upstream drifted; "
                "re-derive anchors from the new output_processor.py",
                file=sys.stderr,
            )
            return 1
        if n > 1:
            print(f"{LOG} FATAL: ambiguous anchor ({name}, {n} hits)", file=sys.stderr)
            return 1
    src = src.replace(IMPORT_OLD, IMPORT_NEW, 1)
    src = src.replace(HELPER_ANCHOR, HELPER_NEW, 1)
    src = src.replace(SITE_OLD, SITE_NEW, 1)
    TARGET.write_text(src, encoding="utf-8")
    import py_compile

    py_compile.compile(str(TARGET), doraise=True)
    print(f"{LOG} applied: gen_ai.prompt/gen_ai.completion on llm_request spans")
    return 0


sys.exit(main())
