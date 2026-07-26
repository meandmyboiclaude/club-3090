#!/usr/bin/env python3
"""PN129 — stamp finish_reason + a zero-output flag on llm_request trace spans.

WHY
---
V1 tracing (vllm/v1/engine/output_processor.py::do_tracing) emits token counts
and latencies but NOT the request's finish reason, and `instrument_manual` has no
status parameter — so every span lands in the collector with status UNSET. A
request that returned zero tokens is therefore indistinguishable, in the trace
store, from one that answered normally: same span name, same UNSET status, only
`gen_ai.usage.completion_tokens: 0` to give it away.

That is the observability half of the output-loss class. BUG-127's aborts are
HTTP 200 with `finish_reason=abort` and an empty body; the runner-side counters
miss them and, without this patch, so does Phoenix. Measured on the live store
(2026-07-14..26, 31912 traced requests): 4911 spans (15.4%) carry
completion_tokens == 0, ALL of them with decode time exactly 0.0 — and not one
carries a reason, because the field was never recorded.

WHAT IT ADDS
------------
  - gen_ai.response.finish_reason — "stop" / "length" / "abort" / "error" /
    "repetition" (lowercased FinishReason name), or "none" if unfinished.
  - gen_ai.response.zero_output   — True ONLY when the request generated zero
    tokens. Cheap to filter on and impossible to confuse with a short answer.

Both are plain attributes, so no collector or backend change is needed. Cost is
two dict writes per traced request; the patch is inert unless
--otlp-traces-endpoint is set (do_tracing is behind tracing_enabled).

Anchors deliberately avoid the region PN99 rewrites, so the two patches compose
in either order.
"""
import pathlib
import sys

LOG = "[pn129-trace-finish-reason]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/output_processor.py"
)
MARKER = "# PN129:"

# Anchor on the tail of the attributes dict literal. PN99 inserts after the
# `req_state.n` block further down, so these two never touch the same bytes.
SITE_OLD = (
    "            SpanAttributes.GEN_AI_REQUEST_ID: req_state.external_req_id,\n"
    "        }\n"
)
SITE_NEW = (
    "            SpanAttributes.GEN_AI_REQUEST_ID: req_state.external_req_id,\n"
    "        }\n"
    "\n"
    "        # PN129: record WHY the request ended and flag zero-token results.\n"
    "        # Upstream records neither, and instrument_manual cannot set span\n"
    "        # status, so an aborted request is otherwise an UNSET 'success' span.\n"
    "        _pn129_fr = engine_core_output.finish_reason\n"
    '        attributes["gen_ai.response.finish_reason"] = (\n'
    '            _pn129_fr.name.lower() if _pn129_fr is not None else "none"\n'
    "        )\n"
    "        if not metrics.num_generation_tokens:\n"
    '            attributes["gen_ai.response.zero_output"] = True\n'
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"{LOG} already applied — skipping")
        return 0
    if "gen_ai.response.finish_reason" in src:
        print(f"{LOG} upstream now records finish_reason — patch self-retired, drop PN129")
        return 0

    n = src.count(SITE_OLD)
    if n == 0:
        print(
            f"{LOG} FATAL: anchor-not-found (site) — upstream drifted; "
            "re-derive anchors from the new output_processor.py",
            file=sys.stderr,
        )
        return 1
    if n > 1:
        print(f"{LOG} FATAL: ambiguous anchor (site, {n} hits)", file=sys.stderr)
        return 1

    src = src.replace(SITE_OLD, SITE_NEW, 1)
    TARGET.write_text(src, encoding="utf-8")
    import py_compile

    py_compile.compile(str(TARGET), doraise=True)
    print(f"{LOG} applied: gen_ai.response.finish_reason + zero_output on llm_request spans")
    return 0


sys.exit(main())
