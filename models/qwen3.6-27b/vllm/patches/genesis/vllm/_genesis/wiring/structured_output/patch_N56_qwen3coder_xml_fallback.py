# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN56 — vllm#41466 backport: qwen3coder XML parse fallback.

Backport of upstream PR #41466 (ToastyTheBot, OPEN). When
`_parse_xml_function_call` throws or returns None inside
`extract_tool_calls_streaming`, the original code leaves
`prev_tool_call_arr[i]["arguments"]` with the placeholder `"{}"`
from the header-sent step. Serving layer's remaining-args check
later double-emits `{"arguments":"{}"}`, breaking strict OpenAI
clients (Vercel AI SDK, OpenAI Node SDK).

Composes with our existing P64 (vllm#39598) — P64 changed the
post-`except` flow but did NOT modify the try block where PN56
inserts. Anchor stable on both pristine and post-P64 file states.

Sub-patches (2):
  A — insert `parse_succeeded = False/True` flags around the try
  B — append fallback block: when parse_succeeded=False, restore
      `prev_tool_call_arr[i]["arguments"]` from
      `streamed_args_for_tool[i] + "}"` (matches what the serving
      layer's remainder check will compute).

Affects ALL Genesis configs that use qwen3_coder tool parser:
- 27B Lorbus + MTP K=3 + tools (PROD)
- 27B FP8 short/long + tools
- 35B FP8 + DFlash/MTP + tools

Default OFF — defensive backport. Risk: low (extra branch per parse,
data-only fallback). Enable after live verify against tool-call sweep.

dev1060 SELF-RETIRE (2026-07-13)
--------------------------------
dev1060 REMOVED ``tool_parsers/qwen3coder_tool_parser.py``. The
qwen3_coder parser is now an 8-line shim
(``tool_parsers/qwen3_engine_tool_parser.py`` →
``Qwen3ParserToolAdapter`` → ``vllm/parser/qwen3.py:Qwen3Parser`` →
``vllm/parser/engine/{parser_engine,streaming_parser_engine,
incremental_lexer,token_id_scanner}.py``) — pure Python end to end, no
compiled/Rust leg in this chain.

The {}-placeholder double-emit this patch guarded against is
STRUCTURALLY IMPOSSIBLE on dev1060 (verified against
/home/user/engines/vllm-build, exact installed source):

  1. No placeholder exists: the tool-name delta carries only
     ``DeltaFunctionCall(name=...)`` (``parser_engine.py``
     ``_emit_name_delta``); the tool-end fallback emits
     ``arguments=remaining or ""`` — never ``"{}"``.
  2. The serving-layer remainder re-emit path is GONE:
     ``entrypoints/openai/chat_completion/serving.py`` has zero
     references to ``prev_tool_call_arr`` / ``streamed_args_for_tool``;
     streaming deltas flow through ``parser.parse_delta`` and the
     engine's own ``finish()``.
  3. A failed/partial arg parse returns ``None`` deltas
     (``_compute_arg_delta`` / ``_flush_arg_converter`` catch
     ``JSONDecodeError`` → ``None``) and streamed args obey a strict
     ``startswith(prev)`` prefix invariant — nothing is re-emitted.

On dev1060 layouts ``apply()`` therefore self-retires ("upstream
parser engine absorbed") instead of the ambiguous "target not found"
skip. Historical anchors below are kept verbatim for pre-dev1060 pins.

Author: Sandermage backport (ToastyTheBot, vllm#41466).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn56_qwen3coder_xml_fallback")

GENESIS_PN56_MARKER = "Genesis PN56 qwen3coder XML parse fallback (vllm#41466)"

# [dev1060 drift marker 2026-07-13] Surfaced in the skip reason (and log)
# when the dev1060 parser-engine layout is detected, so audit/shadow
# tooling can distinguish "retired by design" from "target went missing".
GENESIS_PN56_DEV1060_RETIRE_MARKER = (
    "PN56 dev1060 self-retire: upstream parser engine absorbed"
)


def _dev1060_parser_engine_present() -> bool:
    """dev1060 layout probe (2026-07-13).

    True iff the install carries the new parser-engine chain that
    replaced qwen3coder_tool_parser.py. Either file is sufficient
    evidence; both are checked so a partial layout still retires
    cleanly.
    """
    return (
        resolve_vllm_file("tool_parsers/qwen3_engine_tool_parser.py") is not None
        or resolve_vllm_file("parser/engine/parser_engine.py") is not None
    )


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Sub-A: wrap try with parse_succeeded flag
ANCHOR_A_OLD = (
    "                if func_content_end != -1:\n"
    "                    func_content = tool_text[func_start:func_content_end]\n"
    "                    try:\n"
    "                        parsed_tool = self._parse_xml_function_call(\n"
    "                            func_content,\n"
    "                        )\n"
    "                        if parsed_tool and self.current_tool_index < len(\n"
    "                            self.prev_tool_call_arr\n"
    "                        ):\n"
    "                            self.prev_tool_call_arr[self.current_tool_index][\n"
    "                                \"arguments\"\n"
    "                            ] = parsed_tool.function.arguments\n"
    "                    except Exception:\n"
    "                        logger.debug(\n"
    "                            \"Failed to parse tool call during streaming: %s\",\n"
    "                            tool_text,\n"
    "                            exc_info=True,\n"
    "                        )"
)

ANCHOR_A_NEW = (
    "                if func_content_end != -1:\n"
    "                    func_content = tool_text[func_start:func_content_end]\n"
    "                    # [Genesis PN56 vllm#41466] Track parse success to know\n"
    "                    # if fallback below should fire (else \"{}\" placeholder leaks).\n"
    "                    _pn56_parse_succeeded = False\n"
    "                    try:\n"
    "                        parsed_tool = self._parse_xml_function_call(\n"
    "                            func_content,\n"
    "                        )\n"
    "                        if parsed_tool and self.current_tool_index < len(\n"
    "                            self.prev_tool_call_arr\n"
    "                        ):\n"
    "                            self.prev_tool_call_arr[self.current_tool_index][\n"
    "                                \"arguments\"\n"
    "                            ] = parsed_tool.function.arguments\n"
    "                            _pn56_parse_succeeded = True\n"
    "                    except Exception:\n"
    "                        logger.debug(\n"
    "                            \"Failed to parse tool call during streaming: %s\",\n"
    "                            tool_text,\n"
    "                            exc_info=True,\n"
    "                        )\n"
    "                    # [Genesis PN56 vllm#41466] When parse failed, prev_tool_call_arr\n"
    "                    # still has \"{}\" placeholder. Restore from incrementally\n"
    "                    # streamed args + closing brace so serving layer remainder\n"
    "                    # check produces correct output instead of double-emit \"{}\".\n"
    "                    if (\n"
    "                        not _pn56_parse_succeeded\n"
    "                        and self.current_tool_index < len(self.prev_tool_call_arr)\n"
    "                        and self.current_tool_index < len(self.streamed_args_for_tool)\n"
    "                    ):\n"
    "                        # [Audit A-14 fix 2026-05-05] Guard against double `}}`\n"
    "                        # if streamed_args already ends with closing brace\n"
    "                        # (P64 may have written it, or a prior partial close).\n"
    "                        _pn56_streamed = self.streamed_args_for_tool[\n"
    "                            self.current_tool_index\n"
    "                        ]\n"
    "                        _pn56_suffix = \"\" if _pn56_streamed.rstrip().endswith(\"}\") else \"}\"\n"
    "                        self.prev_tool_call_arr[self.current_tool_index][\n"
    "                            \"arguments\"\n"
    "                        ] = _pn56_streamed + _pn56_suffix"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("tool_parsers/qwen3coder_tool_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN56 qwen3coder XML fallback (vllm#41466)",
        target_file=str(target),
        marker=GENESIS_PN56_MARKER,
        sub_patches=[TextPatch(
            name="pn56_xml_fallback",
            anchor=ANCHOR_A_OLD,
            replacement=ANCHOR_A_NEW,
            required=True,
        )],
        upstream_drift_markers=[
            "_pn56_parse_succeeded",
            "parse_succeeded = False",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN56")
    log_decision("PN56", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        # [dev1060 self-retire 2026-07-13] Old target removed AND the new
        # parser-engine chain is present → the bug this patch fixed is
        # structurally impossible (see module docstring, "dev1060
        # SELF-RETIRE"). Retire loudly so the skip is auditable as
        # by-design, not drift.
        if _dev1060_parser_engine_present():
            log.info(
                "[PN56] %s — qwen3coder_tool_parser.py removed on dev1060; "
                "qwen3_coder now streams via the parser engine "
                "(vllm/parser/engine/*), which never emits a '{}' "
                "arguments placeholder and has no serving-layer remainder "
                "re-emit. Nothing to patch; PN56 retires on this pin.",
                GENESIS_PN56_DEV1060_RETIRE_MARKER,
            )
            return "skipped", (
                f"{GENESIS_PN56_DEV1060_RETIRE_MARKER} — "
                "qwen3coder_tool_parser.py removed; parser engine "
                "(vllm/parser/engine/*) streams name deltas without a "
                "'{}' placeholder and serving.py no longer re-emits "
                "remaining args. vllm#41466 class fixed upstream by "
                "architecture."
            )
        return "skipped", "qwen3coder_tool_parser.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "PN56 applied: XML parse failure no longer leaks {} placeholder"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return "skipped", f"{msg} — likely upstream merged or P64 reshaped block"
    return "failed", failure.reason if failure else "unknown failure"
