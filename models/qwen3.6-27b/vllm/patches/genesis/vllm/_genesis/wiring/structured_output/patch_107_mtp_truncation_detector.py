# SPDX-License-Identifier: Apache-2.0
"""Wiring for P107 — vllm#41467 backport: MTP truncation detector.

При MTP K≥1 + tools + reasoning parser возможна редкая (≈0.25% по
автору на Qwen3.6-27B-FP8) ситуация: модель производит EOS на
boundary reasoning→tool_call. finish_reason=stop + tools configured +
ни tool_calls, ни content не отдано → silent client-side error.

Решение: defensive guard в `chat_completion_stream_generator` которая
detect'ит этот combo и raise GenerationError (retryable) вместо
silent stop. Клиент получит SSE error event и retry.

Affects прямо наш PROD: 27B Lorbus + MTP K=3 + tools — exactly the
config which author reported. P107 — safety net на уже defended path
(P59/P60/P61/P62/P64/P68/P69 family).

Default OFF — defensive backport. Risk: low, добавляет один extra
branch в hot path, не аффектит happy path.

Author: Sandermage backport (ToastyTheBot, vllm#41467).

Rebase note (vllm-new nightly 9c7f7741, 2026-06-07):
  * NOT obsolete. Upstream `_raise_if_error()` (entrypoints/openai/engine/
    serving.py:192) only raises for finish_reason == "error"; it does NOT
    cover the P107 case (finish_reason == "stop" + tools configured + only
    reasoning produced). The defensive guard is still required.
  * Anchor re-derived: the unified-parsing churn (vllm#44267) dropped the
    local `auto_tools_called` flag from `chat_completion_stream_generator`.
    In vllm-new the finish-reason branch is keyed solely on `tools_streamed[i]`
    (serving.py:821). The pristine `if (...)` head therefore no longer carries
    the leading `auto_tools_called` OR-term — ANCHOR updated to match.
  * Guard condition simplified: removed `and not auto_tools_called` (the var
    is no longer in scope in the streaming generator — it now lives only in
    `chat_completion_full_generator`). `not tools_streamed[i]` is the unified
    "no tool emitted" signal and fully subsumes the old check.
  * Import fixed: `GenerationError` was never in `chat_completion/protocol.py`
    (true for both vllm-old and vllm-new). It is defined in and imported from
    `vllm.entrypoints.openai.engine.serving` (serving.py:50). The local import
    now targets that module.

Rebase note (vllm-new nightly 1033ffac, 2026-06-13):
  * serving.py finish-reason head dropped the `self.use_harmony and
    harmony_tools_streamed[i]` OR-term (serving.py:821 → 674) — same churn class
    as the 2026-06-07 auto_tools_called drop. ANCHOR head re-derived to the bare
    `if tools_streamed[i] and not tool_choice_function_name:` (count=1 in dev491;
    direct patcher.apply() == APPLIED). else/finish_reason_/choice_data tail +
    detector body unchanged; reasoning_parser + GenerationError import still in scope.
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

log = logging.getLogger("genesis.wiring.p107_mtp_truncation_detector")

GENESIS_P107_MARKER = "Genesis P107 MTP truncation detector (vllm#41467)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_P107_MTP_TRUNCATION_DETECTOR", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor on the finish_reason_ if/else block in chat_completion_stream_generator.
# vllm-new nightly 9c7f7741: serving.py:821-828 (was lines 887-896 in vllm-old,
# which carried the now-removed `auto_tools_called` OR-term; vllm#44267 folded
# that signal into tools_streamed[i]).
ANCHOR_OLD = (
    "                        if tools_streamed[i] and not tool_choice_function_name:\n"
    "                            finish_reason_ = \"tool_calls\"\n"
    "                        else:\n"
    "                            finish_reason_ = (\n"
    "                                output.finish_reason if output.finish_reason else \"stop\"\n"
    "                            )\n"
    "                        choice_data = ChatCompletionResponseStreamChoice("
)

ANCHOR_NEW = (
    "                        if tools_streamed[i] and not tool_choice_function_name:\n"
    "                            finish_reason_ = \"tool_calls\"\n"
    "                        else:\n"
    "                            finish_reason_ = (\n"
    "                                output.finish_reason if output.finish_reason else \"stop\"\n"
    "                            )\n"
    "\n"
    "                        # [Genesis P107 vllm#41467] MTP truncation detector.\n"
    "                        # ~0.25% rate on Qwen3.6 27B-FP8 + MTP K=3 (per upstream\n"
    "                        # author): EOS at reasoning→tool_call boundary leaves\n"
    "                        # finish_reason=stop with no content/tool_calls. Raise\n"
    "                        # retryable error so client retries instead of seeing\n"
    "                        # silent empty response. Defensive: AND-chained, no\n"
    "                        # impact on happy path. (vllm-new: auto_tools_called no\n"
    "                        # longer in this scope post vllm#44267 — not tools_streamed[i]\n"
    "                        # is the unified 'no tool emitted' signal.)\n"
    "                        # dev491 scope fix: chat_completion_stream_generator has NO\n"
    "                        # `reasoning_parser` local (that lives in _create_chat_completion);\n"
    "                        # use the instance-level configured-parser class instead, which\n"
    "                        # IS in scope via self. Referencing the old local NameError'd\n"
    "                        # and turned this branch into a 500 on streaming tool calls.\n"
    "                        if (\n"
    "                            finish_reason_ == \"stop\"\n"
    "                            and request.tools\n"
    "                            and not tools_streamed[i]\n"
    "                            and self.reasoning_parser_cls is not None\n"
    "                            and delta_message is not None\n"
    "                            and not delta_message.content\n"
    "                            and not delta_message.tool_calls\n"
    "                        ):\n"
    "                            try:\n"
    "                                # [2026-07-14] dev1060 moved GenerationError to engine.protocol;\n"
    "                                # the old engine.serving module was deleted (import was silently\n"
    "                                # dead behind the outer try/except -> P107 never fired).\n"
    "                                from vllm.entrypoints.openai.engine.protocol import GenerationError as _P107_GenError\n"
    "                            except ImportError:\n"
    "                                from vllm.entrypoints.openai.engine.serving import GenerationError as _P107_GenError\n"
    "                            logger.warning(\n"
    "                                \"[Genesis P107] MTP truncation detected for request %s: \"\n"
    "                                \"finished with 'stop' but tools configured and only \"\n"
    "                                \"reasoning produced.\",\n"
    "                                request_id,\n"
    "                            )\n"
    "                            raise _P107_GenError(\n"
    "                                \"MTP speculative decoding truncated tool call \"\n"
    "                                \"generation. Please retry.\"\n"
    "                            )\n"
    "                        choice_data = ChatCompletionResponseStreamChoice("
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "entrypoints/openai/chat_completion/serving.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="P107 MTP truncation detector (vllm#41467)",
        target_file=str(target),
        marker=GENESIS_P107_MARKER,
        sub_patches=[TextPatch(
            name="p107_mtp_truncation",
            anchor=ANCHOR_OLD,
            replacement=ANCHOR_NEW,
            required=True,
        )],
        upstream_drift_markers=[
            "MTP truncation detected",
            "MTP speculative decoding truncated",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P107")
    log_decision("P107", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "serving.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "P107 applied: MTP truncation now raises retryable error"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return "skipped", f"{msg} — likely upstream merged"
    return "failed", failure.reason if failure else "unknown failure"
