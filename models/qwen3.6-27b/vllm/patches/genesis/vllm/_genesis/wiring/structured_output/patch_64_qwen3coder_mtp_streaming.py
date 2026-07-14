# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 64 — qwen3coder MTP streaming early-return removal.

Backport of vllm-project/vllm#39598 (kotori-yan, OPEN at time of writing).

================================================================
Targets MTP/spec-decode streaming tool-call edge case where a single
delta bundles multiple semantic tokens (last parameter value AND
</function> together). The pre-PR code has an early `return` after
emitting parameter fragments, so the `</function>` block never
executes — leaving prev_tool_call_arr with stale `"{}"` arguments and
streamed_args_for_tool without the closing `}`. Final chunk then has
no valid arguments → empty `tool_calls`.
================================================================

2 sub-patches ACTIVE (parser-side); 2 sub-patches now OBSOLETE
(serving-side, absorbed upstream by #44267); plus 1 deferred:

1. **qwen3coder_tool_parser.py — remove early return** — Restructure
   `extract_tool_calls_streaming` to drop early `return` and accumulate
   into `combined`. **ACTIVE.**

2. **qwen3coder_tool_parser.py — unify </function> emit** — Inside the
   `</function>` branch, append `"}"` to `combined` and fall through
   to a single emit point at end of method (instead of a separate
   early `return result`). **ACTIVE.**

3. **serving.py — _should_check_for_unstreamed_tool_arg_tokens widen** —
   Fire safety-net on `finish_reason` presence alone (not gated by
   non-empty `tool_calls`). **OBSOLETE (absorbed by #44267).**

4. **serving.py — call-site guard for tool_calls[0]** — Defensive
   `if delta_message.tool_calls and isinstance(...)` guard before
   indexing into `tool_calls[0]`. **OBSOLETE (absorbed by #44267).**

DEFERRED (audit A-17 documented):
  D. **serving.py — _create_remaining_args_delta Pydantic-null fix** —
     SERVING_CRD_OLD/NEW constants are DEFINED in this module but NOT
     wired into sub_patches. Now also OBSOLETE-by-restructure: the whole
     `_create_remaining_args_delta` helper was deleted from serving.py
     by #44267 (see REBASE note). Constants retained as historical doc.

REBASE 2026-06-07 (vllm-new nightly 9c7f7741) — serving-side rework
-------------------------------------------------------------------
PR vllm-project/vllm#44267 "[Refactor] Unify reasoning + tool-call
parsing behind Parser.parse()" (merged 2026-06-02, sha 68dafcca7)
DELETED the entire serving-side streaming safety-net that sub-patches
3, 4 (and deferred D) targeted:

  * `_should_check_for_unstreamed_tool_arg_tokens()` — GONE from
    serving.py (was old serving.py:1547; not present in new tree
    except an unrelated `step3p5_tool_parser.parser_should_check_*`).
  * `_create_remaining_args_delta()` — GONE from serving.py (zero
    matches anywhere in vllm-new).
  * the `should_check` / `latest_delta_len` call-site block (old
    serving.py:829-878) — GONE.

The equivalent safety-net now lives in the unified Parser layer:
  * `vllm/entrypoints/parser/abstract_parser.py`
    `DelegatingParser.parse_delta(... , *, finished)` calls
    `self._append_unstreamed_tool_args(delta_message)` when
    `finished` is True (abstract_parser.py:902-903).
  * `_append_unstreamed_tool_args` (abstract_parser.py:781-794)
    already implements BOTH things sub-patches 3+4 added:
      - it fires on `finished` alone — no non-empty `tool_calls`
        precondition (this IS sub-patch 3's "widen"), and
      - it guards `delta_message and delta_message.tool_calls and
        (last_tc := delta_message.tool_calls[-1]).function` before
        touching the last tool call (this IS sub-patch 4's "call-site
        guard" — no unguarded `tool_calls[0]` index).
  * the remaining-args diff is `ToolParser.get_remaining_unstreamed_args()`
    (vllm/tool_parsers/abstract_tool_parser.py:82-99), and the new
    serving.py call site passes `finished=output.finish_reason is not
    None` (entrypoints/openai/chat_completion/serving.py:725).

Therefore the serving patcher is DROPPED from the active patcher list
(its anchors `SERVING_SHOULD_OLD`/`SERVING_CALLSITE_OLD` no longer match
and the upstream replacement is strictly equivalent). Its constants and
a serving-side drift marker are kept so a future regression check can
re-detect this state. The PARSER patcher stays active: the qwen3coder
parser streaming method is byte-identical between vllm-old and vllm-new
in this region — the early-return at parser line 526 / `return result`
at line 589 STILL exist, so the primary MTP param-drop bug is NOT fixed
upstream and sub-patches 1+2 are still required.

Status: opt-in via `GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1`.

Compatibility
-------------
- Streaming-only fix; non-streaming tool-call code path unaffected.
- Idempotent (marker check).
- Auto-no-op once #39598 lands upstream (parser drift marker:
  `[Genesis P64 vllm#39598]` / `Do NOT early-return here. With MTP`).
- Serving-side sub-patches auto-skip post-#44267 via the serving drift
  marker `finished=output.finish_reason is not None` — but as of the
  2026-06-07 rebase the serving patcher is simply not added to the
  active list, so the marker is belt-and-braces.

Risks acknowledged
------------------
- After the 2026-06-07 rebase only the PARSER patcher is active (2
  sub-patches, one file: qwen3coder_tool_parser.py). The serving
  patcher (sub-patches 3, 4) and the deferred Pydantic null fix
  (`SERVING_CRD_*`) are OBSOLETE — their targets were deleted from
  serving.py by #44267. Their behavior is now upstream in
  `Parser._append_unstreamed_tool_args`.
- All-or-nothing apply: anchor drift in one → whole group skips. **Audit P1
  caveat (genesis_deep_cross_audit_2026-05-05):** the current commit-loop
  treats `TextPatchResult.SKIPPED` as success and only short-circuits on
  `FAILED`. A subset can therefore install while the rest silently skip.
  Tracked for migration to `MultiFilePatchTransaction` once dry-run learns
  anchor-uniqueness checking; until then operators should grep for the
  specific marker if quality regresses unexpectedly.
- Test coverage limited to streaming clients (LibreChat, OpenWebUI).
  Synthetic streaming reproducer in test suite.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p64_qwen3coder_mtp_streaming")

GENESIS_P64_MARKER = "Genesis P64 qwen3coder MTP streaming early-return fix v7.13"


# ─── Sub-patch A: qwen3coder_tool_parser.py — remove early return ───────────
# The bug: early `return DeltaMessage(...)` after `if json_fragments:` block
# prevents the </function> handling block from executing when MTP bundles
# the last parameter + </function> in the same delta.
#
# Fix: collapse fragment accumulation into a `combined` string that gets
# extended by the </function> path (adding "}"), and emit ONE return at the
# end with the merged content.

QWEN3CODER_OLD = (
    "            if json_fragments:\n"
    "                combined = \"\".join(json_fragments)\n"
    "\n"
    "                if self.current_tool_index < len(self.streamed_args_for_tool):\n"
    "                    self.streamed_args_for_tool[self.current_tool_index] += combined\n"
    "                else:\n"
    "                    logger.warning(\n"
    "                        \"streamed_args_for_tool out of sync: index=%d len=%d\",\n"
    "                        self.current_tool_index,\n"
    "                        len(self.streamed_args_for_tool),\n"
    "                    )\n"
    "\n"
    "                return DeltaMessage(\n"
    "                    tool_calls=[\n"
    "                        DeltaToolCall(\n"
    "                            index=self.current_tool_index,\n"
    "                            function=DeltaFunctionCall(arguments=combined),\n"
    "                        )\n"
    "                    ]\n"
    "                )\n"
)

QWEN3CODER_NEW = (
    "            # [Genesis P64 vllm#39598] Do NOT early-return here. With MTP\n"
    "            # speculative decoding a single delta can bundle the last\n"
    "            # parameter value AND </function> together. An early return\n"
    "            # would skip the </function> block below, leaving\n"
    "            # prev_tool_call_arr with stale \"{}\" and streamed_args_for_tool\n"
    "            # without the closing \"}\". Accumulate into `combined`, let\n"
    "            # </function> path append \"}\", emit ONE return at end.\n"
    "            combined = \"\".join(json_fragments) if json_fragments else \"\"\n"
)


# ─── Sub-patch B: append "}" to combined inside </function> branch ──────────
# The </function> branch currently emits its own `result = DeltaMessage(... arguments="}")`
# and `return result`. Change to: append "}" to `combined`, fall through to
# unified emit below.

QWEN3COD_FNEND_OLD = (
    "                if self.current_tool_index < len(self.streamed_args_for_tool):\n"
    "                    self.streamed_args_for_tool[self.current_tool_index] += \"}\"\n"
    "                else:\n"
    "                    logger.warning(\n"
    "                        \"streamed_args_for_tool out of sync: index=%d len=%d\",\n"
    "                        self.current_tool_index,\n"
    "                        len(self.streamed_args_for_tool),\n"
    "                    )\n"
    "\n"
    "                result = DeltaMessage(\n"
    "                    tool_calls=[\n"
    "                        DeltaToolCall(\n"
    "                            index=self.current_tool_index,\n"
    "                            function=DeltaFunctionCall(arguments=\"}\"),\n"
    "                        )\n"
    "                    ]\n"
    "                )\n"
    "\n"
    "                self.in_function = False\n"
    "                self.json_closed = True\n"
    "                self.accumulated_params = {}\n"
    "\n"
    "                return result\n"
)

QWEN3COD_FNEND_NEW = (
    "                # [Genesis P64 vllm#39598] Append \"}\" to combined and fall\n"
    "                # through to unified emit below — no early return.\n"
    "                # [Audit A-07 fix 2026-05-05] NOTE on `self.json_closed = True`:\n"
    "                # the OLD branch set this here. NEW branch does NOT — it is set\n"
    "                # at the top of the upstream `if not self.json_closed and ...`\n"
    "                # branch above (parser line ~626 in current pin). Not a bug,\n"
    "                # just non-redundant: removing avoids double-set; runtime\n"
    "                # invariant preserved by upstream parser logic.\n"
    "                combined += \"}\"\n"
    "                self.in_function = False\n"
    "                self.accumulated_params = {}\n"
    "\n"
    "            if combined:\n"
    "                if self.current_tool_index < len(self.streamed_args_for_tool):\n"
    "                    self.streamed_args_for_tool[self.current_tool_index] += combined\n"
    "                else:\n"
    "                    logger.warning(\n"
    "                        \"streamed_args_for_tool out of sync: index=%d len=%d\",\n"
    "                        self.current_tool_index,\n"
    "                        len(self.streamed_args_for_tool),\n"
    "                    )\n"
    "\n"
    "                return DeltaMessage(\n"
    "                    tool_calls=[\n"
    "                        DeltaToolCall(\n"
    "                            index=self.current_tool_index,\n"
    "                            function=DeltaFunctionCall(arguments=combined),\n"
    "                        )\n"
    "                    ]\n"
    "                )\n"
)


# ─── Sub-patch C: serving.py — _should_check safety-net widening ────────────

SERVING_SHOULD_OLD = (
    "        return bool(\n"
    "            # if there is a delta message that includes tool calls which\n"
    "            # include a function that has arguments\n"
    "            output.finish_reason is not None\n"
    "            and self.enable_auto_tools\n"
    "            and self.tool_parser\n"
    "            and delta_message\n"
    "            and delta_message.tool_calls\n"
    "            and delta_message.tool_calls[0]\n"
    "            and delta_message.tool_calls[0].function\n"
    "            and delta_message.tool_calls[0].function.arguments is not None\n"
    "        )\n"
)

SERVING_SHOULD_NEW = (
    "        # [Genesis P64 vllm#39598] Widen safety-net: with MTP/spec-decode\n"
    "        # the final delta before finish_reason may carry no tool_calls\n"
    "        # even though tool calls are still in progress. Caller's\n"
    "        # auto_tools_called guard (checks len(prev_tool_call_arr) > 0)\n"
    "        # prevents false positives for plain-text responses.\n"
    "        return bool(\n"
    "            output.finish_reason is not None\n"
    "            and self.enable_auto_tools\n"
    "            and self.tool_parser\n"
    "        )\n"
)


# ─── Sub-patch E: serving.py — call-site guard for tool_calls[0] ────────────
# P64's widened _should_check returns True on finish_reason alone (no tool_calls
# required). The call site in chat_completion_stream_generator then accesses
# delta_message.tool_calls[0] unconditionally → IndexError when the final
# streaming delta carries tool calls in prev_tool_call_arr but an empty
# delta_message.tool_calls list. Guard the inner `if` before [0] access.

SERVING_CALLSITE_OLD = (
    "                        if should_check and tool_parser and auto_tools_called:\n"
    "                            latest_delta_len = 0\n"
    "                            if (\n"
    "                                isinstance(\n"
    "                                    delta_message.tool_calls[0].function,\n"
    "                                    DeltaFunctionCall,\n"
    "                                )\n"
    "                            ) and isinstance(\n"
    "                                delta_message.tool_calls[0].function.arguments, str\n"
    "                            ):\n"
)

SERVING_CALLSITE_NEW = (
    "                        if should_check and tool_parser and auto_tools_called:\n"
    "                            latest_delta_len = 0\n"
    "                            # [Genesis P64 call-site guard] _should_check\n"
    "                            # fires on finish_reason alone; tool_calls may\n"
    "                            # be [] on the final delta — guard before [0].\n"
    "                            if (\n"
    "                                delta_message.tool_calls\n"
    "                                and isinstance(\n"
    "                                    delta_message.tool_calls[0].function,\n"
    "                                    DeltaFunctionCall,\n"
    "                                )\n"
    "                            ) and isinstance(\n"
    "                                delta_message.tool_calls[0].function.arguments, str\n"
    "                            ):\n"
)


# ─── Sub-patch D: serving.py — _create_remaining_args_delta Pydantic fix ────
# AUDIT A-17 (2026-05-05): These SERVING_CRD_* constants are DEFINED but
# NOT WIRED into _make_serving_patcher.sub_patches — this is INTENTIONAL.
# See the explanation block in _make_serving_patcher below ("belt-and-
# braces / leave _create_remaining_args_delta unchanged / parser fix
# closes primary symptom"). The constants stay as living documentation
# of the deferred Pydantic-null fix and as a future-ready replacement
# if the primary parser fix proves insufficient. DO NOT delete them
# without re-evaluating that decision. Audit-A-17-keepalive: SERVING_CRD.

SERVING_CRD_OLD = (
    "        original_tc = next(\n"
    "            (tc for tc in delta_message.tool_calls if tc.index == index),\n"
    "            None,\n"
    "        )\n"
    "        original_fn = original_tc.function if original_tc else None\n"
    "        return DeltaMessage(\n"
    "            tool_calls=[\n"
    "                DeltaToolCall(\n"
    "                    index=index,\n"
    "                    id=original_tc.id if original_tc else None,\n"
    "                    type=original_tc.type if original_tc else None,\n"
    "                    function=DeltaFunctionCall("
)

SERVING_CRD_NEW = (
    "        # [Genesis P64 vllm#39598] Use post-construction assignment\n"
    "        # instead of constructor None so Pydantic v2 treats absent\n"
    "        # fields as *unset* (model_dump(exclude_unset=True) omits them\n"
    "        # instead of emitting \"id\": null which violates OpenAI spec).\n"
    "        original_tc = next(\n"
    "            (tc for tc in delta_message.tool_calls if tc.index == index),\n"
    "            None,\n"
    "        )\n"
    "        original_fn = original_tc.function if original_tc else None\n"
    "\n"
    "        delta_fn = DeltaFunctionCall(arguments=remaining_call)\n"
    "        if original_fn and original_fn.name is not None:\n"
    "            delta_fn.name = original_fn.name\n"
    "\n"
    "        tc = DeltaToolCall(index=index, function=delta_fn)\n"
    "        if original_tc and original_tc.id is not None:\n"
    "            tc.id = original_tc.id\n"
    "            tc.type = original_tc.type or \"function\"\n"
    "\n"
    "        return DeltaMessage(tool_calls=[tc])\n"
    "        # [Genesis P64] above replaces the original constructor-style:\n"
    "        return DeltaMessage(\n"
    "            tool_calls=[\n"
    "                DeltaToolCall(\n"
    "                    index=index,\n"
    "                    id=original_tc.id if original_tc else None,\n"
    "                    type=original_tc.type if original_tc else None,\n"
    "                    function=DeltaFunctionCall("
)


def _make_qwen3cod_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("tool_parsers/qwen3coder_tool_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P64 qwen3coder_tool_parser.py — MTP streaming early-return removal",
        target_file=str(target),
        marker=GENESIS_P64_MARKER + " :: qwen3coder_tool_parser.py",
        sub_patches=[
            TextPatch(name="p64_remove_early_return", anchor=QWEN3CODER_OLD,
                      replacement=QWEN3CODER_NEW, required=True),
            TextPatch(name="p64_unify_emit_at_fnend", anchor=QWEN3COD_FNEND_OLD,
                      replacement=QWEN3COD_FNEND_NEW, required=True),
        ],
        upstream_drift_markers=[
            "[Genesis P64 vllm#39598]",
            "Do NOT early-return here. With MTP",
        ],
    )


def _make_serving_patcher() -> TextPatcher | None:
    """OBSOLETE since vllm-project/vllm#44267 (merged 2026-06-02, sha
    68dafcca7) — RETAINED for documentation / regression detection only.

    #44267 deleted `_should_check_for_unstreamed_tool_arg_tokens`,
    `_create_remaining_args_delta`, and the `should_check`/`latest_delta_len`
    call-site block from serving.py. The equivalent safety-net moved into
    `vllm/entrypoints/parser/abstract_parser.py::DelegatingParser.parse_delta`
    which calls `_append_unstreamed_tool_args` on `finished` (lines 902-903).
    That method already (a) fires on `finished` alone — sub-patch C's
    "widen" — and (b) guards `tool_calls` before indexing the last call —
    sub-patch E's "call-site guard" (abstract_parser.py:781-794).

    Consequently this patcher is NO LONGER added to the active patcher list
    in apply(); its anchors `SERVING_SHOULD_OLD` / `SERVING_CALLSITE_OLD`
    intentionally no longer match vllm-new. The serving-side drift marker
    below (`finished=output.finish_reason is not None`, unique to the
    post-#44267 serving.py call site at serving.py:725) makes apply()
    report 'upstream may have absorbed this fix' should the patcher ever
    be re-wired against a tree that already carries #44267.
    """
    target = resolve_vllm_file("entrypoints/openai/chat_completion/serving.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P64 serving.py — MTP safety-net (OBSOLETE post-#44267)",
        target_file=str(target),
        marker=GENESIS_P64_MARKER + " :: serving.py",
        sub_patches=[
            TextPatch(name="p64_safety_net_widen", anchor=SERVING_SHOULD_OLD,
                      replacement=SERVING_SHOULD_NEW, required=True),
            TextPatch(name="p64_callsite_guard", anchor=SERVING_CALLSITE_OLD,
                      replacement=SERVING_CALLSITE_NEW, required=True),
            # NOTE: D sub-patch DUPLICATES the original return for the second
            # half of _create_remaining_args_delta to keep the function shape
            # syntactically valid (the original DeltaFunctionCall line
            # continues into name= / arguments=). The duplicate emit is
            # unreachable but preserves anchor stability for #D's tail.
            # Simpler approach: only patch the safety-net (sub-patch C); leave
            # _create_remaining_args_delta unchanged. The Pydantic null fix is
            # belt-and-braces; the parser fix (sub-patches A+B) closes the
            # primary symptom.
        ],
        upstream_drift_markers=[
            "[Genesis P64 vllm#39598]",
            # [Rebase 2026-06-07] #44267 moved the serving-side safety-net
            # into Parser.parse_delta(..., finished=...). This kwarg on the
            # serving.py call site is unique to the post-#44267 tree
            # (absent in vllm-old) and signals the targets are gone.
            "finished=output.finish_reason is not None",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P64 streaming-fix backport.

    REBASE 2026-06-07: only the PARSER patcher is active. The serving
    patcher's targets were deleted by vllm#44267 (its safety-net is now
    `Parser._append_unstreamed_tool_args`, abstract_parser.py:781-794,
    fired on `finished` from parse_delta) — it is OBSOLETE and is no
    longer applied. We still *probe* the serving file once (soft, never
    blocking) so the apply log records that #44267 absorbed sub-patches
    3/4/D; an unexpected miss there is logged, not failed.
    """
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P64")
    log_decision("P64", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # Soft probe of the (obsolete) serving patcher — informational only.
    serving_note = ""
    serving_patcher = _make_serving_patcher()
    if serving_patcher is not None and os.path.isfile(serving_patcher.target_file):
        with open(serving_patcher.target_file) as f:
            sc = f.read()
        if any(m in sc for m in serving_patcher.upstream_drift_markers):
            serving_note = (
                " serving-side sub-patches OBSOLETE — vllm#44267 absorbed "
                "them into Parser._append_unstreamed_tool_args (drift marker "
                "present in serving.py)."
            )
            log.info("P64 serving patcher obsolete (vllm#44267 absorbed).")
        else:
            serving_note = (
                " WARNING: serving.py lacks the #44267 drift marker AND the "
                "serving patcher is intentionally inactive — verify the new "
                "Parser.parse_delta/_append_unstreamed_tool_args safety-net "
                "is present before trusting MTP streaming tool calls."
            )
            log.warning(
                "P64 serving drift marker absent; serving patcher inactive."
            )

    # Only the parser patcher is active post-rebase. The MTP param-drop bug
    # (early return at qwen3coder parser line ~526) still exists in vllm-new
    # — verified byte-identical to vllm-old — so sub-patches A+B are required.
    patchers = [_make_qwen3cod_patcher()]
    if any(p is None for p in patchers):
        return "skipped", "parser target file not found"

    # Pre-flight
    for p in patchers:
        if not os.path.isfile(p.target_file):
            return "skipped", f"target disappeared: {p.target_file}"
        with open(p.target_file) as f:
            content = f.read()
        if p.marker in content:
            continue
        for m in p.upstream_drift_markers:
            if m in content:
                return (
                    "skipped",
                    f"upstream drift marker {m!r} in {p.target_file} — "
                    "vllm#39598 likely already merged or backported.",
                )
        for sp in p.sub_patches:
            if sp.required and sp.anchor not in content:
                return (
                    "skipped",
                    f"required anchor for {sp.name!r} not found in "
                    f"{p.target_file} — anchor drifted, P64 cannot apply.",
                )

    results = []
    for p in patchers:
        result, failure = p.apply()
        if result == TextPatchResult.FAILED:
            return "failed", (
                f"{p.patch_name}: {failure.reason if failure else 'unknown'} "
                f"({failure.detail if failure else ''}) — partial state risk; "
                "container should be torn down (compose down + up -d)."
            )
        results.append((p.patch_name, result))

    applied = sum(1 for _, r in results if r == TextPatchResult.APPLIED)
    idempotent = sum(1 for _, r in results if r == TextPatchResult.IDEMPOTENT)

    return "applied", (
        f"P64 applied (parser-only post-rebase): {applied} files modified, "
        f"{idempotent} idempotent. qwen3coder streaming parser no longer "
        "drops parameters when MTP bundles last param + </function> in same "
        "delta." + serving_note
    )
