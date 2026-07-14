# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN66 — multiturn `</think>` leak in DelegatingParser.

Backport of vllm-project/vllm#41696 (panpan0000, OPEN as of 2026-05-05).
Removes the buggy `prompt_reasoning_checked` short-circuit in
`vllm.parser.abstract_parser.DelegatingParser.parse_delta` that walks
the FULL prompt looking for `</think>` and prematurely sets
`reasoning_ended=True` if a previous turn's `</think>` is present in
prompt_token_ids.

================================================================
THE BUG
================================================================

Original code (current pin, abstract_parser.py:654-661):

    if not state.prompt_reasoning_checked and prompt_token_ids is not None:
        state.prompt_reasoning_checked = True
        if self.is_reasoning_end(prompt_token_ids):
            state.reasoning_ended = True

This block fires ONCE per request. `is_reasoning_end()` walks the
prompt backward looking for the reasoning end token (e.g. `</think>`).
On a multi-turn conversation where ANY prior assistant turn closed its
reasoning with `</think>`, this token is in `prompt_token_ids` —
DelegatingParser then sets `reasoning_ended=True` BEFORE the new turn's
generation starts producing tokens. Result: the new turn's reasoning
content is routed to `delta.content` (where users see it leak as
plaintext) instead of `delta.reasoning` (where it belongs).

================================================================
GENESIS APPROACH
================================================================

Two-step text-patch (matches upstream PR #41696 exactly):

  1. Remove the `prompt_reasoning_checked: bool = False` field declaration
     from `StreamState` dataclass (line ~57)
  2. Remove the 4-line short-circuit block in `parse_delta` (lines ~654-661)

Both anchors are unique in the current pin (`0.20.2rc1.dev9+g01d4d1ad3`)
and exact-match against the upstream PR diff.

================================================================
RELATIONSHIP TO OTHER GENESIS PATCHES
================================================================

PN66 is in the SAME file as nothing else (Genesis doesn't currently
patch `vllm/parser/abstract_parser.py`). It composes orthogonally with:
- P59/P61/P61b/P62/P64 — Qwen3 reasoning + tool-call parsers (different
  file, `vllm/reasoning/qwen3_reasoning_parser.py`)
- P107 — MTP truncation detector (different file)
- PN51/PN56 — Qwen3 streaming + Qwen3Coder XML (different files)

================================================================
WHO THIS HELPS
================================================================

Direct beneficiaries:
- Multi-turn DSML / Hermes / Qwen3 chat clients that send full history
  including assistant `</think>` markers in subsequent turns
- DeepSeek V3.2 reasoning users (original reporter)

For Genesis PROD (27B Lorbus + 35B FP8 with Qwen3 parsers): the
DelegatingParser code path is reached through `tool_call_parser`
selection. Whether we hit this exact bug depends on which parser is
active. Defensive backport: ON-by-default would mask the bug for any
user. Default OFF for now; recommend enabling once we have a Genesis
multi-turn reproducer (test_pn66_*.py contract).

================================================================
ENV
================================================================

GENESIS_ENABLE_PN66=1

================================================================
RISK
================================================================

LOW — removes existing buggy behavior. The removed lines never had a
correct purpose: they conflated PROMPT history with CURRENT GENERATION.
Worst case: a niche use of `prompt_reasoning_checked` we don't see in
PROD breaks. Drift markers detect the case.

Author: Sandermage 2026-05-05.
Backport reference: vllm#41696 (panpan0000, OPEN as of 2026-05-05).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn66_multiturn_think_leak")

GENESIS_PN66_MARKER = "Genesis PN66 multiturn think leak fix vllm#41696"

# Anchor 1: StreamState dataclass field declaration
PN66_FIELD_OLD = (
    "    tool_call_text_started: bool = False\n"
    "    prompt_reasoning_checked: bool = False\n"
    "    previous_text: str = \"\"\n"
)
PN66_FIELD_NEW = (
    "    tool_call_text_started: bool = False\n"
    "    previous_text: str = \"\"\n"
)

# Anchor 2: the buggy short-circuit block in parse_delta.
# 2026-07-14 re-anchor (dev1060cherry): upstream evolved the block well
# past the original 4-line form. It now (a) wraps `is_reasoning_end(...)`
# across two lines behind a defensive `self._reasoning_parser is None or`
# guard, (b) adds an `else:` branch that calls
# `self._reasoning_parser.adjust_initial_state_from_prompt(prompt_token_ids)`
# — a NO-OP on our engine-adapter parser stack (parser_engine.py
# `adjust_initial_state_from_prompt` is `return`), and (c) the trailing
# line moved from `current_text = state.previous_text + delta_text` to
# `current_text, current_token_ids = state.advance(delta_text, delta_token_ids)`.
# The multiturn-leak surface is unchanged: the `if ... is_reasoning_end(
# prompt_token_ids): state.reasoning_ended = True` still sets reasoning_ended
# from PRIOR-turn </think> in prompt history. Faithful to vllm#41696, we
# remove ONLY the prompt-history walk and preserve state.advance.
#
# 2026-07-14 BUG-4 fix: the upstream block ALSO carries a load-bearing
# init — `self._reasoning_parser is None` ⇒ `state.reasoning_ended = True`.
# That is the ONLY place reasoning_ended is set for tool-parser-only
# configs (`--tool-call-parser X` with no `--reasoning-parser`); dropping
# it leaves `_in_reasoning_phase()` True forever, `_in_tool_call_phase()`
# never fires, and tool-call markup streams as plain content. #41696's
# intent covers only the is_reasoning_end(prompt_token_ids) HISTORY WALK,
# so BLOCK_NEW keeps the None-guard (idempotent per-delta re-set is
# harmless) and removes only the history walk + no-op else branch.
PN66_BLOCK_OLD = (
    "        state = self._stream_state\n"
    "\n"
    "        if not state.prompt_reasoning_checked and prompt_token_ids is not None:\n"
    "            state.prompt_reasoning_checked = True\n"
    "            if self._reasoning_parser is None or self.is_reasoning_end(\n"
    "                prompt_token_ids\n"
    "            ):\n"
    "                state.reasoning_ended = True\n"
    "            else:\n"
    "                # Reasoning is still open at the end of the prompt; let the\n"
    "                # reasoning parser adjust its initial parsing state so the\n"
    "                # first generated tokens are classified correctly.\n"
    "                self._reasoning_parser.adjust_initial_state_from_prompt(\n"
    "                    prompt_token_ids\n"
    "                )\n"
    "\n"
    "        current_text, current_token_ids = state.advance(delta_text, delta_token_ids)\n"
)
PN66_BLOCK_NEW = (
    "        state = self._stream_state\n"
    "\n"
    "        # [Genesis PN66 multiturn think leak fix vllm#41696] Removed\n"
    "        # the prompt_token_ids walk that prematurely set reasoning_ended\n"
    "        # from a PRIOR turn's </think> in the prompt history. The block\n"
    "        # conflated 'prompt contains </think>' with 'current generation\n"
    "        # already ended reasoning' — wrong on multi-turn chat. The removed\n"
    "        # else-branch (adjust_initial_state_from_prompt) is a no-op on the\n"
    "        # engine-adapter parser, so continuation behavior is unchanged.\n"
    "        # KEPT the upstream no-reasoning-parser init: it is the ONLY\n"
    "        # place reasoning_ended is set for tool-parser-only configs;\n"
    "        # without it _in_tool_call_phase() never fires and tool-call\n"
    "        # markup streams as plain content.\n"
    "        if self._reasoning_parser is None:\n"
    "            state.reasoning_ended = True\n"
    "\n"
    "        current_text, current_token_ids = state.advance(delta_text, delta_token_ids)\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("parser/abstract_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN66 abstract_parser.py — multiturn </think> leak fix (vllm#41696)",
        target_file=str(target),
        marker=GENESIS_PN66_MARKER,
        sub_patches=[
            TextPatch(
                name="pn66_remove_field",
                anchor=PN66_FIELD_OLD,
                replacement=PN66_FIELD_NEW,
                required=True,
            ),
            TextPatch(
                name="pn66_remove_block",
                anchor=PN66_BLOCK_OLD,
                replacement=PN66_BLOCK_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN66",
            # Upstream merge marker: when #41696 lands, the field declaration
            # disappears entirely. If the file no longer contains
            # `prompt_reasoning_checked`, upstream merged → auto-skip.
            # We DON'T add this as a drift marker because the marker check
            # below covers the same case via `_FIELD_OLD not in content`.
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN66 — remove buggy prompt_reasoning_checked short-circuit."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN66")
    log_decision("PN66", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return (
            "skipped",
            "vllm/parser/abstract_parser.py not resolvable — pin may be too old "
            "(file added in vllm V2 parser refactor)",
        )

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target file disappeared: {patcher.target_file}"

    # Pre-flight: detect upstream-merged auto-skip
    with open(patcher.target_file) as f:
        content = f.read()
    if "prompt_reasoning_checked" not in content:
        return (
            "skipped",
            "abstract_parser.py no longer contains `prompt_reasoning_checked` "
            "field — upstream PR #41696 (or equivalent) appears merged",
        )

    if patcher.marker in content:
        return "applied", "PN66 already applied (marker present, idempotent)"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN66 applied: removed prompt_reasoning_checked history walk "
            "in DelegatingParser.parse_delta (no-reasoning-parser "
            "reasoning_ended init KEPT). Multi-turn chat with prior "
            "assistant </think> markers no longer leaks current-turn "
            "reasoning into delta.content. Backport of vllm#41696 "
            "(panpan0000, OPEN at backport time)."
        ),
        patch_name=patcher.patch_name,
    )
