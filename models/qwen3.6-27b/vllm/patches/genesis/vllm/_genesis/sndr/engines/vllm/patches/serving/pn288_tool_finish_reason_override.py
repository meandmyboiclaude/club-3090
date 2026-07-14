# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN288 — qwen3_coder tool_call finish_reason override
(§1.3 of the unified plan, Phase B / dry-run scaffold).

The decision logic lives in the companion middleware module
``sndr.engines.vllm.middleware.pn288_finish_reason_override``. This file
is the text-patch overlay that wires that logic into
``OpenAIServingChat._create_chat_completion`` at two anchors:

  1. **Streaming** (serving.py:821-828 on pin 0.22.1rc1.dev259) — the
     if-block that assigns ``finish_reason_`` inside the choice-data loop.
  2. **Non-streaming** (serving.py:1246-1250) — the
     ``is_finish_reason_tool_calls`` bool assignment.

Anchor strategy
---------------
v1 verified live against the running
``vllm-gemma4-31b-tq-mtp-structured-k4-k4`` container 2026-05-30 (pin
626fa9bb). v2 re-anchored 2026-06-11 for pin
0.22.1rc1.dev259+g303916e93: upstream removed ``auto_tools_called``
from the streaming condition (the variable now exists only in the
non-streaming full generator), so the streaming anchor, the injected
call and its except-fallback dropped every reference to it. Sub-patch 2
is byte-identical to v1 (pristine serving.py:1246-1250 unchanged).
P107 targets the same streaming block — see the apply-order chain note
above ``PN288_STREAMING_OLD``.

v3 (2026-07-13, dev1060): upstream removed the ``use_harmony`` /
``harmony_tools_streamed`` OR-clause from the streaming condition
(harmony now flows through its own parser under the new
``vllm/parser/`` engine), so the v2 streaming anchor no longer
matches. The NON-streaming anchor is byte-identical to v1/v2
(chat_completion/serving.py:960-964 on dev1060, count==1 verified
against /home/user/engines/vllm-build). ``_make_patcher`` sniffs the
target content and selects the matching anchor pair (v2 vs
``*_DEV1060``). dev1060 also removed the legacy ToolParser surface
(``prev_tool_call_arr``) that the middleware's args-validity walk
reads, so the dev1060 replacements feed it a shim built by the
companion module ``pn288_dev1060_probe`` (same directory): the
non-streaming leg probes the parsed ``tool_calls`` FunctionCall list
(exactly what the client receives), the streaming leg probes
``parser._tool_slots[*].streamed_json`` (what was actually streamed).
The override itself is STILL NEEDED on dev1060 — serving.py computes
``auto_tools_called = len(tool_calls) > 0`` with no args-validity or
truncation check before stamping ``finish_reason="tool_calls"``.

Both replacements are wrapped in ``try / except Exception`` so any
import or runtime failure in the middleware logic falls back to the
upstream verdict — observability must not break the request.

Gates
-----
  * ``GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE`` — install
    gate. Default OFF; no text patch is written unless this is ``1``.
  * ``GENESIS_PN288_DRY_RUN`` — Phase B vs Phase C selector, read
    inside the middleware module at decision time (NOT at patch time)
    so an operator can flip it on a live container without re-patching.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import (
    resolve_vllm_file, vllm_install_root,
)
from sndr.kernel import (
    TextPatch, TextPatcher, TextPatchResult,
)


log = logging.getLogger("genesis.wiring.pn288_tool_finish_reason_override")

GENESIS_PN288_MARKER = (
    "Genesis PN288 tool-call finish_reason override v1 (Phase B dry-run)"
)


# ─── Sub-patch 1: streaming anchor ──────────────────────────────────────
#
# v2 (2026-06-11, pin 0.22.1rc1.dev259+g303916e93): upstream REMOVED the
# ``auto_tools_called`` OR-clause from the streaming finish_reason
# condition — on this pin the variable exists only in the non-streaming
# full generator (pristine serving.py:1069+). Anchor refreshed to the
# pristine block at chat_completion/serving.py:821-828 (count==1
# verified against /private/tmp/candidate_pin_current):
#
#     if (tools_streamed[i] and not tool_choice_function_name) or (
#         self.use_harmony and harmony_tools_streamed[i]
#     ):
#         finish_reason_ = "tool_calls"
#     else:
#         finish_reason_ = (
#             output.finish_reason if output.finish_reason else "stop"
#         )
#
# APPLY-ORDER CHAIN with P107: P107 v3's ANCHOR_OLD spans this same
# block PLUS the following ``choice_data =
# ChatCompletionResponseStreamChoice(`` line, and its ANCHOR_NEW keeps
# the block verbatim — so PN288 applies cleanly on BOTH pristine and
# post-P107 content. The reverse order does NOT compose: PN288's
# replacement re-indents the block (+4 spaces) inside the
# except-fallback, destroying P107's anchor. The registry therefore
# declares ``requires_patches: ["P107"]`` on PN288 (ordering-only —
# PN288 still applies standalone when P107 is disabled). Proven in
# tests/unit/integrations/serving/test_pn288_p107_anchor_coordination.py.


PN288_STREAMING_OLD = (
    "                        if (tools_streamed[i] and not tool_choice_function_name) or (\n"
    "                            self.use_harmony and harmony_tools_streamed[i]\n"
    "                        ):\n"
    "                            finish_reason_ = \"tool_calls\"\n"
    "                        else:\n"
    "                            finish_reason_ = (\n"
    "                                output.finish_reason if output.finish_reason else \"stop\"\n"
    "                            )\n"
)


PN288_STREAMING_NEW = (
    "                        # [Genesis PN288] args-validity-aware finish_reason\n"
    "                        # decision. Phase B (dry-run) by default — see\n"
    "                        # sndr.engines.vllm.middleware.pn288_finish_reason_override.\n"
    "                        try:\n"
    "                            from sndr.engines.vllm.middleware.pn288_finish_reason_override import (  # noqa: E501\n"
    "                                decide_streaming_finish_reason as _genesis_pn288_decide_streaming,\n"
    "                            )\n"
    "                            finish_reason_ = _genesis_pn288_decide_streaming(\n"
    "                                tools_streamed_i=tools_streamed[i],\n"
    "                                tool_choice_function_name=tool_choice_function_name,\n"
    "                                use_harmony=self.use_harmony,\n"
    "                                harmony_tools_streamed_i=harmony_tools_streamed[i],\n"
    "                                output=output,\n"
    "                                request=request,\n"
    "                                tool_parser=getattr(self, \"tool_parser\", None),\n"
    "                            )\n"
    "                        except Exception:\n"
    "                            # Defensive fallback — replicate upstream verbatim\n"
    "                            # (pristine serving.py:821-828). The pre-0.22 auto\n"
    "                            # tool-call flag no longer exists in the streaming\n"
    "                            # generator — referencing it here would raise a\n"
    "                            # NameError that ESCAPES this defensive wrapper.\n"
    "                            if (tools_streamed[i] and not tool_choice_function_name) or (\n"
    "                                self.use_harmony and harmony_tools_streamed[i]\n"
    "                            ):\n"
    "                                finish_reason_ = \"tool_calls\"\n"
    "                            else:\n"
    "                                finish_reason_ = (\n"
    "                                    output.finish_reason if output.finish_reason else \"stop\"\n"
    "                                )\n"
)


# ─── Sub-patch 1 (dev1060 v3): streaming anchor ────────────────────────
#
# v3 (2026-07-13, dev1060): the harmony OR-clause is gone from the
# streaming condition — anchor is the bare tools_streamed if-block at
# chat_completion/serving.py:694-699 (count==1 verified against
# /home/user/engines/vllm-build). The P107 apply-order chain above is
# v2-only: P107's v3 anchor spans the old harmony block and self-skips
# on dev1060, and this anchor does not include the following
# ``choice_data =`` line, so no new coupling is introduced.
#
# ``auto_tools_called`` still does not exist in the streaming
# generator on dev1060, so (as on v2) the streaming leg is effectively
# pass-through — the middleware's downgrade trigger requires
# ``auto_tools_called=True``. Retained for marker parity, dry-run
# observability, and so a future trigger re-evaluation has the call
# site already wired. ``parser`` (= ``parsers[i]``, assigned at the
# top of the per-output loop) is in scope at the anchor.


PN288_STREAMING_OLD_DEV1060 = (
    "                        if tools_streamed[i] and not tool_choice_function_name:\n"
    '                            finish_reason_ = "tool_calls"\n'
    "                        else:\n"
    "                            finish_reason_ = (\n"
    '                                output.finish_reason if output.finish_reason else "stop"\n'
    "                            )\n"
)


PN288_STREAMING_NEW_DEV1060 = (
    "                        # [Genesis PN288 dev1060 2026-07-13] args-validity-aware\n"
    "                        # finish_reason decision (v3 anchors: dev1060 dropped the\n"
    "                        # harmony OR-clause; parsing flows through the parser\n"
    "                        # engine). auto_tools_called does not exist in the\n"
    "                        # streaming generator, so the downgrade trigger cannot\n"
    "                        # fire on this leg — kept for marker parity and dry-run\n"
    "                        # observability. See\n"
    "                        # sndr.engines.vllm.middleware.pn288_finish_reason_override.\n"
    "                        try:\n"
    "                            from sndr.engines.vllm.middleware.pn288_finish_reason_override import (  # noqa: E501\n"
    "                                decide_streaming_finish_reason as _genesis_pn288_decide_streaming,\n"
    "                            )\n"
    "                            from sndr.engines.vllm.patches.serving.pn288_dev1060_probe import (  # noqa: E501\n"
    "                                probe_from_parser as _genesis_pn288_probe_streaming,\n"
    "                            )\n"
    "                            finish_reason_ = _genesis_pn288_decide_streaming(\n"
    "                                tools_streamed_i=tools_streamed[i],\n"
    "                                tool_choice_function_name=tool_choice_function_name,\n"
    "                                use_harmony=False,\n"
    "                                harmony_tools_streamed_i=False,\n"
    "                                output=output,\n"
    "                                request=request,\n"
    "                                tool_parser=_genesis_pn288_probe_streaming(parser),\n"
    "                            )\n"
    "                        except Exception:\n"
    "                            # Defensive fallback — replicate upstream verbatim\n"
    "                            # (pristine dev1060 chat_completion/serving.py:694-699).\n"
    "                            if tools_streamed[i] and not tool_choice_function_name:\n"
    '                                finish_reason_ = "tool_calls"\n'
    "                            else:\n"
    "                                finish_reason_ = (\n"
    '                                    output.finish_reason if output.finish_reason else "stop"\n'
    "                                )\n"
)


# ─── Sub-patch 2: non-streaming anchor ─────────────────────────────────


PN288_NONSTREAMING_OLD = (
    "            is_finish_reason_tool_calls = auto_tools_called or (\n"
    "                request.tool_choice\n"
    "                and request.tool_choice == \"required\"\n"
    "                and output.finish_reason == \"stop\"\n"
    "            )\n"
)


PN288_NONSTREAMING_NEW = (
    "            # [Genesis PN288] args-validity-aware bool. Phase B dry-run\n"
    "            # default — see sndr.engines.vllm.middleware.pn288_finish_reason_override.\n"
    "            try:\n"
    "                from sndr.engines.vllm.middleware.pn288_finish_reason_override import (  # noqa: E501\n"
    "                    decide_non_streaming_is_tool_calls as _genesis_pn288_decide_non_streaming,\n"
    "                )\n"
    "                is_finish_reason_tool_calls = _genesis_pn288_decide_non_streaming(\n"
    "                    auto_tools_called=auto_tools_called,\n"
    "                    request=request,\n"
    "                    output=output,\n"
    "                    tool_parser=getattr(self, \"tool_parser\", None),\n"
    "                )\n"
    "            except Exception:\n"
    "                # Defensive fallback — replicate upstream verbatim.\n"
    "                is_finish_reason_tool_calls = auto_tools_called or (\n"
    "                    request.tool_choice\n"
    "                    and request.tool_choice == \"required\"\n"
    "                    and output.finish_reason == \"stop\"\n"
    "                )\n"
)


# ─── Sub-patch 2 (dev1060 v3): non-streaming replacement ───────────────
#
# v3 (2026-07-13, dev1060): anchor text is byte-identical to v1/v2
# (``PN288_NONSTREAMING_OLD`` reused), but the replacement feeds the
# middleware a probe built from the parsed ``tool_calls`` FunctionCall
# list — the legacy ``self.tool_parser.prev_tool_call_arr`` surface no
# longer exists on dev1060 (getattr(self, "tool_parser", None) is None
# there, which made v2's call a permanent KEPT_VALID no-op). Both
# ``tool_calls`` and ``parser`` are in scope at the anchor
# (chat_completion/serving.py:878/960 on dev1060).


PN288_NONSTREAMING_NEW_DEV1060 = (
    "            # [Genesis PN288 dev1060 2026-07-13] args-validity-aware bool.\n"
    "            # dev1060 removed the legacy prev_tool_call_arr surface, so the\n"
    "            # middleware receives a probe built from the parser-engine\n"
    "            # FunctionCall list (what the client actually receives).\n"
    "            # Phase B dry-run default — see\n"
    "            # sndr.engines.vllm.middleware.pn288_finish_reason_override.\n"
    "            try:\n"
    "                from sndr.engines.vllm.middleware.pn288_finish_reason_override import (  # noqa: E501\n"
    "                    decide_non_streaming_is_tool_calls as _genesis_pn288_decide_non_streaming,\n"
    "                )\n"
    "                from sndr.engines.vllm.patches.serving.pn288_dev1060_probe import (  # noqa: E501\n"
    "                    probe_from_tool_calls as _genesis_pn288_probe_non_streaming,\n"
    "                )\n"
    "                is_finish_reason_tool_calls = _genesis_pn288_decide_non_streaming(\n"
    "                    auto_tools_called=auto_tools_called,\n"
    "                    request=request,\n"
    "                    output=output,\n"
    "                    tool_parser=_genesis_pn288_probe_non_streaming(tool_calls),\n"
    "                )\n"
    "            except Exception:\n"
    "                # Defensive fallback — replicate upstream verbatim.\n"
    "                is_finish_reason_tool_calls = auto_tools_called or (\n"
    "                    request.tool_choice\n"
    '                    and request.tool_choice == "required"\n'
    '                    and output.finish_reason == "stop"\n'
    "                )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "entrypoints/openai/chat_completion/serving.py"
    )
    if target is None:
        return None

    # [dev1060 port 2026-07-13] Variant sniff: select the anchor set
    # matching the installed pin. The v2 streaming anchor carries the
    # harmony OR-clause; dev1060 dropped it. Historical (v2) anchors are
    # kept as the default so pre-dev1060 pins behave exactly as before;
    # when neither variant matches, the v2 set is returned and apply()
    # reports required_anchor_missing (anchor drift) as usual.
    variant = "v2"
    try:
        with open(target) as f:
            _content = f.read()
    except OSError:
        _content = ""
    if PN288_STREAMING_OLD in _content:
        variant = "v2"
    elif PN288_STREAMING_OLD_DEV1060 in _content:
        variant = "dev1060"

    if variant == "dev1060":
        sub_patches = [
            TextPatch(
                name="pn288_streaming_finish_reason_dev1060",
                anchor=PN288_STREAMING_OLD_DEV1060,
                replacement=PN288_STREAMING_NEW_DEV1060,
                required=True,
            ),
            TextPatch(
                name="pn288_non_streaming_finish_reason_dev1060",
                anchor=PN288_NONSTREAMING_OLD,
                replacement=PN288_NONSTREAMING_NEW_DEV1060,
                required=True,
            ),
        ]
    else:
        sub_patches = [
            TextPatch(
                name="pn288_streaming_finish_reason",
                anchor=PN288_STREAMING_OLD,
                replacement=PN288_STREAMING_NEW,
                required=True,
            ),
            TextPatch(
                name="pn288_non_streaming_finish_reason",
                anchor=PN288_NONSTREAMING_OLD,
                replacement=PN288_NONSTREAMING_NEW,
                required=True,
            ),
        ]

    return TextPatcher(
        patch_name=(
            "PN288 serving.py — tool-call finish_reason override "
            "(Phase B dry-run)"
        ),
        target_file=str(target),
        marker=GENESIS_PN288_MARKER,
        sub_patches=sub_patches,
        upstream_drift_markers=[
            # Self-collision lint (triage plan §6 2026-06-11): former
            # entries "_genesis_pn288_decide_streaming" /
            # "_genesis_pn288_decide_non_streaming" were baked by our own
            # replacements — residue coverage stays with the
            # "[Genesis PN288" banner.
            "[Genesis PN288",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN288 text patch.

    Phase B semantics: when the gate
    ``GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE=1`` is set, both
    anchors are replaced with the middleware-delegating calls. The
    text-patched code itself dispatches dry-run vs actual override at
    request time via the middleware's ``is_dry_run()`` check, which
    reads ``GENESIS_PN288_DRY_RUN`` live on every call.
    """
    from sndr.dispatcher import should_apply, log_decision
    from sndr.engines.vllm.middleware.pn288_finish_reason_override import (
        setup_prometheus_counters,
    )

    decision, reason = should_apply("PN288")
    log_decision("PN288", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", (
            "vllm/entrypoints/openai/chat_completion/serving.py "
            "not found"
        )

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"

    with open(patcher.target_file) as f:
        content = f.read()

    # Idempotency + upstream-drift check.
    if patcher.marker in content:
        pass  # already applied
    else:
        for m in patcher.upstream_drift_markers:
            if m in content:
                return (
                    "skipped",
                    f"upstream drift marker {m!r} already in "
                    f"{patcher.target_file} — PN288 already injected "
                    "or upstream landed an equivalent fix.",
                )
        # Pre-flight: confirm both anchors are present before we start.
        for sub in patcher.sub_patches:
            if sub.anchor not in content:
                return (
                    "skipped",
                    f"required anchor {sub.name!r} not found in "
                    f"{patcher.target_file} — upstream drift; refresh "
                    "anchor against the current pin.",
                )

    result, failure = patcher.apply()
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )

    # Register the Prometheus counter once the patch is in place.
    prom_ready = setup_prometheus_counters()
    prom_note = (
        " + Prometheus counter registered "
        "(vllm:pn288_finish_reason_override_total{model,channel,action})"
        if prom_ready else
        " (prometheus_client unavailable; module-global dict only)"
    )

    return "applied", (
        "PN288 finish_reason override installed at both serving.py "
        "anchors (streaming + non-streaming). Phase B dry-run is "
        "ACTIVE by default; set GENESIS_PN288_DRY_RUN=0 to enable "
        "Phase C behavior change after evidence review."
        + prom_note
    )


def is_applied() -> bool:
    """Filesystem-level marker check — True iff serving.py carries the
    PN288 patch marker. Cheap; used by audit / shadow CLI."""
    target = resolve_vllm_file(
        "entrypoints/openai/chat_completion/serving.py"
    )
    if target is None:
        return False
    try:
        with open(target) as f:
            return GENESIS_PN288_MARKER in f.read()
    except OSError:
        return False
