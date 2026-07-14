# SPDX-License-Identifier: Apache-2.0
"""PN385 — forced-named empty-params tool schema → JSON object (vendor of vllm#45290).

Upstream bug (vllm#45290, "Constrain forced named tool choice with empty
parameters to a JSON object", OPEN as of 2026-06-13): the PUBLIC
``get_json_schema_from_tools`` in ``vllm/tool_parsers/utils.py`` has two
forced-named branches that return the chosen tool's ``parameters``
verbatim:

    # tool_choice: Forced Function (Responses)
    return tool_map[tool_name].parameters
    # tool_choice: Forced Function (ChatCompletion)
    return tool_map[tool_name].function.parameters

For a no-arg tool (``end_turn`` / ``noop`` / ``handoff`` — the agent
control verbs our loops lean on) that value is:

  * ``None`` — no guided decoding at all, so the generated arguments are
    FREE-FORM text; or
  * ``{}`` — an UNCONSTRAINED schema, so the arguments may be any JSON
    value (a bare string/number), not necessarily an object.

Either way the model can emit ``"bytes"`` / ``"synced"`` / ``42`` as the
tool arguments instead of ``{}``. The upstream author reproduced exactly
this end-to-end on a real engine (A800, ``vllm serve Qwen3-4B``): the
unconstrained branch returned a bare JSON string as ``function.arguments``
on BOTH the ``/v1/chat/completions`` and ``/v1/responses`` paths.

LIVE EXPOSURE on our stack: this is an agent-loop parse-500 on
parameterless tools for 3 of our 4 PROD families — qwen3_xml (35B FP8,
27B int4) and gemma4 (26B-A4B AWQ MoE, 31B AWQ dense) — whose downstream
object-shaped argument parser cannot consume a bare string/number.
qwen3_coder is shielded (its own extraction path tolerates the
non-object shape). The ``tool_choice="required"`` path is already immune:
``_get_tool_schema_from_tool`` normalizes the same way —

    params = params if params else {"type": "object", "properties": {}}

PN385 applies that SAME normalization to BOTH forced-named branches so a
no-arg forced tool yields a JSON object (``{"type": "object",
"properties": {}}``) instead of leaving the arguments unconstrained.

================================================================
DEV1060 RE-ANCHOR (2026-07-13; pin 0.23.1rc1.dev1060 club-dev1060-cherry)
================================================================

dev1060 refactored ``tool_parsers/utils.py`` around the namespace-tool
helper family (``_extract_tool_info`` /
``_get_tool_schema_from_name_and_params`` / ``iter_response_function_``
``tool_info``). The two historical forced-named anchors are GONE
(count==0 on dev1060):

  * The Responses branch now builds a ``responses_tool_map`` via
    ``iter_response_function_tool_info`` (with a ``name.rsplit("__",``
    ``1)[1]`` flat-name alias for namespaced tools) and returns
    ``responses_tool_map[tool_name]``.
  * The ChatCompletion branch now builds a ``chat_tool_map`` and
    returns ``chat_tool_map[tool_name].function.parameters``.

THE BUG IS **NOT** FIXED UPSTREAM on dev1060 — byte-verified against
the installed source (/home/user/engines/vllm-build):

  * Both branches still return the chosen tool's parameters VERBATIM
    (utils.py lines 402 / 415) — ``None`` or ``{}`` flows out raw.
  * The consumer (``abstract_tool_parser.py``,
    ``adjust_request``: ``if json_schema_from_tool is not None``) still
    drops guided decoding entirely on ``None`` and installs an
    unconstrained schema on ``{}`` — the exact vllm#45290 semantic.
  * Only the ``required`` path normalizes, now via the factored helper
    ``_get_tool_schema_from_name_and_params``:
    ``params = params if params else {"type": "object", "properties":``
    `` {}}`` — the SAME idiom our replacement inlines.
  * ``_params_or_empty_object`` (the #45290 helper) is absent
    (count==0), so the drift markers stay valid merge detectors; the
    two historical call-site markers reference the pre-refactor
    ``tool_map`` spelling and can no longer fire on this tree, but the
    helper-def marker still covers the merge case.

Fix shape: the historical sub-patches are kept (required=False, inert
on dev1060) and two ``_dev1060``-suffixed sub-patches key on the
refactored return lines (byte-verified count==1 each on dev1060),
inlining the identical empty-object normalization. We still do NOT
adopt the #45290 ``_params_or_empty_object`` helper — same two reasons
as the original Genesis divergence below, now with
``_get_tool_schema_from_name_and_params`` line 316 as the in-file
idiom source.

================================================================
WHY A TEXT-OVERLAY (NOT A MONKEY-PATCH)
================================================================

The two branches are local ``return`` statements inside the public
``get_json_schema_from_tools`` — there is no clean rebind seam for just
those two lines without re-defining the whole ~30-line dispatcher (which
drifts between pins). A surgical anchor→replacement on each ``return``
line is smaller, verifiable, and self-skips when upstream merges #45290.

This is the public-path sibling of PN70, which wraps the DISJOINT
INTERNAL ``_get_json_schema_from_tools`` (the ``required`` combined-anyOf
builder). PN385 touches only the two PUBLIC forced-named ``return``
lines; PN70 wraps a different function; they share the file but not a
single anchor. PN385 is likewise disjoint from P68 (the auto→required
``tool_choice`` upgrade gate, which lives in the middleware and never
touches this file's schema-emission lines). Verified: no anchor
collision with either.

================================================================
GENESIS DIVERGENCE (documented per iron rule #10)
================================================================

Upstream #45290 factors the literal into a shared helper
``_params_or_empty_object(params)`` and calls it in three places. We do
NOT introduce that helper: our replacement inlines the exact same
expression the in-pin ``required`` path already uses verbatim —
``params if params else {"type": "object", "properties": {}}`` — at each
of the two forced-named branches. Two reasons:

  1. Adding a module-level def would require a THIRD anchor (after the
     existing ``_get_tool_schema_from_tool``) and enlarge the drift
     surface for zero behavioral gain — the inline form is byte-for-byte
     the established in-file idiom.
  2. Self-collision contract (PN369 rule / tools/lint_drift_markers.py):
     the upstream drift markers below are the PR's exact
     ``_params_or_empty_object(...)`` call sites. By NOT emitting that
     helper name anywhere in our replacement text, the markers stay
     usable as upstream-merge detectors without ever matching our own
     output. (Verified: linter stays at 0 self-collisions.)

We deliberately do NOT tighten ``additionalProperties: false`` here even
though the roadmap floats it as optional: the in-pin ``required`` path
emits the loose empty-object form, and matching it byte-for-byte keeps
PN385 a pure parity fix (the empty-properties object already rejects
non-object arguments, which is the entire bug). A stricter grammar would
be a separate, independently-benched change.

================================================================
ACTIVATION
================================================================

Opt-in via ``GENESIS_ENABLE_PN385_FORCED_NAMED_EMPTY_PARAMS=1``
(``default_on=False`` in the registry — candidate for default-ON after a
fleet test on the three exposed families). STRONG RECOMMENDATION: enable
this on any deployment that serves parameterless agent tools
(``end_turn`` / ``noop`` / ``handoff``) under forced ``tool_choice`` with
qwen3_xml or gemma4 — without it those calls intermittently 500 the
agent loop. Self-skips when #45290 lands upstream (the drift markers
below are exact substrings of the PR's merged form).

Anchors byte-verified (count==1 each) against the pristine pin
``0.22.1rc1.dev259+g303916e93`` at
``/private/tmp/candidate_pin_current/vllm/tool_parsers/utils.py``.

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Vendor target: vllm-project/vllm#45290 (OPEN as of 2026-06-13).
"""
from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, result_to_wiring_status

log = logging.getLogger("genesis.wiring.pn385_forced_named_empty_params")

GENESIS_PN385_MARKER = (
    "Genesis PN385 forced-named empty-params -> JSON object "
    "(vendor of vllm#45290) v1"
)

_TARGET_REL = "tool_parsers/utils.py"

# The exact empty-object normalization already used by the in-pin
# `required` path (`_get_tool_schema_from_tool`). We inline this verbatim
# at each forced-named branch so the three schema-emission sites agree.
_EMPTY_OBJECT_EXPR = 'if params else {"type": "object", "properties": {}}'

# Drift markers — exact substrings of #45290's merged form, taken from
# `gh pr diff 45290` on 2026-06-13. Absent in the pristine pin tree
# (g303916e93: all count 0) and deliberately NOT substrings of our own
# replacement text: we never emit the `_params_or_empty_object` helper
# name. When upstream merges #45290 these markers appear in the file and
# PN385 self-skips (lint_drift_markers self-collision contract).
#
# dev1060 note (2026-07-13): all three markers remain count==0 on
# dev1060. The two call-site markers use the pre-refactor `tool_map`
# spelling, so on dev1060-shaped trees only the helper-def marker can
# fire on an upstream merge — sufficient (the helper def is the one
# part of #45290 a rebase cannot respell).
_DRIFT_MARKERS = (
    # The PR's shared helper definition head.
    "def _params_or_empty_object(params: dict | None) -> dict:",
    # The PR's Responses-branch call site.
    "return _params_or_empty_object(tool_map[tool_name].parameters)",
    # The PR's ChatCompletion-branch call site.
    "return _params_or_empty_object(tool_map[tool_name].function.parameters)",
)

# ── Sub-patch 1 (required): Responses forced-named branch ─────────────
# Anchor: the bare `return tool_map[tool_name].parameters` of the
# `# tool_choice: Forced Function (Responses)` block. Unique in the file
# (count==1 byte-verified vs the pristine pin); the ChatCompletion branch
# returns `.function.parameters`, so this `.parameters`-only line is
# distinct.

PN385_RESPONSES_OLD = (
    "        return tool_map[tool_name].parameters\n"
)

PN385_RESPONSES_NEW = (
    # [Genesis PN385 vendor of vllm#45290] A forced-named Responses tool
    # with falsey parameters (None = no guided decoding, {} =
    # unconstrained) otherwise lets the model emit a bare string/number
    # as arguments. Normalize to an empty JSON object exactly like the
    # in-pin `required` path (_get_tool_schema_from_tool) so the
    # arguments are constrained to an object even for no-arg tools.
    "        # [Genesis PN385 vendor of vllm#45290] Constrain a no-arg\n"
    "        # forced-named tool to a JSON object (parity with the\n"
    "        # `required` path) instead of leaving arguments free-form.\n"
    "        params = tool_map[tool_name].parameters\n"
    "        return params " + _EMPTY_OBJECT_EXPR + "\n"
)

# ── Sub-patch 2 (required): ChatCompletion forced-named branch ────────
# Anchor: the bare `return tool_map[tool_name].function.parameters` of
# the `# tool_choice: Forced Function (ChatCompletion)` block. Unique in
# the file (count==1 byte-verified vs the pristine pin).

PN385_CHAT_OLD = (
    "        return tool_map[tool_name].function.parameters\n"
)

PN385_CHAT_NEW = (
    "        # [Genesis PN385 vendor of vllm#45290] Constrain a no-arg\n"
    "        # forced-named tool to a JSON object (parity with the\n"
    "        # `required` path) instead of leaving arguments free-form.\n"
    "        params = tool_map[tool_name].function.parameters\n"
    "        return params " + _EMPTY_OBJECT_EXPR + "\n"
)

# ─────────────────────────────────────────────────────────────────────
# DEV1060 RE-ANCHOR (2026-07-13). The two historical anchors above are
# gone on 0.23.1rc1.dev1060 (count==0): the forced-named branches were
# rebuilt around the namespace-tool helper family (_extract_tool_info /
# _get_tool_schema_from_name_and_params / iter_response_function_tool_
# info). The bug itself PERSISTS — both branches still return the
# chosen tool's parameters verbatim (None / {} escape unnormalized) and
# only the `required` path normalizes, now via
# _get_tool_schema_from_name_and_params:
#     params = params if params else {"type": "object", "properties": {}}
# The dev1060 sub-patches below key on the refactored return lines
# (byte-verified count==1 each against the installed dev1060 source)
# and inline that same idiom, exactly like the historical pair.
# ─────────────────────────────────────────────────────────────────────

# ── Sub-patch 1' (dev1060): Responses forced-named branch ─────────────
# Anchor: the bare `return responses_tool_map[tool_name]` that closes
# the `# tool_choice: Forced Function (Responses)` block. The map value
# is the raw `params` yielded by iter_response_function_tool_info
# (dict | None) — returned verbatim, so None/{} still escape. Unique in
# the file: the only other `responses_tool_map[...]` uses are the map
# build and the membership check.

PN385_RESPONSES_DEV1060_OLD = (
    "        return responses_tool_map[tool_name]\n"
)

PN385_RESPONSES_DEV1060_NEW = (
    "        # [Genesis PN385 vendor of vllm#45290] Constrain a no-arg\n"
    "        # forced-named tool to a JSON object (parity with the\n"
    "        # `required` path) instead of leaving arguments free-form.\n"
    "        params = responses_tool_map[tool_name]\n"
    "        return params " + _EMPTY_OBJECT_EXPR + "\n"
)

# ── Sub-patch 2' (dev1060): ChatCompletion forced-named branch ────────
# Anchor: the bare `return chat_tool_map[tool_name].function.parameters`
# that closes the `# tool_choice: Forced Function (ChatCompletion)`
# block. Unique in the file (the historical `tool_map` spelling is gone;
# `chat_tool_map` appears only in this branch).

PN385_CHAT_DEV1060_OLD = (
    "        return chat_tool_map[tool_name].function.parameters\n"
)

PN385_CHAT_DEV1060_NEW = (
    "        # [Genesis PN385 vendor of vllm#45290] Constrain a no-arg\n"
    "        # forced-named tool to a JSON object (parity with the\n"
    "        # `required` path) instead of leaving arguments free-form.\n"
    "        params = chat_tool_map[tool_name].function.parameters\n"
    "        return params " + _EMPTY_OBJECT_EXPR + "\n"
)


def build_patcher_for_target(target_file: str) -> TextPatcher:
    """Construct the PN385 TextPatcher against an explicit target file.

    Factored out so tests can drive the patch against a temp copy of the
    pristine source without the dispatcher / install-root resolution. The
    production builder ``_make_patcher`` calls this after resolving the
    live vllm path.
    """
    return TextPatcher(
        patch_name=(
            "PN385 tool_parsers/utils.py — forced-named empty-params -> "
            "JSON object (vendor of vllm#45290)"
        ),
        target_file=target_file,
        marker=GENESIS_PN385_MARKER,
        sub_patches=[
            # Historical pair (pin g303916e93 shape). dev1060 re-anchor
            # (2026-07-13): these anchors are count==0 on dev1060 (the
            # forced-named branches were rebuilt around the namespace-
            # tool helper family), so they are now required=False and
            # soft-skip there; on the historical pin they still land.
            # Exactly one pair matches per pin — the anchor spellings
            # (`tool_map` vs `responses_tool_map`/`chat_tool_map`) are
            # mutually exclusive, so the pairs can never double-apply.
            TextPatch(
                name="pn385_responses_forced_named",
                anchor=PN385_RESPONSES_OLD,
                replacement=PN385_RESPONSES_NEW,
                required=False,
            ),
            TextPatch(
                name="pn385_chatcompletion_forced_named",
                anchor=PN385_CHAT_OLD,
                replacement=PN385_CHAT_NEW,
                required=False,
            ),
            # dev1060 pair (2026-07-13): refactored return lines,
            # byte-verified count==1 each on the installed dev1060
            # source. Same normalization semantic as the pair above.
            TextPatch(
                name="pn385_responses_forced_named_dev1060",
                anchor=PN385_RESPONSES_DEV1060_OLD,
                replacement=PN385_RESPONSES_DEV1060_NEW,
                required=False,
            ),
            TextPatch(
                name="pn385_chatcompletion_forced_named_dev1060",
                anchor=PN385_CHAT_DEV1060_OLD,
                replacement=PN385_CHAT_DEV1060_NEW,
                required=False,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_MARKERS),
    )


def _make_patcher() -> TextPatcher | None:
    """Production builder — resolve the live vllm target then delegate.

    Returns None when the target file is absent on this pin (the wiring
    then reports a clean skip). Name follows the ``_make*patcher``
    convention so tools/lint_drift_markers.py + pin_preflight discover it.
    """
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return None
    return build_patcher_for_target(str(target))


def apply() -> tuple[str, str]:
    """Apply PN385 — forced-named empty-params -> JSON object. Never raises.

    Opt-in: gated through the dispatcher on
    ``GENESIS_ENABLE_PN385_FORCED_NAMED_EMPTY_PARAMS`` (default_on=False
    in the registry — candidate default-ON after the fleet test on the
    three exposed PROD families).
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN385")
    log_decision("PN385", decision, reason)
    if not decision:
        return "skipped", reason

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", f"PN385: target file {_TARGET_REL} not found"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN385 applied: both forced-named branches of the public "
            "get_json_schema_from_tools now normalize falsey parameters "
            "(None / {}) to a JSON object {\"type\": \"object\", "
            "\"properties\": {}} — parity with the in-pin `required` "
            "path. A no-arg forced tool (end_turn/noop/handoff) can no "
            "longer yield a bare string/number as arguments, killing the "
            "agent-loop parse-500 on parameterless tools for qwen3_xml "
            "and gemma4 (vllm#45290). Disjoint from PN70 (internal "
            "_get_json_schema_from_tools) and P68 (auto->required gate)."
        ),
        patch_name=patcher.patch_name,
    )


def is_applied() -> bool:
    """Return True iff our marker is present in the target file."""
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file, encoding="utf-8") as f:
            return patcher.marker in f.read()
    except (OSError, UnicodeDecodeError):
        return False
