# SPDX-License-Identifier: Apache-2.0
"""PN525 — drop incomplete tool-call markup in non-streaming (vllm#47562).

================================================================
UPSTREAM BUG (vllm#47562, fixes issue #47137) — CLIENT-VISIBLE GARBAGE
================================================================

``DelegatingParser._extract_tool_calls`` (vllm/parser/abstract_parser.py)
is the shared non-streaming auto-tool-choice path for EVERY engine tool
parser (qwen3_xml on our 35B/27B lanes, gemma4 on the G4 lanes). When
generation terminates INSIDE a ``<tool_call>`` opener (``max_tokens`` or
a ``stop`` string), the engine parser promotes no tool call and returns
an ``ExtractedToolCallInformation`` whose ``content`` has the incomplete
markup already stripped — but the pristine else-branch ignores it and
returns the RAW ``content``:

    else:
        # No tool calls.
        return None, content

so the client receives raw ``"<tool_call>\\n<function"`` garbage, and the
non-streaming result diverges from streaming (which drops the truncated
opener). Tool calls are a first-class validated capability on our lanes
(7/7 gates per lane) — this is a correctness fix, no crash involved.

================================================================
THE FIX — return the parser's cleaned content, raw fallback
================================================================

PR #47562 guards on ``tool_call_info is not None`` and returns
``tool_call_info.content or None``, falling back to the raw ``content``
when the parser returned no result object. PN525 vendors that logic
with a byte-divergent shape (``cleaned if cleaned else None``) and
Genesis-worded comments, so the PR's exact comment head AND its code
line stay usable as SELF_COLLISION-safe drift markers
(tools/lint_drift_markers.py, PN369 contract).

================================================================
SAFETY MODEL
================================================================

  * Branch only runs when NO complete tool call was parsed; the
    tools_called promotion branch is byte-untouched. When the parser's
    content equals the raw content (the overwhelmingly common
    no-tool-markup case) behavior is identical.
  * default_on=True (work-order verdict): stream/non-stream parity is a
    correctness property of the shared path every tool lane uses; the
    env flag remains an operator OFF switch.
  * Same-file hygiene (grep-verified 2026-07-05): PN66 and PN392 anchor
    ``parse_delta`` — a disjoint function; no anchor overlap.
  * Anchor byte-verified count==1 in pristine dev748 (2dfaae752, gh
    api, 2026-07-05: '# No tool calls.' count==1, fix ABSENT).
  * Upper range bound capped <0.24.0 pending the #47562 merge (drift
    markers self-skip earlier if it lands within the window).

================================================================
DEV1060 RE-ANCHOR (2026-07-14; pin 0.23.1rc1.dev1060 club-dev1060-cherry)
================================================================

The historical 3-line else-anchor is count==0 on dev1060: upstream
expanded the else-branch with a required/named empty-content guard
(``return [], None``) BEFORE the raw-content return. That guard is a
DIFFERENT fix — the PN525 bug persists: the closing
``return None, content`` still ignores ``tool_call_info.content`` on
the auto path, so truncated ``<tool_call>`` markup still reaches the
client. The dev1060 sub-patch keys on the guard-tail + raw-return pair
(byte-verified count==1 against the installed dev1060 source; the
20-space ``return [], None`` alone appears 2x, the pair is unique) and
inserts the same cleaned-content preference between them.
``tool_call_info`` is in scope (assigned in the is_auto_tool_choice
arm). Historical + dev1060 anchors are mutually exclusive (the guard
block does not exist on dev748), so the pair can never double-apply.

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#47562 (OPEN as of 2026-07-05); fixes issue #47137.
"""

from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatch,
    TextPatcher,
)

log = logging.getLogger("genesis.wiring.pn525_nonstream_truncated_toolcall_markup")

GENESIS_PN525_MARKER = (
    "Genesis PN525 non-streaming truncated tool-call markup drop "
    "(vendor of vllm#47562) v1"
)

# ── Sub-patch (required): the cleaned-content else branch ─────────────
# Anchor: the 3-line else-branch closing the auto-tool-choice arm of
# _extract_tool_calls. The '# No tool calls.' comment is count==1 in
# pristine dev748 (2dfaae752, gh api 2026-07-05), pinning the block.

PN525_NO_TOOLCALL_OLD = (
    "            else:\n"
    "                # No tool calls.\n"
    "                return None, content\n"
)

PN525_NO_TOOLCALL_NEW = (
    "            else:\n"
    "                # [Genesis PN525 vendor of vllm#47562] No complete tool\n"
    "                # call was promoted. Prefer the parser's cleaned text: it\n"
    "                # drops incomplete tool-call markup (a <tool_call> opener\n"
    "                # truncated by max_tokens or a stop string), so the\n"
    "                # non-streaming result matches streaming (#47137). Fall\n"
    "                # back to the raw content when the parser returned no\n"
    "                # result object (original behavior).\n"
    "                if tool_call_info is not None:\n"
    "                    cleaned = tool_call_info.content\n"
    "                    return None, cleaned if cleaned else None\n"
    "                return None, content\n"
)

# ── Sub-patch (dev1060): guard-tail + raw-return pair ────────────────
# Anchor: the required/named empty-content guard's return (20-space)
# followed by the closing raw-content return (16-space). Byte-verified
# count==1 on installed dev1060 (`return [], None` alone is 2x; the
# pair is unique). Mutually exclusive with the historical anchor.

PN525_NO_TOOLCALL_DEV1060_OLD = (
    "                    return [], None\n"
    "                return None, content\n"
)

PN525_NO_TOOLCALL_DEV1060_NEW = (
    "                    return [], None\n"
    "                # [Genesis PN525 vendor of vllm#47562] No complete\n"
    "                # tool call was promoted. Prefer the parser's cleaned\n"
    "                # text: it drops incomplete tool-call markup (a\n"
    "                # <tool_call> opener truncated by max_tokens or a stop\n"
    "                # string), so the non-streaming result matches\n"
    "                # streaming (#47137). Fall back to the raw content\n"
    "                # when the parser returned no result object.\n"
    "                if tool_call_info is not None:\n"
    "                    cleaned = tool_call_info.content\n"
    "                    return None, cleaned if cleaned else None\n"
    "                return None, content\n"
)

# Drift markers — #47562's exact comment head and code line (from
# `gh pr diff 47562`, 2026-07-05). Byte-verified absent in pristine
# dev748 (count 0). Our replacement rewords the comment and uses the
# byte-divergent `cleaned if cleaned else None` shape, so neither marker
# appears in our emitted text — SELF_COLLISION-safe (PN369).
_DRIFT_MARKERS = (
    "# No complete tool calls: return the tool parser's content,\n",
    "return None, tool_call_info.content or None\n",
    # Defended convention entry (our own banner).
    "[Genesis PN525",
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("parser/abstract_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN525 parser/abstract_parser.py — drop incomplete tool-call "
            "markup in non-streaming (vendor of vllm#47562)"
        ),
        target_file=str(target),
        marker=GENESIS_PN525_MARKER,
        sub_patches=[
            # Historical anchor (dev748 shape). dev1060 re-anchor
            # (2026-07-14): count==0 there (else-branch expanded with the
            # required/named guard), so it is now required=False and
            # soft-skips; on the historical pin it still lands. Exactly
            # one of the pair matches per pin (the guard block does not
            # exist pre-dev1060), so they can never double-apply.
            TextPatch(
                name="pn525_no_toolcall_cleaned_content",
                anchor=PN525_NO_TOOLCALL_OLD,
                replacement=PN525_NO_TOOLCALL_NEW,
                required=False,
            ),
            TextPatch(
                name="pn525_no_toolcall_cleaned_content_dev1060",
                anchor=PN525_NO_TOOLCALL_DEV1060_OLD,
                replacement=PN525_NO_TOOLCALL_DEV1060_NEW,
                required=False,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_MARKERS),
    )


def apply() -> tuple[str, str]:  # noqa: PLR0911 - dispatcher early-return cascade: distinct skip/self-retire reasons per gate
    """Apply PN525 — non-streaming truncated tool-call markup drop.

    Gated through the dispatcher on
    ``GENESIS_ENABLE_PN525_NONSTREAM_TOOLCALL_MARKUP_DROP``
    (default_on=True — stream/non-stream parity on the shared tool-call
    path; the flag is an operator OFF switch). Never raises.
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN525")
    log_decision("PN525", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/parser/abstract_parser.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file, encoding="utf-8") as f:
        content = f.read()
    if patcher.marker in content:
        return "skipped", f"{patcher.patch_name}: already applied (marker present)"
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} present — upstream PR "
                "#47562 (or equivalent fix) appears merged (upstream_merged)",
            )

    try:
        result, failure = patcher.apply()
    except Exception as e:  # noqa: BLE001 - dispatcher contract: never raise
        return "failed", f"PN525 apply raised {e!r}"

    from sndr.kernel import TextPatchResult

    if result == TextPatchResult.FAILED:
        return "failed", f"PN525: {failure.reason if failure else 'unknown'}"
    if result == TextPatchResult.SKIPPED:
        return "skipped", f"PN525: {failure.reason if failure else 'unknown'}"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN525 already applied (idempotent)"

    return (
        "applied",
        "PN525 applied: DelegatingParser._extract_tool_calls now returns "
        "the tool parser's cleaned content when no complete tool call was "
        "promoted, so a <tool_call> opener truncated by max_tokens/stop is "
        "dropped in non-streaming exactly like streaming (vllm#47562 / "
        "#47137). Raw-content fallback preserved when the parser returns "
        "no result object.",
    )


def is_applied() -> bool:
    """Return True iff our marker is present in the target file."""
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file, encoding="utf-8") as f:
            return patcher.marker in f.read()
    except OSError:
        return False
