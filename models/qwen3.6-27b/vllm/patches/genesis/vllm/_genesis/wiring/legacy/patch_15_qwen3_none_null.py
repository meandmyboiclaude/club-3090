# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 15 — Qwen3 chat-template `None` vs `null` tool-call fix.

Problem
-------
Qwen3.5+ chat templates use Jinja's `| string` filter for scalar tool-call
arguments, which produces Python's `repr()` form `None` instead of the JSON
literal `null`. The qwen3coder tool parser only recognises lowercase `null`,
so a `None` slips through as the literal string `"None"` and breaks any
tool with a nullable parameter.

Reference: vLLM PR [#38996](https://github.com/vllm-project/vllm/pull/38996)
            issue [#38885](https://github.com/vllm-project/vllm/issues/38885).

Fix
---
Accept both `null` and `none` (case-insensitive) in `_convert_param_value`:

    # before:
    if param_value.lower() == "null":
        return None
    # after:
    if param_value.lower() in ("null", "none"):
        return None

Platform compatibility: vendor-agnostic — pure Python parser logic.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p15_qwen3_none_null")

GENESIS_P15_MARKER = "Genesis P15 Qwen3 None/null tool arg v7.0"

UPSTREAM_DRIFT_MARKERS = [
    # If PR #38996 merges, the file will contain the multi-value tuple.
    '("null", "none")',
    "'null', 'none'",
]


# 2026-07-14 re-anchor (dev1060cherry): upstream deleted
# `tool_parsers/qwen3coder_tool_parser.py` (engine-adapter rewrite). Arg
# coercion moved to the shared `tool_parsers/utils.py::coerce_to_schema_type`
# state-machine helper, whose null branch (live L736-739) still accepts ONLY
# JSON "null" — dropping the Jinja `| string` repr "None". Same fix, new site.
# NOTE: utils.py is a SHARED coercion helper used by every adapter parser, so
# the widening is global (harmless — identical semantics) and schema-gated
# (fires only when the param's type set includes `null`).
_OLD = (
    '        if candidate_type == "null":\n'
    '            if value.lower() == "null":\n'
    "                return None\n"
    "            continue"
)

_NEW = (
    '        if candidate_type == "null":\n'
    "            # [Genesis P15] Qwen3.5+ chat template emits Python repr 'None'\n"
    "            # (Jinja `| string`) instead of JSON 'null'. Accept both\n"
    '            # case-insensitively (was: only "null"). Ref vLLM PR #38996.\n'
    '            if value.lower() in ("null", "none"):\n'
    "                return None\n"
    "            continue"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("tool_parsers/utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P15 Qwen3 None/null tool arg",
        target_file=target,
        marker=GENESIS_P15_MARKER,
        sub_patches=[
            TextPatch(
                name="p15_none_null",
                anchor=_OLD,
                replacement=_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    """Apply P15 wiring. Never raises."""
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "tool_parsers/utils.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "None/none mapping added to tool param parser"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
