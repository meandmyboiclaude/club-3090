#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""REVIEW-genesis-dispatcher-20260727 H1 / H2 / M1 regression test.

H1 — lane 1 had no lifecycle gate: `lifecycle: retired` was decorative, so a
     stale GENESIS_ENABLE_* carried across a pin bump re-armed six retired
     patches on the 07-27 boot (P60b actually rewrote two files). Lane 2 has
     the gate (sndr/dispatcher/decision.py `_check_lifecycle_gate`, GAP4).

H2 — the boot summary's "✓ APPLIED (n)" table printed should_apply()'s INTENT.
     12 of its 42 rows were false on that boot, including two anchor-drift
     cases the same log flagged at WARNING three lines earlier.

M1 — `validate_apply_plan` checked requires_patches against those same
     decisions, so `P67b requires P67` read "clean" on a boot where P67
     announced APPLY and then DRIFT-skipped.

Pure stdlib, no pytest, no vllm import (the `vllm` package here is the Genesis
namespace tree). Run:  python3 test_retired_gate_and_truthful_summary.py
"""
from __future__ import annotations

import os
import pathlib
import sys

# _genesis package root (…/genesis/vllm/_genesis) — tests/ sits directly under it.
_GEN = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_GEN.parent.parent))  # …/genesis  → import vllm._genesis

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


# The six lane-1 ids the review caught armed-while-retired on the 07-27 boot,
# with the env flag that armed each one.
RETIRED_ARMED = {
    "P59": "GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY",
    "P60": "GENESIS_ENABLE_P60_GDN_NGRAM_FIX",
    "P60b": "GENESIS_ENABLE_P60B_TRITON_KERNEL",
    "P62": "GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING",
    "P67": "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL",
    "P87": "GENESIS_ENABLE_P87",
}

SKIP_LINE = "retired — pass GENESIS_ALLOW_RETIRED to engage"


# ── H1: the lifecycle gate ────────────────────────────────────────────────

def test_retired_gate(disp) -> None:
    print("\nH1 — lifecycle gate on lane 1")

    # Arrange: exactly the live container's state — every flag set, no override.
    for flag in RETIRED_ARMED.values():
        os.environ[flag] = "1"
    os.environ.pop("GENESIS_ALLOW_RETIRED", None)

    for pid in RETIRED_ARMED:
        decision, reason = disp.should_apply(pid)
        check(decision is False,
              f"{pid}: retired + flag=1 → SKIP (was APPLY before the gate)")
        check(SKIP_LINE in reason,
              f"{pid}: skip reason names the override ({reason[:48]!r}…)")
        check(reason.lower().startswith("lifecycle:"),
              f"{pid}: reason is LIFECYCLE-classed, not 'opt-in env (neutral)'")

    # The escape hatch must still work — it is the diagnostic path.
    os.environ["GENESIS_ALLOW_RETIRED"] = "1"
    engaged = [pid for pid in RETIRED_ARMED if disp.should_apply(pid)[0]]
    check(sorted(engaged) == sorted(RETIRED_ARMED),
          f"GENESIS_ALLOW_RETIRED=1 re-engages all six ({len(engaged)}/6)")
    os.environ.pop("GENESIS_ALLOW_RETIRED", None)

    # A non-retired opt-in patch armed by env must be untouched by the gate.
    os.environ["GENESIS_ENABLE_P71_BLOCK_VERIFY"] = "1"
    decision, reason = disp.should_apply("P71")
    check(decision is True and "lifecycle" not in reason.lower(),
          "P71 (stable, opt-in, flag=1) still APPLYs — gate is retirement-only")
    os.environ.pop("GENESIS_ENABLE_P71_BLOCK_VERIFY", None)

    # Synthetic: the gate must key on lifecycle, not on an id allowlist.
    saved = dict(disp.PATCH_REGISTRY)
    try:
        disp.PATCH_REGISTRY["ZZ99"] = {
            "title": "synthetic retired probe",
            "lifecycle": "retired",
            "env_flag": "GENESIS_ENABLE_ZZ99",
            "default_on": True,          # default_on must not bypass the gate
            "category": "stability",
        }
        os.environ.pop("GENESIS_ENABLE_ZZ99", None)
        decision, reason = disp.should_apply("ZZ99")
        check(decision is False and SKIP_LINE in reason,
              "synthetic retired + default_on=True → SKIP (default_on cannot bypass)")
        disp.PATCH_REGISTRY["ZZ99"]["lifecycle"] = "stable"
        decision, _ = disp.should_apply("ZZ99")
        check(decision is True,
              "same entry flipped to lifecycle=stable → APPLY (gate is the only change)")
    finally:
        disp.PATCH_REGISTRY.clear()
        disp.PATCH_REGISTRY.update(saved)

    for flag in RETIRED_ARMED.values():
        os.environ.pop(flag, None)


# ── H1 trap: P67 gated to SKIP must not break P67b ────────────────────────

def test_p67b_dependency(disp) -> None:
    print("\nH1 trap — P67b requires P67, and P67 is now a permanent SKIP")

    os.environ["GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL"] = "1"
    os.environ.pop("GENESIS_ALLOW_RETIRED", None)
    try:
        p67, _ = disp.should_apply("P67")
        p67b, _ = disp.should_apply("P67b")
        check(p67 is False, "P67 (retired kernel patch) SKIPs")
        check(p67b is True,
              "P67b (the live half) still APPLYs via its env_flag_alias")

        meta = disp.PATCH_REGISTRY["P67b"]
        check("P67" in disp._coerce_list(
                  meta.get("requires_satisfied_by_retirement")),
              "P67b declares requires_satisfied_by_retirement: ['P67']")

        issues = disp.validate_apply_plan({"P67b"}, outcomes={})
        errors = [i for i in issues if i.severity == "ERROR"]
        check(not errors,
              "validate_apply_plan({P67b}) emits no ERROR "
              f"(got {[i.message[:40] for i in errors]})")
        check(not [i for i in issues if i.patch_id == "P67b"],
              "…and no WARNING either — the edge is explicitly acknowledged")
    finally:
        os.environ.pop("GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL", None)


# ── H2: truthful boot summary ─────────────────────────────────────────────

def test_truthful_summary(disp) -> None:
    print("\nH2 — the boot summary reports outcomes, not intent")

    saved_dec = list(disp._DECISIONS)
    saved_out = disp.get_outcomes()
    try:
        disp._DECISIONS.clear()
        disp.clear_outcomes()

        # Four APPLY decisions, mirroring the 07-27 boot shape.
        disp.log_decision("PN59", True, "opt-in env (config: neutral)")
        disp.log_decision("PN25", True, "opt-in env (config: neutral)")
        disp.log_decision("PN52", True, "opt-in env (config: neutral)")
        disp.log_decision("PN70", True, "opt-in env (config: neutral)")
        disp.log_decision("PN58", False, "opt-in only — set GENESIS_ENABLE_PN58=1")

        # With NO outcomes reported the summary must degrade to the old
        # decision view rather than silently emptying the table.
        out = disp.dump_structured_boot_summary()
        check("✓ APPLIED (4)" in out,
              "no outcome data → summary falls back to the decision view")

        # Now the outcomes, exactly as apply_all reports them.
        disp.log_outcome("PN59 spec reasoning boundary", "applied",
                         "text patch applied")
        disp.log_outcome("PN25 fork-safe registration", "skipped",
                         "required anchor 'def _init_workers' not found")
        disp.log_outcome("PN52 prompt-logprobs eviction", "skipped",
                         "qwen3_reasoning_parser.py not found")
        # PN70 reports nothing at all.

        out = disp.dump_structured_boot_summary()
        check("✓ APPLIED (1)" in out,
              "APPLIED table counts only outcome=applied (1 of 4 announced)")
        check("ANNOUNCED-BUT-NOT-APPLIED (2)" in out,
              "drifted + refused land in their own bucket")
        check("ANNOUNCED-NO-OUTCOME (1)" in out,
              "an announced patch that never reported is not counted as applied")
        check("Anchor drift" in out,
              "the drift sub-bucket is labelled as anchor drift")
        check("Outcomes: 1 applied" in out and "1 drift-skipped" in out,
              "counter line carries the outcome breakdown")

        # The specific lie the review measured: a drifted id must not appear
        # under ✓ APPLIED.
        applied_block = out.split("✓ APPLIED")[1].split("ANNOUNCED")[0]
        check("PN25" not in applied_block,
              "the drift-skipped id is absent from the ✓ APPLIED block")
        check("PN59" in applied_block,
              "the genuinely-applied id is present in the ✓ APPLIED block")

        # Long registration names must resolve to registry ids.
        check(disp.resolve_outcome_ids("PN16 Lazy-reasoner request hook")
              == ["PN16"],
              "resolve_outcome_ids: long name → leading id")
        # Combined registrations ("P32/P33 …") map to every half the registry
        # knows; P33 has no registry row, so only P32 is credited.
        check(disp.resolve_outcome_ids(
                  "P32/P33 TurboQuant cu_2 + synth_seq_lens preallocs")
              == ["P32"],
              "resolve_outcome_ids: combined registration → the known half")
        disp.PATCH_REGISTRY["P33"] = {"title": "synthetic P33"}
        try:
            check(sorted(disp.resolve_outcome_ids(
                      "P32/P33 TurboQuant cu_2 + synth_seq_lens preallocs"))
                  == ["P32", "P33"],
                  "resolve_outcome_ids: both halves credited when both exist")
        finally:
            disp.PATCH_REGISTRY.pop("P33", None)

        # Effective applied set = what actually landed.
        eff = disp.get_effective_applied_set()
        check(eff == {"PN59", "PN70"},
              f"get_effective_applied_set() drops drifted/refused (got {sorted(eff)})")
    finally:
        disp._DECISIONS.clear()
        disp._DECISIONS.extend(saved_dec)
        disp.clear_outcomes()
        for pid, rec in saved_out.items():
            disp._OUTCOMES[pid] = rec


# ── M1: the dependency validator reads outcomes ───────────────────────────

def test_validator_reads_outcomes(disp) -> None:
    print("\nM1 — validate_apply_plan evaluates requires_patches on OUTCOMES")

    registry = {
        "AA1": {"title": "dependency", "lifecycle": "stable"},
        "AA2": {"title": "dependent", "requires_patches": ["AA1"]},
        "AA3": {"title": "retired dependency", "lifecycle": "retired"},
        "AA4": {"title": "depends on retired", "requires_patches": ["AA3"]},
        "AA5": {"title": "acknowledged", "requires_patches": ["AA3"],
                "requires_satisfied_by_retirement": ["AA3"]},
    }

    # (a) Both landed → clean. Regression guard for the happy path.
    issues = disp.validate_apply_plan(
        {"AA1", "AA2"}, registry=registry,
        outcomes={"AA1": {"status": "applied", "reason": "ok"}},
    )
    check(not issues, f"AA1 applied + AA2 applied → clean (got {issues})")

    # (b) THE M1 CASE — AA1 announced APPLY, then drift-skipped. Old code fed
    # the decision set, AA1 stayed in it, and the validator read clean.
    issues = disp.validate_apply_plan(
        {"AA2"}, registry=registry,
        outcomes={"AA1": {"status": "drift",
                          "reason": "required anchor 'def foo' not found"}},
    )
    check(len(issues) == 1, f"drift-skipped dependency produces one issue ({issues})")
    check(issues and issues[0].severity == "WARNING",
          "…at WARNING (the boot is degraded, not misconfigured)")
    check(issues and "drift" in issues[0].message,
          "…and the message names the drift, not just 'currently SKIP'")
    check(issues and "announced APPLY" in issues[0].message,
          "…and says the dependency announced APPLY")

    # (c) No outcome at all for a genuinely-absent dependency → ERROR, as before.
    issues = disp.validate_apply_plan({"AA2"}, registry=registry, outcomes={})
    check(len(issues) == 1 and issues[0].severity == "ERROR",
          f"never-enabled dependency still ERRORs ({issues})")

    # (d) Unacknowledged requires → retired: WARNING with a fix hint, not a
    # per-boot ERROR nobody can action.
    issues = disp.validate_apply_plan({"AA4"}, registry=registry, outcomes={})
    check(len(issues) == 1 and issues[0].severity == "WARNING",
          f"requires→retired without acknowledgement warns ({issues})")
    check(issues and "requires_satisfied_by_retirement" in issues[0].message,
          "…and the warning names the acknowledgement field")

    # (e) Acknowledged → silent.
    issues = disp.validate_apply_plan({"AA5"}, registry=registry, outcomes={})
    check(not issues, f"acknowledged requires→retired is silent ({issues})")

    # (f) Conflicts are unchanged.
    conflict_registry = {
        "BB1": {"title": "x", "conflicts_with": ["BB2"]},
        "BB2": {"title": "y", "conflicts_with": ["BB1"]},
    }
    issues = disp.validate_apply_plan(
        {"BB1", "BB2"}, registry=conflict_registry, outcomes={})
    check(len(issues) == 1 and issues[0].severity == "ERROR",
          f"conflict pair still reported once, as ERROR ({issues})")


def main() -> int:
    try:
        import vllm._genesis.dispatcher as disp
    except Exception as exc:  # pragma: no cover - import-env dependent
        print(f"SKIP: cannot import the dispatcher here ({exc})")
        return 0

    print("REVIEW-genesis-dispatcher H1/H2/M1 — retired gate + truthful summary")
    test_retired_gate(disp)
    test_p67b_dependency(disp)
    test_truthful_summary(disp)
    test_validator_reads_outcomes(disp)

    print(f"\n== RESULT: {len(failures)} fail ==")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
