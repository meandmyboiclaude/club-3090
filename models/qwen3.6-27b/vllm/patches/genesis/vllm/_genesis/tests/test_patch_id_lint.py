# SPDX-License-Identifier: Apache-2.0
"""Gate the patch-id / env-flag invariants from the 2026-07-25 collision audit.

The linter itself lives in `vllm/_genesis/utils/patch_id_lint.py` (pure
stdlib, AST-only, also runnable as a standalone script). This module is the
pytest face of it, so it runs the way every other genesis test runs.

Three invariants, one per failure mode the audit found:

  A. no two registry rows declare the same `env_flag` — otherwise neither of
     the pair can be enabled, disabled or A/B'd on its own
  B. no house `/fixes` id silently shares a number with an unrelated
     dispatcher-lane patch outside `_HOUSE_COLLIDING_IDS`
  C. no id violates the shape the boot recorder can parse
     (`[A-Za-z]+\\d+[a-zA-Z]*` for dispatcher lanes, `[a-z0-9_-]+` for house
     log slugs) — a violating id is truncated on the way into
     `vllmops.boot_patches` and its apply/skip state becomes unobservable

Each invariant gets a positive test (the live tree is clean) AND a negative
test (the check really fires), so a check that silently stopped working
cannot pass as "no violations".
"""
from __future__ import annotations

import pytest

from vllm._genesis.utils import patch_id_lint as lint


@pytest.fixture(scope="module")
def report() -> lint.LintReport:
    return lint.run()


# ═══════════════════════════════════════════════════════════════════════════
#                        POSITIVE — the live tree is clean
# ═══════════════════════════════════════════════════════════════════════════

def test_tree_has_no_patch_id_violations(report: lint.LintReport) -> None:
    assert report.ok, (
        f"{len(report.violations)} patch-id violation(s):\n  "
        + "\n  ".join(report.violations)
    )


def test_lanes_were_actually_parsed(report: lint.LintReport) -> None:
    """A linter that found nothing because it parsed nothing is not a gate."""
    assert len(report.lane1_ids) > 100, "lane-1 registry did not parse"
    assert len(report.lane2_ids) > 300, "lane-2 registry did not parse"
    assert len(report.house_ids) > 30, "house /fixes lane did not parse"


def test_the_two_split_flags_kept_their_legacy_name(
    report: lint.LintReport,
) -> None:
    """The P67/P67b and PN40/PN40c flag splits must not have orphaned the
    legacy shared name — every existing compose still sets it."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY

    p67b = PATCH_REGISTRY["P67b"]
    assert p67b["env_flag"] == "GENESIS_ENABLE_P67B_SPEC_VERIFY_ROUTING"
    assert "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL" in (
        p67b["env_flag_aliases"]
    )
    assert (
        PATCH_REGISTRY["P67"]["env_flag"]
        == "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL"
    )

    pn40c = PATCH_REGISTRY["PN40c"]
    assert pn40c["env_flag"] == "GENESIS_ENABLE_PN40C_WORKLOAD_CLASSIFIER"
    assert "GENESIS_ENABLE_PN40_DFLASH_OMNIBUS" in pn40c["env_flag_aliases"]
    assert "PN40-classifier" not in PATCH_REGISTRY


def test_six_audited_collisions_are_now_guarded() -> None:
    """PN102/PN104/PN105/PN106/PN108/PN118 were the unguarded house↔lane-2
    id collisions in the audit's §4 table."""
    from vllm._genesis.patches.sndr_lane import _HOUSE_COLLIDING_IDS

    for pid in ("PN102", "PN104", "PN105", "PN106", "PN108", "PN118"):
        assert pid in _HOUSE_COLLIDING_IDS, (
            f"{pid} is a known house↔lane-2 id collision and must carry the "
            f"S-alias guard"
        )


# ═══════════════════════════════════════════════════════════════════════════
#                     NEGATIVE — each check really fires
# ═══════════════════════════════════════════════════════════════════════════

def test_duplicate_env_flag_is_detected() -> None:
    rep = lint.LintReport()
    lint._check_duplicate_env_flags(
        "synthetic",
        {
            "PX1": {"env_flag": "GENESIS_ENABLE_PX1_THING"},
            "PX1b": {"env_flag": "GENESIS_ENABLE_PX1_THING"},
            "PX2": {"env_flag": "GENESIS_ENABLE_PX2_OTHER"},
        },
        rep,
    )
    assert len(rep.violations) == 1
    assert "DUPLICATE env_flag" in rep.violations[0]
    assert "PX1b" in rep.violations[0]


def test_duplicate_env_flag_baseline_downgrades_to_note() -> None:
    rep = lint.LintReport()
    lint._check_duplicate_env_flags(
        "synthetic",
        {
            "PX1": {"env_flag": "GENESIS_ENABLE_PX1_THING"},
            "PX1b": {"env_flag": "GENESIS_ENABLE_PX1_THING"},
        },
        rep,
        baseline=frozenset({("GENESIS_ENABLE_PX1_THING", ("PX1", "PX1b"))}),
    )
    assert rep.violations == []
    assert len(rep.notes) == 1


def test_illegal_id_shapes_are_detected() -> None:
    rep = lint.LintReport()
    lint._check_id_shape(
        "synthetic",
        {"PN40-classifier", "PN119_ROUTER", "PN40c", "P67b", "H119"},
        rep,
    )
    flagged = {v.split("id shape ")[1].split(":")[0].strip("'")
               for v in rep.violations}
    assert flagged == {"PN40-classifier", "PN119_ROUTER"}
    # and the message names what the recorder WOULD have banked
    assert any("'PN40'" in v for v in rep.violations)
    assert any("'PN119'" in v for v in rep.violations)


def test_unguarded_cross_lane_collision_is_detected() -> None:
    rep = lint.LintReport()
    lint._check_cross_lane(
        {"PN108", "PN999"},          # house ids
        set(),                       # lane-1 ids
        {"PN108", "PN999"},          # lane-2 ids
        {"PN108"},                   # guarded — PN999 deliberately is not
        rep,
    )
    assert len(rep.violations) == 1
    assert "PN999" in rep.violations[0]
    assert any("PN108" in n for n in rep.notes)


def test_guarded_cross_lane_collision_passes() -> None:
    rep = lint.LintReport()
    lint._check_cross_lane({"PN108"}, set(), {"PN108"}, {"PN108"}, rep)
    assert rep.violations == []
    assert len(rep.notes) == 1


def test_house_id_absent_from_both_lanes_is_not_a_collision() -> None:
    rep = lint.LintReport()
    lint._check_cross_lane({"PN114"}, {"P67b"}, {"PN71"}, set(), rep)
    assert rep.violations == []
