# SPDX-License-Identifier: Apache-2.0
"""BUG-133 — patch-id / env-flag collisions must FAIL, not merely print.

BUG-133's title is "**Unguarded** patch-id collisions". The lint gate
(`_genesis/utils/patch_id_lint.py`) has existed since 2026-07-25, but its only
pytest face lives in `_genesis/tests/test_patch_id_lint.py`, which imports
`vllm._genesis` — so it runs only where the vllm package is importable, i.e.
inside the container. Nothing under `fixes/` ran it, and on 2026-07-26 two new
house patches (PN122, PN129) landed on numbers lane-2 already held while the
linter sat at `violations=2` with nobody reading it.

This module is the `/fixes` face of the same gate:

  * pure stdlib — the linter is AST-only and is loaded BY PATH, so no vllm,
    no torch, no CUDA, no container, no GPU;
  * `test_lint_gate_is_clean` FAILS the suite on any new violation;
  * `test_lanes_were_actually_parsed` stops a linter that found nothing
    because it parsed nothing from passing as "clean";
  * every rule gets a synthetic-collision test, so a check that silently
    stopped firing cannot masquerade as green.

Run (pytest, anywhere):

    python3 -m pytest -q --noconftest fixes/test_bug133_patch_id_collisions.py

Run (no pytest installed — the bare host has none):

    python3 fixes/test_bug133_patch_id_collisions.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

FIXES = Path(__file__).resolve().parent
_GEN_REL = "vllm/_genesis"
# Host checkout first, then the in-container layout (`/fixes` bind-mount +
# genesis overlaid onto site-packages), so the same file runs either place.
_CANDIDATES = [
    FIXES.parent / "models/qwen3.6-27b/vllm/patches/genesis" / _GEN_REL,
    Path("/usr/local/lib/python3.12/dist-packages") / _GEN_REL,
]
GENESIS = next(
    (c for c in _CANDIDATES if (c / "utils/patch_id_lint.py").is_file()),
    _CANDIDATES[0],
)
LINT_PY = GENESIS / "utils/patch_id_lint.py"


def _load_lint():
    """Import the linter by path. No package context, no vllm import."""
    assert LINT_PY.is_file(), (
        f"linter missing — tried {[str(c) for c in _CANDIDATES]}"
    )
    spec = importlib.util.spec_from_file_location("_bug133_patch_id_lint", LINT_PY)
    mod = importlib.util.module_from_spec(spec)
    # Must be in sys.modules BEFORE exec: @dataclass resolves the owning
    # module out of sys.modules on py3.12+ and raises AttributeError if absent.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


lint = _load_lint()

# The house↔lane-2 number collisions found on 2026-07-26. Both house patches
# were minted that morning onto numbers lane-2 already owned; neither is an
# exact env-flag collision (the descriptive suffixes differ), so they get the
# same S-alias guard as the six rows from the 2026-07-25 audit.
BUG133_GUARDED_IDS = ("PN122", "PN129")

# Cross-lane env_flag couplings where the lane-2 side is NET-NEW (not
# suppressed by `apply_policy` step 1), i.e. one operator switch really does
# arm two patches. Pinned so a NEW one is a test failure rather than a note
# nobody reads. Splitting an existing one is a behaviour change (audit §4).
KNOWN_LIVE_ALIAS_COUPLINGS = {
    ("GENESIS_ENABLE_PN29_GDN_SCALE_FOLD", "PN298"),
}


def _report():
    return lint.run()


# ═══════════════════════════════════════════════════════════════════════════
#                    POSITIVE — the live tree must be clean
# ═══════════════════════════════════════════════════════════════════════════

def test_lint_gate_is_clean():
    """The whole point of BUG-133: a collision FAILS, it does not just print."""
    rep = _report()
    assert rep.ok, (
        f"{len(rep.violations)} patch-id/env-flag violation(s) — a new patch "
        f"collided with an existing one:\n  " + "\n  ".join(rep.violations)
    )


def test_lanes_were_actually_parsed():
    """A gate that parsed nothing is not a gate."""
    rep = _report()
    assert len(rep.lane1_ids) > 100, "lane-1 dispatcher registry did not parse"
    assert len(rep.lane2_ids) > 300, "lane-2 sndr registry did not parse"
    assert len(rep.house_ids) > 30, "house /fixes lane did not parse"


def test_bug133_ids_carry_the_s_alias_guard():
    """PN122/PN129 regression pin — the collision that was live on 07-26."""
    guarded = lint._extract_name_set(
        GENESIS / "patches/sndr_lane.py", "_HOUSE_COLLIDING_IDS",
    )
    for pid in BUG133_GUARDED_IDS:
        assert pid in guarded, (
            f"{pid} names BOTH a house /fixes patch and an unrelated lane-2 "
            f"patch and must carry the GENESIS_ENABLE_S<bare> alias guard"
        )


def test_no_new_live_cross_lane_flag_coupling():
    """Duplicate env flags across lanes stay at the known, documented set."""
    rep = _report()
    live = set()
    for note in rep.notes:
        if "NET-NEW and NOT suppressed" not in note:
            continue
        flag = note.split()[1]
        pid = note.split("lane-2 ")[1].split()[0]
        live.add((flag, pid))
    assert live == KNOWN_LIVE_ALIAS_COUPLINGS, (
        f"cross-lane env_flag couplings changed.\n"
        f"  new:  {sorted(live - KNOWN_LIVE_ALIAS_COUPLINGS)}\n"
        f"  gone: {sorted(KNOWN_LIVE_ALIAS_COUPLINGS - live)}"
    )


# ═══════════════════════════════════════════════════════════════════════════
#                  NEGATIVE — every rule really fires
# ═══════════════════════════════════════════════════════════════════════════

def test_synthetic_id_collision_fails_rule_b():
    rep = lint.LintReport()
    lint._check_cross_lane(
        {"PN122", "PN777"},   # house ids
        set(),                # lane-1
        {"PN122", "PN777"},   # lane-2
        {"PN122"},            # guarded — PN777 deliberately is not
        rep,
    )
    assert len(rep.violations) == 1, rep.violations
    assert "PN777" in rep.violations[0]
    assert "UNGUARDED id collision" in rep.violations[0]


def test_synthetic_cross_lane_env_flag_collision_fails_rule_d():
    """Two DIFFERENT ids, different lanes, one flag — must FAIL."""
    rep = lint.LintReport()
    lint._check_cross_lane_env_flags(
        {"PN777": {"env_flag": "GENESIS_ENABLE_PN777_THING"}},          # lane-1
        {"PN778": {"env_flag": "GENESIS_ENABLE_PN777_THING"}},          # lane-2
        {}, rep,
    )
    assert len(rep.violations) == 1, rep.violations
    assert "DUPLICATE env_flag" in rep.violations[0]
    assert "GENESIS_ENABLE_PN777_THING" in rep.violations[0]


def test_house_flag_colliding_with_a_lane_flag_fails_rule_d():
    """The exact scenario BUG-133 calls latent: a house patch adopting the
    lane-2 patch's descriptive suffix on the same number."""
    rep = lint.LintReport()
    lint._check_cross_lane_env_flags(
        {},
        {"PN122": {"env_flag": "GENESIS_ENABLE_PN122_CG_DISPATCH_TRACE"}},
        {"GENESIS_ENABLE_PN122_CG_DISPATCH_TRACE": "PN122H"},   # house
        rep,
    )
    assert len(rep.violations) == 1, rep.violations
    assert "/fixes:PN122H" in rep.violations[0]


def test_same_id_in_both_lanes_is_not_a_flag_collision():
    """The vendored-copy case: one patch, two lanes, one flag — never a fail."""
    rep = lint.LintReport()
    lint._check_cross_lane_env_flags(
        {"P64": {"env_flag": "GENESIS_ENABLE_P64_X"}},
        {"P64": {"env_flag": "GENESIS_ENABLE_P64_X"}},
        {}, rep,
    )
    assert rep.violations == [], rep.violations


def test_baselined_cross_lane_flag_downgrades_to_note():
    rep = lint.LintReport()
    lint._check_cross_lane_env_flags(
        {"PX1": {"env_flag": "GENESIS_ENABLE_PX1_THING"}},
        {"PX1b": {"env_flag": "GENESIS_ENABLE_PX1_THING"}},
        {}, rep,
        baseline=frozenset({("GENESIS_ENABLE_PX1_THING", ("PX1", "PX1b"))}),
    )
    assert rep.violations == []
    assert len(rep.notes) == 1


def test_house_declared_flags_ignores_flags_a_patch_merely_reads():
    """`patch_pn100_*` reads GENESIS_ENABLE_PN16_LAZY_REASONER; attributing
    that to PN100 would make rule D scream about every consumer."""
    declared = lint._house_declared_flags(FIXES)
    assert declared.get("GENESIS_ENABLE_PN122_STRUCTURED_FORCE_GUARD") == "PN122"
    assert "GENESIS_ENABLE_PN16_LAZY_REASONER" not in declared


def test_duplicate_env_flag_within_a_lane_still_fails_rule_a():
    rep = lint.LintReport()
    lint._check_duplicate_env_flags(
        "synthetic",
        {
            "PX1": {"env_flag": "GENESIS_ENABLE_PX1_THING"},
            "PX1b": {"env_flag": "GENESIS_ENABLE_PX1_THING"},
        },
        rep,
    )
    assert len(rep.violations) == 1
    assert "DUPLICATE env_flag" in rep.violations[0]


def test_illegal_id_shape_still_fails_rule_c():
    rep = lint.LintReport()
    lint._check_id_shape("synthetic", {"PN40-classifier", "PN40c"}, rep)
    assert len(rep.violations) == 1
    assert "PN40-classifier" in rep.violations[0]


# ═══════════════════════════════════════════════════════════════════════════
# Bare-python fallback: the aibox host has no pytest, and a gate that cannot
# be run on the box it guards is not a gate.
# ═══════════════════════════════════════════════════════════════════════════

def _main() -> int:
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"ok   {name}")
    print("FAILED" if failed else "PASSED", f"({failed} failure(s))")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
