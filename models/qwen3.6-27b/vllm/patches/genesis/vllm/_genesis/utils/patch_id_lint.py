# SPDX-License-Identifier: Apache-2.0
"""Patch-id / env-flag lint gate for all three Genesis apply lanes.

Turns the invariants from the 2026-07-25 patch-id collision audit
(`~/shared/PATCH-ID-COLLISION-AUDIT-20260725.md`) from a convention into a
gate. Three rules, one per failure mode the audit found:

  A. **No duplicate `env_flag`.** Two registry rows declaring the same
     `env_flag` string cannot be enabled, disabled or A/B'd independently —
     flipping one flips both. A sub-patch that must ride its parent's flag
     declares the parent flag in `env_flag_aliases` and keeps a canonical
     name of its own (this is what P67b and PN40c do).

  B. **No unguarded cross-lane id collision.** House `/fixes` ids and the
     vendored lane-2 sndr registry mint numbers from overlapping ranges, so
     the SAME id can name two unrelated patches. Every such pair must be
     listed in `patches/sndr_lane.py::_HOUSE_COLLIDING_IDS`, which gives the
     lane-2 member a unique `GENESIS_ENABLE_S<bare>` alias. Without that
     guard nothing prevents the two sides from converging on one env var —
     the BUG-122 shape, where a patch gates on the wrong var and still
     reports "applied".

  D. **No CROSS-LANE env_flag collision.** Rule A is per-lane, so it cannot
     see the failure the audit actually feared: two *different* patches, in
     *different* lanes, canonically declaring one flag. That is the BUG-122
     shape with teeth — the operator sets one var and two unrelated patches
     arm. Cross-lane reuse of a flag by the SAME id is not a collision (lane-2
     is a vendored copy of that patch and `sndr_lane.apply_policy` step 1
     suppresses it); reuse by different ids is. The two pre-existing pairs are
     frozen into `LANE2_DUP_FLAG_BASELINE` alongside the per-lane ones.
     Alias-level couplings (a lane-1 flag listed in an unrelated lane-2 row's
     `env_flag_aliases`) are recorded as notes with their shadow status, since
     splitting one is a behaviour change, not a naming fix.

  C. **Recorder-legal id shape.** `ops/vllm-patch-guard/vllm-patch-record.py`
     extracts dispatcher ids with ``[A-Za-z]+\\d+[a-zA-Z]*`` and house slugs
     with ``^\\[([a-z0-9_-]+)\\]``. An id that does not FULL-match its lane's
     shape is silently truncated on the way into `vllmops.boot_patches`
     (`PN40-classifier` recorded as `PN40`), so its apply/skip state can
     never be observed.

Scope of enforcement — deliberate, and the reason this gate can be green:

  * lane-1 (`_genesis/dispatcher.py`) and the house `/fixes` lane are ours:
    **enforced strictly**.
  * lane-2 (`_genesis/sndr/**`) is the vendored sndr tree, kept
    byte-identical so future `git read-tree` grafts from the sndr remote are
    conflict-free (see `patches/sndr_lane.py`). Renaming inside it is tier-5
    in the audit's migration ladder — explicitly rejected. Its pre-existing
    violations are therefore frozen into `LANE2_SHAPE_BASELINE` /
    `LANE2_DUP_FLAG_BASELINE`: the live violation set must stay a SUBSET of
    the baseline, so a NEW violation fails the gate while the existing ones
    are recorded rather than silently tolerated.

Pure stdlib, AST-only: nothing here imports vllm, torch or the registries,
so it runs on a bare python3 outside the container.

Run standalone:

    python3 vllm/_genesis/utils/patch_id_lint.py            # from repo root
    python3 -m vllm._genesis.utils.patch_id_lint

Exit code 0 = clean, 1 = violations (printed one per line).
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ─── shapes the boot recorder can actually parse ──────────────────────────
# ops/vllm-patch-guard/vllm-patch-record.py:
#   disp_names  = re.findall(r"APPLY ([A-Z]+\d+[a-zA-Z]*)", log)
#   drift_names = re.findall(r"DRIFT skipped:\s*([A-Za-z]+\d+[a-zA-Z]*)", log)
#   house_names = re.match(r"^\[([a-z0-9_-]+)\]", line)
DISPATCHER_ID_SHAPE = re.compile(r"^[A-Za-z]+\d+[a-zA-Z]*$")
HOUSE_SLUG_SHAPE = re.compile(r"^[a-z0-9_-]+$")
# Leading id token of a house slug / filename stem: "pn108-plateau-cap" →
# "pn108", "pn91g_48475_..." → "pn91g", "h119-lens-router" → "h119".
HOUSE_ID_TOKEN = re.compile(r"^([a-z]+\d+[a-z]*)(?:[-_]|$)")

# ─── frozen vendored-lane baselines (see module docstring) ────────────────
# Ids in the vendored sndr registry that violate DISPATCHER_ID_SHAPE. These
# record into boot_patches truncated (every `G4_nn` banks as `G4`). Fixing
# them means renaming inside `_genesis/sndr/**`, which forfeits the
# byte-identical vendor-graft invariant — audit tier 5, rejected. Frozen so
# the set can only shrink.
LANE2_SHAPE_BASELINE = frozenset({
    "G4_01", "G4_02", "G4_03", "G4_04", "G4_05", "G4_06", "G4_07", "G4_08",
    "G4_09", "G4_10", "G4_11", "G4_12", "G4_13", "G4_14", "G4_15", "G4_16",
    "G4_17", "G4_18", "G4_19", "G4_19B", "G4_19C", "G4_23", "G4_24",
    "G4_25", "G4_26", "G4_31", "G4_32", "G4_60A", "G4_60B", "G4_60C",
    "G4_60D", "G4_60E", "G4_60G", "G4_60H", "G4_60K", "G4_60L", "G4_61",
    "G4_62", "G4_67", "G4_68", "G4_69", "G4_70", "G4_70B", "G4_70C",
    "G4_71", "G4_71B", "G4_72", "G4_73", "G4_74", "G4_75", "G4_76",
    "G4_78", "G4_79", "G4_80", "G4_81", "G4_82", "G4_83", "G4_84", "G4_85",
    "G4_T1", "P18B_TEXT", "P23_WIRE", "P29_HEAL", "PN-FP8MOE-KPAD",
    "PN118_V2_MD5_TURBOQUANT_ATTN", "PN118_V2_MD5_WORKSPACE", "PN16_V6",
    "PN40-classifier", "PN521_SPLIT_K", "PN79_V2_MD5_CHUNK",
    "PN79_V2_MD5_CHUNK_DELTA_H", "SNDR_EAGLE3_AUX_HIDDEN_001",
    "SNDR_MTP_DYNAMIC_K_001", "SNDR_WORKSPACE_001",
})

# Duplicate `env_flag` values inside the vendored sndr registry. Both are the
# lane-2 COPIES of pairs lane-1 has already split (P67/P67b, PN40/PN40c) —
# and both copies are inert here: `sndr_lane.apply_policy` step 1 suppresses
# every shared id so lane-1 stays authoritative. Frozen, not fixed, for the
# same vendor-graft reason.
LANE2_DUP_FLAG_BASELINE = frozenset({
    ("GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL", ("P67", "P67b")),
    ("GENESIS_ENABLE_PN40_DFLASH_OMNIBUS", ("PN40", "PN40-classifier")),
})

# House patches that do NOT live in `fixes/patch_<id>_*.py` and so cannot be
# discovered from a filename. Declared explicitly, and VERIFIED: the linter
# fails if the file is gone or no longer contains the marker flag, so the
# table cannot rot into a lie.
#   id -> (path relative to the repo root, a flag string that must be present)
HOUSE_IDS_OUTSIDE_FIXES: dict[str, tuple[str, str]] = {
    "PN102": (
        "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis/middleware/"
        "answer_rescue.py",
        "GENESIS_ENABLE_PN102_CONTRACT",
    ),
    # answer_rescue.py Leg 3 — the premature-close gate. Renumbered
    # PN118 -> PN123 on 2026-07-26 (BUG-144): lane-2 owns a bare `PN118` row
    # plus PN118_V2_MD5_* which the boot recorder truncates to `PN118`, so the
    # old number could never be observed distinctly in `vllmops.boot_patches`.
    # PN123 is the canonical id and flag.
    "PN123": (
        "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis/middleware/"
        "answer_rescue.py",
        "GENESIS_ENABLE_PN123_CLOSEGATE",
    ),
    # The old number stays REGISTERED, deliberately, because the rename is
    # partial by design: `GENESIS_ENABLE_PN118_CLOSEGATE` is still a working
    # legacy alias, so the house really does still claim the number PN118.
    # Keeping the row is what makes two things stay true:
    #   * rule B keeps pairing house PN118 with lane-2's PN118 and reporting it
    #     as guarded by `_HOUSE_COLLIDING_IDS` (dropping the row would instead
    #     emit a "no house patch claims that number — remove the entry" note,
    #     which would be wrong advice while the alias is live), and
    #   * the marker check pins the alias itself: if the legacy flag is ever
    #     deleted from answer_rescue.py this row goes red, which is the signal
    #     to retire the `_HOUSE_COLLIDING_IDS["PN118"]` guard in the same edit.
    # This is an alias registration, not a second patch: same file, same leg.
    "PN118": (
        "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis/middleware/"
        "answer_rescue.py",
        "GENESIS_ENABLE_PN118_CLOSEGATE",
    ),
    # answer_rescue.py Leg 5 — the BUG-155 budget-truth guard. Lives in the
    # mounted genesis tree rather than `fixes/patch_pn155_*.py` because the
    # middleware IS the deployment artefact (the tree is bind-mounted read-only
    # into site-packages), so there is no file for an applier to rewrite.
    "PN155": (
        "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis/middleware/"
        "answer_rescue.py",
        "GENESIS_ENABLE_PN155_BUDGET_TRUTH",
    ),
}


@dataclass
class LintReport:
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    lane1_ids: set[str] = field(default_factory=set)
    lane2_ids: set[str] = field(default_factory=set)
    house_ids: set[str] = field(default_factory=set)
    house_slugs: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations


# ─── AST extraction (no imports, no side effects) ─────────────────────────

def _extract_registry(path: Path) -> dict[str, dict[str, Any]]:
    """Return the module-level ``PATCH_REGISTRY`` literal from *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        target: Optional[ast.Name] = None
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "PATCH_REGISTRY":
                    target = t
        elif isinstance(node, ast.AnnAssign):
            if (isinstance(node.target, ast.Name)
                    and node.target.id == "PATCH_REGISTRY"):
                target = node.target
        if target is None or not isinstance(node.value, ast.Dict):
            continue
        out: dict[str, dict[str, Any]] = {}
        for key_node, val_node in zip(node.value.keys, node.value.values):
            try:
                key = ast.literal_eval(key_node)
            except Exception:
                continue
            meta: dict[str, Any] = {}
            if isinstance(val_node, ast.Dict):
                for k2, v2 in zip(val_node.keys, val_node.values):
                    try:
                        mk = ast.literal_eval(k2)
                    except Exception:
                        continue
                    try:
                        meta[mk] = ast.literal_eval(v2)
                    except Exception:
                        meta[mk] = None  # computed value — shape is enough
            out[str(key)] = meta
        return out
    raise ValueError(f"no module-level PATCH_REGISTRY literal in {path}")


def _extract_name_set(path: Path, name: str) -> set[str]:
    """Return a module-level ``name = {...}`` / ``frozenset({...})`` literal."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        value = node.value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id in ("frozenset", "set") and value.args):
            value = value.args[0]
        try:
            return {str(x) for x in ast.literal_eval(value)}
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"{name} in {path} is not a literal: {exc}")
    raise ValueError(f"no module-level {name} in {path}")


def _extract_name_dict(
    path: Path, name: str, default: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        return {
            str(k): str(v) for k, v in ast.literal_eval(node.value).items()
        }
    if default is not None:
        return dict(default)
    raise ValueError(f"no module-level {name} in {path}")


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return []


# ─── layout discovery ─────────────────────────────────────────────────────

def find_genesis_root(start: Optional[Path] = None) -> Path:
    """Directory holding ``vllm/_genesis`` (the genesis package root)."""
    here = (start or Path(__file__)).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "vllm" / "_genesis" / "dispatcher.py").is_file():
            return parent
    raise FileNotFoundError("could not locate the genesis tree root")


def find_fixes_dir(genesis_root: Path) -> Optional[Path]:
    """The always-on house ``/fixes`` lane, if this checkout carries it.

    Genesis is also exported standalone (without the club-3090 `/fixes`
    lane); the house checks then report a note instead of failing.
    """
    for parent in genesis_root.resolve().parents:
        candidate = parent / "fixes"
        if candidate.is_dir() and any(candidate.glob("patch_*.py")):
            return candidate
    return None


def find_repo_root(genesis_root: Path) -> Optional[Path]:
    fixes = find_fixes_dir(genesis_root)
    return fixes.parent if fixes else None


# ─── the checks ───────────────────────────────────────────────────────────

def _check_duplicate_env_flags(
    lane: str, registry: dict[str, dict[str, Any]], report: LintReport,
    baseline: frozenset = frozenset(),
) -> None:
    by_flag: dict[str, list[str]] = {}
    for pid, meta in registry.items():
        flag = meta.get("env_flag")
        if isinstance(flag, str) and flag:
            by_flag.setdefault(flag, []).append(pid)
    for flag, pids in sorted(by_flag.items()):
        if len(pids) < 2:
            continue
        key = (flag, tuple(sorted(pids)))
        if key in baseline:
            report.notes.append(
                f"[{lane}] duplicate env_flag {flag} on {sorted(pids)} "
                f"— frozen baseline (vendored tree, not renameable)"
            )
            continue
        report.violations.append(
            f"[{lane}] DUPLICATE env_flag {flag!r} declared by "
            f"{sorted(pids)}: neither patch can be enabled, disabled or "
            f"A/B'd on its own. Give each a canonical flag and put the "
            f"shared legacy name in the sub-patch's `env_flag_aliases`."
        )


def _check_id_shape(
    lane: str, ids: set[str], report: LintReport,
    baseline: frozenset = frozenset(),
) -> None:
    for pid in sorted(ids):
        if DISPATCHER_ID_SHAPE.match(pid):
            continue
        if pid in baseline:
            report.notes.append(
                f"[{lane}] id {pid!r} violates the recorder shape "
                f"[A-Za-z]+\\d+[a-zA-Z]* — frozen baseline "
                f"(vendored tree, not renameable)"
            )
            continue
        truncated = re.match(r"[A-Za-z]+\d+[a-zA-Z]*", pid)
        report.violations.append(
            f"[{lane}] ILLEGAL id shape {pid!r}: the boot recorder matches "
            f"[A-Za-z]+\\d+[a-zA-Z]* and would bank it as "
            f"{truncated.group(0)!r} — the patch can never appear "
            f"distinctly in vllmops.boot_patches. Use letters-digits"
            f"[-letters] with no separator (e.g. PN40c)."
        )


def _house_declared_flags(fixes_dir: Path) -> dict[str, str]:
    """Flags a house `/fixes` patch declares as ITS OWN gate → declaring id.

    A house patch mentions plenty of flags it merely READS (patch_pn100 reads
    `GENESIS_ENABLE_PN16_LAZY_REASONER` to see whether the lazy reasoner is on).
    Only a flag whose id segment matches the patch's own id token is a
    declaration — `patch_pn122_structured_force_guard.py` declaring
    `GENESIS_ENABLE_PN122_STRUCTURED_FORCE_GUARD`. That is exactly the segment
    a cross-lane id collision puts at risk, so it is the one rule D must see.
    """
    out: dict[str, str] = {}
    for path in sorted(fixes_dir.glob("patch_*.py")):
        m = HOUSE_ID_TOKEN.match(path.stem[len("patch_"):])
        if not m:
            continue
        pid = m.group(1).upper()
        src = path.read_text(encoding="utf-8")
        own = re.compile(
            r"\b(?:GENESIS|SNDR)_ENABLE_" + re.escape(pid) + r"(?:_[A-Z0-9_]+)?\b"
        )
        for flag in sorted(set(own.findall(src))):
            out.setdefault(flag, pid)
    return out


def _check_cross_lane_env_flags(
    lane1: dict[str, dict[str, Any]], lane2: dict[str, dict[str, Any]],
    house_flags: dict[str, str], report: LintReport,
    baseline: frozenset = frozenset(),
) -> None:
    """Rule D — one env_flag canonically declared by two DIFFERENT ids.

    Same id in both lanes is the vendored-copy case and is fine. Different ids
    means one operator switch arms two unrelated patches.
    """
    canon: dict[str, set[tuple[str, str]]] = {}
    alias: dict[str, set[tuple[str, str]]] = {}
    for lane, registry in (("lane-1", lane1), ("lane-2", lane2)):
        for pid, meta in registry.items():
            flag = meta.get("env_flag")
            if isinstance(flag, str) and flag:
                canon.setdefault(flag, set()).add((lane, pid))
            for a in _coerce_list(meta.get("env_flag_aliases")):
                alias.setdefault(a, set()).add((lane, pid))
    for flag, pid in house_flags.items():
        canon.setdefault(flag, set()).add(("/fixes", pid))

    for flag, rows in sorted(canon.items()):
        ids = sorted({pid for _, pid in rows})
        lanes = {lane for lane, _ in rows}
        # Same id in lane-1 AND lane-2 is the vendored-copy case and is fine.
        # A house patch is NEVER a vendored copy of a dispatcher patch, so
        # house-meets-lane on one flag is a collision even at the same number
        # — otherwise an id already parked in `_HOUSE_COLLIDING_IDS` (rule B
        # satisfied) could still adopt the lane's exact flag unnoticed, which
        # is the very convergence the audit named.
        house_meets_lane = "/fixes" in lanes and len(lanes) > 1
        if len(ids) < 2 and not house_meets_lane:
            continue
        if (flag, tuple(ids)) in baseline:
            report.notes.append(
                f"[cross-lane] env_flag {flag} canonically declared by "
                f"{ids} — frozen baseline (vendored tree, not renameable)"
            )
            continue
        where = sorted(f"{lane}:{pid}" for lane, pid in rows)
        why = (
            "a house /fixes patch and a dispatcher-lane patch declare it "
            "canonically — house is never a vendored copy, so these are two "
            "unrelated patches on one switch"
            if house_meets_lane else
            f"{len(ids)} different ids declare it canonically across lanes"
        )
        report.violations.append(
            f"[cross-lane] DUPLICATE env_flag {flag!r} ({', '.join(where)}): "
            f"{why} — the BUG-122 shape, where one patch gates on the wrong "
            f"var and still reports 'applied'. Give the newer patch a flag "
            f"carrying its own id, or fold it into the other as an "
            f"`env_flag_aliases` entry so the coupling is declared."
        )

    # Alias-level couplings: recorded, not failed. Splitting one changes
    # behaviour (audit §4) and needs a bench, so the gate only keeps them visible.
    lane1_ids, lane2_ids = set(lane1), set(lane2)
    for flag, rows in sorted(alias.items()):
        holders = {pid for _, pid in rows} | {
            pid for _, pid in canon.get(flag, set())
        }
        if len(holders) < 2:
            continue
        for lane, pid in sorted(rows):
            if lane != "lane-2" or pid in canon.get(flag, set()):
                continue
            shared = pid in lane1_ids and pid in lane2_ids
            report.notes.append(
                f"[cross-lane] {flag} is an alias on lane-2 {pid} while "
                f"{sorted(holders - {pid})} declare it canonically — "
                + ("lane-2 side is a SHARED id, suppressed by "
                   "apply_policy step 1, so lane-1 stays authoritative"
                   if shared else
                   "lane-2 side is NET-NEW and NOT suppressed: setting the "
                   "flag arms BOTH patches")
            )


def _check_cross_lane(
    house_ids: set[str], lane1_ids: set[str], lane2_ids: set[str],
    house_colliding: set[str], report: LintReport,
) -> None:
    """Rule B — a house `/fixes` id must not silently share a number with an
    unrelated dispatcher-lane patch.

    lane-1 ∩ lane-2 is NOT a collision: lane-2 is a vendored copy of the same
    patch and `sndr_lane.apply_policy` step 1 suppresses it so lane-1 stays
    authoritative (audit §1c). What must not happen is a house id meeting a
    lane-1 or lane-2 id with no `_HOUSE_COLLIDING_IDS` guard.
    """
    for house_id in sorted(house_ids):
        for lane, ids in (("lane-2", lane2_ids), ("lane-1", lane1_ids)):
            if house_id not in ids:
                continue
            if house_id in house_colliding:
                report.notes.append(
                    f"[cross-lane] {house_id} exists in house /fixes AND "
                    f"{lane} — guarded by _HOUSE_COLLIDING_IDS (S-alias)"
                )
                continue
            report.violations.append(
                f"[cross-lane] UNGUARDED id collision {house_id!r}: named by "
                f"a house /fixes patch AND by an unrelated {lane} patch. "
                f"Nothing stops the two from converging on one env var "
                f"(BUG-122 shape). Add {house_id!r} to "
                f"`_HOUSE_COLLIDING_IDS` in patches/sndr_lane.py so the "
                f"lane-2 member gets its unique GENESIS_ENABLE_S<bare> "
                f"alias, or rename one side."
            )


def run(genesis_root: Optional[Path] = None) -> LintReport:
    """Run every check. Returns a `LintReport` (see `.ok` / `.violations`)."""
    root = Path(genesis_root) if genesis_root else find_genesis_root()
    gen = root / "vllm" / "_genesis"
    report = LintReport()

    lane1 = _extract_registry(gen / "dispatcher.py")
    lane2 = _extract_registry(gen / "sndr" / "dispatcher" / "registry.py")
    report.lane1_ids = set(lane1)
    report.lane2_ids = set(lane2)

    sndr_lane_py = gen / "patches" / "sndr_lane.py"
    house_colliding = _extract_name_set(sndr_lane_py, "_HOUSE_COLLIDING_IDS")
    # Optional: only present once a lane-1 id has been renamed away from its
    # (unrenameable) lane-2 counterpart.
    renamed_shared = _extract_name_dict(
        sndr_lane_py, "_LANE1_RENAMED_SHARED", default={},
    )

    # ── rule A — duplicate env_flag ──────────────────────────────────────
    _check_duplicate_env_flags("lane-1", lane1, report)
    _check_duplicate_env_flags(
        "lane-2", lane2, report, baseline=LANE2_DUP_FLAG_BASELINE,
    )

    # An alias must not be another row's canonical flag *in the same lane*
    # unless that row is the declared parent — otherwise arming the parent
    # silently arms a stranger. (P67b→P67 and PN40c→PN40 are parents.)
    for lane, registry in (("lane-1", lane1), ("lane-2", lane2)):
        canonical = {
            meta.get("env_flag"): pid for pid, meta in registry.items()
            if isinstance(meta.get("env_flag"), str)
        }
        for pid, meta in sorted(registry.items()):
            for alias in _coerce_list(meta.get("env_flag_aliases")):
                if alias == meta.get("env_flag"):
                    report.violations.append(
                        f"[{lane}] {pid}: env_flag_aliases repeats its own "
                        f"canonical flag {alias!r}"
                    )
                owner = canonical.get(alias)
                if owner and owner != pid:
                    report.notes.append(
                        f"[{lane}] {pid} rides {owner}'s flag {alias} as an "
                        f"alias (declared parent/child coupling)"
                    )

    # ── rule C — id shape ────────────────────────────────────────────────
    _check_id_shape("lane-1", report.lane1_ids, report)
    _check_id_shape(
        "lane-2", report.lane2_ids, report, baseline=LANE2_SHAPE_BASELINE,
    )

    # ── house /fixes lane ────────────────────────────────────────────────
    fixes_dir = find_fixes_dir(root)
    repo_root = fixes_dir.parent if fixes_dir else None
    if fixes_dir is None:
        report.notes.append(
            "house /fixes lane not present in this checkout — house checks "
            "skipped (standalone genesis export)"
        )
    else:
        slug_owner: dict[str, str] = {}
        for path in sorted(fixes_dir.glob("patch_*.py")):
            stem = path.stem[len("patch_"):]
            m = HOUSE_ID_TOKEN.match(stem)
            if m:
                report.house_ids.add(m.group(1).upper())
            src = path.read_text(encoding="utf-8")
            log_m = re.search(r'^LOG\s*=\s*"\[([^\]]*)\]"', src, re.M)
            if not log_m:
                continue
            slug = log_m.group(1)
            report.house_slugs[slug] = path.name
            if not HOUSE_SLUG_SHAPE.match(slug):
                report.violations.append(
                    f"[/fixes] ILLEGAL log slug '[{slug}]' in {path.name}: "
                    f"the boot recorder matches ^\\[([a-z0-9_-]+)\\] — this "
                    f"line will not be banked into boot_patches at all."
                )
            if slug in slug_owner:
                report.violations.append(
                    f"[/fixes] DUPLICATE log slug '[{slug}]' in "
                    f"{slug_owner[slug]} and {path.name}: both patches bank "
                    f"onto the same boot_patches row."
                )
            slug_owner[slug] = path.name

        for pid, (rel, marker) in sorted(HOUSE_IDS_OUTSIDE_FIXES.items()):
            target = (repo_root / rel) if repo_root else None
            if target is None or not target.is_file():
                # In-container the genesis tree is overlaid onto site-packages
                # and `/fixes` is a bind-mount, so `repo_root` is `/` and the
                # repo-relative path resolves nowhere. Re-anchor on the genesis
                # package itself before calling the table stale.
                _, _, tail = rel.partition("_genesis/")
                if tail and (gen / tail).is_file():
                    target = gen / tail
            if target is None or not target.is_file():
                report.violations.append(
                    f"[/fixes] HOUSE_IDS_OUTSIDE_FIXES[{pid}] points at "
                    f"{rel} which does not exist — update the table."
                )
                continue
            if marker not in target.read_text(encoding="utf-8"):
                report.violations.append(
                    f"[/fixes] HOUSE_IDS_OUTSIDE_FIXES[{pid}] declares "
                    f"{marker} in {rel} but the flag is not there any more "
                    f"— update the table."
                )
                continue
            report.house_ids.add(pid)

    # ── rule B — cross-lane id collisions ────────────────────────────────
    _check_cross_lane(
        report.house_ids, report.lane1_ids, report.lane2_ids,
        house_colliding, report,
    )

    # ── rule D — cross-lane env_flag collisions ──────────────────────────
    _check_cross_lane_env_flags(
        lane1, lane2,
        _house_declared_flags(fixes_dir) if fixes_dir is not None else {},
        report, baseline=LANE2_DUP_FLAG_BASELINE,
    )

    # ── guard-table consistency ──────────────────────────────────────────
    for pid in sorted(house_colliding):
        if pid not in report.lane2_ids:
            report.violations.append(
                f"[guard] _HOUSE_COLLIDING_IDS lists {pid!r} which is not a "
                f"lane-2 id — the S-alias is a no-op; remove it."
            )
        elif fixes_dir is not None and pid not in report.house_ids:
            # A house patch may claim the NUMBER under a suffixed id — PN106 vs
            # house PN106D, PN91 vs house PN91G. That is still exactly the
            # collision the guard exists for (same number, unrelated patch), so
            # the entry is correct and must not be reported as stale. Only a
            # number no house patch claims under ANY spelling is a dead entry.
            suffixed = sorted(
                h for h in report.house_ids
                if h != pid and re.fullmatch(re.escape(pid) + r"[A-Z]+", h)
            )
            if suffixed:
                continue
            report.notes.append(
                f"[guard] _HOUSE_COLLIDING_IDS lists {pid!r} but no house "
                f"patch claims that number under any spelling (archived or "
                f"removed) — the S-alias names nothing; remove the entry"
            )
    for old, new in sorted(renamed_shared.items()):
        if old not in report.lane2_ids:
            report.violations.append(
                f"[guard] _LANE1_RENAMED_SHARED maps {old!r} which is not a "
                f"lane-2 id — the shared-suppression re-link is dead."
            )
        if new not in report.lane1_ids:
            report.violations.append(
                f"[guard] _LANE1_RENAMED_SHARED maps {old!r} -> {new!r} but "
                f"{new!r} is not a lane-1 id."
            )

    return report


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    verbose = "-v" in argv or "--verbose" in argv
    root = None
    for arg in argv:
        if not arg.startswith("-"):
            root = Path(arg)
    report = run(root)
    if verbose:
        for note in report.notes:
            print(f"note: {note}")
    print(
        f"patch-id lint: lane-1={len(report.lane1_ids)} "
        f"lane-2={len(report.lane2_ids)} house={len(report.house_ids)} "
        f"slugs={len(report.house_slugs)} "
        f"notes={len(report.notes)} violations={len(report.violations)}"
    )
    for v in report.violations:
        print(f"FAIL: {v}")
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
