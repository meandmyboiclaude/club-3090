# SPDX-License-Identifier: Apache-2.0
"""club-3090 lane-2 bridge — run the vendored sndr v12 dispatcher.

Two-lane apply architecture (sndr-v12-import, 2026-07-13):

  Lane 1  — our proven monolith (`vllm._genesis.patches.apply_all` body +
            `vllm._genesis.dispatcher`).  Owns every patch id that existed
            in the club-3090 registry before the import (the "shared" set,
            126 ids incl. our dev1060 re-anchors PN8/PN12/P34/P83).  This
            is what serves :8020 today; it stays byte-identical.

  Lane 2  — Sander's sndr v12 dispatcher (`sndr.apply.run`), vendored
            byte-identical under `vllm/_genesis/sndr/`.  Owns everything
            NET-NEW (209 registry ids incl. the 59 Gemma4 G4_* patches),
            all default-OFF until an enable-wave flips a flag.

Why the vendor location: the live compose mounts ONLY `vllm/_genesis` into
the container, so the sndr package must live inside that directory.  This
bridge registers it as top-level `sndr` (his modules import `from sndr.x`).

Policy layer (applied IN-MEMORY every boot — his files stay unmodified so
future `git read-tree` grafts from the sndr remote are conflict-free):

  1. SHARED SUPPRESSION — every patch id present in BOTH registries gets
     `GENESIS_DISABLE_<bare>=1` injected process-locally before lane-2
     runs.  Our compose enables 20+ GENESIS_ENABLE_* flags for lane-1;
     without this, lane-2 would see the same env and text-patch his
     dev748-era form ON TOP of our dev1060 form (double-apply).
     Migration lever: `GENESIS_SNDR_OWNS_<bare>=1` hands a single shared
     patch to lane-2 (skips the DISABLE injection here AND makes lane-1's
     should_apply skip it) — this is how enable-waves adopt his newer
     forms one patch at a time, with an A/B path back.

  2. NET-NEW DEFAULT-OFF — his registry ships 28 net-new entries with
     default_on=True (13 Gemma4 + 15 others).  ALL are forced to
     default_on=False in-memory.  Import mandate: nothing changes live
     behavior until an enable-wave sets an explicit flag.  (The G4_*
     entries would additionally self-skip on model detection; we do not
     rely on that alone.)

  3. S-PREFIX ALIASES — his in-registry PN71..PN95 numbers collide with
     our /fixes house series of the same numbers (different patches!).
     Zero exact env-var collisions exist, but to keep enable-waves and
     log greps unambiguous, each colliding entry gets an extra alias
     `GENESIS_ENABLE_S<bare>` (e.g. GENESIS_ENABLE_SPN71_THINKING_TAG_
     NORMALIZE).  Compose files should use the S-form for these ids.

Kill-switch: GENESIS_SNDR_LANE=0 disables lane-2 entirely (lane-1 alone =
pre-import behavior, byte-for-byte).

See IMPORT-NOTES.md at the repo root for the full decision record.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("genesis.sndr_lane")

# Our /fixes house series numbers that also exist (as DIFFERENT patches)
# in Sander's registry — S-prefix alias targets (policy step 3).
_HOUSE_COLLIDING_IDS = {
    "PN71", "PN72", "PN73", "PN79", "PN80", "PN82", "PN90", "PN91", "PN92",
}

# Shared ids that LANE-1 hard-owns and applies (its wiring is the live form
# serving :8020). For these, the blanket GENESIS_DISABLE_<bare> shared
# suppression (policy step 1) must NOT run: <bare> is ALSO lane-1's own
# enable-flag stem (e.g. P74 → GENESIS_ENABLE_P74_CHUNK_CLAMP), so injecting
# GENESIS_DISABLE_P74_CHUNK_CLAMP shadows the operator's ENABLE. Instead,
# lane-2's copy of the id is repointed to a UNIQUE S-alias flag (unset in the
# compose) so the compose's GENESIS_ENABLE_<bare>=1 engages lane-1 ONLY, with
# no DISABLE conflict. See P74 rename directive (name-collision → rename).
_LANE1_HARDOWNED_SHARED = {"P74"}


def _lane_enabled() -> bool:
    return os.environ.get("GENESIS_SNDR_LANE", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def bootstrap_sndr_alias() -> Any:
    """Register `vllm/_genesis/sndr` as the top-level `sndr` package.

    Must run before any `import sndr...` — his modules use absolute
    `from sndr.x import y` imports.  Uses an explicit spec (not sys.path
    insertion) so no other `vllm/_genesis` subdirectory can shadow a
    top-level module name.
    """
    if "sndr" in sys.modules:
        return sys.modules["sndr"]
    pkg_dir = Path(__file__).resolve().parent.parent / "sndr"
    init_py = pkg_dir / "__init__.py"
    if not init_py.is_file():
        raise ImportError(f"vendored sndr package not found at {pkg_dir}")
    spec = importlib.util.spec_from_file_location(
        "sndr", str(init_py), submodule_search_locations=[str(pkg_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so intra-package `from sndr.x import y` resolves.
    sys.modules["sndr"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop("sndr", None)
        raise
    return mod


def _bare(flag: str) -> str:
    for prefix in ("SNDR_ENABLE_", "GENESIS_ENABLE_"):
        if flag.startswith(prefix):
            return flag[len(prefix):]
    return flag


def _sndr_owns(bare: str) -> bool:
    return os.environ.get(f"GENESIS_SNDR_OWNS_{bare}", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply_policy() -> dict[str, Any]:
    """Apply the club-3090 policy overlay to the (already imported) sndr
    registry.  Returns a summary dict for logging.  Idempotent."""
    from sndr import dispatcher as sndr_dispatcher
    from vllm._genesis.dispatcher import PATCH_REGISTRY as OUR_REGISTRY

    his_reg = sndr_dispatcher.PATCH_REGISTRY
    shared = sorted(set(his_reg) & set(OUR_REGISTRY))
    net_new = sorted(set(his_reg) - set(OUR_REGISTRY))

    # 1. shared suppression (unless operator handed the id to lane-2)
    suppressed, handed_over, lane1_hardowned = [], [], []
    for pid in shared:
        meta2 = his_reg[pid]
        flag = meta2.get("env_flag")
        if not flag:
            continue
        bare = _bare(flag)
        if _sndr_owns(bare):
            handed_over.append(pid)
            continue
        if pid in _LANE1_HARDOWNED_SHARED:
            # Lane-1 owns + applies this shared patch (its wiring is the live
            # form). Do NOT inject GENESIS_DISABLE_<bare>: <bare> is ALSO the
            # stem of lane-1's own enable flag (P74 → GENESIS_ENABLE_P74_
            # CHUNK_CLAMP), so a DISABLE here would shadow the operator ENABLE.
            # Instead repoint lane-2's copy to a UNIQUE S-alias flag (unset in
            # the compose) — mirrors the step-3 S-prefix mechanism — so
            # GENESIS_ENABLE_<bare>=1 engages lane-1 ONLY and lane-2's copy
            # stays opt-in on the (unset) S-flag. No DISABLE, no conflict.
            s_flag = f"GENESIS_ENABLE_S{bare}"
            # BUG-5 fix 2026-07-14: the canonical flag must NOT remain in
            # env_flag_aliases. sndr's should_apply → _resolve_env_state
            # treats an ENABLED alias as truthy (decision.py alias loop), so
            # keeping GENESIS_ENABLE_<bare> addressable would re-arm lane-2
            # whenever the compose enables lane-1 — the exact double-apply
            # shadowing this repoint exists to end. Strip it if present:
            # lane-2's copy is reachable ONLY via the unique S-flag.
            aliases = [
                a for a in (meta2.get("env_flag_aliases") or []) if a != flag
            ]
            meta2["env_flag"] = s_flag
            meta2["env_flag_aliases"] = aliases
            meta2["default_on"] = False  # unset S-flag ⇒ lane-2 opt-in skip
            meta2.setdefault(
                "club3090_note",
                f"lane-1 hard-owned shared id; lane-2 flag repointed to "
                f"{s_flag} so GENESIS_ENABLE_{bare}=1 engages lane-1 only "
                f"(no DISABLE shadow).",
            )
            lane1_hardowned.append(f"{pid}→{s_flag}")
            continue
        os.environ[f"GENESIS_DISABLE_{bare}"] = "1"
        suppressed.append(pid)

    # 2. net-new default-off — UNLESS the operator trusts the sndr registry's
    # own default_on judgment (2026-07-13 enable-wave: GENESIS_SNDR_TRUST_DEFAULT_ON=1
    # lets Sander's engine decide; per-patch config_detect/arch gates still apply,
    # and any of the 28 can be individually vetoed via GENESIS_DISABLE_<flag>=1).
    forced_off = []
    trust_default_on = os.environ.get(
        "GENESIS_SNDR_TRUST_DEFAULT_ON", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    for pid in net_new:
        meta = his_reg[pid]
        if meta.get("default_on"):
            if trust_default_on:
                meta.setdefault(
                    "club3090_note",
                    "default_on TRUSTED (GENESIS_SNDR_TRUST_DEFAULT_ON=1, "
                    "2026-07-13 enable-wave).",
                )
                continue
            meta["default_on"] = False
            meta.setdefault(
                "club3090_note",
                "default_on forced False by club-3090 import policy "
                "(sndr_lane.apply_policy) — enable via env flag only.",
            )
            forced_off.append(pid)

    # 3. S-prefix aliases for house-series numeric collisions
    aliased = []
    for pid in _HOUSE_COLLIDING_IDS:
        meta = his_reg.get(pid)
        if not meta or pid in OUR_REGISTRY:
            continue  # shared ids are lane-1-owned; alias only net-new
        flag = meta.get("env_flag")
        if not flag:
            continue
        s_alias = f"GENESIS_ENABLE_S{_bare(flag)}"
        aliases = list(meta.get("env_flag_aliases") or [])
        if s_alias not in aliases:
            aliases.append(s_alias)
            meta["env_flag_aliases"] = aliases
            aliased.append(f"{pid}→{s_alias}")

    return {
        "shared_suppressed": len(suppressed),
        "handed_to_lane2": handed_over,
        "net_new": len(net_new),
        "default_on_forced_off": forced_off,
        "s_aliases": aliased,
        "lane1_hardowned": lane1_hardowned,
    }


def run_lane2(dry: bool) -> Optional[Any]:
    """Run lane-2 (sndr dispatcher).  Returns his PatchStats, or None when
    the lane is disabled, or a RuntimeError is raised on setup failure."""
    if not _lane_enabled():
        log.info("[Genesis lane-2/sndr] disabled via GENESIS_SNDR_LANE=0")
        return None

    log.info("═" * 78)
    log.info("[Genesis lane-2/sndr] sndr v12 dispatcher (net-new patches)")
    log.info("═" * 78)

    # Force the spec-driven apply path. The default ("legacy") loop calls
    # the ~95 hand-written apply_patch_X functions UNCONDITIONALLY (each
    # self-gates internally) — measured on dev1060 it text-patches 29
    # patches with no env flag set (incl. shared P15/P26 forms on top of
    # lane-1's), because the legacy functions ignore GENESIS_DISABLE_* and
    # in-registry default_on. The spec-driven path routes EVERY patch
    # through dispatcher.should_apply, where the policy overlay above is
    # authoritative. This is the no-regression invariant's load-bearing
    # switch — do not remove.
    os.environ.setdefault("SNDR_APPLY_VIA_SPECS", "1")

    bootstrap_sndr_alias()
    summary = apply_policy()
    log.info(
        "[Genesis lane-2/sndr] policy: %d shared ids suppressed "
        "(lane-1 owns), %d net-new ids available, default_on forced off "
        "for %d (%s), S-aliases: %s, lane-1 hard-owned (repointed): %s, "
        "handed to lane-2: %s",
        summary["shared_suppressed"],
        summary["net_new"],
        len(summary["default_on_forced_off"]),
        ",".join(summary["default_on_forced_off"]) or "-",
        ",".join(summary["s_aliases"]) or "-",
        ",".join(summary["lane1_hardowned"]) or "-",
        ",".join(summary["handed_to_lane2"]) or "-",
    )

    from sndr.apply import run as sndr_run
    return sndr_run(verbose=True, apply=not dry)
