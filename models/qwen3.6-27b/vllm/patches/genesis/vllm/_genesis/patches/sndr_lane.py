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

  3. S-PREFIX ALIASES — his in-registry PN71..PN118 numbers collide with
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
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("genesis.sndr_lane")

# Bares whose GENESIS_DISABLE_<bare> was injected by apply_policy step 1 (NOT
# by the operator). Module-level so a second apply_policy() call (it advertises
# idempotency) does not misread its OWN first-call injection as an operator
# opt-out. bare -> patch id.
_POLICY_INJECTED_DISABLES: dict[str, str] = {}

_TRUTHY = ("1", "true", "yes", "on")

# Sub-patch name as sndr's text_patch bakes it into a failure detail, e.g.
#   "PN79: required_anchor_missing — sub-patch 'chunk_inplace': anchor not found"
_SUB_PATCH_RE = re.compile(r"sub-patch ['\"]([^'\"]+)['\"]")

# Our /fixes house series numbers that also exist (as DIFFERENT patches)
# in Sander's registry — S-prefix alias targets (policy step 3).
# PN95/PN96 added 2026-07-14: house /fixes patch_pn95 (TQ prefill workspace
# sizing #43357) / patch_pn96 (structured-output marker FSM #44993) collide
# with Sander's PN95 (tier-aware KV offload) / PN96 (emergency-demote PoC).
# The S-aliases (GENESIS_ENABLE_SPN95_TIER_AWARE_CACHE /
# GENESIS_ENABLE_SPN96_EMERGENCY_DEMOTE) keep Sander's pair individually
# addressable with zero collision; both remain opt-in (default_on=False,
# offload/tiering PoCs — no-op-to-risky on a single-card rig).
#
# PN102/PN104/PN105/PN106/PN108/PN118 added 2026-07-25 (patch-id collision
# audit). Each of these lane-2 ids ALSO names a completely different house
# /fixes patch:
#
#   PN102  house middleware/answer_rescue.py Leg-1 envelope contract
#          (GENESIS_ENABLE_PN102_CONTRACT)          vs lane-2 PrefetchOffloader
#          pinned-allocator prewarm pool (…_PN102_PARAM_POOL)
#   PN104  house patch_pn104_mamba_align_gather_clamp.py (always-on)
#          vs lane-2 cpu-offload → Prefetch redirect
#   PN105  house patch_pn105_nan_logits_abort.py (always-on)
#          vs lane-2 AutoRound INT4 offload compat
#   PN106  house patch_pn106d_bug076_nan_slot_audit.py (house id PN106D)
#          vs lane-2 GDN scratch tensor pool
#   PN108  house patch_pn108_plateau_cap.py (GENESIS_ENABLE_PN108_PLATEAU_CAP)
#          vs lane-2 GDN fused_recurrent prefill dispatch
#   PN118  house answer_rescue.py Leg-3 premature-close gate
#          (GENESIS_ENABLE_PN118_CLOSEGATE) vs lane-2 TurboQuant workspace
#          graceful-fallback (bare GENESIS_ENABLE_PN118)
#
# No pair collides on the EXACT env-var today — but nothing prevented it, and
# that is precisely the BUG-122 failure shape (a patch gating on the wrong var
# while the boot record says "applied"). Bringing them under the same S-alias
# guard as the other 11 is purely ADDITIVE: step 3 only appends
# GENESIS_ENABLE_S<bare> to env_flag_aliases, and sndr's _resolve_env_state
# ignores an alias whose env var is unset (verified: none of the six S-names
# appears in any compose or model_config). The composes that DO arm these
# lane-2 patches set the canonical flag explicitly, so the BUG-122 mirror below
# (`flag not in os.environ`) never fires for them either.
#
# [2026-07-25] PN79 REMOVED — dead entry, and the only one of the three
# `[guard]`-noted ids that was actually stale. Its house side is ARCHIVED:
# fixes/_archive/patch_pn79_tq_decode_scratch_cudagraph_safe.py, retired
# 2026-07-23 (BUG-119 — the vllm#46067 crash class is handled in-tree). With
# no house patch of any spelling claiming 79, there is no collision left to
# guard, so the alias named nothing. (That archived house patch was an
# unrelated TQ decode-scratch cudagraph fix; lane-2's PN79 is the in-place SSM
# state backport of vllm#41824, itself parked after a PROD IMA.)
#
# PN91 and PN106 are DELIBERATELY KEPT even though the lint's guard check used
# to note them: their house counterparts are live, merely suffixed —
# patch_pn91g_48475_gdn_spec_state_index_clamp.py (PN91G) and
# patch_pn106d_bug076_nan_slot_audit.py (PN106D). Same number, different
# patch, which is exactly the collision the audit §4 table and
# tests/test_patch_id_lint.py::test_six_audited_collisions_are_now_guarded
# require to stay guarded. patch_id_lint now recognises a `<id><SUFFIX>` house
# id as claiming the number, so it no longer mis-flags them.
#
# Net effect of the PN79 removal: exactly one alias disappears
# (GENESIS_ENABLE_SPN79_INPLACE_SSM_STATE); the other 16 S-aliases are
# byte-identical, and that variable appears in no compose, script or
# model_config (grepped), so no boot ever resolved through it. Rule B in
# patch_id_lint is the safety net: if a house patch ever re-claims a bare
# PN79, the lint FAILS (not notes) and tells you to re-add it here.
#
# PN122 / PN129 added 2026-07-26 (BUG-133). Two house patches minted that day —
# fixes/patch_pn122_structured_force_guard.py (03:48) and
# fixes/patch_pn129_trace_finish_reason_zero_output.py (03:52) — took numbers
# lane-2 already holds (PN122 = CG dispatch trace, renamed there 2026-05-14;
# PN129 = slot-mapping warmup, live via GENESIS_ENABLE_PN129_SLOT_MAPPING_WARMUP
# in 5 composes). Neither is an EXACT env collision — the descriptive suffixes
# differ and boot_patches keys house rows by log slug, not by id — so this is
# the same latent shape as the six 2026-07-25 audit rows above, and gets the
# same treatment: the S-alias, not a rename. Renaming the house side (H122/H129,
# the H119 precedent) would have to edit the entrypoints in endgame8020.yml and
# tcbench8021.yml, which is a larger blast radius than the risk warrants.
_HOUSE_COLLIDING_IDS = {
    "PN71", "PN72", "PN73", "PN80", "PN82", "PN90", "PN91", "PN92",
    "PN95", "PN96",
    "PN102", "PN104", "PN105", "PN106", "PN108", "PN118",
    "PN122", "PN129",
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

# Lane-1 ids that were RENAMED (id-shape / collision fixes) while lane-2 still
# carries the OLD key — the vendored tree stays byte-identical, so a lane-1
# rename cannot be mirrored there. Without this map the lane-2 copy drops out
# of the `shared` intersection below and is treated as NET-NEW, silently losing
# the step-1 suppression that keeps lane-1 authoritative for the pair.
#   lane-2 key  ->  lane-1 key
_LANE1_RENAMED_SHARED = {
    # 2026-07-25 patch-id lint: hyphen made the id unrecordable (the boot
    # recorder truncated "PN40-classifier" to "PN40").
    "PN40-classifier": "PN40c",
}


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


def _env_is_enabled(bare: str) -> bool:
    """Mirror of `sndr.env.is_enabled` (SNDR_ wins over GENESIS_)."""
    val = os.environ.get(f"SNDR_ENABLE_{bare}")
    if val is None:
        val = os.environ.get(f"GENESIS_ENABLE_{bare}")
    return val is not None and val.strip().lower() in _TRUTHY


def _env_is_disabled(bare: str) -> bool:
    """Mirror of `sndr.env.is_disabled` (SNDR_ wins over GENESIS_)."""
    val = os.environ.get(f"SNDR_DISABLE_{bare}")
    if val is None:
        val = os.environ.get(f"GENESIS_DISABLE_{bare}")
    return val is not None and val.strip().lower() in _TRUTHY


def audit_env_conflicts(his_reg: dict[str, Any]) -> dict[str, list[str]]:
    """Classify every ENABLE+DISABLE conflict lane-2's dispatcher will resolve
    as "DISABLE wins".

    Why this exists (observability gap, 2026-07-25): `decision._check_disable_
    gate` logs ONE warning per conflict, scattered through a ~500-line boot
    block. On the 07-25 fixwave boot that was 42 separate warnings — nothing
    aggregated them, so an operator who set `GENESIS_ENABLE_X=1` in a compose
    had no way to see that lane-2 overrode it, and no way to tell the 42 apart.

    Most of them are BY DESIGN: `apply_policy` step 1 injects
    `GENESIS_DISABLE_<bare>=1` for every shared id precisely so lane-1 stays
    authoritative. Those are EXPECTED and carry no operator signal. A conflict
    whose DISABLE came from the operator's own env is GENUINE — that is the one
    worth a warning.

    Resolution is deliberately NOT changed here (DISABLE-wins is the contract);
    this only makes the set visible and separates the two classes. Mirrors
    `decision._resolve_env_state` (canonical flag, then enabled-and-not-disabled
    aliases) so the classification tracks what the dispatcher actually decides.
    """
    expected: list[str] = []
    genuine: list[str] = []
    for pid, meta in his_reg.items():
        if not isinstance(meta, dict):
            continue
        flag = meta.get("env_flag")
        if not flag:
            continue
        bare = _bare(flag)
        if not _env_is_disabled(bare):
            continue
        if _env_is_enabled(bare):
            via = bare
        else:
            via = None
            for alias in (meta.get("env_flag_aliases") or ()):
                abare = _bare(alias)
                if _env_is_enabled(abare) and not _env_is_disabled(abare):
                    via = f"{bare}<-alias {alias}"
                    break
            if via is None:
                continue  # DISABLE only — no conflict, operator intent is clear
        entry = f"{pid}({via})"
        if _POLICY_INJECTED_DISABLES.get(bare) is not None:
            expected.append(entry)
        else:
            genuine.append(entry)
    return {"expected": sorted(expected), "genuine": sorted(genuine)}


def log_env_conflicts(conflicts: dict[str, list[str]]) -> None:
    """One summary line for the whole conflict set (WARNING iff genuine)."""
    expected = conflicts.get("expected") or []
    genuine = conflicts.get("genuine") or []
    total = len(expected) + len(genuine)
    if not total:
        return
    if genuine:
        log.warning(
            "[Genesis lane-2/sndr] env conflicts: %d ENABLE+DISABLE pair(s) — "
            "%d expected (policy-injected shared-id suppression, lane-1 owns "
            "them), %d GENUINE operator conflict(s) silently resolved "
            "DISABLE-wins against stated intent: %s. Drop one env var per "
            "pair to clear.",
            total, len(expected), len(genuine), ", ".join(genuine),
        )
    else:
        log.info(
            "[Genesis lane-2/sndr] env conflicts: %d ENABLE+DISABLE pair(s), "
            "all policy-injected (expected — lane-1 owns these shared ids); "
            "0 genuine operator conflicts.",
            total,
        )


def log_partial_apply_warnings(stats: Any) -> None:
    """Lane-2 mirror of lane-1's partial-apply enumeration.

    Why (observability gap, 2026-07-25): under the spec-driven loop that this
    bridge forces (`SNDR_APPLY_VIA_SPECS=1`), `sndr.apply.orchestrator.run()`
    returns immediately after `_run_via_specs(stats)` + `log.info("Genesis %s")`
    — it never reaches the enumeration block that the LEGACY path runs. Net
    effect on the 07-25 fixwave boot: lane-1 printed its 4 warnings one by one,
    lane-2 printed the string "25 ⚠️ partial-apply warning(s)" and dropped
    every detail. A lane-2 patch landing 1 of 4 sub-patches was invisible.

    Emitted from the bridge rather than the vendored tree so `sndr/` stays
    byte-identical for `git read-tree` grafts.
    """
    try:
        warnings = list(getattr(stats, "partial_apply_warnings", None) or ())
    except Exception as exc:  # pragma: no cover - reporting must not break boot
        log.warning("[Genesis lane-2/sndr] partial-apply enumeration "
                    "unavailable: %s", exc)
        return
    if not warnings:
        return
    # sndr's BENIGN list (apply/_state.py) does not know about registry entries
    # that carry no apply_module — pure documentation rows that can never touch
    # source. On the 07-25 boot they were 15 of the 24 warnings and would bury
    # the 9 that matter. Collapse them to one line here rather than editing the
    # vendored BENIGN tuple (the sndr/ tree stays byte-identical).
    informational = [r for r in warnings
                     if "no apply_module declared" in (getattr(r, "reason", "") or "")]
    actionable = [r for r in warnings if r not in informational]
    if actionable:
        log.warning(
            "[Genesis lane-2/sndr] %d partial-apply warning(s) — patch(es) "
            "failed to match expected source pattern. Review below to confirm "
            "anchor drift vs upstream change vs config issue:",
            len(actionable),
        )
        for r in actionable:
            reason = getattr(r, "reason", "") or ""
            m = _SUB_PATCH_RE.search(reason)
            log.warning(
                "[Genesis lane-2/sndr] ⚠️  %s [sub-patch: %s] — %s",
                getattr(r, "name", "?"), m.group(1) if m else "-", reason,
            )
    if informational:
        log.info(
            "[Genesis lane-2/sndr] %d further warning-classified skip(s) are "
            "registry rows with no apply_module (documentation entries, "
            "nothing to apply): %s",
            len(informational),
            ", ".join(str(getattr(r, "name", "?")).split(" ")[0]
                      for r in informational),
        )


def apply_policy() -> dict[str, Any]:
    """Apply the club-3090 policy overlay to the (already imported) sndr
    registry.  Returns a summary dict for logging.  Idempotent."""
    from sndr import dispatcher as sndr_dispatcher
    from vllm._genesis.dispatcher import PATCH_REGISTRY as OUR_REGISTRY

    his_reg = sndr_dispatcher.PATCH_REGISTRY
    shared_set = set(his_reg) & set(OUR_REGISTRY)
    shared_set |= {
        old for old, new in _LANE1_RENAMED_SHARED.items()
        if old in his_reg and new in OUR_REGISTRY
    }
    shared = sorted(shared_set)
    net_new = sorted(set(his_reg) - shared_set)

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
        # Record WHOSE disable this is before writing it, so the conflict
        # audit below can tell "policy injected it so lane-1 keeps the id"
        # apart from "the operator set both ENABLE and DISABLE themselves".
        # Guard on the module-level ledger too: apply_policy advertises
        # idempotency, and on a second call our own first-call injection
        # would otherwise read as pre-existing operator intent.
        if (
            bare not in _POLICY_INJECTED_DISABLES
            and os.environ.get(f"GENESIS_DISABLE_{bare}") is None
            and os.environ.get(f"SNDR_DISABLE_{bare}") is None
        ):
            _POLICY_INJECTED_DISABLES[bare] = pid
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
    alias_mirrored = []
    for pid in _HOUSE_COLLIDING_IDS:
        meta = his_reg.get(pid)
        if not meta or pid in shared_set:
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
        # BUG-122 fix 2026-07-25: the alias is honored at the ENTRY level only
        # (decision._resolve_env_state) — it decides should_apply=True and the
        # dispatcher announces "APPLY <pid>". But each module's own apply()
        # re-gates on the CANONICAL flag via its private _enabled(), which never
        # learns about aliases, reads it unset, and returns "skipped" silently.
        # Net effect before this fix: compose enables SPN71/73/92, the record DB
        # says "applied", and the targets were never touched (0 markers).
        # Mirror a truthy S-alias onto the canonical flag so the module-level
        # gate agrees with the entry-level decision. Safe for these ids: they
        # are net-new (pid not in OUR_REGISTRY), and the canonical flag carries
        # the sndr module's own descriptive suffix, so it cannot shadow lane-1's
        # same-numbered patch (verified: no lane-1 flag shares these names).
        if os.environ.get(s_alias, "").strip().lower() in (
            "1", "true", "yes", "on"
        ) and flag not in os.environ:
            os.environ[flag] = "1"
            alias_mirrored.append(f"{pid}:{s_alias}→{flag}")

    return {
        "shared_suppressed": len(suppressed),
        "handed_to_lane2": handed_over,
        "net_new": len(net_new),
        "default_on_forced_off": forced_off,
        "s_aliases": aliased,
        "s_alias_mirrored": alias_mirrored,
        "lane1_hardowned": lane1_hardowned,
        # Audited AFTER all three policy steps so the S-alias mirroring above
        # (which writes ENABLE vars) is reflected in the classification.
        "env_conflicts": audit_env_conflicts(his_reg),
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
        "for %d (%s), S-aliases: %s, S-alias mirrored to canonical: %s, "
        "lane-1 hard-owned (repointed): %s, "
        "handed to lane-2: %s",
        summary["shared_suppressed"],
        summary["net_new"],
        len(summary["default_on_forced_off"]),
        ",".join(summary["default_on_forced_off"]) or "-",
        ",".join(summary["s_aliases"]) or "-",
        ",".join(summary["s_alias_mirrored"]) or "-",
        ",".join(summary["lane1_hardowned"]) or "-",
        ",".join(summary["handed_to_lane2"]) or "-",
    )
    # Single summary line for the whole ENABLE/DISABLE conflict set. The
    # dispatcher's per-conflict warnings stay where they are; this is what
    # makes them readable as a SET and separates by-design suppression from a
    # genuine operator conflict. Best-effort: never take the boot down.
    try:
        log_env_conflicts(summary.get("env_conflicts") or {})
    except Exception as exc:  # pragma: no cover - reporting must not break boot
        log.warning("[Genesis lane-2/sndr] env-conflict audit failed: %s", exc)

    from sndr.apply import run as sndr_run
    stats = sndr_run(verbose=True, apply=not dry)

    # BUG-122 fix 2026-07-25: lane-2 previously logged only DECISIONS
    # ("[Genesis Dispatcher] APPLY <id>") and never RESULTS, so a module whose
    # private gate disagreed with the entry-level decision skipped silently and
    # every consumer — including vllm-patch-record — counted the announcement
    # as an apply. Lane-1 has always logged per-patch results; emit the lane-2
    # equivalent so "applied" means applied on BOTH lanes. Best-effort: a
    # reporting gap must never take the boot down.
    try:
        results = list(getattr(stats, "results", None) or ())
        for r in results:
            log.info(
                "[Genesis lane-2/sndr] RESULT %s: %s — %s",
                getattr(r, "status", "unknown"),
                getattr(r, "name", "?"),
                (getattr(r, "reason", "") or "")[:120],
            )
        if results:
            counts = Counter(getattr(r, "status", "unknown") for r in results)
            log.info(
                "[Genesis lane-2/sndr] Results: %d applied, %d skipped, "
                "%d failed (of %d dispatched)",
                counts.get("applied", 0), counts.get("skipped", 0),
                counts.get("failed", 0), len(results),
            )
    except Exception as exc:  # pragma: no cover - reporting must not break boot
        log.warning("[Genesis lane-2/sndr] result logging failed: %s", exc)

    # Enumerate the partial-apply warnings the way lane-1 does. The spec-driven
    # orchestrator path returns before sndr's own enumeration block, so without
    # this the aggregate count is the ONLY signal lane-2 emits.
    log_partial_apply_warnings(stats)

    return stats
