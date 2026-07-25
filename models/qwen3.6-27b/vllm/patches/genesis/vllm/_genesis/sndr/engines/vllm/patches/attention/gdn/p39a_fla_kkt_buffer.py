# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 39a — FLA `chunk_scaled_dot_kkt_fwd` persistent A pool.

Replaces per-call `torch.empty(B, T, H, BT, fp32)` with a persistent
pool via `FlaKktBufferManager.acquire`. Monkey-patches
`vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd`
at module level.

Rationale: the GDN chunked-prefill path inside the AOT-compiled model
calls this function once per GDN-bearing layer per chunk. Each alloc is
16 MiB on our config (Qwen3.6 B=1 T≤4096 H=16 BT=64 fp32) but the
N_layers-fold churn saturates the allocator at the yaml=0.93/0.94
boundary with dev134 memory accounting. A single shared pool removes
the churn entirely.

Compatibility
-------------
- NVIDIA CUDA SM 8.0+: wiring applied.
- AMD / CPU / pre-Ampere: wiring skipped, fallback in manager
  (`acquire` returns fresh `torch.empty` when `should_apply()` is
  False).
- Upstream drift: this module's `apply()` dry-imports the target and
  logs skip if the symbol isn't present (future rename or removal).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.3 implementation
"""

# Legacy auto-apply note (audit 2026-05-11): registry env_flag
# `GENESIS_LEGACY_P39A` is synthetic — flag exists for registry/audit
# coherence but has no runtime effect. Patch applies unconditionally
# via dispatcher's legacy auto-apply path (`is_legacy_active` in
# vllm/sndr_core/dispatcher/decision.py). See registry.py "Legacy
# patches" section (~line 2083) for full context.

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from sndr.engines.vllm.detection.guards import is_nvidia_cuda, is_sm_at_least
# v11.1.0 P3.3: surface the FLA KKT persistent A pool through
# PersistentBufferRegistry so operators can `sndr patches show
# buffer_registry` and see this pool listed. Byte-equivalent — the
# actual torch.empty() still happens inside FlaKktBufferManager.acquire
# (allocate-once-keep-forever, pointer-stable via the
# reserve-before-cudagraph pattern). The registry hook only exposes
# the pool name; tensor storage ownership is unchanged.
from sndr.runtime.persistent_buffer_registry import (
    PersistentBufferRegistry,
    POOL_FLA_KKT_PERSISTENT_A,
)

log = logging.getLogger("genesis.wiring.p39a_fla_kkt")

_GENESIS_P39A_MARKER_ATTR = "_genesis_p39a_wrapped"

# ─── P39b warm-up hints (LAZY, cached) ──────────────────────────────────────
#
# BUG-129 (2026-07-25): these used to be resolved ONCE inside `apply()`.
# `apply()` runs under the compose entrypoint's
#     python3 -m vllm._genesis.patches.apply_all
# which is a standalone process with no engine in it, so
# `get_current_vllm_config()` ALWAYS raises there ("Current vLLM config is
# not set") and P39b silently fell back to max_T=4096 / max_B=2 on every
# single boot. On this rig that is materially wrong: chunked prefill
# dispatches chunks of `mamba_block_size` = `max_num_batched_tokens` = 4128
# tokens, i.e. 32 MORE than the 4096 default, so the pool would be born
# undersized and then GROW on the first real chunk — a pool pointer swap,
# which is precisely the CUDA-graph invalidation P39b exists to prevent.
#
# Fix: resolve LAZILY at first kernel call (inside the serving process,
# where a config context can exist) and cache. Precedence mirrors P73:
#   1. GENESIS_FLA_KKT_MAX_T / _MAX_B  (explicit operator override)
#   2. central P73 `prealloc_budget.resolve_token_budget()` for max_T —
#      which itself consults GENESIS_PREALLOC_TOKEN_BUDGET, the domain env,
#      then the live `scheduler_config.max_num_batched_tokens`
#   3. live `scheduler_config.max_num_seqs` for max_B
#   4. conservative defaults below
#
# BUG-071 lesson (see prealloc_budget.py "Priority 4"): a *default* result
# is NEVER cached. Only a value that came from a real signal is pinned, so
# an early call made before the engine exists cannot poison every later one.
_DEFAULT_MAX_T = 4096
_DEFAULT_MAX_B = 2
_ENV_MAX_T = "GENESIS_FLA_KKT_MAX_T"
_ENV_MAX_B = "GENESIS_FLA_KKT_MAX_B"

# key -> (value, source). `source` is "" for an unresolved/default result,
# which is deliberately not stored.
_HINT_CACHE: dict[str, tuple[int, str]] = {}


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name, "")
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return None


def _probe_scheduler_config() -> Any:
    """Return the live `scheduler_config`, or None when no config context.

    Mirrors `prealloc_budget._probe_vllm_config`: dev1060+ raises from
    `get_current_vllm_config()` outside the context, so prefer the
    `_or_none` accessor where the build has it.
    """
    try:
        try:
            from vllm.config import get_current_vllm_config_or_none
            cfg = get_current_vllm_config_or_none()
        except ImportError:
            from vllm.config import get_current_vllm_config
            cfg = get_current_vllm_config()
        if cfg is None:
            return None
        return getattr(cfg, "scheduler_config", None)
    except Exception:
        return None


def _resolve_max_T() -> tuple[int, str]:
    env = _env_int(_ENV_MAX_T)
    if env is not None:
        return env, _ENV_MAX_T
    try:
        from sndr.runtime.prealloc_budget import (
            resolve_token_budget, get_cached,
        )
        value = int(resolve_token_budget(domain_env=_ENV_MAX_T))
        # `get_cached()` is non-None only when P73 resolved from a REAL
        # source (env / live scheduler_config). When it returns None the
        # value we got back is P73's own uncached fallback — treat it as
        # unresolved so we keep re-probing.
        if get_cached() is not None and value > 0:
            return value, "P73 prealloc_budget"
    except Exception as e:  # noqa: BLE001
        log.debug("[Genesis P39b] prealloc_budget probe failed: %s", e)
    return _DEFAULT_MAX_T, ""


def _resolve_max_B() -> tuple[int, str]:
    env = _env_int(_ENV_MAX_B)
    if env is not None:
        return env, _ENV_MAX_B
    sched = _probe_scheduler_config()
    if sched is not None:
        ms = getattr(sched, "max_num_seqs", None)
        if ms:
            return int(ms), "vllm scheduler_config.max_num_seqs"
    return _DEFAULT_MAX_B, ""


def resolve_kkt_hints() -> tuple[int, int]:
    """Lazily resolve (max_T, max_B) for `FlaKktBufferManager.acquire`.

    Safe to call on every kernel invocation: once a field resolves from a
    real source it is a plain dict read — no env reads, no config probes
    inside the hot path (dynamo-safe, same contract as P73).
    """
    out: list[int] = []
    for field, resolver, default in (
        ("max_T", _resolve_max_T, _DEFAULT_MAX_T),
        ("max_B", _resolve_max_B, _DEFAULT_MAX_B),
    ):
        cached = _HINT_CACHE.get(field)
        if cached is not None:
            out.append(cached[0])
            continue
        value, source = resolver()
        if source:
            _HINT_CACHE[field] = (value, source)
            log.info(
                "[Genesis P39b] %s resolved → %d (via %s) — pool pre-sized "
                "before first grow, pointer-stable for CUDA-graph capture",
                field, value, source,
            )
        else:
            # Unresolved: use the default for THIS call but do not pin it.
            log.debug(
                "[Genesis P39b] %s unresolved (no env / no vLLM config "
                "context yet) → using default %d for this call, will "
                "re-probe on the next one",
                field, default,
            )
        out.append(value)
    return out[0], out[1]


def reset_hints_for_tests() -> None:
    """TESTS ONLY — drop the lazy hint cache."""
    _HINT_CACHE.clear()


def ensure_pool_registered() -> None:
    """Idempotent registry hook — exposes POOL_FLA_KKT_PERSISTENT_A in
    PersistentBufferRegistry for operator visibility. No allocation,
    no behavior change.

    The real FLA KKT `A` tensor (B, T, H, BT, fp32) is owned by
    sndr.engines.vllm.kernels_legacy.fla_kkt_buffer.FlaKktBufferManager via the
    reserve-before-cudagraph pattern (P39b). Its allocation semantics
    are GROW-IN-PLACE + SLICE-ON-ACQUIRE keyed by (H, BT, device, dtype)
    — variable first two dims (B, T) → fixed last two — which matches
    PersistentSlicePool exactly.

    v11.3.0 bug fix: this was previously calling `get_pool()` which
    creates a BufferPool (free-list acquire/release semantics, wrong
    pool type for P39a). Switched to `get_slice_pool()` which matches
    the actual allocation pattern. Operator-visibility only — does
    not change any runtime allocation behavior; the actual storage
    still lives in FlaKktBufferManager via GPB.
    """
    PersistentBufferRegistry().get_slice_pool(POOL_FLA_KKT_PERSISTENT_A)

# Module paths we target. Primary + candidates for future renames.
# [2026-07-25] vllm#48500 moved fla ops to third_party/flash_linear_attention;
# new path first, old path kept for prod's dev1060cherry pin.
_CANDIDATE_MODULE_PATHS = (
    "vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt",
    "vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt",
)
_FN_NAME = "chunk_scaled_dot_kkt_fwd"

# ─── P39a self-install (exec-survival) ──────────────────────────────────────
#
# The shipped compose entrypoint is
#     python3 -m vllm._genesis.patches.apply_all
#     exec vllm serve "$@"
# `exec` REPLACES the process, so every setattr/monkey-patch made by apply()
# is discarded before a single token is served. Only TEXT patches (which
# write files under the vllm install root) survive. P39a has always been
# setattr-only, so it logged "[Genesis] applied: P39a" each boot and then
# did nothing where it mattered — the same incident class documented in
# `wiring/hybrid/patch_103_fla_cliff2_chunked.py` (cross-rig club-3090#19,
# 2026-05-02).
#
# Fix (copies P103's sanctioned mechanism verbatim in shape): text-patch
# `chunk_scaled_dot_kkt.py` to APPEND a module-import-time hook that calls
# `_genesis_p39a_install_at_import(globals())`. Every fresh import — main
# process, `exec vllm serve`, spawned workers — re-installs the pooled
# wrapper.
#
# GUARDED: this is a live behavioural change on a kernel path, so it is
# opt-in and DEFAULTS OFF. With the flag unset apply() does not write a
# single byte to chunk_scaled_dot_kkt.py and the appended hook (if a prior
# opt-in boot left one behind) short-circuits at import.
_ENV_SELFINSTALL = "GENESIS_ENABLE_P39A_SELFINSTALL"
_TRUTHY = ("1", "true", "yes", "on")

_GENESIS_P39A_SELFINSTALL_MARKER = (
    "Genesis P39a self-install hook (exec-survival, club-3090#19 class)"
)


def _selfinstall_enabled() -> bool:
    return os.environ.get(_ENV_SELFINSTALL, "").strip().lower() in _TRUTHY


def should_apply() -> bool:
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


def _make_pooled_kkt_fwd(ns_get: Any) -> Any:
    """Build the pooled drop-in for `chunk_scaled_dot_kkt_fwd`.

    `ns_get(name, default=None)` resolves symbols out of the TARGET
    module's namespace. Two callers, two bindings:

      * `apply()`               -> `getattr(mod, name, default)`
      * the self-install hook   -> `module_globals.get(name, default)`

    Both are late-bound (read at call time, not build time) so a later
    text-patch that swaps `FLA_CHUNK_SIZE` or re-JITs the kernel is picked
    up without rebuilding the wrapper.

    Caller is responsible for setting the marker attrs.
    """
    from sndr.engines.vllm.kernels_legacy.fla_kkt_buffer import (
        FlaKktBufferManager,
    )

    # Lazily-probed set of parameter names the Triton kernel DECLARES.
    # List-cell so the nested wrapper can write without `nonlocal` churn.
    _KKT_DECLARED: list[Any] = [None]

    def _genesis_pooled_chunk_scaled_dot_kkt_fwd(
        k,
        g=None,
        beta=None,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=None,
        output_dtype=None,
        use_exp2=False,
    ):
        """Signature-compatible drop-in around the original.

        Replaces the `A = torch.empty(B, T, H, BT, ...)` line with a
        pooled acquire + same Triton kernel call. Everything else
        (heuristics, autotune, store layout) is untouched because we
        pass a same-shape same-stride view.

        P39b (reserve-before-cudagraph): `max_T` and `max_B` are passed
        on every call so the pool grows to its final size on the FIRST
        call (typically at profile_run with small batch) — afterwards
        all calls reuse the same buffer pointer, eliminating any risk
        of CUDA-graph invalidation from pool pointer-swap on growth.
        The hints are resolved LAZILY here (first call, cached) because
        apply() runs in a process that has no vLLM config context, and
        because under the self-install hook there is no `apply()` in
        this process at all.
        """
        import triton
        import torch

        # Resolve defaults by asking the module (in case upstream bumps
        # FLA_CHUNK_SIZE or changes output dtype default).
        if chunk_size is None:
            chunk_size = ns_get("FLA_CHUNK_SIZE", 64)
        if output_dtype is None:
            output_dtype = torch.float32

        B, T, Hg, K = k.shape
        H = beta.shape[-1]
        BT = chunk_size
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = ns_get("prepare_chunk_indices")(cu_seqlens, BT)
        NT = (
            triton.cdiv(T, BT)
            if cu_seqlens is None
            else len(chunk_indices)
        )

        # POOLED acquire — P39a core + P39b pre-sizing hints (lazy)
        _max_t, _max_b = resolve_kkt_hints()
        A = FlaKktBufferManager.acquire(
            B=B, T=T, H=H, BT=BT,
            device=k.device, dtype=output_dtype,
            max_T=_max_t,
            max_B=_max_b,
        )

        # Constexpr kwargs that the upstream launcher passes but our
        # hand-rolled launch would otherwise DROP. Triton treats a
        # `tl.constexpr` parameter with no default as REQUIRED, so a
        # dropped one is a hard `missing 1 required positional argument`
        # at the first GDN prefill — not a silent degradation.
        #
        #   USE_EXP2             — added by the PN354 text-patch.
        #     [Genesis PN354 composition fix v2 2026-06-10] v1 passed the
        #     flag only when truthy and crashed boot when the env was off.
        #     Correct rule: pass it whenever the KERNEL declares the
        #     parameter, regardless of the env value; omit only when the
        #     kernel is unpatched.
        #   CAST_DOT_TO_K_DTYPE  — added UPSTREAM by nightly-0ba2aa35's
        #     RDNA/WMMA rework (present on dev1474cherry*, absent on
        #     dev1060cherry). Its value is the target module's own
        #     `_CAST_DOT_TO_K_DTYPE` global (False on every NVIDIA rig).
        #     This wrapper never passed it, which is exactly why enabling
        #     P39a for real needs this fix landed in the same commit.
        #
        # Probe the declared names ONCE (both text-patches apply at an
        # earlier ordinal than P39a, so the kernel signature is final).
        if _KKT_DECLARED[0] is None:
            _kern = ns_get("chunk_scaled_dot_kkt_fwd_kernel")
            _names = getattr(_kern, "arg_names", None)
            if not _names:
                _names = getattr(getattr(_kern, "fn", None), "arg_names", None)
            _KKT_DECLARED[0] = frozenset(_names or ())
        _declared = _KKT_DECLARED[0]

        _kkt_extra = {}
        if "USE_EXP2" in _declared:
            _kkt_extra["USE_EXP2"] = bool(use_exp2)
        if "CAST_DOT_TO_K_DTYPE" in _declared:
            _kkt_extra["CAST_DOT_TO_K_DTYPE"] = bool(
                ns_get("_CAST_DOT_TO_K_DTYPE", False)
            )

        ns_get("chunk_scaled_dot_kkt_fwd_kernel")[(NT, B * H)](
            k=k,
            g=g,
            beta=beta,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            BT=BT,
            **_kkt_extra,
        )
        return A

    return _genesis_pooled_chunk_scaled_dot_kkt_fwd


def _genesis_p39a_install_at_import(module_globals: Any) -> bool:
    """Install the pooled wrapper into chunk_scaled_dot_kkt.py's globals.

    Called from the text-patched bottom of chunk_scaled_dot_kkt.py at
    MODULE-IMPORT TIME. Survives `exec vllm serve` and worker spawn
    because every fresh import of the module runs the appended block.

    Returns True if installed (or already installed), False if skipped.
    NEVER raises — a failure here must not break the module's import.
    """
    try:
        if not _selfinstall_enabled():
            return False

        original = module_globals.get(_FN_NAME)
        if original is None:
            return False
        if getattr(original, _GENESIS_P39A_MARKER_ATTR, False):
            return True  # idempotent

        # P49 interface contract, self-install flavour: the wrapper calls
        # these by name out of the module dict. Missing symbol => upstream
        # drift => leave the original alone.
        for required in (
            "chunk_scaled_dot_kkt_fwd_kernel",
            "prepare_chunk_indices",
        ):
            if required not in module_globals:
                log.debug(
                    "[Genesis P39a self-install] interface drift: %r missing "
                    "from chunk_scaled_dot_kkt.py globals — skipping",
                    required,
                )
                return False

        try:
            wrapper = _make_pooled_kkt_fwd(
                lambda name, default=None: module_globals.get(name, default)
            )
        except Exception as e:  # noqa: BLE001
            log.debug(
                "[Genesis P39a self-install] pool manager unavailable: %s", e,
            )
            return False

        setattr(wrapper, _GENESIS_P39A_MARKER_ATTR, True)
        setattr(wrapper, "_genesis_p39a_original", original)
        module_globals[_FN_NAME] = wrapper
        log.info(
            "[Genesis P39a self-install] pooled chunk_scaled_dot_kkt_fwd "
            "installed at module-import time (survives `exec vllm serve` + "
            "worker spawn)"
        )
        return True
    except Exception as e:  # noqa: BLE001
        log.debug("[Genesis P39a self-install] non-fatal failure: %s", e)
        return False


# ─── text-patch: append the self-install hook ───────────────────────────────
#
# Anchor policy (dual-pin safe). The tail of `chunk_scaled_dot_kkt_fwd`
# differs across every pin AND across our own earlier patches:
#
#   dev1060cherry-20260713            ...  BT=BT,\n    )\n    return A
#   dev1474cherry*-20260725           ...  BT=BT,\n  CAST_DOT_TO_K_DTYPE=..
#   either of the above + PN354       ...  + USE_EXP2=use_exp2,
#
# So we anchor on the invariant two-line file tail instead of on any of the
# four launch-site shapes, and assert it is UNIQUE before splicing. Verified
# count == 1 against all three live image pins.
_P39A_SELF_INSTALL_ANCHOR = "    )\n    return A\n"

_P39A_SELF_INSTALL_BLOCK = (
    "\n\n"
    "# ============================================================\n"
    "# [Genesis P39a self-install] — module-import-time hook\n"
    "# ============================================================\n"
    "# When GENESIS_ENABLE_P39A_SELFINSTALL=1, rebind\n"
    "# chunk_scaled_dot_kkt_fwd to the pooled version at module-import\n"
    "# time so the P39a buffer pool survives any startup mechanism:\n"
    "# `exec vllm serve` from an entrypoint shell, worker spawn, etc.\n"
    "# The setattr-only path died on the entrypoint pattern\n"
    "# (`python3 -m vllm._genesis.patches.apply_all && exec vllm serve`)\n"
    "# — same incident class as P103, see club-3090#19 (2026-05-02).\n"
    "#\n"
    "# Lazy import — if the sndr tree isn't on sys.path (test env,\n"    "# partial install), the try/except keeps this module importable.\n"
    "try:\n"
    "    import os as _genesis_p39a_os\n"
    "    if _genesis_p39a_os.environ.get(\n"
    "        \"GENESIS_ENABLE_P39A_SELFINSTALL\", \"\"\n"
    "    ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\"):\n"
    "        from sndr.engines.vllm.patches.attention.gdn"
    ".p39a_fla_kkt_buffer import (\n"
    "            _genesis_p39a_install_at_import as "
    "_genesis_p39a_install,\n"
    "        )\n"
    "        _genesis_p39a_install(globals())\n"
    "except Exception:  # noqa: BLE001\n"
    "    # Never break chunk_scaled_dot_kkt.py import — P39a is opt-in.\n"
    "    pass\n"
)


def _make_self_install_text_patcher():
    """Build the text-patch that appends the self-install hook.

    Returns None when the vllm tree is unresolvable or the anchor is not
    uniquely present (upstream drift) — the caller treats None as "skip".
    """
    from sndr.engines.vllm.detection.guards import resolve_vllm_file
    from sndr.kernel import TextPatch, TextPatcher

    # NOTE: `resolve_vllm_file` returns a **str | None**, NOT a Path — and it
    # already aliases the vllm#48500 fla move in BOTH directions, so passing
    # either home resolves on either pin. (Treating the return as a Path cost
    # a boot cycle on 2026-07-25; see 92647851 -> 748d2a0b.)
    target = resolve_vllm_file(
        "third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py"
    )
    if not target:
        return None

    # Content-sniff: never splice an anchor we have not confirmed is present
    # exactly once in THIS pin's file. Wrapped in `except Exception` because a
    # read failure here must degrade to "skip", never to a boot abort.
    try:
        with open(target, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:  # noqa: BLE001
        log.debug("[Genesis P39a] anchor sniff failed to read %s: %s", target, e)
        return None

    occurrences = content.count(_P39A_SELF_INSTALL_ANCHOR)
    if occurrences != 1:
        log.debug(
            "[Genesis P39a] anchor sniff: tail anchor found %d times in %s "
            "(need exactly 1) — skipping self-install text-patch",
            occurrences, target,
        )
        return None

    return TextPatcher(
        patch_name=(
            "P39a fla/ops/chunk_scaled_dot_kkt.py — self-install hook "
            "(exec-survival)"
        ),
        target_file=str(target),
        marker=_GENESIS_P39A_SELFINSTALL_MARKER,
        sub_patches=[
            TextPatch(
                name="p39a_self_install_at_kkt_py_end",
                anchor=_P39A_SELF_INSTALL_ANCHOR,
                replacement=(
                    _P39A_SELF_INSTALL_ANCHOR + _P39A_SELF_INSTALL_BLOCK
                ),
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # Specific to our own insertion (re-runs hit Layer 2 IDEMPOTENT
            # first via the wiring marker).
            "[Genesis P39a self-install]",
        ],
    )


def _apply_self_install_text_patch() -> tuple[str, str]:
    """Run the self-install text-patch. Returns (status, reason).

    Never raises. Returns ("skipped", ...) when the opt-in flag is unset —
    which means the target file is left BYTE-IDENTICAL.
    """
    if not _selfinstall_enabled():
        return "skipped", f"{_ENV_SELFINSTALL} not set (default OFF)"
    try:
        from sndr.engines.vllm.detection.guards import vllm_install_root
        from sndr.kernel import TextPatchResult

        if vllm_install_root() is None:
            return "skipped", "vllm tree not resolvable"
        patcher = _make_self_install_text_patcher()
        if patcher is None:
            return "skipped", "target file missing or anchor not unique"
        result, failure = patcher.apply()
        if result == TextPatchResult.APPLIED:
            return "applied", (
                "chunk_scaled_dot_kkt.py self-install hook appended "
                "(survives `exec vllm serve` + worker spawn)"
            )
        if result == TextPatchResult.IDEMPOTENT:
            return "idempotent", "self-install hook already present"
        return "skipped", (
            f"text-patch did not land: "
            f"{failure.reason if failure else 'unknown'} — "
            f"{failure.detail if failure and failure.detail else 'unknown'}"
        )
    except Exception as e:  # noqa: BLE001
        log.debug("[Genesis P39a] self-install text-patch raised: %s", e)
        return "skipped", f"text-patch raised: {e}"


def _import_target() -> tuple[Any, Any] | None:
    """Return (module, original_fn) or None on failure."""
    import importlib
    for modpath in _CANDIDATE_MODULE_PATHS:
        try:
            mod = importlib.import_module(modpath)
        except ImportError:
            continue
        except Exception as e:
            log.warning("[Genesis P39a] import %s: %s", modpath, e)
            continue
        fn = getattr(mod, _FN_NAME, None)
        if fn is not None:
            return mod, fn
    return None


def apply() -> tuple[str, str]:
    """Rebind `chunk_scaled_dot_kkt_fwd` to the pooled version.

    Never raises. Returns (status, reason).
    """
    if not should_apply():
        return "skipped", "platform: NVIDIA SM 8.0+ required"

    # P53 (v7.9): Hybrid-active dispatch gate. chunk_scaled_dot_kkt_fwd is
    # FLA-GDN only. Pure-attention models may not even have the FLA module
    # imported — the target-import check below would skip, but we log
    # the dispatch reason up-front.
    try:
        from sndr.engines.vllm.detection.model_detect import is_hybrid_model, log_skip
        if not is_hybrid_model():
            log_skip(
                "P39a FLA chunk_scaled_dot_kkt pool",
                "pure-attention model (no GDN chunked-prefill)",
            )
            return "skipped", "P53 dispatch: model has no hybrid linear-attention layers"
    except Exception as e:
        log.debug("[Genesis P39a] model_detect probe failed (proceeding): %s", e)

    # ─── Step 1: durable text-patch (the ONLY step that survives exec) ──
    # Opt-in, default OFF. With the flag unset this writes nothing and the
    # next boot is byte-for-byte unchanged.
    ti_status, ti_reason = _apply_self_install_text_patch()

    target = _import_target()
    if target is None:
        return "skipped", (
            f"FLA module {_CANDIDATE_MODULE_PATHS[0]!r} or symbol "
            f"{_FN_NAME!r} not available (not an FLA-GDN build); "
            f"self-install={ti_status} ({ti_reason})"
        )
    mod, original = target

    # P49 interface contract check (v7.8): our replacement calls
    # `mod.chunk_scaled_dot_kkt_fwd_kernel`, `mod.FLA_CHUNK_SIZE`,
    # and `mod.prepare_chunk_indices`. If upstream renamed ANY of
    # these, we bail rather than calling into a missing symbol at
    # first forward.
    #
    # Note: Triton `@triton.jit`-decorated kernels are `JITFunction`
    # instances that are NOT `callable()` in the Python sense (you
    # invoke via `kernel[grid](*args)`). So for the kernel symbol we
    # use `required_attrs={...: ANY}` (presence check) instead of
    # `required_methods` (callable check). For `chunk_scaled_dot_kkt_fwd`
    # (the regular Python wrapper) and `prepare_chunk_indices` (also
    # plain Python), `required_methods` works fine.
    try:
        from sndr.runtime.interface_guard import (
            validate_impl, ANY,
        )
        validate_impl(
            mod,
            role="FLA chunk_scaled_dot_kkt module (P39a)",
            required_attrs={
                "chunk_scaled_dot_kkt_fwd_kernel": ANY,  # Triton JIT
                "FLA_CHUNK_SIZE": int,
            },
            required_methods=[
                "chunk_scaled_dot_kkt_fwd",
                "prepare_chunk_indices",
            ],
        )
    except Exception as e:
        if "GenesisInterfaceMismatch" in type(e).__name__:
            return "skipped", f"P49 interface drift: {e}"

    if getattr(original, _GENESIS_P39A_MARKER_ATTR, False):
        return "applied", (
            f"already wrapped (idempotent — self-install hook fired or a "
            f"prior apply is still live in this process); "
            f"self-install={ti_status} ({ti_reason})"
        )

    try:
        from sndr.engines.vllm.kernels_legacy.fla_kkt_buffer import FlaKktBufferManager
    except Exception as e:
        return "failed", f"kernel import failed: {e}"

    # v11.1.0 P3.3: expose the pool name in the registry — no allocation,
    # purely operator-visibility surface.
    try:
        ensure_pool_registered()
    except Exception as e:
        log.debug("[P39a] registry pool registration failed (proceeding): %s", e)

    # P39b: warm the (max_T, max_B) hints now if — and only if — a real
    # signal is already available. Under the shipped entrypoint
    # (`python3 -m vllm._genesis.patches.apply_all` then `exec vllm serve`)
    # apply() runs in a standalone process with no engine, so there is no
    # vLLM config context here and this is EXPECTED, not an incident: the
    # real resolution happens lazily at the first kernel call inside the
    # serving process. Hence DEBUG, not INFO — see resolve_kkt_hints().
    _t, _b = resolve_kkt_hints()
    if _HINT_CACHE:
        log.info(
            "[Genesis P39b] warm-up hints available at apply time: "
            "max_T=%d max_B=%d (%s)",
            _t, _b,
            ", ".join(f"{k}<-{v[1]}" for k, v in sorted(_HINT_CACHE.items())),
        )
    else:
        log.debug(
            "[Genesis P39b] no vLLM config context at apply time (normal for "
            "the `apply_all` + `exec vllm serve` entrypoint) — hints will be "
            "resolved lazily at the first kernel call; defaults meanwhile "
            "are max_T=%d max_B=%d",
            _t, _b,
        )

    # Build the pooled drop-in. Shared, module-level factory — the SAME
    # builder the module-import-time self-install hook uses, so the two
    # install paths can never drift.
    _genesis_pooled_chunk_scaled_dot_kkt_fwd = _make_pooled_kkt_fwd(
        lambda name, default=None: getattr(mod, name, default)
    )

    # Marker + preserve the original so revert can restore it.
    setattr(
        _genesis_pooled_chunk_scaled_dot_kkt_fwd,
        _GENESIS_P39A_MARKER_ATTR, True,
    )
    setattr(
        _genesis_pooled_chunk_scaled_dot_kkt_fwd,
        "_genesis_p39a_original", original,
    )

    setattr(mod, _FN_NAME, _genesis_pooled_chunk_scaled_dot_kkt_fwd)

    # ALSO rebind on any already-imported callers. FLA internal code
    # typically imports via `from .chunk_scaled_dot_kkt import
    # chunk_scaled_dot_kkt_fwd` — those modules will retain the ORIGINAL
    # reference. To fix, we walk the chunk_delta_h importer.
    # However: callers inside the AOT-compiled model path resolve the
    # symbol from the `mod` namespace at call time when accessed as
    # attribute. Most FLA internal calls DO `from ... import ...` →
    # they capture the original. To cover both, we also rebind the
    # symbol inside `vllm.model_executor.layers.fla.ops.chunk_delta_h`
    # and siblings if they imported it.
    import sys as _sys
    rebound_callers = []
    # [2026-07-25] both fla homes (vllm#48500 move) — old kept for prod pin.
    fla_ops_prefixes = (
        "vllm.third_party.flash_linear_attention.ops",
        "vllm.model_executor.layers.fla.ops",
    )
    for mod_name, caller_mod in list(_sys.modules.items()):
        if caller_mod is None:
            continue
        if not mod_name.startswith(fla_ops_prefixes):
            continue
        if mod_name in _CANDIDATE_MODULE_PATHS:
            continue
        existing = getattr(caller_mod, _FN_NAME, None)
        if existing is original:
            try:
                setattr(
                    caller_mod, _FN_NAME,
                    _genesis_pooled_chunk_scaled_dot_kkt_fwd,
                )
                rebound_callers.append(mod_name)
            except Exception as e:
                log.debug(
                    "[Genesis P39a] couldn't rebind in %s: %s",
                    mod_name, e,
                )

    log.info(
        "[Genesis P39a] rebound %s.%s (+%d caller mods: %s)",
        _CANDIDATE_MODULE_PATHS[0], _FN_NAME,
        len(rebound_callers), rebound_callers,
    )
    return "applied", (
        f"self-install={ti_status} ({ti_reason}); module-level fn replaced "
        f"({len(rebound_callers)} caller module(s) also rebound — pool "
        f"shared across GDN layers). NOTE: the setattr half does NOT "
        f"survive `exec vllm serve`; only the self-install text-patch does."
    )


def is_applied() -> bool:
    target = _import_target()
    if target is None:
        return False
    _mod, fn = target
    return getattr(fn, _GENESIS_P39A_MARKER_ATTR, False)


def revert() -> bool:
    """Restore the original function. For tests only."""
    target = _import_target()
    if target is None:
        return False
    mod, fn = target
    if not getattr(fn, _GENESIS_P39A_MARKER_ATTR, False):
        return False
    original = getattr(fn, "_genesis_p39a_original", None)
    if original is None:
        return False
    setattr(mod, _FN_NAME, original)
    # Restore in caller mods too
    import sys as _sys
    for mod_name, caller_mod in _sys.modules.items():
        if caller_mod is None or mod_name == _CANDIDATE_MODULE_PATHS[0]:
            continue
        existing = getattr(caller_mod, _FN_NAME, None)
        if existing is fn:
            setattr(caller_mod, _FN_NAME, original)
    return True
