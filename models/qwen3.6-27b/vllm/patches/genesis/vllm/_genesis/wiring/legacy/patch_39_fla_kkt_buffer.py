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
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least

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
        from vllm._genesis.prealloc_budget import (
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

# Module paths we target. Primary + candidates for future renames.
# [2026-07-25] vllm#48500 moved fla ops to third_party/flash_linear_attention;
# new path first, old path kept for prod's dev1060cherry pin.
_CANDIDATE_MODULE_PATHS = (
    "vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt",
    "vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt",
)
_FN_NAME = "chunk_scaled_dot_kkt_fwd"


def should_apply() -> bool:
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


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
        from vllm._genesis.model_detect import is_hybrid_model, log_skip
        if not is_hybrid_model():
            log_skip(
                "P39a FLA chunk_scaled_dot_kkt pool",
                "pure-attention model (no GDN chunked-prefill)",
            )
            return "skipped", "P53 dispatch: model has no hybrid linear-attention layers"
    except Exception as e:
        log.debug("[Genesis P39a] model_detect probe failed (proceeding): %s", e)

    target = _import_target()
    if target is None:
        return "skipped", (
            f"FLA module {_CANDIDATE_MODULE_PATHS[0]!r} or symbol "
            f"{_FN_NAME!r} not available (not an FLA-GDN build)"
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
        from vllm._genesis.interface_guard import (
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
        return "applied", "already wrapped (idempotent)"

    try:
        from vllm._genesis.kernels.fla_kkt_buffer import FlaKktBufferManager
    except Exception as e:
        return "failed", f"kernel import failed: {e}"

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

    # PN354 composition: lazily resolved "does the kernel declare
    # USE_EXP2" flag (None = not yet probed). List-cell so the nested
    # wrapper can write without `nonlocal` churn.
    _KKT_HAS_EXP2 = [None]

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
        apply() runs in a process that has no vLLM config context.
        """
        import triton
        import torch

        # Resolve defaults by asking the module (in case upstream bumps
        # FLA_CHUNK_SIZE or changes output dtype default).
        if chunk_size is None:
            try:
                chunk_size = mod.FLA_CHUNK_SIZE
            except AttributeError:
                chunk_size = 64
        if output_dtype is None:
            output_dtype = torch.float32

        B, T, Hg, K = k.shape
        H = beta.shape[-1]
        BT = chunk_size
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = mod.prepare_chunk_indices(cu_seqlens, BT)
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

        # [Genesis PN354 composition fix v2 2026-06-10] the PN354
        # text-patch adds `USE_EXP2: tl.constexpr` (NO default — Triton
        # treats it as REQUIRED) to the kernel this wrapper launches.
        # v1 of this fix passed the flag only when truthy and crashed
        # boot with "dynamic_func() missing 1 required positional
        # argument: 'USE_EXP2'" when env was off. Correct rule: pass
        # USE_EXP2 whenever the KERNEL declares the parameter (PN354
        # text applied — P39a applies at a later ordinal so the state
        # is final by now), regardless of the env value; omit it only
        # when the kernel is unpatched.
        if _KKT_HAS_EXP2[0] is None:
            _kern = mod.chunk_scaled_dot_kkt_fwd_kernel
            _names = getattr(_kern, "arg_names", None)
            if not _names:
                _names = getattr(getattr(_kern, "fn", None), "arg_names", None)
            _KKT_HAS_EXP2[0] = bool(_names) and "USE_EXP2" in _names
        _kkt_extra = (
            {"USE_EXP2": bool(use_exp2)} if _KKT_HAS_EXP2[0] else {}
        )
        mod.chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
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
        f"module-level fn replaced ({len(rebound_callers)} caller "
        f"module(s) also rebound — pool shared across GDN layers)"
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
