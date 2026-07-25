# SPDX-License-Identifier: Apache-2.0
"""P39b lazy warm-up-hint resolution (BUG-129).

P39b used to read `get_current_vllm_config()` inside `apply()`. `apply()`
runs under the shipped entrypoint as

    python3 -m vllm._genesis.patches.apply_all   # standalone process
    exec vllm serve "$@"                          # engine lives HERE

so there is never a vLLM config context at apply time and P39b fell back
to `max_T=4096 / max_B=2` on every boot — silently, at INFO. On this rig
that is materially wrong (`max_num_batched_tokens=4128`), so the pool
would be born undersized and pointer-swap on the first real chunk.

These tests pin the fixed contract:
  * config absent  -> defaults are USED but NOT cached (keeps re-probing)
  * config present -> real values, cached, one INFO line
  * env override   -> wins over everything
  * an early unresolved call must not poison a later resolved one
    (the BUG-071 "poisoned cache" regression)

Runs with the stdlib only — no torch, no vLLM, no pytest required:

    python3 vllm/_genesis/tests/test_p39b_lazy_hints.py

and is also collected normally by pytest inside the container.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_GENESIS = _HERE.parent                      # vllm/_genesis
_TARGET = _GENESIS / "wiring" / "legacy" / "patch_39_fla_kkt_buffer.py"
_BUDGET = _GENESIS / "prealloc_budget.py"

_ENVS = (
    "GENESIS_FLA_KKT_MAX_T",
    "GENESIS_FLA_KKT_MAX_B",
    "GENESIS_PREALLOC_TOKEN_BUDGET",
)


class _SchedulerConfig:
    def __init__(self, max_num_batched_tokens, max_num_seqs):
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs


class _VllmConfig:
    def __init__(self, scheduler_config):
        self.scheduler_config = scheduler_config


def _install_stubs():
    """Minimal `vllm` namespace so the target module imports without torch.

    `prealloc_budget` is loaded for REAL (stdlib-only) so the P73
    precedence chain under test is the shipped one, not a mock of it.
    """
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            del sys.modules[name]

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []  # namespace-ish
    genesis = types.ModuleType("vllm._genesis")
    genesis.__path__ = [str(_GENESIS)]

    guards = types.ModuleType("vllm._genesis.guards")
    guards.is_nvidia_cuda = lambda: True
    guards.is_sm_at_least = lambda *_a, **_k: True

    config = types.ModuleType("vllm.config")
    config._current = None

    def get_current_vllm_config_or_none():
        return config._current

    def get_current_vllm_config():
        if config._current is None:
            raise RuntimeError("Current vLLM config is not set.")
        return config._current

    config.get_current_vllm_config_or_none = get_current_vllm_config_or_none
    config.get_current_vllm_config = get_current_vllm_config

    sys.modules["vllm"] = vllm
    sys.modules["vllm._genesis"] = genesis
    sys.modules["vllm._genesis.guards"] = guards
    sys.modules["vllm.config"] = config

    budget = _load("vllm._genesis.prealloc_budget", _BUDGET)
    sys.modules["vllm._genesis.prealloc_budget"] = budget
    return config, budget


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fresh():
    """Fresh (target_module, fake_vllm_config, prealloc_budget) triple."""
    for key in _ENVS:
        os.environ.pop(key, None)
    config, budget = _install_stubs()
    budget.reset_for_tests()
    p39 = _load(
        "vllm._genesis.wiring.legacy.patch_39_fla_kkt_buffer", _TARGET,
    )
    p39.reset_hints_for_tests()
    return p39, config, budget


# ── the tests ───────────────────────────────────────────────────────────────

def test_no_config_context_uses_defaults_and_does_not_cache():
    """apply_all's standalone process: no engine, no env → defaults, uncached."""
    p39, config, _ = _fresh()
    config._current = None

    assert p39.resolve_kkt_hints() == (
        p39._DEFAULT_MAX_T, p39._DEFAULT_MAX_B,
    ) == (4096, 2)
    # Nothing pinned — this is the BUG-071 rule. A default must never be
    # cached or the serving process can never correct it.
    assert p39._HINT_CACHE == {}


def test_live_config_resolves_our_real_values():
    """Serving process, this rig's actual config."""
    p39, config, _ = _fresh()
    config._current = _VllmConfig(
        _SchedulerConfig(max_num_batched_tokens=4128, max_num_seqs=4)
    )

    max_t, max_b = p39.resolve_kkt_hints()
    assert (max_t, max_b) == (4128, 4), (max_t, max_b)
    assert p39._HINT_CACHE["max_T"][1] == "P73 prealloc_budget"
    assert p39._HINT_CACHE["max_B"][1] == "vllm scheduler_config.max_num_seqs"

    # THE MATERIAL POINT: the old apply-time read produced 4096, and 4096
    # < 4128 means the pool grows on the first full chunk → pointer swap.
    assert max_t > p39._DEFAULT_MAX_T
    assert max_b > p39._DEFAULT_MAX_B


def test_prealloc_token_budget_env_is_honoured():
    """GENESIS_PREALLOC_TOKEN_BUDGET is set on this rig (boot log: 4160)."""
    p39, config, _ = _fresh()
    config._current = None
    os.environ["GENESIS_PREALLOC_TOKEN_BUDGET"] = "4160"

    max_t, max_b = p39.resolve_kkt_hints()
    assert max_t == 4160, max_t
    # max_B has no central resolver and no config context here.
    assert max_b == p39._DEFAULT_MAX_B
    assert "max_B" not in p39._HINT_CACHE  # still re-probing


def test_domain_env_override_wins():
    p39, config, _ = _fresh()
    config._current = _VllmConfig(
        _SchedulerConfig(max_num_batched_tokens=4128, max_num_seqs=4)
    )
    os.environ["GENESIS_PREALLOC_TOKEN_BUDGET"] = "4160"
    os.environ["GENESIS_FLA_KKT_MAX_T"] = "8192"
    os.environ["GENESIS_FLA_KKT_MAX_B"] = "9"

    assert p39.resolve_kkt_hints() == (8192, 9)


def test_early_default_call_does_not_poison_later_resolution():
    """The regression this fix exists for.

    Call once with no config (as `apply()` now does), then again once the
    engine exists. The second call MUST see the real values.
    """
    p39, config, _ = _fresh()

    config._current = None
    assert p39.resolve_kkt_hints() == (4096, 2)

    config._current = _VllmConfig(
        _SchedulerConfig(max_num_batched_tokens=4128, max_num_seqs=4)
    )
    assert p39.resolve_kkt_hints() == (4128, 4)


def test_resolution_is_cached_once_real():
    """After a real resolution, later config changes are ignored (stable
    pool size — no pointer swap from a mid-run re-read)."""
    p39, config, _ = _fresh()
    config._current = _VllmConfig(
        _SchedulerConfig(max_num_batched_tokens=4128, max_num_seqs=4)
    )
    assert p39.resolve_kkt_hints() == (4128, 4)

    config._current = _VllmConfig(
        _SchedulerConfig(max_num_batched_tokens=999, max_num_seqs=1)
    )
    assert p39.resolve_kkt_hints() == (4128, 4)


def test_wrapper_signature_accepts_pn354_use_exp2():
    """PN354 text-patches chunk.py to call kkt_fwd with `use_exp2=`.

    The lane-1 copy had drifted and lacked the kwarg — it would have
    raised TypeError at the first forward if P39a ever went live.
    """
    import inspect
    src = _TARGET.read_text(encoding="utf-8")
    assert "use_exp2=False," in src
    assert 'if _KKT_HAS_EXP2[0] else {}' in src
    del inspect


def _main() -> int:
    failures = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failures.append((name, e))
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    for key in _ENVS:
        os.environ.pop(key, None)
    print(f"\n{'FAILED' if failures else 'ALL PASSED'} "
          f"({len(failures)} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
