# SPDX-License-Identifier: Apache-2.0
"""P39a self-install hook — exec-survival, dual-pin anchor, flag-off no-op.

The shipped compose entrypoint is

    python3 -m vllm._genesis.patches.apply_all   # apply runs here
    exec vllm serve "$@"                          # exec REPLACES the process

`exec` discards Python state, so P39a's setattr-only install never reached
the serving process: it logged "[Genesis] applied: P39a" every boot and did
nothing. Same incident class as P103 (club-3090#19, 2026-05-02), and the
sanctioned fix is P103's: a text-patched module-import-time self-install
hook.

These tests pin the contract:
  * flag OFF  -> target file is BYTE-IDENTICAL (a live kernel change must
                 not ride an unrelated boot)
  * flag ON   -> hook appended once, idempotent on re-apply
  * the tail anchor is present EXACTLY ONCE in all four live file shapes
    (dev1060cherry / dev1474cherry*, each with and without PN354)
  * the import-time hook installs the pooled wrapper into a module dict
  * the kernel-kwarg probe passes USE_EXP2 / CAST_DOT_TO_K_DTYPE if and
    only if the kernel DECLARES them — dropping CAST_DOT_TO_K_DTYPE on the
    dev1474 pins is a hard TypeError at the first GDN prefill, which is why
    it had to land in the same commit as the hook.

Runs with the stdlib only — no torch, no vLLM, no triton, no pytest:

    python3 vllm/_genesis/tests/test_p39a_self_install.py

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
_TEXT_PATCH = _GENESIS / "wiring" / "text_patch.py"

_FLAG = "GENESIS_ENABLE_P39A_SELFINSTALL"
_REL = "third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py"

# ── real file tails, verbatim from the live images ──────────────────────────
# Extracted 2026-07-25 with
#   podman run --rm --entrypoint cat localhost/vllm-qwen36-endgame:<tag> \
#       <vllm>/…/chunk_scaled_dot_kkt.py
# dev1474cherrymax-1757 and dev1474cherry-1711 are byte-identical (diff = 0).
_TAIL_DEV1060 = (
    "        K=K,\n"
    "        BT=BT,\n"
    "    )\n"
    "    return A\n"
)
_TAIL_DEV1474 = (
    "        K=K,\n"
    "        BT=BT,\n"
    "        CAST_DOT_TO_K_DTYPE=_CAST_DOT_TO_K_DTYPE,\n"
    "    )\n"
    "    return A\n"
)
# PN354 anchors on the `BT=BT,` line alone and splices `USE_EXP2=use_exp2,`
# directly after it — so both pins have a second, post-PN354 shape.
_TAIL_DEV1060_PN354 = _TAIL_DEV1060.replace(
    "        BT=BT,\n", "        BT=BT,\n        USE_EXP2=use_exp2,\n",
)
_TAIL_DEV1474_PN354 = _TAIL_DEV1474.replace(
    "        BT=BT,\n", "        BT=BT,\n        USE_EXP2=use_exp2,\n",
)

_ALL_SHAPES = {
    "dev1060cherry-20260713": _TAIL_DEV1060,
    "dev1060cherry-20260713 + PN354": _TAIL_DEV1060_PN354,
    "dev1474cherry*-20260725": _TAIL_DEV1474,
    "dev1474cherry*-20260725 + PN354": _TAIL_DEV1474_PN354,
}

_HEAD = (
    "import torch\n"
    "\n"
    "_CAST_DOT_TO_K_DTYPE = False\n"
    "\n"
    "def chunk_scaled_dot_kkt_fwd(k, g=None, beta=None):\n"
    "    A = torch.empty(1)\n"
    "    chunk_scaled_dot_kkt_fwd_kernel[(1, 1)](\n"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeKernel:
    """Stand-in for a `triton.jit` JITFunction (arg_names, [grid](**kw))."""

    def __init__(self, arg_names):
        self.arg_names = list(arg_names)
        self.calls = []

    def __getitem__(self, grid):
        def _launch(**kwargs):
            self.calls.append((grid, kwargs))
        return _launch


def _install_stubs(vllm_root: Path):
    """Minimal `vllm` namespace — no torch, no triton, no real vLLM."""
    for name in list(sys.modules):
        if name in ("vllm", "triton", "torch") or name.startswith(
            ("vllm.", "triton.", "torch.")
        ):
            del sys.modules[name]

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    genesis = types.ModuleType("vllm._genesis")
    genesis.__path__ = [str(_GENESIS)]

    guards = types.ModuleType("vllm._genesis.guards")
    guards.is_nvidia_cuda = lambda: True
    guards.is_sm_at_least = lambda *_a, **_k: True
    guards.vllm_install_root = lambda: str(vllm_root)

    def resolve_vllm_file(relative_path: str):
        # Mirrors the shipped contract: returns a **str**, not a Path.
        full = os.path.join(str(vllm_root), relative_path)
        return full if os.path.exists(full) else None

    guards.resolve_vllm_file = resolve_vllm_file

    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    triton = types.ModuleType("triton")
    triton.cdiv = lambda a, b: -(-a // b)

    kernels_pkg = types.ModuleType("vllm._genesis.kernels")
    kernels_pkg.__path__ = []
    kkt_buf = types.ModuleType("vllm._genesis.kernels.fla_kkt_buffer")

    class _FlaKktBufferManager:
        acquired = []

        @classmethod
        def acquire(cls, **kwargs):
            cls.acquired.append(kwargs)
            return "POOLED_A"

    kkt_buf.FlaKktBufferManager = _FlaKktBufferManager

    sys.modules["vllm"] = vllm
    sys.modules["vllm._genesis"] = genesis
    sys.modules["vllm._genesis.guards"] = guards
    sys.modules["vllm._genesis.kernels"] = kernels_pkg
    sys.modules["vllm._genesis.kernels.fla_kkt_buffer"] = kkt_buf
    sys.modules["torch"] = torch
    sys.modules["triton"] = triton

    wiring = types.ModuleType("vllm._genesis.wiring")
    wiring.__path__ = [str(_GENESIS / "wiring")]
    sys.modules["vllm._genesis.wiring"] = wiring
    sys.modules["vllm._genesis.wiring.text_patch"] = _load(
        "vllm._genesis.wiring.text_patch", _TEXT_PATCH,
    )
    return _FlaKktBufferManager


def _fresh(tmp_root: Path, tail: str = _TAIL_DEV1474):
    """Build a fake vllm tree containing the kkt file, load the patch module."""
    target = tmp_root / _REL
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_HEAD + tail, encoding="utf-8")
    pool = _install_stubs(tmp_root)
    p39 = _load(
        "vllm._genesis.wiring.legacy.patch_39_fla_kkt_buffer", _TARGET,
    )
    return p39, target, pool


# ── the tests ───────────────────────────────────────────────────────────────

def test_flag_off_leaves_target_byte_identical():
    """The guard that matters: an unrelated boot must not pick this up."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        os.environ.pop(_FLAG, None)
        p39, target, _ = _fresh(Path(td))
        before = target.read_bytes()

        status, reason = p39._apply_self_install_text_patch()

        assert status == "skipped", (status, reason)
        assert _FLAG in reason
        assert target.read_bytes() == before, "flag-off wrote to the target"
        assert not p39._selfinstall_enabled()


def test_flag_on_appends_hook_and_is_idempotent():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        os.environ[_FLAG] = "1"
        try:
            p39, target, _ = _fresh(Path(td))

            status, reason = p39._apply_self_install_text_patch()
            assert status == "applied", (status, reason)

            body = target.read_text(encoding="utf-8")
            # TextPatcher prepends its own `[Genesis wiring marker: …]` line;
            # that line IS the framework's idempotency signal.
            assert p39._GENESIS_P39A_SELFINSTALL_MARKER in body
            assert body.splitlines()[0].startswith("# [Genesis wiring marker:")
            assert body.count("_genesis_p39a_install(globals())") == 1
            # The original function body must survive untouched.
            assert _HEAD in body
            assert "    return A\n" in body
            # The appended hook is syntactically valid Python.
            compile(body, str(target), "exec")

            # Re-apply -> idempotent, no second hook.
            status2, _ = p39._apply_self_install_text_patch()
            assert status2 in ("idempotent", "skipped"), status2
            body2 = target.read_text(encoding="utf-8")
            assert body2.count("_genesis_p39a_install(globals())") == 1
        finally:
            os.environ.pop(_FLAG, None)


def test_anchor_is_unique_in_every_live_pin_shape():
    """Dual-pin: dev1060 + dev1474, each with and without PN354."""
    import tempfile
    os.environ[_FLAG] = "1"
    try:
        for label, tail in _ALL_SHAPES.items():
            with tempfile.TemporaryDirectory() as td:
                p39, target, _ = _fresh(Path(td), tail=tail)
                content = target.read_text(encoding="utf-8")
                assert content.count(p39._P39A_SELF_INSTALL_ANCHOR) == 1, (
                    f"{label}: anchor not uniquely present"
                )
                assert p39._make_self_install_text_patcher() is not None, label
                status, reason = p39._apply_self_install_text_patch()
                assert status == "applied", (label, status, reason)
                compile(
                    target.read_text(encoding="utf-8"), str(target), "exec",
                )
    finally:
        os.environ.pop(_FLAG, None)


def test_missing_or_ambiguous_anchor_skips_instead_of_splicing():
    import tempfile
    os.environ[_FLAG] = "1"
    try:
        # (a) anchor absent entirely
        with tempfile.TemporaryDirectory() as td:
            p39, target, _ = _fresh(Path(td))
            target.write_text("def f():\n    return 1\n", encoding="utf-8")
            before = target.read_bytes()
            assert p39._make_self_install_text_patcher() is None
            assert p39._apply_self_install_text_patch()[0] == "skipped"
            assert target.read_bytes() == before

        # (b) anchor present twice -> refuse to guess
        with tempfile.TemporaryDirectory() as td:
            p39, target, _ = _fresh(Path(td))
            target.write_text(
                _HEAD + _TAIL_DEV1474 + "\n" + _HEAD + _TAIL_DEV1474,
                encoding="utf-8",
            )
            before = target.read_bytes()
            assert p39._make_self_install_text_patcher() is None
            assert p39._apply_self_install_text_patch()[0] == "skipped"
            assert target.read_bytes() == before
    finally:
        os.environ.pop(_FLAG, None)


def _module_globals(kernel_arg_names, cast=False):
    """A stand-in for chunk_scaled_dot_kkt.py's `globals()`."""
    def _original(k, **_kw):
        return "ORIGINAL_A"

    return {
        "chunk_scaled_dot_kkt_fwd": _original,
        "chunk_scaled_dot_kkt_fwd_kernel": _FakeKernel(kernel_arg_names),
        "prepare_chunk_indices": lambda cu, bt: [0, 1],
        "FLA_CHUNK_SIZE": 64,
        "_CAST_DOT_TO_K_DTYPE": cast,
    }


def test_install_at_import_is_gated_installs_and_is_idempotent():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        os.environ.pop(_FLAG, None)
        p39, _, _ = _fresh(Path(td))

        # flag OFF -> no install, module dict untouched
        g = _module_globals(["k", "BT"])
        original = g["chunk_scaled_dot_kkt_fwd"]
        assert p39._genesis_p39a_install_at_import(g) is False
        assert g["chunk_scaled_dot_kkt_fwd"] is original

        # flag ON -> installs, marker set, original preserved for revert
        os.environ[_FLAG] = "1"
        try:
            assert p39._genesis_p39a_install_at_import(g) is True
            wrapper = g["chunk_scaled_dot_kkt_fwd"]
            assert wrapper is not original
            assert getattr(wrapper, p39._GENESIS_P39A_MARKER_ATTR) is True
            assert getattr(wrapper, "_genesis_p39a_original") is original

            # idempotent: second call is a no-op, not a double wrap
            assert p39._genesis_p39a_install_at_import(g) is True
            assert g["chunk_scaled_dot_kkt_fwd"] is wrapper
        finally:
            os.environ.pop(_FLAG, None)


def test_install_at_import_bails_on_interface_drift():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        os.environ[_FLAG] = "1"
        try:
            p39, _, _ = _fresh(Path(td))
            g = _module_globals(["k", "BT"])
            del g["prepare_chunk_indices"]
            original = g["chunk_scaled_dot_kkt_fwd"]
            assert p39._genesis_p39a_install_at_import(g) is False
            assert g["chunk_scaled_dot_kkt_fwd"] is original
        finally:
            os.environ.pop(_FLAG, None)


class _K:
    """Minimal `k` tensor stand-in: `.shape` + `.device`."""
    shape = (1, 256, 8, 128)
    device = "cuda:0"


def _call_wrapper(p39, kernel_arg_names, cast=False, use_exp2=False):
    g = _module_globals(kernel_arg_names, cast=cast)
    assert p39._genesis_p39a_install_at_import(g) is True
    wrapper = g["chunk_scaled_dot_kkt_fwd"]

    class _Beta:
        shape = (1, 256, 8)

    out = wrapper(_K(), g=None, beta=_Beta(), use_exp2=use_exp2)
    return out, g["chunk_scaled_dot_kkt_fwd_kernel"].calls[-1][1]


def test_kernel_kwarg_probe_matches_the_declared_signature():
    """CAST_DOT_TO_K_DTYPE is REQUIRED on dev1474 — dropping it is a crash."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        os.environ[_FLAG] = "1"
        try:
            p39, _, pool = _fresh(Path(td))

            # dev1060 shape: neither constexpr declared -> pass neither
            out, kw = _call_wrapper(p39, ["k", "g", "beta", "A", "BT"])
            assert out == "POOLED_A", "wrapper must return the pooled buffer"
            assert "CAST_DOT_TO_K_DTYPE" not in kw
            assert "USE_EXP2" not in kw

            # dev1474 shape: CAST declared -> pass it, sourced from the module
            _, kw = _call_wrapper(
                p39, ["k", "BT", "CAST_DOT_TO_K_DTYPE"], cast=True,
            )
            assert kw["CAST_DOT_TO_K_DTYPE"] is True
            assert "USE_EXP2" not in kw

            # dev1474 + PN354: both declared -> both passed, even when the
            # env flag is off (Triton constexpr with no default is REQUIRED)
            _, kw = _call_wrapper(
                p39,
                ["k", "BT", "USE_EXP2", "CAST_DOT_TO_K_DTYPE"],
                cast=False,
                use_exp2=False,
            )
            assert kw["USE_EXP2"] is False
            assert kw["CAST_DOT_TO_K_DTYPE"] is False

            # the pool really was consulted, with P39b sizing hints
            assert pool.acquired, "FlaKktBufferManager.acquire never called"
            assert "max_T" in pool.acquired[-1]
            assert "max_B" in pool.acquired[-1]
        finally:
            os.environ.pop(_FLAG, None)


def _main() -> int:
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as e:  # noqa: BLE001
                import traceback
                failures.append(name)
                print(f"  FAIL  {name}: {e}")
                traceback.print_exc()
    print(f"\n{'FAILED' if failures else 'OK'} — {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
