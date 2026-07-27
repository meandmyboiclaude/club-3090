#!/usr/bin/env python3
"""BUG-177 VRAM-budget-wall tests — no GPU, no container, no engine.

Two halves, both against the code that will actually boot:

  * the APPLIER is run for real against a copy of the PINNED vllm sources
    (/var/tmp/led-vllm-pin/vllm), including the vram_guardian applied first, so
    the anchors, the markers, the ordering and the emitted Python are the ones
    the boot produces;
  * the INJECTED code is then sliced back off disk and exec'd against a stub
    ``torch``, so the fraction math, the env gating and the guardian clamp are
    exercised as written, not as re-implemented here.

    python3 fixes/test_bug177_vram_budget_wall.py
"""
import importlib.util
import os
import pathlib
import shutil
import sys
import tempfile
import types

PIN = pathlib.Path("/var/tmp/led-vllm-pin/vllm")
HERE = pathlib.Path(__file__).parent
PATCH = HERE / "patch_bug177_vram_budget_wall.py"
GUARDIAN = HERE / "patch_vram_guardian.py"
REL = "v1/worker/gpu_worker.py"

MARK_HELPER = "# BUG-177 graft: VRAM budget wall"
MARK_CALL = "# BUG-177 graft: arm wall"
FLAG = "GENESIS_ENABLE_VRAM_BUDGET_WALL"

MiB = 1024 ** 2

# Live values measured on the prod card, 2026-07-27 (see the patch header).
BOX_TOTAL = 24109 * MiB          # torch.cuda.get_device_properties().total_memory
BOX_OVERHEAD = 889 * MiB         # nvidia-smi used - allocator reserved, 20/20 samples
BOX_PEAK_RESERVED = 22596 * MiB  # steady-state allocator high-water mark

fails = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        fails.append(what)


def load_patch(base: pathlib.Path):
    """Import the applier with BASE pointed at the temp tree. Importing does
    not apply anything — main() does (``if __name__ == '__main__'``)."""
    os.environ["BUG177_VLLM_BASE"] = str(base)
    spec = importlib.util.spec_from_file_location("bug177patch", PATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.BASE = base
    mod.TARGET = base / REL
    return mod


def apply_guardian(base: pathlib.Path):
    """Run the real vram_guardian applier against the temp tree, so the wall
    patch meets the same file the boot's second applier meets."""
    spec = importlib.util.spec_from_file_location("vgpatch", GUARDIAN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # no-op: its hardcoded TARGET does not exist
    mod.TARGET = base / REL
    mod.apply()
    return mod


def slice_helper(src: str) -> str:
    """The injected module-level block, verbatim off disk."""
    start = src.index(MARK_HELPER)
    # It is injected immediately before the class anchor.
    end = src.index("class Worker", start)
    return src[start:end]


def make_stub_torch(total, free, reserved, record):
    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    memory = types.ModuleType("torch.cuda.memory")

    def _set(f, device=None):
        record.append((f, device))

    cuda.set_per_process_memory_fraction = _set
    memory.set_per_process_memory_fraction = _set
    cuda.is_available = lambda: True
    cuda.current_device = lambda: 0
    cuda.get_device_properties = lambda d=0: types.SimpleNamespace(
        total_memory=total
    )
    cuda.mem_get_info = lambda d=0: (free, total)
    cuda.memory_reserved = lambda d=0: reserved
    cuda.memory = memory
    torch.cuda = cuda
    for name, m in (("torch", torch), ("torch.cuda", cuda),
                    ("torch.cuda.memory", memory)):
        sys.modules[name] = m
    return torch


def drop_stub_torch():
    for name in ("torch.cuda.memory", "torch.cuda", "torch"):
        sys.modules.pop(name, None)


def with_env(**kw):
    """Context-manager-ish: returns a restore callable."""
    old = {k: os.environ.get(k) for k in kw}

    def restore():
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return restore


def main() -> int:
    if not PIN.is_dir():
        print(f"SKIP: pinned tree missing at {PIN}")
        return 0

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="bug177-"))
    base = tmp / "vllm"
    (base / "v1/worker").mkdir(parents=True, exist_ok=True)
    shutil.copy2(PIN / REL, base / REL)

    print("T0: the vram_guardian applies first, exactly as the boot does")
    apply_guardian(base)
    g_src = (base / REL).read_text(encoding="utf-8")
    check("_vg_start_guardian()" in g_src, "guardian call present on disk")

    mod = load_patch(base)

    print("T1: the wall applier runs clean against the guarded pinned source")
    restore = with_env(**{FLAG: "1"})   # armed => anchor drift would be FATAL
    try:
        check(mod.main() == 0, "main() returns 0")
    finally:
        restore()

    src = (base / REL).read_text(encoding="utf-8")

    print("T2: markers present exactly once")
    check(src.count(MARK_HELPER) == 1, f"{MARK_HELPER!r} x1")
    check(src.count(MARK_CALL) == 1, f"{MARK_CALL!r} x1")
    check(src.count("_gvw_arm_wall()") == 2, "helper def + call site, no more")

    print("T3: idempotent — re-running is a no-op")
    before = src
    check(mod.main() == 0, "second main() returns 0")
    check((base / REL).read_text(encoding="utf-8") == before,
          "file unchanged on the second pass")

    print("T4: the grafted file still compiles")
    try:
        compile(src, REL, "exec")
        check(True, "gpu_worker.py compiles")
    except SyntaxError as exc:
        check(False, f"gpu_worker.py compiles ({exc})")

    print("T5: ordering — the wall arms AFTER the guardian's hard cap")
    check(src.index("_vg_start_guardian()") < src.index(MARK_CALL),
          "_vg_start_guardian() precedes the arm call")
    check(src.index(MARK_HELPER) < src.index(MARK_CALL),
          "helper is defined before its call site")
    check(src.index(MARK_CALL) < src.index("def load_model"),
          "the arm call is inside init_device, not after it")

    print("T6: fraction math — pure, exercised off the grafted text")
    ns = {}
    exec(slice_helper(src), ns)
    resolve = ns["_gvw_resolve_fraction"]
    headroom = ns["_gvw_binding_headroom"]

    f, mode, _ = resolve(BOX_TOTAL, BOX_OVERHEAD, "auto", 256 * MiB, 1024 * MiB)
    check(mode == "auto", "auto mode selected")
    check(abs(int(BOX_TOTAL * f) // MiB - 22829) <= 1,
          f"auto cap == 22829MiB on the live box (got {int(BOX_TOTAL * f) // MiB})")
    check(int(BOX_TOTAL * f) > BOX_PEAK_RESERVED,
          "auto wall sits ABOVE the observed steady-state peak (no false aborts)")
    check(headroom(BOX_TOTAL, f, BOX_OVERHEAD) > 0,
          "auto wall BINDS (positive physical headroom)")

    for raw in ("", None, "AUTO", " auto "):
        f2, m2, _ = resolve(BOX_TOTAL, BOX_OVERHEAD, raw, 256 * MiB, 1024 * MiB)
        check(m2 == "auto" and f2 == f, f"{raw!r} == auto")

    f3, m3, _ = resolve(BOX_TOTAL, BOX_OVERHEAD, "0.9", 256 * MiB, 1024 * MiB)
    check((m3, f3) == ("explicit", 0.9), "explicit float honoured verbatim")

    for bad in ("0", "-0.5", "1.5", "abc", "nan", "inf", "0.95x"):
        fb, mb, _ = resolve(BOX_TOTAL, BOX_OVERHEAD, bad, 256 * MiB, 1024 * MiB)
        check(fb is None and mb == "invalid", f"{bad!r} rejected")
    check(resolve(0, 0, "auto", 0, 0)[0] is None, "total_bytes 0 rejected")
    check(resolve(BOX_TOTAL, BOX_TOTAL, "auto", MiB, 0)[0] is None,
          "overhead >= total rejected (no negative fraction)")

    print("T7: the overhead floor is what stops auto over-reaching")
    # An init-time reading understates the steady-state overhead; the floor is
    # what keeps a boot-time measurement from placing the wall above physical.
    f_lo, _, _ = resolve(BOX_TOTAL, 400 * MiB, "auto", 256 * MiB, 1024 * MiB)
    check(headroom(BOX_TOTAL, f_lo, BOX_OVERHEAD) > 0,
          "floor 1024MiB still binds when init-time overhead reads only 400MiB")
    f_nofloor, _, _ = resolve(BOX_TOTAL, 400 * MiB, "auto", 256 * MiB, 0)
    check(headroom(BOX_TOTAL, f_nofloor, BOX_OVERHEAD) <= 0,
          "without the floor the same reading would NOT bind (why the floor exists)")

    print("T8: the headroom check catches the decorative fractions")
    check(headroom(BOX_TOTAL, 0.975, BOX_OVERHEAD) < 0,
          "0.975 does NOT bind on this card (-286MiB)")
    check(headroom(BOX_TOTAL, 0.99, BOX_OVERHEAD) < 0,
          "VRAM_GUARDIAN_HARD_PCT=99 has never been able to bind (-648MiB)")
    # boundary: total*(1-f) == overhead  =>  f == 1 - 889/24109 == 0.96313
    check(headroom(BOX_TOTAL, 0.9640, BOX_OVERHEAD) < 0
          and headroom(BOX_TOTAL, 0.9620, BOX_OVERHEAD) > 0,
          "the binding boundary is 1 - overhead/total == 0.96313")

    print("T9: env gating — dark means untouched")
    record = []
    make_stub_torch(BOX_TOTAL, BOX_TOTAL - BOX_OVERHEAD, 0, record)
    try:
        for val in (None, "0", "false", "off", ""):
            record.clear()
            r = with_env(**{FLAG: val})
            try:
                ns["_gvw_arm_wall"]()
            finally:
                r()
            check(record == [], f"{FLAG}={val!r} -> set_per_process_memory_fraction "
                                "never called")
            check(ns["_GVW"]["armed"] is False, f"{FLAG}={val!r} -> not armed")

        print("T10: armed -> the wall is set on the current device")
        record.clear()
        r = with_env(**{FLAG: "1", "GENESIS_VRAM_WALL_FRACTION": None,
                        "GENESIS_VRAM_WALL_RESERVE_MB": None,
                        "GENESIS_VRAM_WALL_OVERHEAD_FLOOR_MB": None})
        try:
            ns["_gvw_arm_wall"]()
        finally:
            r()
        check(len(record) == 1, "exactly one fraction call")
        check(record and abs(record[0][0] - f) < 1e-12,
              "the fraction set equals the auto value")
        check(record and record[0][1] == 0, "device index passed through")
        check(ns["_GVW"]["armed"] is True, "state marked armed")

        print("T11: the clamp holds the wall against the guardian, both orders")
        # (a) guardian AFTER us: its 0.99 must come down to the wall.
        import torch as stub  # the stub installed above

        record.clear()
        stub.cuda.set_per_process_memory_fraction(0.99)
        check(record and abs(record[0][0] - f) < 1e-12,
              "a later 0.99 is clamped down to the wall")
        record.clear()
        stub.cuda.set_per_process_memory_fraction(0.5)
        check(record and record[0][0] == 0.5,
              "a later LOWER fraction still passes through")
        check(stub.cuda.memory.set_per_process_memory_fraction
              is stub.cuda.set_per_process_memory_fraction,
              "torch.cuda.memory binding clamped too")

        # non-float must reach the original so torch's own TypeError fires
        record.clear()
        stub.cuda.set_per_process_memory_fraction("0.99")
        check(record and record[0][0] == "0.99",
              "non-float passes through untouched (torch keeps raising TypeError)")

        # idempotent install: arming twice must not double-wrap
        wrapped = stub.cuda.set_per_process_memory_fraction
        ns["_gvw_install_clamp"](stub, 0.5)
        check(stub.cuda.set_per_process_memory_fraction is wrapped,
              "re-install is idempotent (no wrapper stacking)")
        check(stub.cuda._gvw_wall == 0.5, "re-install updates the wall value")

        # (b) guardian BEFORE us: order-independence. Fresh stub.
        drop_stub_torch()
        record2 = []
        stub2 = make_stub_torch(BOX_TOTAL, BOX_TOTAL - BOX_OVERHEAD, 0, record2)
        ns2 = {}
        exec(slice_helper(src), ns2)
        stub2.cuda.set_per_process_memory_fraction(0.99)   # guardian first
        record2.clear()
        r = with_env(**{FLAG: "1", "GENESIS_VRAM_WALL_FRACTION": None,
                        "GENESIS_VRAM_WALL_RESERVE_MB": None,
                        "GENESIS_VRAM_WALL_OVERHEAD_FLOOR_MB": None})
        try:
            ns2["_gvw_arm_wall"]()
        finally:
            r()
        check(record2 and abs(record2[0][0] - f) < 1e-12,
              "guardian-first: the wall still ends up as the effective cap")

        print("T12: a bad fraction refuses to arm rather than guessing")
        record2.clear()
        r = with_env(**{FLAG: "1", "GENESIS_VRAM_WALL_FRACTION": "banana"})
        try:
            ns2["_gvw_arm_wall"]()
        finally:
            r()
        check(record2 == [], "invalid fraction -> no fraction call at all")

        print("T13: arming never raises, whatever torch does")
        drop_stub_torch()
        boom = types.ModuleType("torch")
        cuda = types.ModuleType("torch.cuda")
        cuda.is_available = lambda: (_ for _ in ()).throw(RuntimeError("nope"))
        boom.cuda = cuda
        sys.modules["torch"] = boom
        sys.modules["torch.cuda"] = cuda
        ns3 = {}
        exec(slice_helper(src), ns3)
        r = with_env(**{FLAG: "1"})
        try:
            ns3["_gvw_arm_wall"]()
            check(True, "a throwing torch is swallowed (boot survives)")
        except Exception as exc:
            check(False, f"a throwing torch is swallowed (raised {exc!r})")
        finally:
            r()
    finally:
        drop_stub_torch()

    print("T14: a SECOND, un-guarded tree still gets a correct graft")
    tmp2 = pathlib.Path(tempfile.mkdtemp(prefix="bug177b-"))
    base2 = tmp2 / "vllm"
    (base2 / "v1/worker").mkdir(parents=True, exist_ok=True)
    shutil.copy2(PIN / REL, base2 / REL)
    mod2 = load_patch(base2)
    r = with_env(**{FLAG: "1"})
    try:
        check(mod2.main() == 0, "applies with no guardian present")
    finally:
        r()
    s2 = (base2 / REL).read_text(encoding="utf-8")
    check(s2.count(MARK_CALL) == 1, "call injected exactly once")
    check(s2.index(MARK_CALL) < s2.index("def load_model"),
          "call still lands inside init_device")
    try:
        compile(s2, REL, "exec")
        check(True, "un-guarded graft compiles")
    except SyntaxError as exc:
        check(False, f"un-guarded graft compiles ({exc})")

    print("T15: anchor drift is FATAL when armed, silent when dark")
    tmp3 = pathlib.Path(tempfile.mkdtemp(prefix="bug177c-"))
    base3 = tmp3 / "vllm"
    (base3 / "v1/worker").mkdir(parents=True, exist_ok=True)
    (base3 / REL).write_text("x = 1\n", encoding="utf-8")   # no anchors at all
    mod3 = load_patch(base3)
    for val, want in (("1", 1), (None, 0)):
        r = with_env(**{FLAG: val})
        try:
            mod3.main()
            check(False, f"{FLAG}={val!r} should SystemExit")
        except SystemExit as exc:
            check(exc.code == want,
                  f"{FLAG}={val!r} -> exit {want} (got {exc.code})")
        finally:
            r()

    for d in (tmp, tmp2, tmp3):
        shutil.rmtree(d, ignore_errors=True)

    print()
    if fails:
        print(f"FAILED {len(fails)}:")
        for f_ in fails:
            print("  -", f_)
        return 1
    print("all BUG-177 vram-budget-wall tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
