#!/usr/bin/env python3
"""BUG-177 graft (2026-07-27) — a REAL VRAM budget wall for the engine process.

WHY (the BUG-177 mechanism, measured, not lore)
-----------------------------------------------
``gpu_memory_utilization`` is a **boot-time sizing hint**: vLLM profiles once,
subtracts weights+activations, and hands the remainder to the KV cache. Nothing
enforces it afterwards. Prod (util 0.91, seqs 6, maxlen 82560, KV
turboquant_3bit_nc / 188,708 tok) therefore runs with its steady-state footprint
sitting exactly ON the budget, and a transient allocation — several long-prompt
requests aligning, TQ continuation/upcast buffers scaling with prompt length
(P38 class) — walks straight past it into physical exhaustion. On 2026-07-27 the
engine died at 4 MiB free.

Live numbers from this box (``[vram_diag]`` lines the vram_guardian prints,
``journalctl -u vllm-endgame-8020``, 2026-07-27 07:33-07:46Z, 20 samples):

    torch total_memory              24109 MiB   (nvidia-smi says 24564; torch's
                                                 "total visible" is the number
                                                 the fraction multiplies)
    allocator reserved (steady)     22256..22596 MiB
    nvidia-smi used                 23145..23485 MiB
    nvidia - reserved               889 MiB on EVERY sample, no drift

That 889 MiB is the CUDA context + cuBLAS/NCCL workspaces + anything allocated
outside the torch caching allocator. It matters enormously here — see NUMERACY.

WHAT THIS GRAFT DOES
--------------------
At the end of ``Worker.init_device`` (the same injection point family as
``patch_vram_guardian.py``) it calls
``torch.cuda.set_per_process_memory_fraction(F)``, so the torch caching
allocator refuses to grow past ``F x total_memory`` and raises
``torch.cuda.OutOfMemoryError`` instead. That exception is exactly what
``patch_oom_resilience.py`` (v7, ``v1/engine/core.py``, wraps ``self.step_fn()``)
already catches: it aborts the running request(s), notifies their clients and
keeps EngineCore alive. So the wall converts "engine dies, podman restarts,
every in-flight request lost" into "the offending step is aborted, engine stays
up" — the failure mode we want at the ceiling.

NUMERACY — WHY 0.975 WOULD BE A NO-OP HERE, AND WHY 'auto' IS THE DEFAULT
--------------------------------------------------------------------------
``set_per_process_memory_fraction`` caps the **caching allocator's** bytes
(torch/cuda/memory.py docstring: "limit an caching allocator to allocated memory
on a CUDA device ... allowed memory equals total_memory * fraction"). The CUDA
context is NOT charged against it. So the physical ceiling the cap corresponds
to is ``F x total + 889 MiB``, and the wall only binds when

    F x total + non_allocator_overhead  <  total

i.e. ``F < 1 - overhead/total``. On this card (total 24109 MiB, overhead
889 MiB) that means **F < 0.96313**.

  * F = 0.975 -> allocator cap 23506 MiB, + 889 = 24395 MiB > 24109. The card
    runs out of physical VRAM BEFORE the cap is reached. The wall never fires.
  * The vram_guardian's existing hard cap, ``VRAM_GUARDIAN_HARD_PCT=99``
    (endgame8020.yml:743), is 0.99 -> 23868 + 889 = 24757 MiB. Same story — and
    this is not theory: the engine crashed at physical exhaustion on 2026-07-27
    having never once hit that cap, while the guardian's soft leg fired ~900
    times reclaiming 0 MB. The guardian's "hard backstop" has never been able
    to bind. That is the live proof that the cap is on allocator bytes and that
    a fraction chosen without subtracting the overhead is decorative.
  * Observed steady-state peak reserved is 22596 MiB, so a wall must also stay
    ABOVE that or it will abort healthy traffic. The usable window on this box
    is therefore [22596, 23220] MiB -> F in [0.9372, 0.96313] — 624 MiB wide.

Hence ``GENESIS_VRAM_WALL_FRACTION=auto`` (the default), which computes

    F = (total - overhead - reserve) / total

with ``overhead = max(measured_at_init, GENESIS_VRAM_WALL_OVERHEAD_FLOOR_MB)``
and ``reserve = GENESIS_VRAM_WALL_RESERVE_MB``. Defaults 1024 / 256 MiB give
F ~= 0.94691 (allocator cap 22829 MiB) on this card: 233 MiB above the observed
steady peak, 391 MiB of physical slack under the wall. The floor exists because
the overhead measured at ``init_device`` time is only the context + NCCL — later
cuBLAS/kernel workspaces add to it, so trusting the init-time reading alone
would place the wall too high. An explicit float is always honoured verbatim.

An armed boot prints ``physical_headroom_at_wall``; if that is <= 0 the wall
cannot bind and the patch says so LOUDLY on its own log line rather than
pretending to protect anything.

INTERACTION WITH patch_vram_guardian.py (REQUIRED READING)
----------------------------------------------------------
The guardian (``fixes/patch_vram_guardian.py:57``) calls
``torch.cuda.set_per_process_memory_fraction(hard_pct/100)`` **once,
synchronously, inside ``_vg_start_guardian()``** — i.e. also from
``init_device``, before its polling thread starts. With
``VRAM_GUARDIAN_HARD_PCT=99`` that is a 0.99 cap. Naively grafting a 0.947 wall
would be silently overwritten upward by whichever call ran last.

Two mechanisms make this deterministic, in this order:

 1. **Ordering.** The entrypoint runs this applier AFTER
    ``patch_vram_guardian.py``, and both grafts append to the end of
    ``init_device``; this one anchors *specifically on the
    ``_vg_start_guardian()`` line the guardian left behind* when that line is
    present, so the wall is armed after the guardian's cap is set. The applier
    asserts the resulting order and refuses (when armed) if it cannot.
 2. **A clamp, so order does not actually matter.** Arming installs a wrapper
    on ``torch.cuda.set_per_process_memory_fraction`` that lets any LATER call
    only *lower* the cap, never raise it (``min(requested, wall)``), logging
    each clamp. The guardian's 0.99 is therefore clamped to the wall whichever
    way round the two run, and any future reactive raise is likewise contained.
    Non-float arguments are passed through untouched so torch's own TypeError
    still fires.

The guardian is otherwise untouched: its soft leg (gc + empty_cache at
``VRAM_GUARDIAN_SOFT_PCT``) and its ``[vram_diag]`` telemetry are unchanged and
remain useful — the wall bounds live buffers, which empty_cache never could.

WHAT THIS WALL DOES *NOT* COVER
-------------------------------
Only the torch caching allocator. Raw ``cudaMalloc`` from a custom kernel, the
CUDA context itself, NCCL and cuBLAS workspaces are outside it. The TQ
continuation/upcast buffers suspected in BUG-177 are ordinary torch tensors, so
they are covered; a hypothetical raw-cudaMalloc path would not be. The
``physical_headroom_at_wall`` reserve is what absorbs growth in the uncovered
part.

STATUS: DARK. With ``GENESIS_ENABLE_VRAM_BUDGET_WALL`` unset or 0 the injected
``_gvw_arm_wall()`` returns on its first line: no fraction is set, no wrapper is
installed, nothing is logged, and the process behaves byte-for-byte as it does
today. Flipping it to 1 needs no re-patch (the flag is read at call time), but
it DOES need a boot — ``init_device`` runs once.

ARM:      GENESIS_ENABLE_VRAM_BUDGET_WALL=1   (+ boot)
ROLLBACK: unset it (or =0) + boot. Nothing else to undo; the graft is inert.

Idempotent by marker. Anchor drift is FATAL only when the flag is ARMED — a
dark patch must never be able to fail a boot it would not have changed.

Tests: ``fixes/test_bug177_vram_budget_wall.py`` (no GPU, no container).
GPU proof: ``fixes/gpu_probe_bug177_vram_wall.py`` (standalone process; needs a
free card — see its header).
"""
import os
import pathlib
import sys

LOG = "[patch_bug177_vram_budget_wall]"
BASE = pathlib.Path(
    os.environ.get(
        "BUG177_VLLM_BASE", "/usr/local/lib/python3.12/dist-packages/vllm"
    )
)
TARGET = BASE / "v1/worker/gpu_worker.py"

FLAG = "GENESIS_ENABLE_VRAM_BUDGET_WALL"

MARK_HELPER = "# BUG-177 graft: VRAM budget wall"
MARK_CALL = "# BUG-177 graft: arm wall"

GUARDIAN_CALL = "_vg_start_guardian()"

# ── Module-level block, injected just above the Worker class. Every name is
#    _gvw_-prefixed so it cannot collide with the module's own globals or with
#    the vram_guardian's _vg_ block sitting right above it.
HELPER_SRC = '''
# BUG-177 graft: VRAM budget wall (dark unless GENESIS_ENABLE_VRAM_BUDGET_WALL=1)
# See /fixes/patch_bug177_vram_budget_wall.py for the full rationale, the
# guardian interaction and the numeracy behind the default 'auto' fraction.
import os as _gvw_os
import logging as _gvw_logging

_gvw_log = _gvw_logging.getLogger("vram_budget_wall")
_gvw_log.setLevel(_gvw_logging.INFO)
if not _gvw_log.handlers:
    _gvw_log.addHandler(_gvw_logging.StreamHandler())

_GVW = {"armed": False, "fraction": None, "clamped": 0}


def _gvw_flag_on(name, default="0"):
    return str(_gvw_os.environ.get(name, default)).strip().lower() in (
        "1", "true", "yes", "on",
    )


def _gvw_int_env(name, default):
    """int(env) with a documented fallback — never raises."""
    try:
        return int(str(_gvw_os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return int(default)


def _gvw_resolve_fraction(total_bytes, overhead_bytes, raw, reserve_bytes,
                          overhead_floor_bytes=0):
    """PURE. -> (fraction | None, mode, note). No torch, no CUDA, no I/O.

    raw in (None, "", "auto")  -> (total - eff_overhead - reserve) / total
                                  where eff_overhead = max(overhead_bytes,
                                                           overhead_floor_bytes)
    raw a float in (0, 1]      -> that value, verbatim
    anything else              -> (None, "invalid", why)
    """
    if not isinstance(total_bytes, (int, float)) or total_bytes <= 0:
        return None, "invalid", "total_bytes must be a positive number"
    txt = "auto" if raw is None else str(raw).strip().lower()
    if txt in ("", "auto"):
        ov = max(
            overhead_bytes if isinstance(overhead_bytes, (int, float))
            and overhead_bytes > 0 else 0,
            overhead_floor_bytes if isinstance(overhead_floor_bytes, (int, float))
            and overhead_floor_bytes > 0 else 0,
        )
        rs = reserve_bytes if isinstance(reserve_bytes, (int, float)) \\
            and reserve_bytes > 0 else 0
        usable = total_bytes - ov - rs
        if usable <= 0:
            return None, "invalid", (
                "auto: overhead %d + reserve %d leaves nothing of total %d"
                % (ov, rs, total_bytes)
            )
        return (
            usable / float(total_bytes),
            "auto",
            "auto: (total %d - overhead %d - reserve %d) / total"
            % (total_bytes, ov, rs),
        )
    try:
        frac = float(txt)
    except (TypeError, ValueError):
        return None, "invalid", "not a float and not 'auto': %r" % (raw,)
    if frac != frac or frac in (float("inf"), float("-inf")):
        return None, "invalid", "fraction is not finite: %r" % (raw,)
    if not 0.0 < frac <= 1.0:
        return None, "invalid", "fraction out of range (0, 1]: %r" % (frac,)
    return frac, "explicit", "explicit fraction %r" % (frac,)


def _gvw_binding_headroom(total_bytes, fraction, overhead_bytes):
    """PURE. Physical bytes still free when the allocator sits ON the wall.

    <= 0 means the wall CANNOT bind: real VRAM runs out first and the cap is
    decorative (this is exactly the state VRAM_GUARDIAN_HARD_PCT=99 is in).
    """
    return int(total_bytes - (total_bytes * fraction) - (overhead_bytes or 0))


def _gvw_install_clamp(torch, fraction):
    """Make the boot wall a CEILING.

    Any later set_per_process_memory_fraction() may only LOWER the cap. The
    vram_guardian sets its own 0.99 hard cap from this very method, so without
    this the wall could be silently undone by call order. Idempotent; the
    clamp value lives on torch.cuda so re-arming just updates it.
    """
    fn = torch.cuda.set_per_process_memory_fraction
    if getattr(fn, "_gvw_clamped", False):
        torch.cuda._gvw_wall = fraction
        return fn

    def _gvw_set_fraction(f, device=None):
        wall = getattr(torch.cuda, "_gvw_wall", None)
        # Non-floats fall through so torch's own TypeError still fires.
        if wall is not None and isinstance(f, float) and f > wall:
            _GVW["clamped"] += 1
            _gvw_log.info(
                "[vram_wall] clamped set_per_process_memory_fraction(%.4f) -> "
                "%.4f (boot wall wins)", f, wall,
            )
            f = wall
        return fn(f, device)

    _gvw_set_fraction._gvw_clamped = True
    _gvw_set_fraction._gvw_orig = fn
    torch.cuda._gvw_wall = fraction
    torch.cuda.set_per_process_memory_fraction = _gvw_set_fraction
    # The symbol is defined in torch.cuda.memory and re-exported; keep both
    # bindings in sync so a `from torch.cuda.memory import ...` caller is
    # clamped too.
    try:
        import torch.cuda.memory as _gvw_mem

        _gvw_mem.set_per_process_memory_fraction = _gvw_set_fraction
    except Exception:
        pass
    return _gvw_set_fraction


def _gvw_arm_wall():
    """Arm the budget wall. Called at the very end of init_device, after the
    vram_guardian's own start (see the applier). DARK unless the flag is on."""
    if not _gvw_flag_on("GENESIS_ENABLE_VRAM_BUDGET_WALL"):
        return
    try:
        import torch

        if not torch.cuda.is_available():
            _gvw_log.warning(
                "[vram_wall] no CUDA device visible — wall NOT armed"
            )
            return
        dev = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(dev).total_memory
        free, _total_drv = torch.cuda.mem_get_info(dev)
        reserved = torch.cuda.memory_reserved(dev)
        # Everything resident on the device that the caching allocator does
        # NOT account for: our CUDA context, NCCL/cuBLAS workspaces, other
        # processes. The cap is on ALLOCATOR bytes, so this is precisely the
        # amount by which "allocator sitting on the wall" overshoots physical.
        overhead = max(0, (total - free) - reserved)

        raw = _gvw_os.environ.get("GENESIS_VRAM_WALL_FRACTION", "auto")
        reserve_b = _gvw_int_env("GENESIS_VRAM_WALL_RESERVE_MB", 256) * 1024 ** 2
        floor_b = (
            _gvw_int_env("GENESIS_VRAM_WALL_OVERHEAD_FLOOR_MB", 1024) * 1024 ** 2
        )
        frac, mode, note = _gvw_resolve_fraction(
            total, overhead, raw, reserve_b, floor_b
        )
        if frac is None:
            _gvw_log.error(
                "[vram_wall] NOT armed — GENESIS_VRAM_WALL_FRACTION=%r rejected "
                "(%s)", raw, note,
            )
            return

        headroom = _gvw_binding_headroom(total, frac, overhead)
        _gvw_install_clamp(torch, float(frac))
        torch.cuda.set_per_process_memory_fraction(float(frac), dev)
        _GVW["armed"] = True
        _GVW["fraction"] = float(frac)
        _gvw_log.info(
            "[vram_wall] ARMED dev=%d f=%.5f (%s) allocator_cap=%dMB "
            "total=%dMB measured_non_allocator_overhead=%dMB "
            "physical_headroom_at_wall=%dMB clamped_guardian=%d",
            dev, frac, mode, int(total * frac) // 1024 ** 2,
            total // 1024 ** 2, overhead // 1024 ** 2,
            headroom // 1024 ** 2, _GVW["clamped"],
        )
        if headroom <= 0:
            _gvw_log.error(
                "[vram_wall] INEFFECTIVE: the wall sits at or above physical "
                "capacity (headroom %dMB) — the device will exhaust before the "
                "allocator cap is reached, exactly like "
                "VRAM_GUARDIAN_HARD_PCT=99 does today. Lower "
                "GENESIS_VRAM_WALL_FRACTION or use 'auto'.",
                headroom // 1024 ** 2,
            )
    except Exception as exc:  # never let the wall break a boot
        _gvw_log.error(
            "[vram_wall] arm failed (%s: %s) — continuing WITHOUT a wall",
            type(exc).__name__, exc,
        )

'''

CALL_SRC = "\n        " + MARK_CALL + "\n        _gvw_arm_wall()"


def _armed() -> bool:
    return str(os.environ.get(FLAG, "0")).strip().lower() in (
        "1", "true", "yes", "on",
    )


def _fatal(msg: str) -> None:
    """Loud stop — but only when the flag is ARMED. A dark patch must not be
    able to fail a boot whose behaviour it would not have changed."""
    if _armed():
        print(f"{LOG} FATAL: {msg}", file=sys.stderr)
        raise SystemExit(1)
    print(f"{LOG} SKIP (dark): {msg}", file=sys.stderr)
    raise SystemExit(0)


def _insert_call(src: str) -> str:
    """Append the arm call to the end of Worker.init_device.

    Preferred anchor: immediately after the vram_guardian's own
    ``_vg_start_guardian()`` line, so the wall is set after the guardian's
    0.99 hard cap. Fallback (guardian absent / not yet applied): the same
    walk-back-over-trailing-comments logic the guardian itself uses.
    """
    idx = src.find("def init_device(self)")
    if idx < 0:
        idx = src.find("def init_device(")
    if idx < 0:
        _fatal("init_device anchor not found in " + str(TARGET))

    guard = src.find(GUARDIAN_CALL, idx)
    next_def = src.find("\n    def ", src.find("\n", idx) + 1)
    if next_def < 0:
        _fatal("cannot find the end of init_device")

    if 0 <= guard < next_def:
        eol = src.find("\n", guard)
        if eol < 0:
            _fatal("malformed guardian call line")
        return src[:eol] + CALL_SRC + src[eol:]

    insert_at = next_def
    while True:
        prev_nl = src.rfind("\n", 0, insert_at)
        if prev_nl < 0:
            break
        line = src[prev_nl + 1:insert_at].strip()
        if line and not line.startswith("#"):
            break
        insert_at = prev_nl
    return src[:insert_at] + CALL_SRC + src[insert_at:]


def main() -> int:
    if not TARGET.exists():
        _fatal(f"target missing: {TARGET}")

    src = TARGET.read_text(encoding="utf-8")

    if MARK_HELPER in src or MARK_CALL in src:
        n_h, n_c = src.count(MARK_HELPER), src.count(MARK_CALL)
        if n_h == 1 and n_c == 1:
            print(f"{LOG} already applied — no-op")
            return 0
        _fatal(
            f"partial/duplicated graft on disk (helper x{n_h}, call x{n_c}) — "
            "recreate the container so the stock file is restored"
        )

    src = _insert_call(src)

    cls = "class GpuWorker"
    if cls not in src:
        cls = "class Worker"
        if cls not in src:
            _fatal("Worker class anchor not found")
    cidx = src.find(cls)
    src = src[:cidx] + HELPER_SRC + "\n" + src[cidx:]

    if src.count(MARK_HELPER) != 1 or src.count(MARK_CALL) != 1:
        _fatal("post-graft marker count wrong — refusing to write")

    # Ordering assertion: the helper must be defined before the call site, and
    # the call must land after the guardian's start when the guardian is here.
    if src.index(MARK_HELPER) > src.index(MARK_CALL):
        _fatal("helper injected after its call site — refusing to write")
    g = src.find(GUARDIAN_CALL)
    if g >= 0 and g > src.index(MARK_CALL):
        _fatal(
            "arm call landed BEFORE _vg_start_guardian() — the guardian's "
            "0.99 hard cap would run last. (The clamp would still hold the "
            "wall, but this applier requires the documented order.)"
        )

    # BUG-172 compile gate: a quiet half-apply on upstream re-indent is worse
    # than a loud refusal.
    try:
        compile(src, str(TARGET), "exec")
    except SyntaxError as exc:
        _fatal(f"patched file does not compile: {exc}")

    TARGET.write_text(src, encoding="utf-8")

    cache = TARGET.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(TARGET.stem + ".*.pyc"):
            try:
                pyc.unlink()
            except OSError:
                pass

    # Informational only. The AUTHORITATIVE state is the env as EngineCore sees
    # it at init_device time; the injected function re-reads the flag there.
    state = (
        f"ARMED: {FLAG}=1 at apply time, fraction="
        f"{os.environ.get('GENESIS_VRAM_WALL_FRACTION', 'auto')}"
        if _armed()
        else f"dark: {FLAG} unset/0 at apply time"
    )
    print(f"{LOG} applied to {TARGET} ({state})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
