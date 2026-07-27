#!/usr/bin/env python3
"""BUG-177 — standalone GPU proof that the wall mechanism actually bites.

WHY THIS IS A SEPARATE SCRIPT AND WAS NOT RUN ON 2026-07-27
-----------------------------------------------------------
The claim the wall rests on is a torch runtime behaviour:
``torch.cuda.set_per_process_memory_fraction(F)`` makes the caching allocator
raise ``torch.cuda.OutOfMemoryError`` once its reserved bytes would exceed
``F x total_memory`` — WITHOUT the device actually being full. Proving that
needs a real CUDA context.

At authoring time prod (`vllm-qwen36-endgame`) held 23550 MiB of the card and
`nvidia-smi` reported **560 MiB free**. A fresh CUDA context on this 4090 costs
roughly 300-500 MiB before a single tensor is allocated, so starting a second
process would have consumed most of the very headroom BUG-177 is about, on a
live engine that dies from ~1.2 GB transients. The GPU proof was therefore
DEFERRED, not skipped silently: run this at the screened-boot window, with the
engine stopped or before it is started.

    python3 fixes/gpu_probe_bug177_vram_wall.py            # needs >= ~1 GiB free
    python3 fixes/gpu_probe_bug177_vram_wall.py --min-free-mb 800

What it proves, in order:
  1. the fraction cap raises OutOfMemoryError while physical VRAM is still free
     (the whole point: a soft wall, not a hard crash);
  2. the raised error is an instance of ``torch.cuda.OutOfMemoryError``, i.e.
     exactly the type ``patch_oom_resilience.py`` v7 matches on
     (``fixes/patch_oom_resilience.py:108``), so the engine would abort the
     request and stay up;
  3. allocations UNDER the cap still succeed, so the wall does not simply
     poison the allocator;
  4. the clamp from the graft holds a later higher fraction down (the
     vram_guardian's 0.99), on real torch.

It refuses to run unless the card has the requested free memory, never
allocates more than ~256 MiB of tensors, and frees everything it takes.
"""
import argparse
import sys

MiB = 1024 ** 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-mb", type=int, default=1024,
                    help="refuse to run below this much free VRAM (default 1024)")
    ap.add_argument("--probe-mb", type=int, default=64,
                    help="size of each probe tensor (default 64 MiB)")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print(
            "SKIP: no torch on this interpreter. The host has no torch — run\n"
            "  this INSIDE the engine image, at the screened-boot window, with\n"
            "  the prod container STOPPED:\n\n"
            "    sudo podman run --rm --device nvidia.com/gpu=all \\\n"
            "      -v /home/user/club-3090/fixes:/fixes:ro \\\n"
            "      localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725 \\\n"
            "      python3 /fixes/gpu_probe_bug177_vram_wall.py\n"
        )
        return 0

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device visible")
        return 0

    dev = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(dev)
    print(f"device {dev}: total={total // MiB}MiB free={free // MiB}MiB")
    if free < args.min_free_mb * MiB:
        print(
            f"REFUSING: only {free // MiB}MiB free, need >= {args.min_free_mb}MiB.\n"
            "  Prod is almost certainly still holding the card. Stop the engine\n"
            "  (or run this before booting it) — do NOT lower --min-free-mb to\n"
            "  squeeze this in next to a live engine; that is the exact failure\n"
            "  BUG-177 is about."
        )
        return 2

    fails = []

    def check(cond, what):
        print(("  ok   " if cond else "  FAIL ") + what)
        if not cond:
            fails.append(what)

    probe = args.probe_mb * MiB
    # A cap a little above one probe tensor: the first allocation fits, the
    # second must not. Deliberately tiny relative to the card so physical VRAM
    # is nowhere near exhausted when the wall fires — that IS the proof.
    cap_bytes = int(probe * 1.6)
    frac = cap_bytes / float(total)
    print(f"\nsetting per-process fraction {frac:.6f} "
          f"(cap {cap_bytes // MiB}MiB of {total // MiB}MiB)")
    torch.cuda.set_per_process_memory_fraction(frac, dev)

    held = []
    try:
        print("T1: an allocation under the cap succeeds")
        held.append(torch.empty(probe, dtype=torch.uint8, device=f"cuda:{dev}"))
        check(True, f"{args.probe_mb}MiB allocated under a "
                    f"{cap_bytes // MiB}MiB cap")

        print("T2: crossing the cap raises, with the card still far from full")
        free_before, _ = torch.cuda.mem_get_info(dev)
        raised = None
        try:
            held.append(
                torch.empty(probe, dtype=torch.uint8, device=f"cuda:{dev}")
            )
        except Exception as exc:
            raised = exc
        check(raised is not None, "second allocation raised")
        check(isinstance(raised, torch.cuda.OutOfMemoryError),
              f"raised torch.cuda.OutOfMemoryError "
              f"(got {type(raised).__name__ if raised else 'nothing'}) — the "
              f"exact type patch_oom_resilience.py:108 matches")
        check(free_before > 4 * probe,
              f"physical VRAM was NOT exhausted when it fired "
              f"({free_before // MiB}MiB still free) — a SOFT wall")

        print("T3: the graft's clamp holds a later higher fraction down")
        # Reproduce _gvw_install_clamp exactly as the graft installs it.
        orig = torch.cuda.set_per_process_memory_fraction
        wall = frac
        calls = []

        def clamped(f, device=None):
            if isinstance(f, float) and f > wall:
                calls.append(("clamped", f, wall))
                f = wall
            else:
                calls.append(("passed", f, None))
            return orig(f, device)

        torch.cuda.set_per_process_memory_fraction = clamped
        try:
            torch.cuda.set_per_process_memory_fraction(0.99, dev)  # guardian
            check(calls and calls[-1][0] == "clamped",
                  "a later 0.99 was clamped to the wall")
            raised2 = None
            try:
                held.append(
                    torch.empty(probe, dtype=torch.uint8, device=f"cuda:{dev}")
                )
            except Exception as exc:
                raised2 = exc
            check(isinstance(raised2, torch.cuda.OutOfMemoryError),
                  "the wall still binds after the guardian's 0.99 — the clamp "
                  "is not cosmetic")
        finally:
            torch.cuda.set_per_process_memory_fraction = orig
    finally:
        held.clear()
        torch.cuda.empty_cache()
        # Release the cap so the process leaves nothing behind.
        try:
            torch.cuda.set_per_process_memory_fraction(1.0, dev)
        except Exception:
            pass
        free_after, _ = torch.cuda.mem_get_info(dev)
        print(f"\nfreed; free={free_after // MiB}MiB "
              f"(started at {free // MiB}MiB)")

    if fails:
        print(f"\nFAILED {len(fails)}:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nGPU wall probe PASSED — the fraction cap raises "
          "torch.cuda.OutOfMemoryError with the card still mostly free")
    return 0


if __name__ == "__main__":
    sys.exit(main())
