#!/usr/bin/env python3
"""Tests for patch_bug077_repetition_detection_default.py.

Runs on the HOST with plain python3 — no vLLM import, no GPU, no server:

    python3 /home/user/club-3090/fixes/test_bug077_repetition_detection.py

Covers
  1. the injected policy helper across the env matrix (dark by default, scope
     gating, caller-override, refusal of the unsafe min_pattern_size=1),
  2. vLLM's own `check_sequence_repetition` semantics (verbatim replica of
     vllm/v1/core/sched/utils.py as of image dev1474cherrymax-1757-20260725)
     against the real BUG-077 token cycle and against the digit-run shape that
     made min_pattern_size=1 false-positive,
  3. the applier: anchor detection, idempotency, and FATAL on drift.
"""
import importlib.util
import os
import pathlib
import random
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PATCH = HERE / "patch_bug077_repetition_detection_default.py"

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILS.append(name)


# ── load the patch module (import only; main() is under __main__ guard) ──────
spec = importlib.util.spec_from_file_location("bug077_patch", PATCH)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


class _RDP:
    """Stand-in for vllm.sampling_params.RepetitionDetectionParams."""

    def __init__(self, min_pattern_size=0, max_pattern_size=0, min_count=0):
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.min_count = min_count

    def __repr__(self):
        return (f"RDP(min={self.min_pattern_size},max={self.max_pattern_size},"
                f"count={self.min_count})")


class _Req:
    def __init__(self, thinking=None, explicit=None):
        self.chat_template_kwargs = (
            None if thinking is None else {"enable_thinking": thinking}
        )
        self.repetition_detection = explicit


def _load_helper():
    ns = {"RepetitionDetectionParams": _RDP}
    exec(patch.HELPER_SRC, ns)
    return ns["_bug077_repetition_default"]


def _env(**kw):
    for k in ("GENESIS_ENABLE_BUG077_REPETITION_DETECT", "BUG077_REP_SCOPE",
              "BUG077_REP_MIN_PATTERN", "BUG077_REP_MAX_PATTERN",
              "BUG077_REP_MIN_COUNT"):
        os.environ.pop(k, None)
    for k, v in kw.items():
        os.environ[k] = str(v)


def test_policy():
    print("policy helper")
    fn = _load_helper()

    _env()
    check("dark by default (thinking req)", fn(_Req(thinking=None)) is None)
    check("dark by default (nothink req)", fn(_Req(thinking=False)) is None)

    _env(GENESIS_ENABLE_BUG077_REPETITION_DETECT=1)
    check("flag on + thinking unset -> no injection",
          fn(_Req(thinking=None)) is None)
    check("flag on + enable_thinking=True -> no injection",
          fn(_Req(thinking=True)) is None)
    got = fn(_Req(thinking=False))
    check("flag on + enable_thinking=False -> injects", isinstance(got, _RDP), got)
    check("defaults are the measured ones (2/32/16)",
          isinstance(got, _RDP) and (got.min_pattern_size, got.max_pattern_size,
                                     got.min_count) == (2, 32, 16), got)

    explicit = _RDP(3, 9, 5)
    check("caller-supplied params are never overridden",
          fn(_Req(thinking=False, explicit=explicit)) is explicit)
    check("caller params respected even in thinking lane",
          fn(_Req(thinking=None, explicit=explicit)) is explicit)

    _env(GENESIS_ENABLE_BUG077_REPETITION_DETECT=1, BUG077_REP_SCOPE="all")
    check("scope=all injects for a thinking request",
          isinstance(fn(_Req(thinking=None)), _RDP))

    _env(GENESIS_ENABLE_BUG077_REPETITION_DETECT=1, BUG077_REP_SCOPE="bogus")
    check("unknown scope fails closed", fn(_Req(thinking=False)) is None)

    _env(GENESIS_ENABLE_BUG077_REPETITION_DETECT=1, BUG077_REP_MIN_PATTERN=1)
    check("min_pattern_size=1 is refused (digit-run false positives)",
          fn(_Req(thinking=False)) is None)

    _env(GENESIS_ENABLE_BUG077_REPETITION_DETECT=1, BUG077_REP_MIN_COUNT=1)
    check("min_count<2 is refused", fn(_Req(thinking=False)) is None)

    _env(GENESIS_ENABLE_BUG077_REPETITION_DETECT=1, BUG077_REP_MIN_PATTERN="x")
    check("non-numeric env falls back to the default",
          isinstance(fn(_Req(thinking=False)), _RDP))
    _env()


# ── verbatim replica of vllm/v1/core/sched/utils.py ─────────────────────────
def _has_repeating_pattern(token_ids, pattern_len, repetition_min_count):
    for n in range(1, pattern_len + 1):
        target_token = token_ids[-n]
        for m in range(1, repetition_min_count):
            if token_ids[-(pattern_len * m + n)] != target_token:
                return False
    return True


def check_sequence_repetition(token_ids, min_pattern_size, max_pattern_size,
                              min_count):
    if min_pattern_size <= 0:
        min_pattern_size = 1
    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return False
    for pattern_len in range(min_pattern_size, max_pattern_size + 1):
        if pattern_len * min_count > len(token_ids):
            return False
        if _has_repeating_pattern(token_ids, pattern_len, min_count):
            return True
    return False


def first_trip(ids, min_p, max_p, min_c):
    for i in range(1, len(ids) + 1):
        if check_sequence_repetition(ids[:i], min_p, max_p, min_c):
            return i
    return None


# Real token ids from the live tokenizer (:8021 /tokenize) for the BUG-077
# cycle unit `CCA GCT GCC CCA GCT CCA ACC ` — 12 tokens once the run is
# established (the leading `CCA` merges differently at the start).
CYCLE = [351, 4887, 469, 1123, 43441, 351, 4887, 469, 1123, 351, 4887, 25016]
# The one false positive min_pattern_size=1 produced on real traffic:
# " If the speed was $0.99999999" — token 24 is `9`.
DIGITRUN = [13, 1368, 279, 4478, 557, 393, 15, 13] + [24] * 12


def test_detector():
    print("detector semantics (vLLM replica)")
    rng = random.Random(1977)
    prose = [rng.randrange(1000, 200000) for _ in range(4000)]
    check("non-periodic token stream never trips (2/32/16)",
          first_trip(prose, 2, 32, 16) is None)

    loop = [7] * 3 + CYCLE * 40
    t = first_trip(loop, 2, 32, 16)
    check("BUG-077 12-token cycle trips at 2/32/16", t is not None, t)
    check("trips after ~16 cycle repeats (not earlier)",
          t is not None and 16 * 12 <= t <= 17 * 12 + 3, t)

    check("digit run does NOT trip with min_pattern_size=2",
          not check_sequence_repetition(DIGITRUN, 2, 32, 16))
    check("digit run DOES trip with min_pattern_size=1 (why we refuse it)",
          check_sequence_repetition(DIGITRUN[:8 + 8], 1, 32, 8))

    # a 12-token cycle repeated only 8x must survive the shipped default
    check("8 repeats of the cycle survive min_count=16",
          not check_sequence_repetition([7] * 3 + CYCLE * 8, 2, 32, 16))


# ── applier ─────────────────────────────────────────────────────────────────
FIXTURE = '''from vllm.sampling_params import RepetitionDetectionParams


class ChatCompletionRequest(OpenAIBaseModel):
    def to_sampling_params(self):
        return SamplingParams.from_optional(
            n=self.n,
            skip_clone=True,
            repetition_detection=self.repetition_detection,
        )
'''


def _run_applier(base):
    env = dict(os.environ, BUG077_VLLM_BASE=str(base))
    return subprocess.run([sys.executable, str(PATCH)], env=env,
                          capture_output=True, text=True)


def test_applier():
    print("applier")
    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)
        tgt = base / "entrypoints/openai/chat_completion/protocol.py"
        tgt.parent.mkdir(parents=True)
        tgt.write_text(FIXTURE)

        r = _run_applier(base)
        check("applies cleanly", r.returncode == 0, r.stdout + r.stderr)
        out = tgt.read_text()
        check("helper injected", patch.MARK_HELPER in out)
        check("call site rewritten",
              "repetition_detection=_bug077_repetition_default(self)," in out)
        check("original call site gone",
              "repetition_detection=self.repetition_detection,\n" not in out)
        check("patched file is valid python",
              compile(out, "protocol.py", "exec") is not None)

        r2 = _run_applier(base)
        check("idempotent (second run is a no-op)",
              r2.returncode == 0 and "already applied" in r2.stdout, r2.stdout)
        check("second run did not double-inject", tgt.read_text() == out)

        # anchor drift
        tgt.write_text(FIXTURE.replace(
            "            repetition_detection=self.repetition_detection,\n", ""))
        r3 = _run_applier(base)
        check("FATAL on missing call anchor", r3.returncode == 1, r3.stderr)

        # build predating the native detector
        tgt.write_text(FIXTURE.replace("RepetitionDetectionParams", "Nope"))
        r4 = _run_applier(base)
        check("FATAL when the build has no native detector",
              r4.returncode == 1 and "predates" in r4.stderr, r4.stderr)

        # half-applied
        tgt.write_text(FIXTURE + "\n" + patch.MARK_HELPER + "\n")
        r5 = _run_applier(base)
        check("FATAL on half-applied file", r5.returncode == 1, r5.stderr)


if __name__ == "__main__":
    test_policy()
    test_detector()
    test_applier()
    print()
    if FAILS:
        print(f"FAILED {len(FAILS)}: {FAILS}")
        sys.exit(1)
    print("all tests passed")


def test_dark_mode_preserves_explicit_caller_value():
    """Review 2026-07-27 H1: with the flag OFF, an explicitly-supplied
    repetition_detection must pass through untouched (dark == identity)."""
    fn = _load_helper()
    _env()  # clears all BUG077 env -> dark
    rd = _RDP(3, 9, 7)
    out = fn(_Req(explicit=rd))
    check("dark_explicit_passthrough", out is rd, f"got {out!r}")

test_dark_mode_preserves_explicit_caller_value()
