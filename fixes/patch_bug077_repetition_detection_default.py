#!/usr/bin/env python3
"""BUG-077 graft (2026-07-26) — server-side default for vLLM's NATIVE
repetition detector, scoped to the lane that has no other containment.

WHAT BUG-077 ACTUALLY IS
------------------------
TC gptq-pro-v2 occasionally locks into an exact token cycle on low-entropy
structured spans (a codon run, a digit run, a near-identical CSV block). It is
a decode-level degeneracy, not a reasoning failure: the cycle unit is
byte-identical and the model never re-acquires an exit token on its own.

Measured on the 2026-07-26 config (image dev1474cherrymax-1757-20260725, TC
gptq-pro-v2, TQ3-nc KV, seqs 6, util 0.91, MTP n=3 probabilistic/standard,
temp 1.0 / top_p 0.95 / top_k 20 from generation_config):

  * 1 genuine cycle in 407 GPQA generations (~650K completion tokens) across
    four arms (champion / nst4 / seqs7 / nst2greedy). The single hit is
    `chatcmpl-a5fb09a6e41526eb` (champion, 16:26:10Z): a 12-token codon unit
    (`CCA GCT GCC CCA GCT CCA ACC `) repeated 56x, ~1.6K chars, ~500 tokens.
  * It occurred in ONE of the four arms on the same prompt, so on today's
    config it is STOCHASTIC, not the "identical across configs/runs" trait the
    original 07-18 entry recorded. (Original observation was the hindsight
    structured-JSON drain, a different lane — see SCOPE below.)

WHY THIS IS OFF BY DEFAULT AND SCOPED
-------------------------------------
vLLM already ships the mechanism: `SamplingParams.repetition_detection`
(`RepetitionDetectionParams`), consumed by `v1/core/sched/utils.check_stop`,
which ENDS the request with `finish_reason="repetition"`. It is exposed on the
OpenAI chat/completions bodies today; nothing here adds a new mechanism, this
graft only supplies a DEFAULT for callers that do not set one.

The detector is a request KILLER, not a loop breaker. On the one real hit that
is the wrong trade:

    loop trips at token 3708/4618 (80.3%)
    PN100's thinking-budget nudge landed at token 4286/4618 (92.8%)
    ...and after the nudge the model closed </think> and answered correctly.

So in the THINKING lane the budget already converts the runaway into a correct
answer, and terminating on repetition would have destroyed a correct answer.
That is why the default scope is `nothink`:

    auto_budget.py:324 — an explicit `chat_template_kwargs.enable_thinking=false`
    makes PN100 skip the request entirely, so NOTHING caps a runaway there; it
    burns to max_tokens and the caller gets garbage. That is the lane BUG-077
    was originally filed from (the hindsight drain replay sends exactly
    `{"chat_template_kwargs":{"enable_thinking":false},"pool":"base"}`).

PARAMETER CHOICE (measured, not guessed)
----------------------------------------
Replaying `check_sequence_repetition` over the token ids of all 407 real
generations:

    min_pattern_size=1, max=32, min_count=8   -> 2 trips: 1 true, 1 FALSE
                                                 (`0.99999999` — eight
                                                 consecutive `9` tokens)
    min_pattern_size>=2  OR  min_count>=16    -> 1 trip in 407: the true one

`min_pattern_size=1` is specifically unsafe for this model's traffic because a
single-token run is exactly what a legitimate long digit literal looks like —
including BUG-077's own `"qty": 10^27` example. Hence the defaults below:

    min_pattern_size = 2   max_pattern_size = 32   min_count = 16

which caught the real 12-token cycle with zero false positives on the corpus.

STATUS: DARK. With `GENESIS_ENABLE_BUG077_REPETITION_DETECT` unset the graft
injects nothing and `to_sampling_params()` behaves byte-identically. Flipping
it ON is a behavioural change (some requests start finishing with
`finish_reason="repetition"` and a truncated body) and wants a bench arm plus a
caller-side check that `repetition` is handled like `length`.

Applies to the chat-completions surface only; `/v1/completions` has no thinking
lane to scope against and no observed BUG-077 traffic.

Idempotent by marker; anchor drift = FATAL exit 1, mirroring the other
/fixes appliers.
"""
import os
import pathlib
import sys

LOG = "[patch_bug077_repetition_detection_default]"
BASE = pathlib.Path(
    os.environ.get("BUG077_VLLM_BASE", "/usr/local/lib/python3.12/dist-packages/vllm")
)
CHAT = BASE / "entrypoints/openai/chat_completion/protocol.py"

# ── The injected helper. Self-contained: no imports beyond os, reads env at
#    CALL time so the flag can be flipped without a rebuild of the module. ──
MARK_HELPER = "# BUG-077 graft: repetition_detection default"
HELPER_SRC = '''
# BUG-077 graft: repetition_detection default (dark unless flag is set).
# Supplies a default RepetitionDetectionParams for callers that send none.
# See /fixes/patch_bug077_repetition_detection_default.py for the measurement
# that picked these numbers (1 trip in 407 real generations, 0 false).
def _bug077_repetition_default(request):
    import os as _os

    # never override a caller that asked for something explicitly — checked
    # BEFORE the flag so dark mode stays byte-identical for explicit callers
    # (review 2026-07-27 H1: the old order dropped a caller's value when dark)
    if getattr(request, "repetition_detection", None) is not None:
        return getattr(request, "repetition_detection")
    if _os.environ.get("GENESIS_ENABLE_BUG077_REPETITION_DETECT", "0") != "1":
        return None
    if getattr(request, "repetition_detection", None) is not None:
        return getattr(request, "repetition_detection")

    scope = _os.environ.get("BUG077_REP_SCOPE", "nothink").strip().lower()
    if scope == "nothink":
        # PN100's thinking budget (auto_budget.py) already truncates a runaway
        # in the thinking lane and yields a usable answer; only the explicitly
        # non-thinking lane is uncontained.
        ctk = getattr(request, "chat_template_kwargs", None) or {}
        try:
            thinking = ctk.get("enable_thinking", None)
        except AttributeError:
            thinking = None
        if thinking is not False:
            return None
    elif scope != "all":
        return None

    def _int(name, default):
        try:
            return int(_os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    min_pattern = _int("BUG077_REP_MIN_PATTERN", 2)
    max_pattern = _int("BUG077_REP_MAX_PATTERN", 32)
    min_count = _int("BUG077_REP_MIN_COUNT", 16)
    # min_pattern_size=1 false-positives on legitimate digit runs
    # ("0.99999999", "10^27" written out) -- refuse to build that config.
    if min_pattern < 2 or max_pattern < min_pattern or min_count < 2:
        return None
    return RepetitionDetectionParams(
        min_pattern_size=min_pattern,
        max_pattern_size=max_pattern,
        min_count=min_count,
    )

'''

ANCH_HELPER = "class ChatCompletionRequest("
ANCH_CALL = "            repetition_detection=self.repetition_detection,\n"
MARK_CALL = "# BUG-077 graft: default injection"
REPL_CALL = (
    "            # BUG-077 graft: default injection (identity when the flag is off)\n"
    "            repetition_detection=_bug077_repetition_default(self),\n"
)


def _fatal(msg):
    print(f"{LOG} FATAL {msg}", file=sys.stderr)
    sys.exit(1)


def _anchor_counts(src, anchor):
    """(total, at_column_0) occurrences of `anchor`.

    Column-anchoring is the difference between a loud FATAL and a quiet
    half-apply: a bare ``class ChatCompletionRequest(`` also matches an INDENTED
    occurrence (nested class, a doc/example string), and splicing a column-0
    helper body next to an indented match produces an IndentationError at
    ``vllm serve`` import time rather than here. Requiring the single match to
    start at a line boundary makes an upstream re-indent fail this applier.
    """
    total = 0
    col0 = 0
    idx = src.find(anchor)
    while idx != -1:
        total += 1
        if idx == 0 or src[idx - 1] == "\n":
            col0 += 1
        idx = src.find(anchor, idx + 1)
    return total, col0


def _require_unique_col0(src, anchor, label):
    total, col0 = _anchor_counts(src, anchor)
    if total != 1:
        _fatal(f"{label} anchor {anchor!r} found {total}x, expected 1")
    if col0 != 1:
        _fatal(
            f"{label} anchor {anchor!r} matches but NOT at column 0 (indented "
            f"occurrence) — the target has been re-indented/re-nested; refusing "
            f"to splice a module-level helper next to it"
        )


def main():
    if not CHAT.exists():
        _fatal(f"missing target {CHAT}")
    src = CHAT.read_text()

    if MARK_HELPER in src and MARK_CALL in src:
        print(f"{LOG} already applied (both markers present) — no-op")
        return 0
    if (MARK_HELPER in src) != (MARK_CALL in src):
        _fatal("half-applied: exactly one marker present, refusing to patch")

    if "RepetitionDetectionParams" not in src:
        _fatal(
            "target has no RepetitionDetectionParams — this vLLM build predates "
            "the native repetition detector; the graft has nothing to default"
        )

    _require_unique_col0(src, ANCH_HELPER, "helper")
    _require_unique_col0(src, ANCH_CALL, "call")

    src = src.replace(ANCH_HELPER, HELPER_SRC.lstrip("\n") + "\n" + ANCH_HELPER, 1)
    src = src.replace(ANCH_CALL, REPL_CALL, 1)

    # Compile BEFORE writing (house model: patch_bug158_…py:265-268). Without
    # this a drifted anchor writes a syntactically broken module, the applier
    # exits 0, `set -e` is satisfied, and the boot only dies ~30 appliers later
    # at `exec vllm serve` import — the quiet half-apply class (review M1).
    try:
        compile(src, str(CHAT), "exec")
    except SyntaxError as exc:
        _fatal(f"patched file does not compile: {exc}")

    CHAT.write_text(src)
    # Informational only — the AUTHORITATIVE state is the env at REQUEST time
    # (the injected helper re-reads it on every call, so a flag flip needs no
    # re-patch). This line reports the env as seen by THIS applier process.
    _armed = os.environ.get("GENESIS_ENABLE_BUG077_REPETITION_DETECT", "0") == "1"
    _state = (
        "ARMED: GENESIS_ENABLE_BUG077_REPETITION_DETECT=1 at apply time"
        if _armed
        else "dark: GENESIS_ENABLE_BUG077_REPETITION_DETECT unset/0 at apply time"
    )
    print(f"{LOG} applied to {CHAT} ({_state})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
