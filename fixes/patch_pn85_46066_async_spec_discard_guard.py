#!/usr/bin/env python3
"""PN85 — skip stale spec-decode frames still pending discard in the async
scheduler's rejection accounting (BUG-042 engine-crash root fix).

Backport of vllm#46066 (merged upstream AFTER the dev799 pin 69715823d), one
guard clause in vllm/v1/core/sched/scheduler.py::update_from_output.

Crash (BUG-042): prompt_logprobs × async-scheduling × chunked-prefill kills
EngineCore with
    "State error: sample_tokens() must be called after execute_model()
     returns None."
(assertion at gpu_model_runner.py execute_model, ~4076; execute_model_state
left non-None because the step machine desynced). Live-reproduced 2026-07-07
03:38 on a PN81 packed-mode /rerank call (packed sends prompt_logprobs; the
stack runs MTP n=3 spec-decode + async scheduling + chunked prefill).

Root cause (upstream #46066): when async-scheduling discards a stale spec
frame (async_tokens_to_discard > 0), update_from_output still ran the
rejection-count arithmetic on that frame. Its pre-reset draft/accept counts
underflow num_computed_tokens / num_output_placeholders, desyncing the
scheduler's view of the batch → the engine issues a new execute_model()
before the previous step's sample_tokens() has run → the state-error assert.

Fix: gate the rejection accounting on `request.async_tokens_to_discard == 0`,
so a frame still pending discard is skipped (its discard is handled
separately). `async_tokens_to_discard` defaults to 0 (request.py:142), so the
added condition is always safe to evaluate.

This is the SECOND prerequisite (with PN84) before flipping
GENESIS_PN81_PACKED on. Chosen over a per-request reject/serialize guard
because it removes the crash at the source (the packed-rerank throughput win
survives) rather than permanently forbidding the combo; the reject/serialize
guard is retained as a documented fallback if soak shows residual crashes.
Retire when #46066 lands in the pin.
"""
import pathlib
import sys

LOG = "[pn85-async-spec-discard-guard]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)
MARKER = "# PN85:"

OLD = (
    "            if scheduled_spec_token_ids and (\n"
    "                generated_token_ids or self.num_sampled_tokens_per_step == 0\n"
    "            ):\n"
)
NEW = (
    "            # PN85: vllm#46066 backport — skip a stale spec frame still\n"
    "            # pending discard (async_tokens_to_discard > 0); its pre-reset\n"
    "            # rejection count underflows num_computed_tokens/placeholders\n"
    "            # under async-scheduling + spec-decode, desyncing the engine\n"
    "            # step machine (BUG-042 EngineCore fatal: \"sample_tokens() must\n"
    "            # be called after execute_model() returns None\"). Attr defaults\n"
    "            # 0 (request.py) so the guard is always safe to evaluate.\n"
    "            if (\n"
    "                scheduled_spec_token_ids\n"
    "                and (generated_token_ids or self.num_sampled_tokens_per_step == 0)\n"
    "                and request.async_tokens_to_discard == 0\n"
    "            ):\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    text = TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} already applied (idempotent)")
        return 0
    # Upstream-merged drift: guard already present.
    if "async_tokens_to_discard == 0" in text and "scheduled_spec_token_ids" in text:
        # Distinguish the accounting-guard site (this fix) from unrelated uses.
        if "and request.async_tokens_to_discard == 0" in text:
            print(f"{LOG} upstream drift: rejection-accounting guard already present "
                  f"— self-retire (no-op)")
            return 0
    if OLD not in text:
        print(f"{LOG} FATAL: anchor-not-found — upstream refactor of "
              f"update_from_output rejection accounting; re-derive (BUG-042 "
              f"engine crash returns without this)", file=sys.stderr)
        return 1
    if text.count(OLD) != 1:
        print(f"{LOG} FATAL: ambiguous anchor", file=sys.stderr)
        return 1
    text = text.replace(OLD, NEW, 1)
    TARGET.write_text(text)
    print(f"{LOG} applied: async-scheduling rejection accounting now skips "
          f"discard-pending spec frames (vllm#46066 backport)")
    return 0


sys.exit(main())
