#!/usr/bin/env python3
"""PN84 — zero (not empty) the CPU prompt-logprobs tensors on creation.

Backport of vllm#42245 (issue #42019), open at the dev799 pin
(69715823d). Pure-Python, 3 token swaps in vllm/v1/outputs.py.

Bug: LogprobsTensors.empty_cpu() allocates with torch.empty (uninitialised
memory). When prefix caching hits N tokens, the prompt-logprobs positions
[0:N] are never written, so those rows read STALE memory left by a prior
request → prompt_logprobs values depend on request order / history / cache
state. Live pointwise rerank is unaffected (it reads the DECODE logprob, not
prompt_logprobs — verified bit-stable 0.985496 across order/history/repeat),
but PN81 PACKED mode reads P(yes) at "Relevant:" slots via prompt_logprobs,
so this stale-memory read makes packed scores order-dependent. This is the
FIRST of two prerequisites before flipping GENESIS_PN81_PACKED on.

Fix: allocate with torch.zeros / torch.zeros_like so cache-skipped positions
read a deterministic 0.0. Retire when upstream #42245 merges into the pin.
"""
import pathlib
import sys

LOG = "[pn84-prompt-logprobs-zeros]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/outputs.py"
)
MARKER = "# PN84:"

OLD = (
    "        logprob_token_ids = torch.empty(\n"
    "            (num_positions, num_tokens_per_position), dtype=torch.int32, device=\"cpu\"\n"
    "        )\n"
    "        logprobs = torch.empty_like(logprob_token_ids, dtype=torch.float32)\n"
    "        selected_token_ranks = torch.empty(\n"
    "            num_positions, dtype=torch.int32, device=\"cpu\"\n"
    "        )\n"
)
NEW = (
    "        # PN84: vllm#42245 backport — zero (not empty) so prefix-cache-skipped\n"
    "        # prompt_logprobs positions read a deterministic 0.0 instead of stale\n"
    "        # reused memory (order-dependent scores; PACKED-rerank prereq).\n"
    "        logprob_token_ids = torch.zeros(\n"
    "            (num_positions, num_tokens_per_position), dtype=torch.int32, device=\"cpu\"\n"
    "        )\n"
    "        logprobs = torch.zeros_like(logprob_token_ids, dtype=torch.float32)\n"
    "        selected_token_ranks = torch.zeros(\n"
    "            num_positions, dtype=torch.int32, device=\"cpu\"\n"
    "        )\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    text = TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} already applied (idempotent)")
        return 0
    # Upstream-merged drift: torch.empty already replaced by torch.zeros.
    if "def empty_cpu" in text:
        body = text.split("def empty_cpu", 1)[1][:800]
        if "torch.empty(" not in body and "torch.zeros(" in body:
            print(f"{LOG} upstream drift: empty_cpu already zero-inits — self-retire (no-op)")
            return 0
    if OLD not in text:
        print(f"{LOG} FATAL: anchor-not-found — upstream refactor of empty_cpu; "
              f"re-derive (PACKED rerank scores go order-dependent without this)",
              file=sys.stderr)
        return 1
    if text.count(OLD) != 1:
        print(f"{LOG} FATAL: ambiguous anchor", file=sys.stderr)
        return 1
    text = text.replace(OLD, NEW, 1)
    TARGET.write_text(text)
    print(f"{LOG} applied: LogprobsTensors.empty_cpu now zero-inits "
          f"(vllm#42245 backport)")
    return 0


sys.exit(main())
