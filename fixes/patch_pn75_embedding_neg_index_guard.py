#!/usr/bin/env python3
"""PN75 — Clamp negative/OOB token ids in VocabParallelEmbedding on tp_size==1.

ROOT CAUSE (BUG-028, the LLM-rerank + thinking_token_budget crash)
------------------------------------------------------------------
Under structured-output (guided-JSON / tool-call) + MTP spec-decode, the
scheduler PADS grammar-rejected draft tokens with the placeholder id -1:

  v1/core/sched/scheduler.py:2018
      spec_token_ids.extend([-1] * num_invalid_tokens)
      num_invalid_spec_tokens[req_id] = num_invalid_tokens     # :2019
  v1/core/sched/scheduler.py:1982   request.spec_token_ids = spec_token_ids (incl -1)

On the NEXT step those -1s are re-emitted into the scheduled batch:

  v1/core/sched/scheduler.py:1135-1136  (_consume_spec_decode_tokens_for_step, P58)
      if request.spec_token_ids:
          spec_token_ids = request.spec_token_ids        # contains [-1,-1,-1]
  -> scheduled_spec_decode_tokens[req] = [-1,-1,-1]       (:602)

They are then written verbatim into the token buffer and fed to embedding:

  v1/worker/gpu_input_batch.py:505   token_ids_cpu[idx, s:e] = spec_token_ids  (-1s)
  v1/worker/gpu_model_runner.py:1958-1962  torch.index_select(... token_ids_cpu_tensor
                                            ...) -> input_ids.cpu  (selects the -1s)
  v1/worker/gpu_model_runner.py:1742  input_ids.copy_to_gpu()  (non-async path)
  model_executor/models/qwen3_moe.py:475/748  embed_input_ids -> embed_tokens(input_ids)
  model_executor/layers/vocab_parallel_embedding.py:472-489  VocabParallelEmbedding.forward:
      if self.tp_size > 1:
          masked_input, input_mask = get_masked_input_and_mask(...)   # maps OOB -> 0
      else:
          masked_input = input_            # <-- tp_size==1: NO MASK; -1 passes through
      ... self.quant_method.embedding(self, masked_input.long())      # F.embedding(-1)

This deployment runs --tensor-parallel-size 1, so the OOB-masking branch is
SKIPPED and F.embedding() receives index -1 -> CUDA device-side assert
("index out of range in self" / indexSelect bounds assert). Because CUDA is
async, the assert SURFACES at the next CPU-touching op, which is the Genesis
P67b synth-seq_lens kernel launch (turboquant_attn.py forward, the
`seq_lens[:B,None] - K1 + 1 + offs[None,:]` line) -- which is why the
traceback fingers P67b even though P67b is only the BYSTANDER, not the cause.

WHY no-think / reasoning:"off" does NOT crash:  enable_thinking=false => MTP
spec-decode never engages => no draft tokens => no -1 placeholders => clean.
WHY reasoning_effort alias does NOT crash:  it only caps max_tokens (PN71),
the long thinking truncates with finish=length BEFORE accumulating a
grammar-rejected spec batch.  WHY thinking_token_budget:N CRASHES:  it forces
real bounded thinking over the ~20K listwise rerank prefill, MTP drafts run,
grammar rejects drafts mid/post-prefill => [-1,-1,-1] => embedding assert.

This is the SAME family on b53b1c7 and 3f5a1e17: the -1-padding path
(scheduler.py:2018 + _consume...:1135) and the unmasked tp==1 embedding both
predate the rebase; only the *frequency* of producing the [-1,-1,-1] shape
differs between builds.

THE FIX (this file)
-------------------
Sanitize input ids on the tp_size==1 embedding path EXACTLY as the tp_size>1
path already does: clamp any out-of-range id (negative OR >= vocab) to 0
before the F.embedding lookup. This is provably correct because those token
positions are speculative DRAFT tokens that the rejection sampler discards
this step (num_invalid_spec_tokens reconciles them out of the accepted-token
accounting at scheduler.py:1625/1989-2019); their embedding output is never
committed. Valid token ids (>=0, < vocab) are untouched -> bit-identical
behavior for every real token.

KEEPS WORKING (does NOT disable anything): MTP spec-decode, P67/P67b
TurboQuant multi-query kernel, structured-output grammar, AND
thinking_token_budget:N over large prompts (the LLM-rerank use case). It only
removes the crash; the spec-verify math is unaffected because rejected drafts
were always going to be dropped.

Idempotent (marker-guarded), fail-soft (non-fatal if anchor drifts), and
auto-skips if upstream ever masks the tp==1 path itself.

Companion (recommended, see BUG-028 deep-dive): a defense-in-depth guard in
Genesis P67b/P67 that skips the synth path for batches containing -1 spec
tokens. That is NOT a substitute for this fix -- P67b is the bystander; the
assert originates at embedding and would still fire with P67 disabled.

Author: BUG-028 root-cause session 2026-06-26.
"""
import sys
from pathlib import Path

TARGET_PATHS = [
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/vocab_parallel_embedding.py",
    "/home/user/engines/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py",
]

# Anchor: the tp_size==1 else-branch of VocabParallelEmbedding.forward that
# assigns masked_input = input_ WITHOUT any OOB clamp. Matched verbatim
# against the live container source (vocab_parallel_embedding.py:483-484).
OLD = '''        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.quant_method.embedding(self, masked_input.long())'''

NEW = '''        else:
            # [Genesis PN75 BUG-028] tp_size==1 skips get_masked_input_and_mask,
            # so out-of-range ids (notably the -1 placeholder that structured-
            # output + MTP spec-decode pads rejected drafts with) reach
            # F.embedding -> CUDA device-side assert (surfaces async at the next
            # op, e.g. the P67b synth-seq_lens kernel launch). Mirror the tp>1
            # path: clamp any id outside [0, num_embeddings) to 0. Those
            # positions are rejected spec drafts whose embedding output is
            # discarded this step, so valid tokens are bit-identical.
            _pn75_n = self.num_embeddings
            masked_input = input_.where(
                (input_ >= 0) & (input_ < _pn75_n),
                input_.new_zeros(()),
            )
        # Get the embeddings.
        output_parallel = self.quant_method.embedding(self, masked_input.long())'''

MARKER = "[Genesis PN75 BUG-028]"

# If upstream ever teaches the tp==1 path to mask, this string appears and our
# OLD anchor won't match -> auto-skip (no double-clamp, no regression).
UPSTREAM_DRIFT = "masked_input = input_\n            input_mask"


def patch(path):
    p = Path(path)
    if not p.exists():
        return f"missing: {path}"
    text = p.read_text()
    if MARKER in text:
        return f"already-applied: {path}"
    if UPSTREAM_DRIFT in text:
        return f"upstream-drift-skip: {path}"
    if OLD not in text:
        return f"anchor-not-found: {path}"
    p.write_text(text.replace(OLD, NEW, 1))
    return f"patched: {path}"


found = 0
for path in TARGET_PATHS:
    result = patch(path)
    print(f"[pn75_embedding_neg_index_guard] {result}", flush=True)
    if result.startswith(("patched", "already-applied")):
        found += 1

if found == 0:
    print(
        "[pn75_embedding_neg_index_guard] WARNING: no targets patched",
        file=sys.stderr,
    )
    sys.exit(0)  # non-fatal — fail-soft, matches /fixes convention
