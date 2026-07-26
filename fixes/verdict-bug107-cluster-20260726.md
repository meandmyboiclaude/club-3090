# VERDICT — BUG-107c / 107d / 107e / 107h / BUG-108(trace-traps)
Adjudicated 2026-07-26 ~22:40 CEST against the LIVE box. No service was restarted,
no benchmark was run, no config was changed. All probes read-only except three
single small chat completions against `:8021` (GPU was at 0% util, no bench in flight).

Yardstick config (present-tense, `sudo podman ps` + boot log
`vllm-boot-logs/vllm-tcbench-8021-20260726-214041.log`):
image `localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725`, engine
`0.1.dev1+g4e2e9bf00.d20260725`, `max_num_batched_tokens=4128`,
`long_prefill_token_threshold=4128`, `max_model_len=82560`, `max_num_seqs=6`,
`gpu_memory_utilization=0.91`, KV `turboquant_3bit_nc`, KV 188,708 tok,
MTP `num_speculative_tokens=3` probabilistic.

---

## BUG-107c — NO LONGER TRUE. Both halves are dead. RETIRE.

The entry has two claims stacked on it. Both fail present-tense.

**Claim 1 — "num_speculative_tokens=4 does NOT BOOT" (the 07-19 filing).** Dead.
Boot log `vllm-boot-logs/vllm-tcbench-8021-20260726-191131.log`, container-local
timestamps 17:09:22–17:09:55Z:

| line | evidence |
|---|---|
| 2953 | `non-default args: … 'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 4, 'draft_sample_method': 'probabilistic'}` |
| 3131 | `gpu_worker.py:652 Available KV cache memory: 2.94 GiB` |
| 3137 | `kv_cache_utils.py:2214 GPU KV cache size: 188,708 tokens` |
| 3139 | `kv_cache_utils.py:2215 Maximum concurrency for 82,560 tokens per request: 2.29x` |
| 3293 | `INFO:     Application startup complete.` |

Zero `AssertionError`, zero `Engine core initialization failed`, zero traceback in
that boot. (`grep -icE 'assertionerror|Engine core initialization failed|Traceback'`
returns 2, and both hits are Genesis *patch-description* strings on lines 399 and
2209 — PN33's changelog text and SNDR-WORKSPACE-001's — not exceptions.)

Then it served a full run: `shared/seqs6-tuning-artifacts/nst4.log` →
`100 requests, 0 errors, 9.1 min`; results
`folderX/qbench45/results/aibox-20260726-nst4-seqs6-util091__gpqa_auto__thinkingcap_router_online_c6.jsonl`
scores **n=100, correct=78, finish_reason `stop` ×100** (zero length-capped).
Occupancy `occ_nst4.txt`: mean running 5.87/6 = 98%.
An independent output audit (`seqs6-tuning-artifacts/CORRUPTION-nst4.md`, 101 real
generations read by hand) closes with *"no evidence that num_speculative_tokens=4
corrupts decoded text"* — 0 floods, 0 decode loops, 0 gibberish over 492 KB.
nst=5 and nst=6 also booted and ran the same evening
(`tcbench-nst5-container.log`, `tcbench-nst6-screen30-container.log`).

**Claim 2 — the 4160 alignment fixed point / "now a BUDGET DECISION".** Also dead,
and this is the part a reader would otherwise carry forward wrongly.

The 07-20 note concluded n=4 *requires* moving the whole alignment to
`mamba_block_size = max_num_batched_tokens = long_prefill_token_threshold = 4160`,
and that even then it costs KV (135,735 tok, wouldn't fit 65,000 ctx without
`max_model_len ≤ 49920` or util > 0.95).

Today's successful n=4 boot used **`max_num_batched_tokens=4128` and
`long_prefill_token_threshold=4128`** — the *original* BUG-076 geometry, untouched.
No 4160 lockstep was applied and no assert fired, so on `dev1474cherrymax` vLLM no
longer raises the attention block to 4160 at n=4. The `validate_block_size` assert
is not reachable at this depth on this pin. The only spec-decode remark in the boot
is the benign `vllm.py:1767 max_num_scheduled_tokens is set to 4128 based on the
speculative decoding settings … Consider increasing max_num_batched_tokens`, i.e. a
performance suggestion, not a failure.

And the KV budget blocker is gone too: n=4 profiled **2.94 GiB / 188,708 tokens at
`max_model_len=82560`** — versus the 07-20 measurement of 135,735 tokens that could
not fit 65,000. For direct comparison, the *current* n=3 boot
(`…-214041.log`, 19:39:05Z) profiles **2.92 GiB / 188,708 tokens** — nst=4 and
nst=3 are KV-identical on this build. "n=4 costs VRAM twice" no longer holds.

**Live consequence of the alignment concern: none.** Nothing needs re-aligning to
4160; no BUG-076 re-opening is implied; there is no budget trade to decide.
Whatever the operator picks among nst 3/4/5/6 is now a pure
throughput/acceptance question on the current build, and the 07-20 trade table
(62.0 / 55.8 / 50.2 tok/s for n=3/4/5) was measured on `dev1060cherry` at a
different geometry and should not be quoted as current.

---

## BUG-107e vs BUG-107h — 107h is RIGHT. 107e is a MISDIAGNOSIS. RETIRE 107e.

**A future reader should believe BUG-107h and ignore BUG-107e.**

### Why 107e is wrong

107e's load-bearing claim is *"protocol.py + serving.py contain no reads of
`thinking_token_budget`"*. That grep was run against the flat path
`vllm/entrypoints/openai/protocol.py`. Live check inside the running container:

```
ls /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/protocol.py
  → No such file or directory
```

The module does not exist on this pin — `entrypoints/openai/` is split into
per-endpoint packages (`chat_completion/`, `completion/`, `engine/`, `models/`,
`responses/`). 107e's zero-hit grep was a **grep-over-a-nonexistent-path**, which
reads identically to "the field isn't handled". It is the exact failure mode
already on file as the grep-missing-path trap.

### Where the field actually lives (all read live, in-container)

```
entrypoints/openai/chat_completion/protocol.py:247   thinking_token_budget: ThinkingTokenBudget = None
entrypoints/openai/chat_completion/protocol.py:797   thinking_token_budget=self.thinking_token_budget,   # inside to_sampling_params (def at :630)
entrypoints/openai/completion/protocol.py:227        thinking_token_budget: ThinkingTokenBudget = Field(…)
entrypoints/openai/completion/protocol.py:375        thinking_token_budget=self.thinking_token_budget,
sampling_params.py:37                                def validate_thinking_token_budget(...)
sampling_params.py:346                               thinking_token_budget: int | None = None
sampling_params.py:432                               thinking_token_budget=thinking_token_budget,
v1/sample/thinking_budget_state.py                   ThinkingBudgetStateHolder
v1/sample/metadata.py:11,55                          holder imported + carried on the sampling metadata
v1/worker/gpu_model_runner.py:4716                   self.input_batch.thinking_budget_state_holder
```

So the payload → `SamplingParams` → V1 sampler holder chain is complete and
unbroken. 107h's line numbers (`:244`, `:774`) are a few lines off from today's
(`:247`, `:797`) only because the Genesis PN71 patch inserts a block at
`chat_completion/protocol.py:676-693` — which itself *reads and writes*
`thinking_token_budget`, further proof the field is load-bearing, not vestigial.

The compose pins `VLLM_USE_V2_MODEL_RUNNER=0` (confirmed in the live container
env), i.e. the V1 runner — the one that implements thinking budgets — is the one
executing. That pin is what makes the chain above the real path and not a dead one.

### Behavioural proof, taken today

Three chat completions to `:8021`, identical prompt / `temperature 0.6` / `seed 7`,
varying only `thinking_token_budget`:

| requested budget | completion_tokens | finish_reason | answer chars |
|---|---|---|---|
| 200 | 254 | stop | 208 |
| 600 | 737 | stop | 354 |
| 1200 | 1271 | stop | 721 |

`completion_tokens` tracks the requested budget almost exactly once the answer tail
is subtracted (≈202 / ≈649 / ≈1091 thinking tokens for 200 / 600 / 1200). A field
that "dies at the API boundary" cannot do that. The holder is tracking, and has
been.

This also lines up with two independent observations already on the box: the
2026-07-26 measurement finding rows force-closed at exactly grant−5 and grant−13 on
an older run, and `CORRUPTION-nst4.md`'s note that all 4 truncated generations in
the nst=4 run *"land exactly on `</think>` (forced think-close shape, not a stream
defect)"* — that is engine budget enforcement, observed by an auditor who was not
looking for it.

### Consequences, restated so they're not re-litigated

1. Tier budgets have been engine-enforced since PN100 shipped 07-18. The "first-ever
   enforcement" framing in 107e is retracted; the darkness it saw was logging-only
   (the 107d slice defect + the 107g handler defect).
2. **PN109 was redundant and the repo already acted on 107h.** `fixes/patch_pn109_*`
   is gone from `fixes/`, and `tcbench8021.yml:872-876` carries
   `# [PN109 RETIRED 2026-07-20 BUG-107h] the 107e diagnosis was wrong —` with the
   invocation commented out pointing at `/fixes/_archive/patch_pn109_budget_bridge.py`.
   So the *code* has believed 107h for six days; only the tracker still says
   otherwise. That gap is the whole hazard here.
3. The capcurve / 3-arm / true-mean results are unaffected — enforcement was real
   under either diagnosis.

### One real defect that survives, and it is NOT what 107e said

In all three probes above, `usage.completion_tokens_details.reasoning_tokens` came
back **0** and `message.reasoning_content` came back **null**, even though 200–1100
thinking tokens were demonstrably generated and billed into `completion_tokens`.
The `content` field begins with a bare `\n\n` and holds only the answer. So the
thinking is generated and enforced, but the qwen3 reasoning parser is not
attributing it on this path (consistent with the chat template opening `<think>` in
the *prompt*, which is the same root shape as BUG-107d/107g). Anyone measuring
`rtok` off the OpenAI usage block on this endpoint will read 0 and conclude "no
thinking happened" — a fresh version of exactly the mistake 107e made. Filed below
as a new open item; **it needs a change to a file I do not own** (the reasoning
parser / usage-accounting path), so I have not touched it.

---

## BUG-107d — untouched, dependency noted only

Not adjudicated here: a concurrent agent owns the boot-receipt audit for the
FIXED-but-unbooted cluster. Recording one incidental read so it isn't lost:
`sudo podman logs vllm-tcbench-8021 | grep -c "PN108: observing think block"`
returns **120** on the current (19:38:32Z) boot, so PN108 is observing live and the
"validation rides next :8021 boot" condition appears discharged. That is a pointer
for the owning agent, not a verdict.

---

## BUG-108 (trace-capture traps) — trap 2 CLOSED, trap 1 half-closed. Downgrade, don't retire.

**Trap 2 — "the var is set TWICE in tcbench8021.yml (~line 137 and ~408), later wins".
NO LONGER TRUE. Closed.**
`grep -n VLLM_TRACE_CONTENT_MAX_CHARS models/qwen3.6-27b/vllm/compose/single/tcbench8021.yml`
now yields exactly two lines, only one of which is an assignment:
- `:697  - VLLM_TRACE_CONTENT_MAX_CHARS=${VLLM_TRACE_CONTENT_MAX_CHARS:-65536}`
- `:945  # Knob: VLLM_TRACE_CONTENT_MAX_CHARS. Self-retires if upstream re-adds.`  (a comment)

Single assignment, no shadowing, and it is now env-overridable rather than
hard-pinned. `sudo podman exec vllm-tcbench-8021 printenv VLLM_TRACE_CONTENT_MAX_CHARS`
→ `65536`, matching the file. The "verify with `podman exec printenv`" advice
remains good practice but the specific duplicate it guarded against is gone.

**Trap 1 — "`=0` means capture OFF, not unbounded". STILL TRUE as a code fact, but
disarmed operationally, and the doc it blamed has already been corrected.**

The code fact holds: `fixes/patch_pn99_trace_content_prompt_completion.py:38`
still defaults `_TRACE_CONTENT_MAX_CHARS` to `8192`, and `:63` still guards
`if _TRACE_CONTENT_MAX_CHARS > 0`. So setting the var to `0` still silently
disables content capture. Nobody should ever set it to 0.

But the doc half is done. `shared/pn108/CALIBRATION-20260719.md:251-256` already
carries the correction inline — it now says to use *"a large FINITE value —
⚠️ CORRECTED 2026-07-20: `0` DISABLES content capture entirely; the PN99 patch
guards with `if _TRACE_CONTENT_MAX_CHARS > 0`. The original '0 = off/unbounded'
reading here was backwards and cost the first capture attempt 30 empty spans"* —
and the same paragraph also warns about the duplicate-line trap. Both traps are
documented at the point of use. The entry's "doc" half is discharged.

**Residual, and the reason this is a downgrade rather than a retirement.** Two stale
numbers remain in that doc, and they are the ones that manufactured a false finding
once already:
- `CALIBRATION-20260719.md:42` still states content is middle-truncated at
  `VLLM_TRACE_CONTENT_MAX_CHARS=8192`. Live value is 65536.
- `:251` still recommends `2000000`; live is 65536.

65536 is 8× the old cap and is why the "83% of prod calls missing the JSON
directive" artifact cannot recur at typical span sizes — but it is *still finite*,
and `CORRUPTION-nst4.md` records a max output of 15,725 chars against inputs that
can be much larger. Any future span-content analysis must check for the
`…[<N> chars truncated]…` marker before drawing a conclusion from span text, exactly
as before, just at a higher threshold. Suggested residual scope: correct those two
numbers in the calibration doc (I do not own it) and keep the marker-check rule.
