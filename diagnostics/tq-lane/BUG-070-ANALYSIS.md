# BUG-070 — first constrained token duplicated (`{{`) under json_schema × MTP spec-decode

Date: 2026-07-13 · Analyst: root-cause lane (read-only) · Live: dev1060
(`nightly-9e57de7197f234f9d9187715d96e07e007048c0f`) + genesis v7.72.2 + /fixes chain.
No fix applied to the live system by this analysis.

**Verdict up front:** upstream vLLM bug, present in STOCK dev799 and dev1060, tracked
upstream as **vllm#48228**, root-caused and fixed by open PR **vllm#44993** (2 pure-Python
files, anchors verified byte-exact in our as-patched tree → text-patchable as a
fixes/-style patcher). Not introduced or worsened by PN82/PN91g/PN93/PN94. Not the same
mechanism as BUG-069.

---

## (a) Mechanism, with as-patched file:line evidence

All line numbers below are from the **live container's as-patched files**
(`sudo podman exec vllm-qwen36-27b-tools-text`), extracted to
`/tmp/claude-1000/-home-user/181666e5-13d3-46c4-9b2f-d30643e86cf5/scratchpad/live-vllm/`.
First fact that frames everything:

> **Genesis P62 (our vllm#36138 backport) DID NOT APPLY on dev1060 — and did not apply
> on dev799 either.** Live boot log: `[Genesis] DRIFT skipped: P62 … required anchor for
> 'p62_grammar_bitmask' not found`. Upstream merged its own reasoning-boundary rework
> (PR #44297, merged 2026-07-04 — the `detect_reasoning_end`/`simulated_buf` serial loop)
> which drifted P62's anchor. So the live structured-output × spec-decode path is **stock
> upstream**, and stock upstream has the bug.

### Token-by-token walkthrough (MTP n=3, json_schema, reasoning parser qwen3)

Let step *k* be the decode step whose spec window contains the reasoning-end marker
`</think>` (MTP drafts it reliably at the end of a thinking block; probability of the
marker landing inside a spec window with ≥1 subsequent constrained token ≈ observed 75%).

1. **Schedule of step k — bitmask side is CORRECT (this is merged #44297, in dev1060).**
   `vllm/v1/structured_output/__init__.py::grammar_bitmask`, serial loop lines 284–340:
   position of `</think>` in the draft window fills an unconstrained row and flips
   `apply_bitmask=True` for subsequent positions (lines 293–313); post-marker draft rows
   and the bonus row (`bonus_apply`, line 336) are constrained **from the grammar's ROOT
   state**; all in-loop FSM advances are rolled back (line 339–340,
   `grammar.rollback(state_advancements)`).

2. **Model runs step k.** The rejection sampler accepts `</think>` and, because the
   post-marker verify position/bonus row was bitmask-constrained at the root state, the
   recovery/bonus token sampled there is the first grammar token — `{`.
   `new_token_ids` for step k = `[..., </think>, {]`.

3. **`update_from_output` step k — the bug.**
   `vllm/v1/core/sched/scheduler.py:1771` calls
   `structured_output_manager.should_advance(request)`.
   `vllm/v1/structured_output/__init__.py::should_advance` lines 394–426: reasoning was
   not yet ended, `is_reasoning_end_streaming` fires on this step's window →
   `structured_req.reasoning_ended = True` (line 403), **and then for every constraint
   type except STRUCTURAL_TAG it `return False`** (the "Defer FSM advance until the next
   pass" block, lines 405–426 — only `STRUCTURAL_TAG` under spec-decode takes the
   record-boundary-and-return-True branch at lines 412–424).
   Consequence: `grammar.accept_tokens` at scheduler.py:1784 is **never called for step
   k** — the emitted `{` **never enters the FSM**. But it was already appended to the
   request (scheduler.py:2032–2047 `_update_request_with_output`) and shipped to the
   client in `EngineCoreOutput.new_token_ids` (scheduler.py:1875). The qwen3 reasoning
   parser routes everything after `</think>` to `content` → the client already has `{`.

4. **Step k+1 — re-emission.** `reasoning_ended` is now True, so
   `should_fill_bitmask` (lines 351–369) returns True and the bitmask is built from the
   FSM's current state — which is still **ROOT** (nothing ever advanced it). Draft
   validation (`scheduler.py:2102/2133 → grammar.validate_tokens`) trims the MTP drafts
   (they continue from real context `…</think>{`, invalid from root), so the step samples
   essentially one token under a root-state bitmask that admits only `{` → the model
   emits **`{` a second time**. This time `should_advance` returns True
   (`reasoning_ended` short-circuit, line 391–392) and the FSM advances past `{`.

5. **Steps k+2… are self-consistent:** FSM state = after one `{`, model context = after
   two `{`. Constrained sampling follows the FSM, so the remainder is grammar-perfect.
   Client content: `{` (step k, orphaned) + `{` (step k+1) + perfect JSON body =
   **`{{"name": …`**, `finish_reason=stop`. The grammar itself never saw a double brace
   — which is why nothing errored.

No-duplication (~25%) case: `</think>` arrives as the lone sampled token of its step
(e.g. all drafts rejected at position 0, or non-spec bonus with `req_tokens` empty —
then the bonus row is unconstrained and step k emits only the marker). Then no orphan
token exists and step k+1's root-state `{` is the first one.

### Upstream confirmation (independent of our code trace)

- **vllm#48228** (OPEN, filed 2026-07-10): byte-identical symptom — Qwen3.6,
  `response_format` json_object/json_schema, MTP spec-decode, `--reasoning-parser
  qwen3`, deterministic `{{"name":…` at temp 0; intermittent across serves (3/5 ≈ our
  75%); *removing `--speculative-config` fixes it; `enable_thinking:false` fixes it;
  xgrammar→guidance backend swap does NOT*.
- **vllm#44993** (OPEN PR, "Advance grammar across reasoning boundary", fixes #43388 +
  #48228) names this exact mechanism: *"the post-marker content tokens produced in the
  marker step never enter the grammar FSM. On the next step the bitmask is prepared with
  the grammar at its initial state, the model emits the opening token again."* A #48228
  reporter cherry-picked #44993's two files → **10/10 valid JSON over 5 restarts**.
- **vllm#44297** (MERGED 2026-07-04, commit `e7c9df94…`, ancestor of our nightly
  9e57de71 — verified via `git merge-base --is-ancestor` in /home/user/engines/vllm) is
  the intra-step bitmask half only; it is what we're running, and it is insufficient.
- **RFC vllm#48197 / draft PR #48200** enumerate the same defect class:
  "`should_advance` defers non-STRUCTURAL_TAG constraints", "`should_advance` mis-infers
  the draft window from `num_output_placeholders`" — the refactor's problem statement is
  a superset of BUG-070. #36138 (P62's upstream) is still OPEN, unmerged.

#44993 also fixes a second latent half we inherit (async scheduling): the
placeholder-derived delta window `num_computed_tokens - num_output_placeholders`
(structured_output/__init__.py:395) misses `</think>` entirely when drafts were rejected
(placeholders stay >0) → `reasoning_ended` never flips → grammar never enforced
(#43388). Our fix should take both halves.

## (b) Does BUG-069 share the mechanism? — No.

BUG-069 (literal `<tool_call>` text in `content` while `tool_calls` parses fine):

- BUG-070's mechanism REQUIRES an active grammar (structured output). Tool calls with
  `tool_choice:auto` run unconstrained — no FSM, no bitmask, nothing to defer. The
  mechanism cannot produce BUG-069 on auto tool choice.
- Upstream's matching track for BUG-069 is **vllm#47194** (Qwen3.6 hybrid + prefix-cache
  + MTP → `<tool_call>` XML leaks as plain text), addressed by open PR **vllm#48361**
  (hybrid-Mamba prefix-cache corruption under MTP/EAGLE: eagle cache-peek overrun on
  recurrent state + align-mode chunk fragmentation) — a state-corruption mechanism,
  orthogonal to grammar bookkeeping. Consistent with BUG-069 reproducing on BOTH dev799
  and dev1060 regardless of Wave-1.
- Shared *trigger surface* only: both live at "marker token inside an MTP spec window."
  Predicted discriminator: applying the #44993 backport will fix BUG-070's `{{` and will
  NOT move BUG-069's leak rate. (#48361 is already flagged for the Wave-3 custom build;
  PN87's header notes it subsumes #43650.)

## (c) Did PN9x introduce/worsen it? — No, proven three ways.

1. **Stock-upstream repro:** #48228 reproduces `{{` on unpatched vLLM (multiple
   checkpoints/quants). The bug does not need any of our patches to exist.
2. **Code inspection (each patcher's hunks read in full):**
   - **PN82** (`patch_pn82_bonus_logprobs_full_vocab_guard.py`) — 4 hunks, all in the
     **logprobs** lane: sampler.py gather-preference guard (`num_logprobs >= 0`), and
     rejection_sampler.py `_get_logprobs_tensors` call-site/signature/specific-token
     gather. It never touches `output_token_ids` / sampled-token placement — cannot
     duplicate a token in the stream.
   - **PN91g** (#48475) — `tl.maximum(i_t, 0)` clamp on the recurrent-state slot index in
     two FLA Triton kernels; affects which SSM state slot is READ on 0-accepted rows.
     No token emission path.
   - **PN93** (#48053) — `capture_error_mode="thread_local"` on cudagraph capture calls.
     Capture-time only; no sampling/bookkeeping semantics.
   - **PN94** (#47833/#47953) — MTP drafter embedding sharing at INIT. Its own header
     documents (and we verified) that on Qwopus3.6-27B target/MTP hidden sizes are equal
     so the runtime branch outcome is unchanged — a no-op today. At most this class of
     patch could change draft *quality* (frequency modulation), never create a
     scheduler-side duplication.
3. **Image A/B:** the dev799 image (`nightly-69715823…`, present locally) contains the
   **identical** #44297-era `detect_reasoning_end` + `trim_reasoning_for_advance` +
   defer-in-`should_advance` code, and P62's anchor is absent there too → P62 was
   drift-skipped on dev799 as well. Same bug, both pins; the dev1060 bump neither
   introduced nor worsened it. The `{{` was simply never probed pre-Wave-1
   (BUG-069's leak on dev799 is the corroborating "boundary was already broken" signal).

No env-disable boot discriminator is needed — (1)+(2) settle it from evidence alone.

## (d) Fix shape — fixes/-style text patcher backporting vllm#44993 (recommended)

**Cite:** vllm#44993 (OPEN; stacked on merged #44297 which we already have). #48361 is
NOT this fix (it's the BUG-069 track); #48200 is a draft refactor — right direction,
wrong vehicle for a hot patch.

#44993 touches exactly 2 runtime files, both pure Python, and **all anchors verified
byte-exact and unique in the LIVE as-patched tree** (checked 2026-07-13 against the
running container's files — no genesis/fixes overlap in these regions; genesis P58/P34
edits sit elsewhere in scheduler.py):

- Hunk 1 — `vllm/v1/core/sched/scheduler.py` (live line 1771), anchor unique:
  `if new_token_ids and self.structured_output_manager.should_advance(request):`
  → pass `new_token_ids=new_token_ids`. (The downstream trim + accept machinery,
  scheduler.py:1779–1795, already exists from #44297 and needs no change.)
- Hunk 2 — `vllm/v1/structured_output/__init__.py::should_advance` signature (live 371):
  add `new_token_ids: list[int] | None = None` (`Iterable` already imported, line 4-ish
  `from collections.abc import Iterable, Sequence`).
- Hunk 3 — same function, lines 394–426: replace the placeholder-derived delta window
  with `start = len(all_token_ids) - len(new_token_ids)` when `new_token_ids` is given
  (fallback preserved), and **delete the STRUCTURAL_TAG-only guard** so EVERY constraint
  type records `reasoning_end_token_index = _find_reasoning_end_index(...)` and
  `return True` at the boundary → the scheduler trims the reasoning prefix and advances
  the FSM through the orphan `{` in the same step it is emitted. Skip #44993's cosmetic
  import removals (unused `logger`/`StructuredOutputOptions` are harmless).

Full PR diff saved:
`/tmp/claude-1000/-home-user/181666e5-13d3-46c4-9b2f-d30643e86cf5/scratchpad/pr44993.diff`.
Suggested name: `patch_pn96_44993_grammar_advance_across_reasoning_boundary.py`, wired in
the entrypoint after PN94. Interaction note: keep `GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=1`
harmless-as-is (it drift-skips), or drop it from the compose in the same change; if
genesis ever re-anchors P62 for dev1060, retire P62 in favor of this backport — they
patch the same function and MUST NOT both apply. Add `def should_advance(\n        self,\n        request: "Request",\n        new_token_ids` as an upstream-drift/self-retire marker
(when #44993 merges, the patcher self-retires).

Residual risk to note in the patcher header: #44993 is still OPEN (not merged), so
review comments may still reshape it; our exposure is bounded to the two functions above
and the canary gate below decides.

## (e) Fix-gate probe

Reproducer (12 rounds, mixed schemas):
`python3 /tmp/claude-1000/-home-user/181666e5-13d3-46c4-9b2f-d30643e86cf5/scratchpad/canary_grammar_mtp.py`
(copy into the repo before the scratchpad is wiped, e.g.
`diagnostics/tq-lane/canary_grammar_mtp.py`).

Minimal one-shot gate (run only when `vllm:num_requests_running` is 0):

```bash
for i in 1 2 3 4; do curl -s localhost:8020/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen3.6","max_tokens":600,
  "messages":[{"role":"user","content":"Give me a person named Ada, age 36, tags [\"math\",\"code\"]."}],
  "response_format":{"type":"json_schema","json_schema":{"name":"person","schema":{
    "type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},
    "tags":{"type":"array","items":{"type":"string"}}},
    "required":["name","age","tags"],"additionalProperties":false}}}}' \
  | python3 -c 'import json,sys; c=json.load(sys.stdin)["choices"][0]["message"]["content"]; print(("DOUBLED" if c.lstrip().startswith("{{") else "OK"), c[:40])'; done
```

Gate: pre-fix baseline ≈ 3/4 DOUBLED; post-fix **0/12 DOUBLED across the full canary,
all rounds valid JSON, finish=stop**, plus one thinking-OFF round (add
`"chat_template_kwargs":{"enable_thinking":false}`) which must be OK both pre- and
post-fix (mechanism discriminator: the bug needs a reasoning boundary). Regression
guard: BUG-069 tool-leak probe expected UNCHANGED by this fix (see (b)).

---

## Addendum — sndr_core_engine recon (coordinator follow-up, 2026-07-13)

Recon tree: `…/scratchpad/sndr-recon/repo` (Sandermage/sndr_core_engine, pin
`0.23.1rc1.dev748+g2dfaae752`).

**Premise correction first (live-probed):** the claim "the live GPU boot lists P62
APPLIED" is a misreading of the boot log. The line
`[Genesis Dispatcher] APPLY P62 … | opt-in env (config: neutral)` is the dispatcher's
env-gate *decision*; the outcome line that follows is
`[Genesis] DRIFT skipped: P62 … required anchor for 'p62_grammar_bitmask' not found …
P62 cannot apply` (both lines present in `sudo podman logs vllm-qwen36-27b-tools-text`,
probed this session). The live files contain **zero** P62 markers and the stock
`should_advance` (evidence in (a) above). P62 is NOT applied — not "applied but
incomplete."

**(a) Our P62 vs Sander's dev748 P62 — diffed:**
- Same lineage, near-identical payloads. His `p62_structured_output_spec_decode_timing.py`
  (31.9K) differs from ours (22.6K) by: (1) DUAL grammar_bitmask anchors —
  `GRAMMAR_BITMASK_OLD` (dev259) + `GRAMMAR_BITMASK_OLD_DEV491`; ours carries only the
  dev491 form (re-anchored 06-13). (2) `DRIFT_MARKER_44297_BITMASK_REWRITE` /
  `DRIFT_MARKER_44993_*` watch entries (his lines 339–381). The NEW replacement hunks
  (update_reasoning_ended / identify_constrained_draft_tokens / 3 scheduler sub-patches)
  are the same text in both.
- **Neither of his anchors matches dev1060 (or dev799).** Both target pre-#44297 code;
  our live file has the #44297 `detect_reasoning_end`/`simulated_buf` rewrite, and his
  scheduler anchor expects `accept_tokens(req_id, new_token_ids)` where dev1060 has the
  #44297 `trim_reasoning_for_advance` + `advance_token_ids` block.
- Decisive: **Sander's own design retires P62 on exactly our pin.** His
  `DRIFT_MARKER_44297_BITMASK_REWRITE = "post_reasoning_end_in_window"` — that string IS
  present in our dev1060 `structured_output/__init__.py` (line 285/313), so his P62
  converts to a named upstream-drift SKIP there by design. His dev748 `anchors.json`
  (generated 2026-07-04) registers **no P62 anchors in either file** (scheduler.py lists
  only P34/P79c/PN388). And his watch-entry comment says it outright: #44297 = intra-step
  half, "#44993 (stacked on #44297) — inter-step state leak … rewriting the scheduler
  call sites our three scheduler sub-patches replace." I.e., Sander's plan for the
  #44297-era pins is *upstream #44297 + #44993*, not a re-anchored P62.
- Same story for the narrower alternative PN58 (vllm#40962): checked all 6 of our PN58
  `*_OLD` anchors against the live dev1060 files — 5 of 6 count 0 (drifted). Dead end too.

**(b) Does the json_schema `{{` path need a separate PN523/PN387-shape guard? No.**
Read both: PN387 (vendor of vllm#45346, RETIRED — native since dev714) and PN523
(vllm#47450) are *request-validation* guards — they 400 degenerate INPUT constraint
params (`json_object: false`, `json: ""`, `structural_tag: ""`, `regex: ""`) to stop an
EngineCore DoS. They act on inputs and cannot see a doubled first token in OUTPUT.
Orthogonal class. (PN523 is separately worth vendoring for its own DoS value — our pin
has the #45346 guards native but not #47450's — but it does nothing for BUG-070.)
The other siblings (p71_pn369_rejection_sampler_consolidated, pn369_relaxed_acceptance,
pn390_streaming_lse_rejection_sampler) are acceptance-policy/perf patches on the
rejection sampler — no boundary bookkeeping; reference-only. No output-side guard is
needed: #44993 removes the STRUCTURAL_TAG-only carve-out, covering json_schema /
json_object / regex / choice / grammar uniformly at the root.

**(c) Concrete fix — unchanged from (d), now with Sander's own corroboration:**
do NOT refresh P62 to dev1060. A re-anchored P62 would double-handle the reasoning
boundary against #44297's native mid-window code in `grammar_bitmask` (upstream now
advances+rolls-back the FSM through the window itself; P62's replacement does the same
job with different bookkeeping — two owners of one invariant across a pin boundary).
The right move is the (d) plan: backport **#44993** as
`patch_pn96_44993_grammar_advance_across_reasoning_boundary.py`, and adopt Sander's
drift-marker convention verbatim — his `DRIFT_MARKER_44993_SHOULD_ADVANCE_SIG`
(`"        new_token_ids: list[int] | None = None,\n    ) -> bool:"`) is exactly the
self-retire marker the new patcher should use. Optionally drop
`GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=1` from the compose in the same change
(it has been a logged no-op on both pins).

### Artifacts
- As-patched tree: `…/scratchpad/live-vllm/` (extracted from the LIVE container).
- Patch-chain replay log: `…/scratchpad/apply-log.txt` (scratch container
  `bug070-scratch`, since removed; P62 DRIFT-skip reproduced identically to live boot).
- PR diff: `…/scratchpad/pr44993.diff`.
- Upstream refs: #48228 (bug) · #44993 (fix, OPEN) · #44297 (merged prerequisite, in
  dev1060 and dev799) · #43388 (async half) · #48197/#48200 (refactor RFC) · #36138/P62
  (unmerged, drift-skipped both pins) · #47194/#48361 (BUG-069 track, distinct).
