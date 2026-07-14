# BUG-042 fix set — PN84 + PN85 activation plan (STAGED, NOT DEPLOYED)

Prepared 2026-07-11. **Nothing here has been executed.** vLLM :8020 was LIVE
and serving at prep time; no live file, venv, container, or the running engine
was touched. Patches authored + application-tested against the read-only dev799
source checkout `/home/user/adawt/vllm-69715823` (git HEAD `69715823d` ==
live `/version` `0.23.1rc1.dev799+g69715823d`).

## What these two patches unblock
Flipping `GENESIS_PN81_PACKED=1` (PN81 packed rerank: 136 docs 2.69s vs
7–17s pointwise). Packed mode reads P(yes) via `prompt_logprobs` and crashed the
engine live 2026-07-07 03:38. Two prerequisites, one patch each:

| Patch | File (fixes/) | Upstream | Target (in container) | Diff |
|---|---|---|---|---|
| PN84 | `patch_pn84_42245_prompt_logprobs_zeros.py` | #42245 (open at pin) | `vllm/v1/outputs.py` `LogprobsTensors.empty_cpu` | `torch.empty*`→`torch.zeros*` (3 swaps) |
| PN85 | `patch_pn85_46066_async_spec_discard_guard.py` | #46066 (merged post-pin) | `vllm/v1/core/sched/scheduler.py` `update_from_output` | +`and request.async_tokens_to_discard == 0` guard |

Both are pure-Python text patches, idempotent, fail-loud on anchor drift,
self-retire (no-op) if upstream lands the fix into the pin. Same convention as
PN80/PN82.

## Async-sched approach: copy #46066 (chosen) vs minimal guard (fallback)
**Chosen: copy #46066 (PN85).** It removes the crash at the source — the
scheduler no longer runs rejection accounting on a stale spec frame still
pending discard, whose pre-reset counts underflow
`num_computed_tokens`/`num_output_placeholders` and desync the
execute_model→sample_tokens step machine (the `gpu_model_runner.py:~4076`
"State error" assert). Pure-Python one-clause guard; upstream carries a
regression test (`tests/v1/core/test_async_scheduler.py::
test_no_placeholder_underflow_on_discarded_spec_frame`).

Rejected as primary: a per-request **reject/serialize guard** (detect
`prompt_logprobs` + async-sched + chunked-prefill and 400 it or force a
serialized/async-off path). It only hides the DoS, permanently forbids or
slows the exact combo packed rerank needs, and leaves the underlying engine
race live for any other caller. **Kept as documented FALLBACK**: if the soak
(step 6) still shows an EngineCore restart, re-gate packed at the PN81 route
(it already 400s when `GENESIS_PN81_PACKED!=1`) and file a follow-up — do NOT
flip the gate on.

Residual uncertainty: #46066's regression test exercises `ngram_gpu` spec
discard, not the `prompt_logprobs`+chunked-prefill entry specifically. Our
stack hits the same `async_tokens_to_discard` underflow path via MTP n=3 +
async scheduling + chunked prefill, so #46066 is the root-cause fix — but the
soak in step 6 is the gate, not an assumption.

## Application order (in a 0-worker / maintenance window ONLY)
Patches mount read-only into the container and run at entrypoint after Genesis
`apply_all`, same as PN80/82 (README §Wiring). Add two lines to the compose
entrypoint (`models/qwen3.6-27b/vllm/compose/single/tools-text-aibox.yml` and
any other Genesis compose in use), AFTER the existing PN80/PN82 invocations:

```bash
python3 /fixes/patch_pn84_42245_prompt_logprobs_zeros.py
python3 /fixes/patch_pn85_46066_async_spec_discard_guard.py
```

Order among themselves is independent (different files). Keep them before
`patch_pn81_rerank_endpoint.py`/PN82 is irrelevant — no shared file. Every
patch aborts boot loud (exit 1) if its anchor is gone.

## Restart procedure (this DOES take :8020 down — schedule it)
1. Confirm 0 in-flight work; announce the window.
2. Add the two entrypoint lines to the compose(s).
3. `systemctl restart vllm-qwen36.service` (system unit, sudo) — the container
   re-applies the full patch chain incl. PN84/PN85 at boot.
4. Boot-guard note: the promote pins tag `validated-qwopus-69715823`. PN84/PN85
   live in `fixes/` (mounted), not in the genesis tag, so no tag move needed;
   but COMMIT the compose edit so a mid-session restart can't revert it
   (checkpoint LESSON: commit genesis/compose edits same-step).
5. Wait for `curl -s localhost:8020/version` to return + a trivial chat to
   succeed (engine healthy, patch chain applied — watch boot log for the
   `[pn84-...] applied` / `[pn85-...] applied` lines; a `FATAL: anchor-not-found`
   means abort and re-derive).

## GENESIS_PN81_PACKED flip
Only AFTER steps below pass with the gate still OFF is confirmed healthy:
- Set `GENESIS_PN81_PACKED=1` in the compose env (leave `GENESIS_PN81_RERANK=1`).
- Restart once more (or set before step 3 and validate packed directly — but
  the safer sequence is: boot with patches + gate OFF, confirm engine stable,
  then flip).

## Validation calls (exact)
Run against `http://localhost:8020`. **A = the call that crashed live; must now
return scores, engine RestartCount unchanged.**

**(A) Packed rerank (the 03:38 crash repro)** — requires `GENESIS_PN81_PACKED=1`:
```bash
RC0=$(systemctl show -p NRestarts --value vllm-qwen36.service 2>/dev/null || echo NA)
curl -s -X POST http://localhost:8020/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query":"clock source correctness","documents":[
        "monotonic clocks avoid wall-clock jumps under NTP slew",
        "bounded resource growth requires backpressure",
        "unrelated: cache eviction is LRU by default",
        "wall-clock reads race with DST transitions"],
       "pack":4}'
# EXPECT: JSON results with relevance_score per index, HTTP 200. NOT a 400
# "packed mode disabled", NOT a dropped connection / engine restart.
RC1=$(systemctl show -p NRestarts --value vllm-qwen36.service 2>/dev/null || echo NA)
echo "RestartCount before=$RC0 after=$RC1  (MUST be equal)"
```
Then hammer it concurrently with chat load (the live crash needed a
prompt_logprobs packed call sharing batches with MTP decode under chunked
prefill):
```bash
# background chat stream to force shared batches + chunked prefill
for i in $(seq 1 8); do
  curl -s http://localhost:8020/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.6","messages":[{"role":"user","content":"count slowly to 300 with commentary"}],"max_tokens":400,"stream":true}' >/dev/null &
done
# fire packed rerank repeatedly during the chat load
for i in $(seq 1 20); do curl -s -X POST localhost:8020/rerank -H 'Content-Type: application/json' \
  -d '{"query":"clock source correctness","documents":["a","b","c","d","e","f","g","h"],"pack":8}' >/dev/null; done
wait
# EXPECT: engine still up, NRestarts unchanged, no EngineCore fatal in journal.
journalctl -u vllm-qwen36.service --since '2 min ago' | grep -i 'State error\|sample_tokens\|EngineCore' && echo "!! CRASH SIGNATURE PRESENT — ABORT, revert gate" || echo "clean"
```

**(B) Bit-stability check (PN84 correctness, 0.985496 pointwise baseline)** —
prompt_logprobs must be order/history-independent after PN84. Pointwise path is
already bit-stable (0.985496 across order/history/repeat — decode logprob);
PN84 makes the PACKED (prompt_logprobs) path deterministic too:
```bash
# same doc set, two request orders — packed scores per doc must match bit-for-bit
A=$(curl -s -X POST localhost:8020/rerank -H 'Content-Type: application/json' \
  -d '{"query":"clock source correctness","documents":["mono clock","backpressure","lru cache"],"pack":3}')
B=$(curl -s -X POST localhost:8020/rerank -H 'Content-Type: application/json' \
  -d '{"query":"clock source correctness","documents":["lru cache","mono clock","backpressure"],"pack":3}')
# EXPECT: the relevance_score for a given document string is identical in A and B
# (reorder-invariant). Any per-doc delta => prefix-cache stale-memory leak =>
# PN84 not effective, do NOT trust packed scores.
echo "$A"; echo "$B"
```
Also re-confirm the pointwise 0.985496 bit-stability is unchanged (regression
guard — PN84/PN85 must not perturb the pointwise path): run the existing
pointwise rerank bit-stability probe from the 07-07 validation set and confirm
the discrimination score (planted doc rank 0, 0.9994 vs 0.0002 noise) holds.

## Rollback
Revert is trivial: remove the two entrypoint lines (or set
`GENESIS_PN81_PACKED=0`) and restart. Patches self-gate; leaving them mounted
with the gate OFF reproduces today's safe state exactly. The dev424 immortal
image + boot-guard rollback (checkpoint) remains the deep rollback.

## Open risks
1. #46066 test covers ngram_gpu discard, not the prompt_logprobs+chunked-prefill
   entry — soak (A) is the real gate, not an assumption.
2. Patches were application-tested on a source COPY, not the container's
   `/usr/local/lib/python3.12/dist-packages` copy. The dist-packages tree should
   be byte-identical to the pinned checkout, but the anchor-uniqueness + fail-loud
   guards catch any drift at boot (abort, don't corrupt).
3. Container Python is 3.12 (TARGET path hardcodes `python3.12/dist-packages`,
   matching PN80/82). If a future image bumps the minor, the `TARGET.exists()`
   FATAL fires — update the path, same as the other PNs.
4. Flip is a two-restart sequence for safety (patches+gate-off healthy, THEN
   gate on); doing it in one restart is fine but loses the isolation datapoint.

## ⚠ BOOT-GUARD BLOCKER (found 2026-07-12 00:25 on first activation attempt)
`boot-guard.sh` (ExecStartPre) does `git checkout -q $TAG` (validated-qwopus-69715823
= d489945) on EVERY boot — it hard-pins to the validated TAG, not just "guard a dirty
tree". A compose commit ON TOP of the tag (0f0806a) is REVERTED at restart → PN84/PN85
never enter the entrypoint. Confirmed: container recreated 00:23:54 came up on the old
entrypoint (PN80/82/81 only, no PN84/85), engine healthy but patches absent.

**So the runbook step "commit the edit" is INSUFFICIENT.** To activate, one of:
 (a) move the `validated-qwopus-69715823` tag to the PN84/85 commit — but that tag MEANS
     "soak-proven", so this pre-declares unvalidated code validated (contract violation);
 (b) temporarily point the tag at the candidate, boot+soak with gate still OFF then ON,
     and only KEEP the tag there if the soak passes (reset to d489945 if it fails);
 (c) add a boot-guard bypass env for a maintenance window.
USER DECISION REQUIRED — moving a validated production tag is beyond a "restart is fine"
go-ahead. Recovery of the wiring commit: `git cherry-pick 0f0806a` (in reflog).
Live state after the aborted attempt: engine healthy on d489945, gate OFF, NRestarts=0 —
byte-identical to pre-attempt. Nothing to roll back.
