# BUG-155 — engine-side guard spec (structured + budget-exhausted ⇒ empty payload)

**Status:** SPEC ONLY — not applied. The harness half is implemented and live in
`~/shared/folderX/qbench45/bench/budget_guard.py` (+ graders/runner/client wiring, 46 tests).
This file is the precise engine change, written for whoever owns
`vllm/_genesis/middleware/answer_rescue.py` and `auto_budget.py` (I do not).

Measured 2026-07-26 on `:8021` (thinkingcap-gptq-pro-v2, PN100 continuous,
`GENESIS_PN100_TOK_PER_STEP=260`, `GENESIS_PN100_STEP_BUDGET_MAP` **unset**).

---

## 1. The failure

With `response_format` enforcing a JSON schema, the cheapest completion the grammar
accepts is the empty container — `{"facts": []}`, ~8 answer tokens. A request whose
thinking budget expires is force-closed by the budget holder and then emits that empty
container with `finish_reason="stop"`. The caller sees a well-formed, schema-valid,
successfully-finished response that carries no data.

`prod-016` (aibox-20260726-guided-prodv3): 0 facts after `rtok=3899` (its cap 3900),
`atok=8`. The unguided arm on the identical chunk: **9 facts** at `rtok=389`, including a
retraction and a bug id. `prod-038` returned 0 facts too — at `rtok=158`, and its chunk
really is empty (`<local-command-stdout>Bye!</local-command-stdout>`). Both arms and the
111-row champion run agree on prod-038. **Empty-at-cap and empty-far-from-cap are the two
cases a caller must be able to separate, and only the engine knows the budget.**

Scale of exposure on that run: 25/40 rows (62%) sit exactly on a PN100 ceiling —
reasoning-token counts repeat across items (2098/2099/2100 ×14, 3099 ×3, 3899 ×8), which
natural stopping does not produce. Unguided control on the same 40 items: 2/40 (5%).

## 2. Why nothing currently catches it

Both are in `middleware/answer_rescue.py`:

1. `_skip_common()` (line ~134) returns **True** when `_has_structured_output(request)`.
   Every PN101 leg — escalate, repair, close-gate — is gated behind
   `not _skip_common(request)`. The one component that exists to rescue empty answers
   deliberately excludes exactly the requests BUG-155 hits.
2. The PN101 guillotine path requires `finish_reason == "length"`. BUG-155 rows finish
   `"stop"` — the grammar closed the JSON legally, so nothing looks truncated.

So the guard cannot be a tweak to an existing leg; it is a new leg that runs **on the
requests the existing gates skip**.

## 3. The change — three parts, shippable independently

### 3a. OBSERVABILITY (ship first; zero behavior change)

The response carries no budget field. Probed live on `:8021`: top-level keys are
`choices, created, ec_transfer_params, id, kv_transfer_params, metrics, model, object,
prompt_logprobs, prompt_text, prompt_token_ids, service_tier, system_fingerprint, usage`;
`usage.completion_tokens_details` has only `reasoning_tokens / accepted_prediction_tokens
/ rejected_prediction_tokens`. Nothing budget-shaped anywhere.

PN100 already writes the grant onto the request (`auto_budget._apply_budget` /
`_apply_tier`: `request.thinking_token_budget = budget`). Echo it on the response:

```python
# answer_rescue.maybe_rescue_answer, non-streaming branch, before any leg
usage = getattr(result, "usage", None)
det = getattr(usage, "completion_tokens_details", None)
b = getattr(request, "thinking_token_budget", None)
if det is not None and isinstance(b, int) and b > 0:
    det.thinking_token_budget = b          # additive field; OpenAI clients ignore it
```

Flag: `GENESIS_PN155_STAMP_BUDGET` (default **1** — it is pure addition).

The harness already reads `thinking_token_budget` / `thinking_budget` /
`genesis_thinking_budget` / `x_genesis_thinking_budget` at top level, on the choice, in
`usage`, in `usage.completion_tokens_details`, and in `metrics`
(`bench/client.py:OpenAIArm._granted_budget`). Any one of those names works and upgrades
the harness guard from an **inferred** cap to an **exact** one with no client change.

Second, cheap, and worth more than the budget echo: the holder in
`vllm/v1/sample/thinking_budget_state.py` knows whether it *forced* `</think>` for a
request (that is what `_apply_forcing_to_logits` does). Publishing a per-request
`budget_forced: bool` through to the output makes detection **exact** instead of a
threshold on token counts. PN121's graft N and PN122's graft X already write to that same
file, so the seat is proven; take the same boot-ids/holder-state route they use.

### 3b. DETECTION + REFUSAL (the guard proper)

New leg `PN155` in `maybe_rescue_answer`, placed **after** the PN123 close-gate and the
PN102 banner-echo net (both may rewrite `content`) and **before** the
`if not _master_on()` early return, so it is independent of the PN101 master flag —
same pattern the BUG-156 banner-echo net uses.

```python
if (_pn155_on() and not hasattr(result, "__aiter__")
        and not getattr(request, "stream", False)):
    try:
        await _maybe_pn155_empty_at_cap(serving, request, result)
    except Exception as exc:      # fail-open, always
        log.warning("PN155: guard failed (%s) — original kept", exc)
```

Fire condition — **all** of:

| # | Condition | How |
|---|---|---|
| 1 | structured decoding | `_has_structured_output(request)` — already exists, line ~121 |
| 2 | a thinking budget applied | `request.thinking_token_budget > 0` (`_bounded`, line ~116) |
| 3 | budget exhausted | `budget_forced` from 3a **if available**, else `reasoning_tokens >= budget - PN155_MARGIN` (default 16; the client-side re-count runs 1-3 low, the engine's own count does not) |
| 4 | payload empty | `_pn155_is_empty(content)` below |

```python
def _pn155_is_empty(content: str) -> bool:
    """The grammar's cheapest legal completion. Schema-free on purpose: any
    container with zero entries counts, and anything unparseable does NOT
    (that is a different failure, already visible)."""
    try:
        obj = json.loads((content or "").strip())
    except Exception:
        return False
    if isinstance(obj, list):
        return len(obj) == 0
    if isinstance(obj, dict):
        if not obj:
            return True
        vals = list(obj.values())
        return all(isinstance(v, list) and not v for v in vals)
    return False
```

Action, by `GENESIS_PN155_MODE`:

- **`observe`** (default on first boot) — `log.warning` + `_STATS["pn155_fired"] += 1` +
  stamp `usage.completion_tokens_details.budget_empty = True`. No behavior change; gives a
  production firing rate before anything changes for callers.
- **`flag`** (recommended target) — stamp as above **and set
  `choice.finish_reason = "length"`**. This is the whole fix in one line: `length` is the
  truthful reason (the response ended because it ran out of budget, not because the model
  was done), every OpenAI-compatible client already treats it as "incomplete", and no
  caller needs to learn a new field to stop trusting the empty array. Keep the original in
  `choice.stop_reason` / a `genesis_finish_reason_original` stamp so nothing is lost.
- **`retry`** — one bounded re-generate at `min(PN100_BUDGET_CEIL, budget * PN155_RETRY_MULT)`
  (default mult 2), then re-run the check. `_maybe_escalate` (line ~1074) already implements
  exactly this re-generate-with-more-room machinery; the only change it needs is to be
  reachable for structured rows under *this* condition (do **not** widen `_skip_common`
  globally — PN101's other legs skip structured rows for good reasons). If the retry empties
  again, fall through to `flag`. Never a second retry.

Cost: on the measured run this fires on 1/40 rows (2.5%). `flag` is free. `retry` costs
~2.5% extra requests at ~3× their tokens. Both bounded to one attempt.

### 3c. What NOT to do

- **Do not make the grammar reject `[]`.** An empty array is legal in the caller's schema
  and legitimately occurs (prod-038 in three separate runs). Removing it would convert a
  detectable failure into a wrong answer.
- **Do not suppress the forced close instead of guarding.** PN122's graft already keeps the
  forced `</think>` out of the constrained region; that fixes malformed JSON, not this. A
  request that reaches its cap still has to end somewhere, and the grammar still offers the
  empty container as the cheapest exit.
- **Do not treat this as fixed by a better budget map.** See §5.

## 4. Validation (no new benchmark needed)

1. `GENESIS_PN155_MODE=observe`, replay `prod_mixed_v3` on the guided arm. Expect the leg
   to fire on the rows the harness guard flags — today exactly `prod-016` — and **never**
   on `prod-038` (rtok 158, no budget pressure) or on any of the 24 cap-pinned rows that
   did carry facts.
2. Unguided control (`thinkingcap_auto_t10_c6`, same 40 items): 0 fires. The leg must be
   unreachable without structured output.
3. Flip to `flag`; assert the flagged rows come back `finish_reason="length"` and that the
   harness's own guard, which is independent, agrees row-for-row.
4. `~/shared/folderX/qbench45` → `python -m pytest bench/tests -q` (46 tests, no GPU) is the
   client-side contract for the field names in 3a.

## 5. Interaction with `GENESIS_PN100_STEP_BUDGET_MAP`

The map is currently **unset**, so budgets are a flat `steps × 260` rounded to 100 — which
is why 62% of guided rows land exactly on a ceiling. Fitting that map (another agent's
work) will shrink the *population at risk* substantially. It does not remove the failure
mode: any request that reaches its budget under a grammar can still take the empty exit,
and a better-fitted map is precisely one that puts budgets *closer* to what rows need,
i.e. more rows near their ceiling by design.

So the guard is an invariant, not a stopgap. After the map lands, its firing rate becomes
a **quality signal for the map** — a step-value whose rows fire disproportionately is a
step-value the map under-grants. Harness side this needs no change: point
`BENCH_BUDGET_GUARD_STEP_BUDGET_MAP` (or the exported `GENESIS_PN100_STEP_BUDGET_MAP`) at
the fitted grid and the inferred caps follow it; ship 3a and the inference is bypassed
entirely.

## 6. Known limits (state these when reporting the fix)

- Catches the **empty** payload only. A row that exhausts its budget and emits one
  shallow fact instead of nine is degraded, not empty, and no budget signal separates it
  from a genuinely thin chunk. That is a different bug and needs content comparison.
- Threshold-based condition 3 has a false-negative if the model stops 20+ tokens short of
  its cap and then emits an empty container anyway. `budget_forced` from the holder (3a)
  removes this; the token-count fallback does not.
- `observe` mode is not a fix. Ship `flag` — an unflagged empty is the bug.
