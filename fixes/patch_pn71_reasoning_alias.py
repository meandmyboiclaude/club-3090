#!/usr/bin/env python3
"""PN71 reasoning-alias — accept `reasoning: off|low|medium|high|xhigh|max` (and a raw int,
and the OpenAI Responses-API `{"effort": ...}` object) on every /v1/chat/completions request.

Behaviour (v3 — bounded thinking AND leak-proof):
- OFF (off/none/0)  -> thinking disabled (reasoning_effort -> "none" -> enable_thinking false).
- a tier/int N      -> `thinking_token_budget = N` (engine forces `</think>` at N) AND a
                       TOTAL bound of N + PN71_ANSWER_GRACE, with the budget clamped so a
                       visible post-`</think>` answer always fits inside the effective
                       max_tokens.
- max / -1          -> uncapped; no budget, no cap.

WHY v3 REVERSES v2's TRADE
--------------------------
v1 (`5192458d`) bounded the THINKING via `thinking_token_budget` and clamped that budget
below max_tokens so a visible answer always fit. 41 minutes later v2 (`4f7e524b`) dropped
the budget entirely — not because the clamp failed, but because of a worse defect measured
on the model of the day: **Qwopus did not stop on the injected `</think>` and kept emitting
reasoning**, which — being after the forced end — the reasoning parser routes to `content`,
i.e. a thinking LEAK into the chat. v2 therefore stopped forcing at all and capped only
TOTAL output, accepting (in its own commit message) that on open-ended prompts the model
over-thinks, fills the budget, gets cut, and returns EMPTY content "rather than polluted".

That trade rested on one empirical claim about one model. **We no longer serve Qwopus.**
Production is ThinkingCap GPTQ-Pro v2 (promoted 2026-07-18), and it HONOURS the forced tag.
Evidence (banked outputs, no GPU, `qbench45/results`, harness semantics in
`bench/client.py:120-160` — `reasoning` = the server parser's `reasoning_content`, so
`answer_tokens = completion_tokens - reasoning_tokens` is exactly the post-`</think>` text):

  * 604 budget-hit rows across every ThinkingCap arm (reasoning_tokens == budget, exact via
    the /tokenize endpoint => the engine forced the close): 604/604 produced non-empty
    post-`</think>` content. Zero rows where the model closed and then emitted nothing.
  * On the promoted quant specifically (`tcgptqpro-v2-confirm-20260718-0045`, 49 forced
    closes) 100% of the rows that ended `finish_reason=stop` parsed under the STRICT grader
    (`ANSWER:\\s*\\(?[A-D]\\)?`) — the model writes a real final answer after the forced tag,
    it does not continue reasoning. Every unparsed row was `finish_reason=length`, i.e. the
    harness envelope ran out, never a leak-and-never-answer.
  * Post-forced-close answers are only ~1.5x longer than post-NATURAL-close answers
    (563 vs 374 median tokens) — a slightly more verbose wrap-up, not a continued trace.
  * Same suite, same harness, Qwopus control: 42% of its forced closes blew the envelope
    vs ThinkingCap's 21%, and its post-close text was half again as long. The June claim was
    real; it was a property of Qwopus, and it does not transfer.

So the leak that motivated v2 does not exist on the served model, while the empty-content
class v2 accepted is real. v3 restores the v1 budget + the `5192458d` clamp.

PN71_ANSWER_GRACE DEFAULT RAISED 512 -> 1024
--------------------------------------------
v2 halved the grace because under v2 it was pure slack on top of a total cap. Under a
restored clamp the grace is the ANSWER's entire allowance, and 512 is far too small for
this model: measured post-forced-close answer lengths on ThinkingCap GPTQ-Pro v2 are
p25=518 / p50=648 / p90=882 (n=93, finish=stop) — **512 covers 23.7% of them**. On real
prod traffic (`prod_mixed_v2`, n=725) answers are longer still: p50=775, p90=1608. 1024 is
the smallest round default that clears the GPQA median with headroom; operators serving
long-form traffic should raise it (`PN71_ANSWER_GRACE`). Note the measurement is
right-censored — the harness ran headroom=1024, so the tail beyond 1024 is not observed.

RELATIONSHIP TO PN100 (auto thinking-budget router)
---------------------------------------------------
Disjoint by construction and verified in both directions: PN100 skips any request carrying
`reasoning`/`reasoning_effort` (`_genesis/middleware/auto_budget.py:19`), and this block only
engages when such a value resolves to a positive tier. PN100's budgets are therefore never
read or written here. The clamp additionally never RAISES a budget — the only budget it can
lower is one already in a state that cannot emit visible content (`budget >= max_tokens`).

Two surgical edits to ChatCompletionRequest in
  vllm/entrypoints/openai/chat_completion/protocol.py
- (A) build_chat_params: normalize the OFF intent (off/none/0) onto reasoning_effort="none".
- (B) to_sampling_params: map reasoning/reasoning_effort tiers (or a raw int) onto
      thinking_token_budget + a clamped total bound. (D) rides inside (B): the
      Qwen3.6-recommended sampling defaults, applied only when the caller omitted them.

Anchors are CONTENT-SNIFFED and COUNTED (never rewritten in place); a drift or an ambiguous
count fails LOUD — a boxed stderr ERROR plus logging.error — because a PN71-family patch that
announces APPLY while silently skipping is exactly BUG-122. Verify without a boot:
    python3 fixes/verify_pn71_anchors.py
"""
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("patch_pn71_reasoning_alias")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

LOG = "[pn71]"
MARKER = "# PATCH: pn71_reasoning_alias_v3"
# v2's marker — a container carrying it is a stale apply, not an idempotent one.
MARKER_V2 = "# PATCH: pn71_reasoning_alias_v2"
TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/protocol.py"
)

# (A) OFF normalize — verbatim anchor at the top of build_chat_params.
A_OLD = (
    "    ) -> ChatParams:\n"
    "        extra_kwargs: dict[str, Any] = dict(\n"
)
A_NEW = (
    "    ) -> ChatParams:\n"
    "        " + MARKER + " (A) — map our `reasoning` OFF intent onto the native\n"
    "        # reasoning_effort->enable_thinking path below (off/none/0 -> \"none\").\n"
    "        _pn71_r = getattr(self, \"reasoning\", None)\n"
    "        if isinstance(_pn71_r, dict):\n"
    "            _pn71_r = _pn71_r.get(\"effort\")\n"
    "        if self.reasoning_effort is None and (\n"
    "            (isinstance(_pn71_r, str) and _pn71_r.strip().lower() in (\"off\", \"none\"))\n"
    "            or (isinstance(_pn71_r, int) and not isinstance(_pn71_r, bool) and _pn71_r == 0)\n"
    "        ):\n"
    "            self.reasoning_effort = \"none\"\n"
    "        extra_kwargs: dict[str, Any] = dict(\n"
)

# (B) budget + clamped total bound — verbatim anchor at the top of to_sampling_params.
B_OLD = (
    "    ) -> SamplingParams:\n"
    "        # Default parameters\n"
)
B_NEW = (
    "    ) -> SamplingParams:\n"
    "        " + MARKER + " (B) — reasoning/reasoning_effort -> thinking budget +\n"
    "        # clamped TOTAL bound. The engine forces </think> at the budget (holder:\n"
    "        # v1/sample/thinking_budget_state.py) and ThinkingCap GPTQ-Pro v2 honours\n"
    "        # it — 604/604 banked budget-hit rows produced a real post-</think> answer\n"
    "        # (see this patch's module docstring for the evidence). The clamp restores\n"
    "        # 5192458d: when the effective max_tokens cannot hold budget + an answer,\n"
    "        # shrink the BUDGET, never the answer, so content is never empty. OFF is\n"
    "        # handled in build_chat_params; max/-1 -> uncapped; a caller's smaller\n"
    "        # max_tokens is always respected.\n"
    "        _pn71_tiers = {\"low\": 1536, \"medium\": 2048, \"high\": 4096, \"xhigh\": 8192, \"max\": -1}\n"
    "\n"
    "        def _pn71_budget(v):\n"
    "            if isinstance(v, bool):\n"
    "                return None\n"
    "            if isinstance(v, int):\n"
    "                return v if v != 0 else None\n"
    "            if isinstance(v, str):\n"
    "                s = v.strip().lower()\n"
    "                if s in _pn71_tiers:\n"
    "                    return _pn71_tiers[s]\n"
    "                if s.lstrip(\"-\").isdigit():\n"
    "                    n = int(s)\n"
    "                    return n if n != 0 else None\n"
    "                return None\n"
    "            if isinstance(v, dict):\n"
    "                return _pn71_budget(v.get(\"effort\"))\n"
    "            return None\n"
    "\n"
    "        _pn71_b = _pn71_budget(getattr(self, \"reasoning\", None))\n"
    "        if _pn71_b is None:\n"
    "            _pn71_b = _pn71_budget(self.reasoning_effort)\n"
    "        if isinstance(_pn71_b, int) and _pn71_b > 0:\n"
    "            import os as _pn71_os\n"
    "            _pn71_grace = int(_pn71_os.environ.get(\"PN71_ANSWER_GRACE\", \"1024\"))\n"
    "            if _pn71_grace < 1:\n"
    "                _pn71_grace = 1\n"
    "            _pn71_cap = _pn71_b + _pn71_grace\n"
    "            if max_tokens is None or max_tokens > _pn71_cap:\n"
    "                max_tokens = _pn71_cap\n"
    "            # Bound the THINKING itself. A budget the caller (or any earlier hook)\n"
    "            # already set wins — we never overwrite an explicit intent.\n"
    "            if getattr(self, \"thinking_token_budget\", None) is None:\n"
    "                try:\n"
    "                    self.thinking_token_budget = _pn71_b\n"
    "                except Exception:\n"
    "                    pass\n"
    "            # 5192458d clamp: a budget that meets or exceeds the effective\n"
    "            # max_tokens guarantees the cut lands INSIDE <think> -> empty content.\n"
    "            # Only ever LOWERS, and only from a state that cannot emit an answer.\n"
    "            _pn71_tb = getattr(self, \"thinking_token_budget\", None)\n"
    "            if (\n"
    "                isinstance(_pn71_tb, int)\n"
    "                and not isinstance(_pn71_tb, bool)\n"
    "                and _pn71_tb > 0\n"
    "                and isinstance(max_tokens, int)\n"
    "                and _pn71_tb >= max_tokens\n"
    "            ):\n"
    "                try:\n"
    "                    self.thinking_token_budget = max(1, max_tokens - _pn71_grace)\n"
    "                except Exception:\n"
    "                    pass\n"
    "        " + MARKER + " (D) — Qwen3.6-recommended sampling defaults, applied ONLY when the\n"
    "        # caller omitted them (request fields are None if omitted). vLLM's unconstrained\n"
    "        # default (temp=1.0, no top_p/top_k) makes the model ramble (~21k reasoning / ~90s).\n"
    "        # off -> instruct profile (.7 / top_p .8 / top_k 20 / presence 1.5); thinking ->\n"
    "        # (.7 / top_p .95 / top_k 20). Explicit caller values are NEVER overridden.\n"
    "        # Tunable via PN71_DEF_TEMPERATURE / PN71_DEF_TOP_P[_OFF] / PN71_DEF_TOP_K /\n"
    "        # PN71_OFF_PRESENCE_PENALTY env.\n"
    "        import os as _pn71d_os\n"
    "        _pn71_rr = getattr(self, \"reasoning\", None)\n"
    "        if isinstance(_pn71_rr, dict):\n"
    "            _pn71_rr = _pn71_rr.get(\"effort\")\n"
    "        _pn71_off = (\n"
    "            (isinstance(_pn71_rr, str) and _pn71_rr.strip().lower() in (\"off\", \"none\"))\n"
    "            or (isinstance(_pn71_rr, int) and not isinstance(_pn71_rr, bool) and _pn71_rr == 0)\n"
    "            or (isinstance(self.reasoning_effort, str) and self.reasoning_effort.strip().lower() == \"none\")\n"
    "        )\n"
    "        if self.temperature is None:\n"
    "            self.temperature = float(_pn71d_os.environ.get(\"PN71_DEF_TEMPERATURE\", \"0.7\"))\n"
    "        if self.top_k is None:\n"
    "            self.top_k = int(_pn71d_os.environ.get(\"PN71_DEF_TOP_K\", \"20\"))\n"
    "        if self.top_p is None:\n"
    "            self.top_p = float(_pn71d_os.environ.get(\"PN71_DEF_TOP_P_OFF\", \"0.8\") if _pn71_off else _pn71d_os.environ.get(\"PN71_DEF_TOP_P\", \"0.95\"))\n"
    "        if _pn71_off and self.presence_penalty in (None, 0.0):\n"
    "            self.presence_penalty = float(_pn71d_os.environ.get(\"PN71_OFF_PRESENCE_PENALTY\", \"1.5\"))\n"
    "        # Default parameters\n"
)

# Content sniff: (B)'s budget arm is inert without the native request field. The
# field has been native since the dev1060 pin (protocol.py `thinking_token_budget:
# ThinkingTokenBudget = None`, forwarded into SamplingParams) — PN109's bridge was
# retired 2026-07-20 for exactly that reason. If a future pin drops it the budget
# silently stops being enforced and we are back to v2's behaviour without saying so.
NATIVE_BUDGET_FIELD = "thinking_token_budget"

HUNKS = (("A", A_OLD, A_NEW), ("B", B_OLD, B_NEW))


def counts(text: str) -> dict:
    """Anchor occurrence counts as the BOOT will see them. 1 is the only good count."""
    return {name: text.count(old) for name, old, _new in HUNKS}


def resolve(text: str):
    """Return (hunks, problems). A problem is any count != 1."""
    problems = []
    for name, old, _new in HUNKS:
        n = text.count(old)
        if n == 0:
            problems.append(f"({name}) anchor NOT FOUND — re-anchor needed")
        elif n > 1:
            problems.append(f"({name}) anchor is AMBIGUOUS ({n} occurrences) — re-anchor needed")
    return (HUNKS if not problems else ()), problems


def _shout(detail: str) -> None:
    """Fail LOUD. A PN71-family patch that announces APPLY while skipping is BUG-122."""
    bar = "=" * 78
    msg = (f"{bar}\n"
           f"{LOG} ERROR: PN71 v3 was REQUESTED but is NOT INSTALLED.\n"
           f"{LOG} ERROR: {detail}\n"
           f"{LOG} ERROR: `reasoning:` stays an unrecognized field — no thinking budget,\n"
           f"{LOG} ERROR: no total bound, no sampling defaults. Requests will look served.\n"
           f"{LOG} ERROR: re-derive the anchors against POST-PATCH content:\n"
           f"{LOG} ERROR:   python3 fixes/verify_pn71_anchors.py\n"
           f"{bar}")
    print(msg, file=sys.stderr, flush=True)
    try:
        logging.getLogger("genesis.pn71").error(msg.replace("\n", " | "))
    except Exception:  # noqa: BLE001 — logging must never break a boot
        pass


def apply() -> int:
    if not TARGET.exists():
        _shout(f"{TARGET} not present on this pin")
        return 1
    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        log.info("%s already applied (v3)", LOG)
        return 0
    if MARKER_V2 in text:
        _shout("this container already carries PN71 v2 — a stale apply, not an "
               "idempotent one. v2 has NO thinking budget; the empty-content class "
               "is live. Rebuild the container rather than layering v3 on top.")
        return 1

    _hunks, problems = resolve(text)
    if problems:
        _shout("; ".join(problems))
        return 1

    if NATIVE_BUDGET_FIELD not in text:
        _shout("protocol.py has no native `thinking_token_budget` field on this pin — "
               "(B)'s budget would be dropped by pydantic and thinking would stay "
               "unbounded. Re-wire /fixes/_archive/patch_pn109_budget_bridge.py first.")
        return 1

    for name, old, new in _hunks:
        text = text.replace(old, new, 1)
        log.info("%s (%s) wired", LOG, name)

    try:
        compile(text, str(TARGET), "exec")
    except SyntaxError as e:
        _shout(f"patched protocol.py does not byte-compile ({e}) — refusing to write")
        return 1

    TARGET.write_text(text, encoding="utf-8")
    log.info("%s v3 applied — thinking budget restored, clamped, grace default %s",
             LOG, os.environ.get("PN71_ANSWER_GRACE", "1024"))
    return 0


if __name__ == "__main__":
    sys.exit(apply())
