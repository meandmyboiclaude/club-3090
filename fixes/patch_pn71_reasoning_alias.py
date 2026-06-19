"""PN71 reasoning-alias — accept `reasoning: off|low|medium|high|xhigh|max` (and a raw int,
and the OpenAI Responses-API `{"effort": ...}` object) on every /v1/chat/completions request.

Behaviour (v2 — leak-proof):
- OFF (off/none/0)  -> thinking disabled (reasoning_effort -> "none" -> enable_thinking false).
- a tier/int N      -> bound TOTAL generation to N + PN71_ANSWER_GRACE (default 512).
- max / -1          -> uncapped.

WHY NOT thinking_token_budget: Qwopus does NOT stop on </think> (proven by instrumenting
vllm/v1/sample/thinking_budget_state.py — the budget force-injects </think> at N, the model
ignores it and keeps generating reasoning, which — being AFTER the forced </think> — gets
routed to `content` = a thinking LEAK into the chat). So we deliberately DON'T force </think>.
The model self-delimits one clean <think>...</think> (extracted to the `reasoning` field;
content stays the answer only) and we cap the TOTAL via max_tokens so reasoning can never
leak into content. Thinking still happens at full quality; it's just on its own path and
bounded. (vLLM 0.23 renamed the response field reasoning_content -> `reasoning`.)

Two surgical edits to ChatCompletionRequest in
  vllm/entrypoints/openai/chat_completion/protocol.py
- (A) build_chat_params: normalize the OFF intent (off/none/0) onto reasoning_effort="none".
- (B) to_sampling_params: map reasoning/reasoning_effort tiers (or a raw int) onto a
      max_tokens (total) bound — NO thinking_token_budget forcing.

Style: a standalone commit-patch like the other /fixes (run from the compose entrypoint after
apply_all). Idempotent (bails if MARKER present); fail-loud if an anchor is missing.
"""
import logging
from pathlib import Path

log = logging.getLogger("patch_pn71_reasoning_alias")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

MARKER = "# PATCH: pn71_reasoning_alias_v2"
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

# (B) TOTAL-bound map — verbatim anchor at the top of to_sampling_params.
B_OLD = (
    "    ) -> SamplingParams:\n"
    "        # Default parameters\n"
)
B_NEW = (
    "    ) -> SamplingParams:\n"
    "        " + MARKER + " (B) — reasoning/reasoning_effort -> TOTAL output bound.\n"
    "        # NO </think> forcing (Qwopus ignores </think> and would leak continued\n"
    "        # reasoning into content). Cap max_tokens so thinking (in its own <think>\n"
    "        # ...</think> -> `reasoning` field) can never spill into the chat. OFF is\n"
    "        # handled in build_chat_params; max/-1 -> uncapped; explicit smaller\n"
    "        # max_tokens is left alone.\n"
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
    "            _pn71_cap = _pn71_b + int(_pn71_os.environ.get(\"PN71_ANSWER_GRACE\", \"512\"))\n"
    "            if max_tokens is None or max_tokens > _pn71_cap:\n"
    "                max_tokens = _pn71_cap\n"
    "        # Default parameters\n"
)


def apply():
    if not TARGET.exists():
        log.warning("[pn71] protocol.py not found at %s", TARGET)
        return
    text = TARGET.read_text()
    if MARKER in text:
        log.info("[pn71] already applied (v2)")
        return

    ok = True
    if A_OLD in text:
        text = text.replace(A_OLD, A_NEW, 1)
        log.info("[pn71] (A) build_chat_params OFF-normalize wired")
    else:
        ok = False
        log.warning("[pn71] (A) anchor NOT found in build_chat_params — re-anchor needed")
    if B_OLD in text:
        text = text.replace(B_OLD, B_NEW, 1)
        log.info("[pn71] (B) to_sampling_params total-bound wired (no </think> forcing)")
    else:
        ok = False
        log.warning("[pn71] (B) anchor NOT found in to_sampling_params — re-anchor needed")

    if not ok:
        log.warning("[pn71] aborting write — at least one anchor missed (no partial patch)")
        return
    TARGET.write_text(text)
    log.info("[pn71] v2 applied (leak-proof total-bound)")


apply()
