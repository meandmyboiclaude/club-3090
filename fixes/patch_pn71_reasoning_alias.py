"""PN71 reasoning-alias — accept `reasoning: off|low|medium|high|xhigh|max` (and a raw int,
and the OpenAI Responses-API `{"effort": ...}` object) on every /v1/chat/completions request,
mapping it onto vLLM's thinking_token_budget (length cap) + reasoning_effort->enable_thinking (off).

Why: on this stack the engine already ENFORCES thinking_token_budget with MTP on, but there is no
ergonomic per-request `reasoning:` knob. This adds one, server-side, for ALL callers (Go/curl/MCP/…).

Two surgical edits to ChatCompletionRequest in
  vllm/entrypoints/openai/chat_completion/protocol.py
- (A) build_chat_params: normalize our custom `reasoning` field's OFF intent (off/none/0) onto the
      native reasoning_effort -> enable_thinking path. `reasoning_effort:"none"` already works natively
      on this base; this just lets the `reasoning` alias reach it too.
- (B) to_sampling_params: map `reasoning`/`reasoning_effort` budget tiers (or a raw int) onto the
      existing thinking_token_budget the method already passes to SamplingParams. Explicit
      thinking_token_budget always wins; OFF/none/0 are handled in (A), not here; unknown = no-op.

Tiers: off/none/0 -> thinking OFF (enable_thinking:false) · low=512 · medium=2048 · high=4096 ·
       xhigh=8192 · max=-1 (uncapped). Empirically: budgets <~256 are a guillotine danger zone; medium
       (2048) is the reliable bounded tier; off is best for trivial/picker calls.

Style: a standalone commit-patch like the other /fixes (run from the compose entrypoint after apply_all).
Idempotent (bails if MARKER present); fail-loud if an anchor is missing (re-anchor needed after a vLLM bump).
"""
import logging
from pathlib import Path

log = logging.getLogger("patch_pn71_reasoning_alias")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

MARKER = "# PATCH: pn71_reasoning_alias_v1"
TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/protocol.py"
)

# (A) OFF normalize — verbatim anchor at the top of build_chat_params (vllm b4c80ec0).
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

# (B) BUDGET map — verbatim anchor at the top of to_sampling_params (vllm b4c80ec0).
B_OLD = (
    "    ) -> SamplingParams:\n"
    "        # Default parameters\n"
)
B_NEW = (
    "    ) -> SamplingParams:\n"
    "        " + MARKER + " (B) — map reasoning budget tiers/int -> thinking_token_budget.\n"
    "        # OFF/none/0 handled in build_chat_params; explicit thinking_token_budget wins.\n"
    "        if self.thinking_token_budget is None:\n"
    "            _pn71_tiers = {\"low\": 1536, \"medium\": 2048, \"high\": 4096, \"xhigh\": 8192, \"max\": -1}\n"
    "\n"
    "            def _pn71_budget(v):\n"
    "                if isinstance(v, bool):\n"
    "                    return None\n"
    "                if isinstance(v, int):\n"
    "                    return v if v != 0 else None\n"
    "                if isinstance(v, str):\n"
    "                    s = v.strip().lower()\n"
    "                    if s in _pn71_tiers:\n"
    "                        return _pn71_tiers[s]\n"
    "                    if s.lstrip(\"-\").isdigit():\n"
    "                        n = int(s)\n"
    "                        return n if n != 0 else None\n"
    "                    return None\n"
    "                if isinstance(v, dict):\n"
    "                    return _pn71_budget(v.get(\"effort\"))\n"
    "                return None\n"
    "\n"
    "            _pn71_b = _pn71_budget(getattr(self, \"reasoning\", None))\n"
    "            if _pn71_b is None:\n"
    "                _pn71_b = _pn71_budget(self.reasoning_effort)\n"
    "            if _pn71_b is not None:\n"
    "                self.thinking_token_budget = _pn71_b\n"
    "        # Default parameters\n"
)


def apply():
    if not TARGET.exists():
        log.warning("[pn71] protocol.py not found at %s", TARGET)
        return
    text = TARGET.read_text()
    if MARKER in text:
        log.info("[pn71] already applied (v1)")
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
        log.info("[pn71] (B) to_sampling_params budget-map wired")
    else:
        ok = False
        log.warning("[pn71] (B) anchor NOT found in to_sampling_params — re-anchor needed")

    if not ok:
        log.warning("[pn71] aborting write — at least one anchor missed (no partial patch)")
        return
    TARGET.write_text(text)
    log.info("[pn71] v1 applied")


apply()
