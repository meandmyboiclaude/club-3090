"""PN76 — engine-level deferred tool-call commit (retires PN73/PN73T).

Regression in the vLLM streaming **parser engine** (`vllm/parser/engine/`), the
code path we'd use once the vendored legacy parsers (PN73/PN73T) are retired.
The qwen3 grammar opens a tool call EAGERLY: the `(CONTENT|REASONING, TOOL_START)`
transition fires `TOOL_CALL_START` the instant the literal `<tool_call>` is lexed
(`qwen3.py` → `streaming_parser_engine._apply_transition`). When the model's prose
merely *mentions* `<tool_call>` (no real `<function=` follows), the engine:
  * has already streamed a bogus `TOOL_CALL_START` to the client,
  * drops every subsequent char (TOOL_PREAMBLE has no `content_events` mapping →
    `_emit_for_state` returns `[]`), and
  * emits a spurious empty `TOOL_CALL_END` at `finish()`.
Result: content truncated at `<tool_call>` + a phantom tool call on the wire.

Upstream #46091 (already in dev424) only recovers the *closed* `<tool_call>…</tool_call>`
empty-block case (it added the `(TOOL_PREAMBLE, TOOL_END)` transition); it does NOT
cover prose with a bare `<tool_call>` and no closing tag / no `<function=`. No upstream
PR fixes this case (verified 2026-06-26).

FIX — port the proven legacy deferred-commit (club-3090 #72 / P61c) INTO the engine:
on `<tool_call>` in CONTENT/REASONING, do NOT open the tool call immediately. Instead
HOLD: buffer the `<tool_call>` literal + following text and wait. If `<function=`
(FUNC_PREFIX) arrives within a slack window, COMMIT — replay the deferred TOOL_START
transition (REASONING_END + TOOL_CALL_START) and process the function normally, so real
tool calls are byte-identical to today. Otherwise (slack exceeded, a different terminal,
or finish()) ABORT — re-emit the buffered text as ordinary content in the origin state
(TEXT_CHUNK / REASONING_CHUNK) and continue. `skip_tool_parsing` passes are untouched
(those are PN72's domain).

Routing the hold accumulation through `_emit_for_state` (every content path funnels
there) means feed()/_on_content need no changes.

Target: vllm/parser/engine/streaming_parser_engine.py (StreamingParserEngine).
Style: standalone idempotent commit-patch like the other /fixes; runs in the compose
entrypoint after apply_all. Never exits non-zero (set -e safe); graceful no-op if any
anchor is gone (vLLM bumped → re-anchor before relying on it).

Intended as an upstream vLLM PR (with PN72) once validated.
"""
import sys
import pathlib

LOG = "[pn76-engine-defer]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/parser/engine/streaming_parser_engine.py"
)
MARKER = "# PN76:"

# --- 1. reset(): initialise the hold state -------------------------------------
RESET_OLD = (
    "        self._scanner.reset()\n"
    "        self._lexer.reset()\n"
    "        self._reset_args_state()\n"
)
RESET_NEW = (
    "        self._scanner.reset()\n"
    "        self._lexer.reset()\n"
    "        self._reset_args_state()\n"
    "        # PN76: deferred tool-call commit hold state.\n"
    "        self._tc_hold = False\n"
    "        self._tc_hold_text = \"\"\n"
    "        self._tc_hold_tag_len = 0\n"
    "        self._tc_hold_origin_state = None\n"
    "        self._tc_hold_from_reasoning = False\n"
)

# --- 2. _emit_for_state(): accumulate while holding ----------------------------
EMIT_OLD = (
    "    def _emit_for_state(self, text: str) -> list[SemanticEvent]:\n"
    "        if self.state == ParserState.TOOL_ARGS:\n"
)
EMIT_NEW = (
    "    def _emit_for_state(self, text: str) -> list[SemanticEvent]:\n"
    "        # PN76: while a <tool_call> commit is deferred, buffer content instead\n"
    "        # of emitting it; flush as content if no <function= arrives within slack.\n"
    "        if self._tc_hold:\n"
    "            self._tc_hold_text += text\n"
    "            if len(self._tc_hold_text) - self._tc_hold_tag_len > self._TC_HOLD_SLACK:\n"
    "                return self._tc_flush_hold()\n"
    "            return []\n"
    "        if self.state == ParserState.TOOL_ARGS:\n"
)

# --- 3. _on_terminal(): wrap with deferred-commit logic ------------------------
TERM_OLD = (
    "    def _on_terminal(self, terminal: str, value: str) -> list[SemanticEvent]:\n"
    "        key = (self.state, terminal)\n"
)
TERM_NEW = (
    "    # PN76: defer opening a tool call until <function= confirms it is real.\n"
    "    _TC_HOLD_SLACK = 64\n"
    "\n"
    "    def _tc_can_confirm(self) -> bool:\n"
    "        \"\"\"PN76: may this config's tool calls EVER be confirmed?\n"
    "\n"
    "        The hold is released by a FUNC_PREFIX terminal, and FUNC_PREFIX is\n"
    "        declared by qwen3 alone. Eight other shipped parsers (deepseek_v4,\n"
    "        deepseek_v32, gemma4, glm47_moe, inkling, kimi_k2, minimax_m2,\n"
    "        seed_oss) declare TOOL_START without it, so an unconditional hold\n"
    "        could never be confirmed and aborted EVERY real tool call into\n"
    "        content -- upstream's own replay suite catches it as 'Tool call\n"
    "        count mismatch: expected 1, got 0'. Deferring only where a\n"
    "        confirmation terminal exists keeps qwen3 byte-identical and leaves\n"
    "        every other parser on the stock eager path.\n"
    "        \"\"\"\n"
    "        cached = getattr(self, \"_tc_confirmable\", None)\n"
    "        if cached is None:\n"
    "            cached = any(\n"
    "                t == \"FUNC_PREFIX\" for (_s, t) in self.config.transitions\n"
    "            )\n"
    "            self._tc_confirmable = cached\n"
    "        return cached\n"
    "\n"
    "    def _tc_flush_hold(self) -> list[SemanticEvent]:\n"
    "        \"\"\"PN76: a held <tool_call> turned out to be prose — re-emit it as\n"
    "        ordinary content in the state we were in when the hold began.\"\"\"\n"
    "        text = self._tc_hold_text\n"
    "        origin = self._tc_hold_origin_state\n"
    "        self._tc_hold = False\n"
    "        self._tc_hold_text = \"\"\n"
    "        self._tc_hold_tag_len = 0\n"
    "        self._tc_hold_origin_state = None\n"
    "        self._tc_hold_from_reasoning = False\n"
    "        if not text:\n"
    "            return []\n"
    "        content_type = self.config.content_events.get(origin, EventType.TEXT_CHUNK)\n"
    "        return [SemanticEvent(content_type, value=text, tool_index=self.tool_index)]\n"
    "\n"
    "    def _on_terminal(self, terminal: str, value: str) -> list[SemanticEvent]:\n"
    "        # PN76: deferred tool-call commit.\n"
    "        if self._tc_hold:\n"
    "            if terminal == \"FUNC_PREFIX\":\n"
    "                # Confirmed real tool call: replay the deferred <tool_call>\n"
    "                # transition, then process this <function= terminal normally.\n"
    "                origin = self._tc_hold_origin_state\n"
    "                # Replay with the held <tool_call> literal (NOT this <function=\n"
    "                # terminal) so TOOL_CALL_START/REASONING_END carry the same value\n"
    "                # the eager path would have emitted — byte-identical real calls.\n"
    "                tag = self._tc_hold_text[: self._tc_hold_tag_len] or value\n"
    "                self._tc_hold = False\n"
    "                self._tc_hold_text = \"\"\n"
    "                self._tc_hold_tag_len = 0\n"
    "                self._tc_hold_origin_state = None\n"
    "                self._tc_hold_from_reasoning = False\n"
    "                started = self._apply_transition(\n"
    "                    self.config.transitions[(origin, \"TOOL_START\")], tag\n"
    "                )\n"
    "                return started + self._on_terminal_inner(terminal, value)\n"
    "            if terminal == \"TOOL_END\":\n"
    "                # A CLOSED block: `<tool_call>...</tool_call>` with no\n"
    "                # `<function=`. Upstream #46091 already handles this via its\n"
    "                # (TOOL_PREAMBLE, TOOL_END) transition, which absorbs the\n"
    "                # block -- TOOL_PREAMBLE has no content_events, so the tags\n"
    "                # AND any interior text are dropped. PN76 exists for the\n"
    "                # UNCLOSED case and must not change this one; flushing the\n"
    "                # hold here instead re-emitted a literal `<tool_call>` into\n"
    "                # content, which upstream's replay suite catches as\n"
    "                # \"expected 'Content after empty tools.'\". Drop the held\n"
    "                # text and the close tag together, staying in the origin\n"
    "                # state -- net-identical to the transition we bypassed.\n"
    "                self._tc_hold = False\n"
    "                self._tc_hold_text = \"\"\n"
    "                self._tc_hold_tag_len = 0\n"
    "                self._tc_hold_origin_state = None\n"
    "                self._tc_hold_from_reasoning = False\n"
    "                return []\n"
    "            # Some other terminal: not a tool call — recover the held text as\n"
    "            # content, then reprocess this terminal in the restored state.\n"
    "            flushed = self._tc_flush_hold()\n"
    "            return flushed + self._on_terminal_inner(terminal, value)\n"
    "        # Not holding: begin a hold when <tool_call> would eagerly open a tool\n"
    "        # call (skip the engine's skip_tool_parsing pass — that is PN72's domain).\n"
    "        # Only defer when THIS config can actually confirm — see _tc_can_confirm.\n"
    "        if (\n"
    "            terminal == \"TOOL_START\"\n"
    "            and not self.skip_tool_parsing\n"
    "            and self._tc_can_confirm()\n"
    "        ):\n"
    "            tr = self.config.transitions.get((self.state, terminal))\n"
    "            if tr is not None and tr.next_state == ParserState.TOOL_PREAMBLE:\n"
    "                self._tc_hold = True\n"
    "                self._tc_hold_text = value\n"
    "                self._tc_hold_tag_len = len(value)\n"
    "                self._tc_hold_origin_state = self.state\n"
    "                self._tc_hold_from_reasoning = (\n"
    "                    self.state == ParserState.REASONING\n"
    "                )\n"
    "                return []\n"
    "        return self._on_terminal_inner(terminal, value)\n"
    "\n"
    "    def _on_terminal_inner(self, terminal: str, value: str) -> list[SemanticEvent]:\n"
    "        key = (self.state, terminal)\n"
)

# --- 4. finish(): flush a still-open hold before the tool-state cleanup --------
FINISH_OLD = (
    "        events.extend(self._process_lex_tokens(self._lexer.flush()))\n"
    "\n"
    "        if self._args_buffer:\n"
)
FINISH_NEW = (
    "        events.extend(self._process_lex_tokens(self._lexer.flush()))\n"
    "\n"
    "        # PN76: a <tool_call> held to end-of-stream with no <function= was prose.\n"
    "        if self._tc_hold:\n"
    "            events.extend(self._tc_flush_hold())\n"
    "\n"
    "        if self._args_buffer:\n"
)

REPLACEMENTS = [
    ("reset hold-init", RESET_OLD, RESET_NEW),
    ("_emit_for_state hold", EMIT_OLD, EMIT_NEW),
    ("_on_terminal wrapper", TERM_OLD, TERM_NEW),
    ("finish hold-flush", FINISH_OLD, FINISH_NEW),
]


def main():
    if not TARGET.exists():
        print(f"{LOG} SKIP: {TARGET} not present on this vLLM build; no-op.", file=sys.stderr)
        return
    text = TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} already applied")
        return
    for name, old, _new in REPLACEMENTS:
        if old not in text:
            print(
                f"{LOG} SKIP: anchor '{name}' not found in {TARGET.name} — engine shape "
                f"changed (vLLM bumped?); re-anchor before relying on this fix. No-op.",
                file=sys.stderr,
            )
            return
    for _name, old, new in REPLACEMENTS:
        text = text.replace(old, new, 1)
    TARGET.write_text(text)
    print(f"{LOG} applied: <tool_call> commit now deferred until <function= confirms")


main()
