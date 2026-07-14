#!/usr/bin/env python3
"""PN98 — drop text-form tool-tag fragments demoted to content (BUG-069 residual).

House-series patch (no upstream PR yet) for the dev1060 pin (9e57de71), 1 file:

  - vllm/parser/engine/streaming_parser_engine.py
      :: _process_lex_tokens (strict-demotion branch)
      :: finish (end-of-output lexer flush)

Bug (BUG-069 residual): non-streaming responses under degenerate
multi-tool-call storms (21+ tool_calls, qwen3_coder parser) leave a literal
'<tool_call>' fragment in message.content (~1-2/15 on an adversarial
repeated-prefix canary). Two leak paths, both in StreamingParserEngine:

  1. TOOL_START/TOOL_END are token_id_terminals for qwen3 (qwen3.py
     token_id_terminals). Non-streaming serving always passes
     model_output_token_ids (serving.py `parser.parse(...,
     model_output_token_ids=token_ids)`), so `_ever_had_token_ids` is True
     and `_process_lex_tokens` runs in strict mode: a `<tool_call>` spelled
     via ordinary BPE text pieces (instead of the dedicated special token)
     lexes as a TOOL_START LexToken and is deliberately demoted to content
     (`strict and tok.terminal in strict` -> `_on_content`). Degenerate
     storms are exactly where the model emits the tag as plain text, so the
     bare tag lands in content: TEXT_CHUNK -> _events_to_delta
     (_deferred_content re-prepended when the final event batch has no tool
     deltas, parser_engine.py ~751) -> parse() content -> message.content.
  2. An unterminated trailing `<tool_call` prefix buffered by
     IncrementalLexer's prefix-match hold flushes as a CONTENT token at
     `finish()` (`self._lexer.flush()` -> _drain(final=True) emits the
     leftover buffer as content) and surfaces the same way.

Fix (conservative, content-preserving — no re-lex):
  1. In the strict-demotion branch: when the demoted terminal is a TOOL
     tag (terminal in `self._tool_terminals`, which intersected with the
     strict token-id set is exactly TOOL_START/TOOL_END for qwen3 — THINK
     tags are untouched) AND tool parsing has occurred this request
     (`self.tool_index >= 0`), DROP the bare tag text instead of emitting
     it as content.
  2. In `finish()`: when tool parsing occurred this request, strip a
     trailing flushed CONTENT token whose entire value is a proper prefix
     (len >= 2) of a tool tag literal (e.g. `<tool_call`, `</tool_ca`). A
     lone `<` is preserved.
  Never touches content when no tool parsing occurred (tool_index < 0):
  plain prose mentioning '<tool_call>' in a normal answer survives, as
  does a leading text-form tag before any real tool call (identical to
  pre-fix behavior). `skip_tool_parsing` reasoning passes keep
  tool_index == -1 and are untouched.

ORDERING: must run AFTER patch_pn76_engine_deferred_toolcall_commit.py in
the compose entrypoint. PN76's finish() anchor (flush line + blank +
`if self._args_buffer:`) is destroyed by PN98's finish() hunk, while PN98's
anchor survives a PN76-applied file (PN76 only inserts AFTER the flush
line). Wiring PN98 after PN96 (which is after PN76 in every shipped chain)
satisfies this. If PN76 is disabled, PN98 applies to the virgin file fine.

Anchors verified byte-exact count==1 against /home/user/engines/vllm-build
(exact source of the installed dev1060 build) 2026-07-14, and against the
PN76-applied variant. No other /fixes patcher touches these regions (PN76
patches reset/_emit_for_state/_on_terminal and inserts after — not into —
the finish() flush line). Retire when the parser engine grows a native
text-form-tag suppression (none upstream as of dev1060).
"""
import pathlib
import sys

LOG = "[pn98-toolcall-text-fragment-demotion]"
VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET = VLLM / "parser/engine/streaming_parser_engine.py"
MARKER = "[Genesis-house PN98"

# ── hunk 1: _process_lex_tokens strict-demotion branch ──
H1_OLD = (
    "    def _process_lex_tokens(self, tokens: list[LexToken]) -> list[SemanticEvent]:\n"
    "        events: list[SemanticEvent] = []\n"
    "        strict = self._token_id_terminal_names if self._ever_had_token_ids else None\n"
    "        for tok in tokens:\n"
    "            if tok.terminal == CONTENT_TERMINAL or (strict and tok.terminal in strict):\n"
    "                events.extend(self._on_content(tok.value))\n"
    "            else:\n"
    "                events.extend(self._on_terminal(tok.terminal, tok.value))\n"
    "        return events\n"
)
H1_NEW = (
    "    def _process_lex_tokens(self, tokens: list[LexToken]) -> list[SemanticEvent]:\n"
    "        events: list[SemanticEvent] = []\n"
    "        strict = self._token_id_terminal_names if self._ever_had_token_ids else None\n"
    "        for tok in tokens:\n"
    "            if tok.terminal == CONTENT_TERMINAL or (strict and tok.terminal in strict):\n"
    "                # [Genesis-house PN98] BUG-069 residual: a tool tag spelled via\n"
    "                # ordinary BPE text pieces (not the dedicated special token) is\n"
    "                # deliberately demoted to content here in token-id mode; in a\n"
    "                # degenerate multi-tool-call storm that leaks a literal\n"
    "                # '<tool_call>' into message.content. Once tool parsing has\n"
    "                # occurred this request (tool_index >= 0), drop the bare tag\n"
    "                # instead of emitting it. Prose mentioning the tag before any\n"
    "                # tool call (tool_index < 0) is preserved unchanged; THINK\n"
    "                # tags are not in _tool_terminals and are untouched.\n"
    "                if (\n"
    "                    strict is not None\n"
    "                    and tok.terminal in strict\n"
    "                    and tok.terminal in self._tool_terminals\n"
    "                    and self.tool_index >= 0\n"
    "                ):\n"
    "                    continue\n"
    "                events.extend(self._on_content(tok.value))\n"
    "            else:\n"
    "                events.extend(self._on_terminal(tok.terminal, tok.value))\n"
    "        return events\n"
)

# ── hunk 2: finish() end-of-output lexer flush ──
# Anchor deliberately spans the def line + flush_pending line so it stays
# unique vs the identical flush line in _process_scanner_items, and still
# matches after PN76 (which inserts AFTER the flush line, not into it).
H2_OLD = (
    "    def finish(self) -> list[SemanticEvent]:\n"
    "        events = self._process_scanner_items(self._scanner.flush_pending())\n"
    "\n"
    "        events.extend(self._process_lex_tokens(self._lexer.flush()))\n"
)
H2_NEW = (
    "    def finish(self) -> list[SemanticEvent]:\n"
    "        events = self._process_scanner_items(self._scanner.flush_pending())\n"
    "\n"
    "        # [Genesis-house PN98] BUG-069 residual: an unterminated trailing\n"
    "        # tool-tag prefix (e.g. '<tool_call') held by the IncrementalLexer's\n"
    "        # prefix-match buffer flushes as a CONTENT token at end-of-output and\n"
    "        # would surface in message.content. When tool parsing occurred this\n"
    "        # request (tool_index >= 0), drop that trailing fragment. A lone '<',\n"
    "        # and all content on requests with no tool parsing, are preserved.\n"
    "        _pn98_flush = self._lexer.flush()\n"
    "        if self.tool_index >= 0 and _pn98_flush:\n"
    "            _pn98_last = _pn98_flush[-1]\n"
    "            if (\n"
    "                _pn98_last.terminal == CONTENT_TERMINAL\n"
    "                and len(_pn98_last.value) >= 2\n"
    "            ):\n"
    "                _pn98_tags = [\n"
    "                    _pn98_text\n"
    "                    for _pn98_name, _pn98_text in (\n"
    "                        self.config.token_id_terminals.items()\n"
    "                    )\n"
    "                    if _pn98_name in self._tool_terminals\n"
    "                ]\n"
    "                if any(\n"
    "                    len(_pn98_last.value) < len(_pn98_tag)\n"
    "                    and _pn98_tag.startswith(_pn98_last.value)\n"
    "                    for _pn98_tag in _pn98_tags\n"
    "                ):\n"
    "                    _pn98_flush = _pn98_flush[:-1]\n"
    "        events.extend(self._process_lex_tokens(_pn98_flush))\n"
)

HUNKS = [
    ("strict-demotion-drop", H1_OLD, H1_NEW),
    ("finish-flush-strip", H2_OLD, H2_NEW),
]


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} SKIP: {TARGET} not present on this vLLM build; no-op.",
              file=sys.stderr)
        return 0
    text = TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} {TARGET.name}: already applied (idempotent)")
        return 0
    if "# PN76:" not in text:
        print(f"{LOG} note: PN76 marker absent — fine if PN76 is disabled; "
              f"if enabled, PN98 must run AFTER it (see docstring ORDERING)")
    for name, old, _ in HUNKS:
        if old not in text:
            print(f"{LOG} SKIP: anchor-not-found ({TARGET.name}/{name}) — "
                  f"engine shape changed (vLLM bumped or patch chain "
                  f"reordered?); re-anchor before relying on this fix "
                  f"(BUG-069 '<tool_call>' fragments in message.content "
                  f"return without it). No-op.", file=sys.stderr)
            return 0
        if text.count(old) != 1:
            print(f"{LOG} SKIP: ambiguous anchor ({TARGET.name}/{name}, "
                  f"{text.count(old)} matches). No-op.", file=sys.stderr)
            return 0
    for _, old, new in HUNKS:
        text = text.replace(old, new, 1)
    try:
        compile(text, str(TARGET), "exec")
    except SyntaxError as exc:
        print(f"{LOG} SKIP: patched result fails to compile ({exc}); "
              f"refusing to write. No-op.", file=sys.stderr)
        return 0
    TARGET.write_text(text)
    print(f"{LOG} {TARGET.name}: applied {len(HUNKS)} hunk(s) — text-form "
          f"tool-tag demotions and trailing '<tool_call' flush fragments are "
          f"now dropped once tool parsing occurred (BUG-069 residual)")
    return 0


sys.exit(main())
