#!/usr/bin/env python3
"""PN135 — streaming `</think>` must terminate reasoning, not become content.

House-series patch (upstream #39697 class; no upstream PR on this pin), 1 file:

  - vllm/parser/engine/streaming_parser_engine.py :: __init__
      (`self._token_id_terminal_names` construction — the ONLY producer of
       the strict set; the only consumer is `_process_lex_tokens`)

────────────────────────────────────────────────────────────────────────────
BUG (AUDIT-leak-paths-20260726.md §L2, class L2)

`_process_lex_tokens` runs in "strict token-id mode" whenever the request
ever supplied delta_token_ids — which streaming serving ALWAYS does
(`chat_completion/serving.py`). In that mode any terminal whose name is in
`_token_id_terminal_names` (qwen3: THINK_START / THINK_END / TOOL_START /
TOOL_END) is demoted to a plain content token when it was matched by the
TEXT lexer rather than by the token-id scanner:

    strict = self._token_id_terminal_names if self._ever_had_token_ids else None
    for tok in tokens:
        if tok.terminal == CONTENT_TERMINAL or (strict and tok.terminal in strict):
            events.extend(self._on_content(tok.value))      # <-- demotion
        else:
            events.extend(self._on_terminal(tok.terminal, tok.value))

The demotion is right for TOOL_START/TOOL_END/THINK_START: an unbacked
`<tool_call>` spelled in prose must not fabricate a tool call. It is WRONG
for the reasoning terminator, and it fails two different ways:

  1. state == REASONING — the text-spelled `</think>` never reaches
     `_on_terminal`, so the REASONING -> CONTENT transition never fires.
     The engine stays in REASONING for the rest of the response: the marker
     AND the entire answer are emitted as `reasoning_content`, and the
     client's `content` stays empty.
  2. state == CONTENT — the `(CONTENT, THINK_END)` "absorb duplicate"
     transition never fires either, so the literal string `</think>` is
     emitted verbatim into the client's `content`.

Both reproduce on this exact image; see fixes/test_pn135_streaming_think_end.py
(T1/T2 red pre-patch, green post-patch).

Reachability: a text-spelled `</think>` arises when the model emits the tag
as ordinary BPE pieces instead of the dedicated special token, and also as
the direct output of SPN71_THINKING_TAG_NORMALIZE, which rewrites
`</thinking>` -> `</think>` in `delta_text` only and leaves `delta_token_ids`
holding the hallucinated sub-word pieces — i.e. SPN71's own repair is then
swallowed by this demotion, which is why SPN71 is inert on the streaming
path. Non-streaming is unaffected in practice (0/1277 banked responses carry
a `</think>`) because there the tag arrives as the real special token id.

────────────────────────────────────────────────────────────────────────────
FIX (one hunk, in the strict-set constructor — deliberately NOT in
`_process_lex_tokens`)

Subtract the config's declared reasoning-close terminals from
`_token_id_terminal_names`, i.e. exactly those terminals with a
`(REASONING, T) -> CONTENT` transition that emits `REASONING_END`. For
qwen3 that is `{THINK_END}` and nothing else:

  * `(REASONING, TOOL_START)` -> TOOL_PREAMBLE — next_state is not CONTENT,
    so TOOL_START is NOT subtracted and unbacked `<tool_call>` text keeps
    being demoted to content. The branch's original purpose is intact.
  * THINK_START has no REASONING_END transition and is NOT subtracted.

With THINK_END out of the strict set, a text-matched `</think>` reaches
`_on_terminal`, which then behaves per the config's own table:

  * REASONING -> CONTENT + REASONING_END          (the intended terminator)
  * CONTENT   -> absorbed silently                (the duplicate rule)
  * any other state (e.g. TOOL_ARGS) -> transition is None, so `_on_terminal`
    falls through to `self._emit_for_state(value)`, which is byte-identical
    to what `_on_content` did. No behaviour change off the two paths above.
  * a config that deliberately wants the old behaviour can still get it via
    `Transition(skip_in_token_id_mode=True)`, which `_on_terminal` honours.

Net: streaming now matches the non-streaming/token-id semantics instead of
diverging from them.

WHY THE __init__ SITE: `_token_id_terminal_names` has exactly two references
in the whole tree (definition here, single use in `_process_lex_tokens`), so
this is behaviourally equivalent to rewriting the condition — but it leaves
the `_process_lex_tokens` body byte-untouched, so it does not collide with
patch_pn98_toolcall_text_fragment_demotion.py (whose hunk-1 anchor is that
whole function) or patch_pn76_engine_deferred_toolcall_commit.py. PN135 is
therefore ORDER-INDEPENDENT with respect to PN76/PN98.

KILL SWITCH: `GENESIS_DISABLE_PN135_STREAM_THINK_END=1` restores stock
demotion at engine construction, without a rebuild or a re-patch.

KNOWN BEHAVIOUR CHANGE: a response that legitimately prints the literal
string `</think>` in its answer prose now has that string absorbed on the
streaming path (it already was on the token-id path). Measured exposure on
this box: `</think>` appears in 0.00% of 770 clean prod rows.

Retire when the parser engine grows native text-form reasoning-close
handling (nothing upstream as of this pin).

Idempotent by marker; anchor drift = loud SKIP no-op (never a silent
half-apply), matching patch_pn98's failure convention.
"""
import pathlib
import sys

LOG = "[pn135-streaming-think-end]"
VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET = VLLM / "parser/engine/streaming_parser_engine.py"
MARKER = "[Genesis-house PN135]"

OLD = (
    "        self._token_id_terminal_names: frozenset[str] = frozenset(\n"
    "            resolved_token_ids.values()\n"
    "        )\n"
)

NEW = (
    "        self._token_id_terminal_names: frozenset[str] = frozenset(\n"
    "            resolved_token_ids.values()\n"
    "        )\n"
    "\n"
    "        # [Genesis-house PN135] AUDIT-leak-paths-20260726 L2 / upstream\n"
    "        # #39697 class. _process_lex_tokens demotes any terminal in this\n"
    "        # set to plain content when it was matched by the TEXT lexer\n"
    "        # instead of the token-id scanner. Streaming serving always\n"
    "        # passes delta_token_ids, so that strict mode is always on and a\n"
    "        # `</think>` spelled as ordinary BPE pieces (or produced by\n"
    "        # SPN71's `</thinking>` -> `</think>` delta_text rewrite, which\n"
    "        # leaves delta_token_ids unrepaired) never reaches _on_terminal:\n"
    "        # in REASONING the engine never transitions and the whole answer\n"
    "        # is stranded in reasoning_content; in CONTENT the literal marker\n"
    "        # is emitted verbatim to the client. Drop the config's declared\n"
    "        # reasoning-close terminals from the strict set so they follow the\n"
    "        # transition table (terminate / absorb) like the token-id path\n"
    "        # already does. Tool terminals are NOT dropped -- (REASONING,\n"
    "        # TOOL_START) lands in TOOL_PREAMBLE, not CONTENT -- so unbacked\n"
    "        # `<tool_call>` text stays content, which is what the strict\n"
    "        # branch was written for.\n"
    "        import os as _pn135_os\n"
    "\n"
    "        if _pn135_os.environ.get(\n"
    "            'GENESIS_DISABLE_PN135_STREAM_THINK_END', ''\n"
    "        ).strip().lower() not in ('1', 'true', 'yes', 'on'):\n"
    "            _pn135_reasoning_close = frozenset(\n"
    "                terminal\n"
    "                for (state, terminal), tr in config.transitions.items()\n"
    "                if state is ParserState.REASONING\n"
    "                and tr.next_state is ParserState.CONTENT\n"
    "                and EventType.REASONING_END in tr.events\n"
    "            )\n"
    "            if _pn135_reasoning_close:\n"
    "                self._token_id_terminal_names = (\n"
    "                    self._token_id_terminal_names - _pn135_reasoning_close\n"
    "                )\n"
)

# Imports the hunk relies on; already module-level on this pin, asserted so a
# future re-pin that moves them fails loudly here instead of at engine build.
REQUIRED_IMPORTS = (
    "from vllm.parser.engine.events import EventType, SemanticEvent",
    "    ParserState,",
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} SKIP: {TARGET} not present on this vLLM build; no-op.",
              file=sys.stderr)
        return 0
    text = TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} {TARGET.name}: already applied (idempotent)")
        return 0
    for imp in REQUIRED_IMPORTS:
        if imp not in text:
            print(f"{LOG} SKIP: required import missing ({imp!r}) — engine "
                  f"shape changed; re-anchor before relying on this fix. "
                  f"No-op.", file=sys.stderr)
            return 0
    count = text.count(OLD)
    if count != 1:
        print(f"{LOG} SKIP: anchor occurs {count}x (need exactly 1) in "
              f"{TARGET.name} — engine shape changed (vLLM bumped or patch "
              f"chain reordered?); re-anchor before relying on this fix "
              f"(streaming `</think>` leaks back into content/reasoning "
              f"without it). No-op.", file=sys.stderr)
        return 0
    text = text.replace(OLD, NEW, 1)
    try:
        compile(text, str(TARGET), "exec")
    except SyntaxError as exc:
        print(f"{LOG} SKIP: patched result fails to compile ({exc}); "
              f"refusing to write. No-op.", file=sys.stderr)
        return 0
    TARGET.write_text(text)
    # Same-second pyc race (2026-07-22): boot scripts import vllm before the
    # text patches rewrite these files; a rewrite landing within the same
    # mtime second leaves a stale pyc that survives timestamp validation.
    cache = TARGET.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(TARGET.stem + ".*.pyc"):
            try:
                pyc.unlink()
                print(f"{LOG} dropped stale pyc {pyc.name}")
            except OSError as exc:
                print(f"{LOG} WARN: could not drop {pyc.name}: {exc}")
    print(f"{LOG} {TARGET.name}: applied 1 hunk — text-matched reasoning-close "
          f"terminals now follow the transition table (streaming `</think>` "
          f"terminates reasoning / duplicates absorbed); tool terminals keep "
          f"strict token-id demotion. Kill switch: "
          f"GENESIS_DISABLE_PN135_STREAM_THINK_END=1")
    return 0


sys.exit(main())
