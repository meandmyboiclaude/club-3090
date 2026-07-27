#!/usr/bin/env python3
"""BUG-158 graft (2026-07-27) — attribute reasoning tokens when the chat
template opens ``<think>`` in PROMPT space.

WHAT BUG-158 ACTUALLY IS (two halves; only ONE is a defect)
-----------------------------------------------------------
The filing reports two symptoms. Measured live on :8021 (image
dev1474cherrymax-1757-20260725, TC gptq-pro-v2) on 2026-07-27:

  * ``usage.completion_tokens_details.reasoning_tokens = 0`` on a request that
    generated hundreds of thinking tokens — **REAL, and this graft fixes it.**
  * ``choices[0].message.reasoning_content = null`` — **NOT a serving defect.**
    This vLLM renamed the response field to ``reasoning``; ``reasoning_content``
    survives only as a deprecated *request*-side alias
    (``entrypoints/openai/chat_completion/protocol.py:509-511``). ``ChatMessage``
    (same file, :55-65) declares ``reasoning``, full stop. A probe that reads
    ``reasoning_content`` off the response reads ``None`` on EVERY request,
    thinking or not. Live proof, one non-streaming call, 134 completion tokens:

        message.reasoning         = " Break 17*23 into (10+7)*(20+3).\\nStep 2: ..."
        message.reasoning_content = <absent>
        usage...reasoning_tokens  = 0        <-- the only thing wrong

    So the reasoning TEXT was never lost; only the COUNT was. (qbench45's
    harness already reads ``reasoning_content or reasoning`` and re-tokenizes
    the text for its own exact rtok — ``bench/client.py:285-310`` — which is why
    no bench number is contaminated by this.)

ROOT CAUSE OF THE REAL HALF
---------------------------
``vllm/parser/engine/parser_engine.py:624-641`` — ``count_reasoning_tokens``
walks the OUTPUT token ids with ``depth = 0`` and only counts a token while
``depth > 0``; depth rises solely on a ``<think>`` **start-token id in the
output**.

``/chat_template.jinja:147`` emits ``<think>\\n`` as part of the generation
prompt, so the opener lives in PROMPT space and the output stream carries only
the closer. depth therefore never leaves 0 and the walk returns 0 for every
thinking request. Reproduced in-container against the real class:

    Qwen3Parser(tok, chat_template_kwargs={"enable_thinking": True})
      -> thinking_enabled=True, initial_state=ParserState.REASONING
      -> count_reasoning_tokens(<24 ids: reasoning, </think>, answer>) == 0

THE STATE ALREADY EXISTS — THIS IS A CONDITIONAL, NOT SURGERY
-------------------------------------------------------------
BUG-160's structural proposal ("hand 'reasoning is open at generation start'
over from the template instead of inferring it from the output") is ALREADY
implemented on this build; it is just not consulted by the counter:

  * ``parser/qwen3.py:226-238`` — ``Qwen3Parser.__init__`` reads
    ``chat_template_kwargs["enable_thinking"]`` (the very kwargs the template
    branches on) and selects ``qwen3_config(thinking=...)``.
  * ``parser/qwen3.py:101`` — that config sets
    ``initial_state = ParserState.REASONING if thinking else ParserState.CONTENT``.
  * ``chat_completion/serving.py:344-349`` builds the parser PER REQUEST with
    those kwargs, and :450-454 forwards them to the engine.

So ``self.parser_engine_config.initial_state == ParserState.REASONING`` IS the
"template opened thinking" bit, on the parser instance, per request. The graft
reads it. Nothing new is plumbed.

WHAT THE GRAFT DOES
-------------------
Replaces the depth-counter with a walk that mirrors the parser's OWN state
machine, seeded from ``initial_state``, so ``reasoning_tokens`` agrees with the
``reasoning`` field the same parser produced:

  * start in reasoning iff ``initial_state == REASONING``;
  * leave reasoning on any terminal whose ``(REASONING, T)`` transition exits
    REASONING — that is ``</think>`` AND the implicit ``<tool_call>`` end
    (``parser/qwen3.py`` transitions, :127-141);
  * re-enter only if the config actually declares a ``(CONTENT, THINK_START)``
    transition (glm47_moe, deepseek_v4 do; qwen3/kimi_k2/minimax_m2 do NOT), so
    on qwen3 a *second* ``<think>`` after the first close stays content —
    exactly what the FSM does with it.

Both the terminator set and the re-entry rule are read off
``parser_engine_config.transitions``; nothing is hard-coded per model.

CONTENT IS NOT TOUCHED — BY CONSTRUCTION
----------------------------------------
``count_reasoning_tokens`` is a pure ``Sequence[int] -> int``. Its only two call
sites are the usage attach points, streaming and non-streaming:

    chat_completion/serving.py:962-972   (streaming, per-choice accumulated ids)
    chat_completion/serving.py:1298-1307 (non-streaming, output.token_ids)

both of which assign ``CompletionTokenUsageInfo(reasoning_tokens=...)`` and
nothing else. The graft edits that one method and no other. It cannot move a
byte of ``message.content``, ``message.reasoning`` or any SSE delta — the split
is done by ``extract_reasoning`` / ``parse_delta``, which this file does not
touch. One fix covers streaming AND non-streaming because both call the same
base method.

BUG-154 / BUG-160 INTERACTION
-----------------------------
The parser transitions ``(REASONING, THINK_END) -> CONTENT`` and absorbs bare
duplicates via ``(CONTENT, THINK_END) -> CONTENT``. The graft copies that
verbatim: first closer ends the span, later closers are inert. A forged
``</think>`` echo (BUG-160) therefore ends the reasoning span in the COUNT at
exactly the point it ends it in the TEXT — the count stays consistent with what
was served, which is the honest reading; it does not and cannot un-corrupt the
split (see BUG-160 note (c)).

STATUS: DARK. With ``GENESIS_ENABLE_BUG158_REASONING_TOKENS`` unset the injected
helper returns ``None`` on its first line and the stock walk runs unchanged, so
``reasoning_tokens`` is byte-identical to today (0 on this template). Flipping it
ON changes ONLY that integer.

Idempotent by marker; anchor drift = FATAL exit 1, mirroring the other /fixes
appliers.
"""
import os
import pathlib
import sys

LOG = "[patch_bug158_reasoning_token_attribution]"
BASE = pathlib.Path(
    os.environ.get("BUG158_VLLM_BASE", "/usr/local/lib/python3.12/dist-packages/vllm")
)
TARGET = BASE / "parser/engine/parser_engine.py"

MARK_HELPER = "# BUG-158 graft: template-opened reasoning attribution"
MARK_CALL = "# BUG-158 graft: delegation"

# ── The injected helper. Reads the flag at CALL time so it can be flipped
#    without a rebuild. Returns None => caller falls through to the stock walk.
HELPER_SRC = '''    # BUG-158 graft: template-opened reasoning attribution (dark unless
    # GENESIS_ENABLE_BUG158_REASONING_TOKENS=1). See
    # /fixes/patch_bug158_reasoning_token_attribution.py for the root cause.
    def _bug158_span_markers(self):
        """(-> starters, enders) token-id sets, read off the transition table.

        enders  = terminals whose (REASONING, T) transition leaves REASONING
                  (``</think>`` and qwen3's implicit ``<tool_call>`` end).
        starters= terminals whose (CONTENT, T) transition enters REASONING
                  (empty on qwen3/kimi_k2/minimax_m2 -- no re-entry).
        """
        cached = getattr(self, "_bug158_markers_cache", None)
        if cached is not None:
            return cached
        cfg = self.parser_engine_config
        vocab = self.vocab
        starters = set()
        enders = set()
        for name, text in cfg.token_id_terminals.items():
            tid = vocab.get(text)
            if tid is None:
                continue
            tr = cfg.transitions.get((ParserState.REASONING, name))
            if tr is not None and tr.next_state != ParserState.REASONING:
                enders.add(tid)
            tr = cfg.transitions.get((ParserState.CONTENT, name))
            if tr is not None and tr.next_state == ParserState.REASONING:
                starters.add(tid)
        cached = (starters, enders)
        self._bug158_markers_cache = cached
        return cached

    def _bug158_count_reasoning_tokens(self, token_ids):
        """Reasoning-token count for a stream whose opener is in prompt space.

        Returns None to defer to the stock walk: flag off, non-thinking
        request (initial_state is CONTENT -- the stock walk is already right
        there), no resolvable ``</think>`` id, or any unexpected shape.
        """
        import os as _os

        if _os.environ.get("GENESIS_ENABLE_BUG158_REASONING_TOKENS", "0") != "1":
            return None
        try:
            cfg = self.parser_engine_config
            if cfg.initial_state != ParserState.REASONING:
                # Reasoning is NOT open at generation start; the stock
                # opener-seeking walk has the same answer. Defer.
                return None
            end_id = self._reasoning_end_token_id
            if end_id is None:
                return None
            start_id = self._reasoning_start_token_id
            starters, enders = self._bug158_span_markers()
            enders = enders | {end_id}
            markers = {end_id}
            if start_id is not None:
                markers.add(start_id)
            count = 0
            in_reasoning = True
            for token_id in token_ids:
                if in_reasoning:
                    if token_id in enders:
                        in_reasoning = False
                        continue
                    if token_id in markers:
                        # a redundant opener inside reasoning: the FSM's
                        # (REASONING, THINK_START) is a no-op, so is this
                        continue
                    count += 1
                elif token_id in starters:
                    in_reasoning = True
            return count
        except Exception:  # never let accounting break a response
            return None

'''

# ── Anchors. The stock method, verbatim from image
#    dev1474cherrymax-1757-20260725 (vllm/parser/engine/parser_engine.py:624).
ANCH_METHOD = (
    "    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:\n"
    "        start_id = self._reasoning_start_token_id\n"
)
REPL_METHOD = (
    "    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:\n"
    "        # BUG-158 graft: delegation (identity when the flag is off --\n"
    "        # the helper returns None on its first line and the stock walk\n"
    "        # below runs unchanged).\n"
    "        _bug158 = self._bug158_count_reasoning_tokens(token_ids)\n"
    "        if _bug158 is not None:\n"
    "            return _bug158\n"
    "        start_id = self._reasoning_start_token_id\n"
)

# The stock body must still be there afterwards — guard against grafting onto
# a build whose walk has already been rewritten by someone else.
ANCH_STOCK_BODY = (
    "        count = 0\n"
    "        depth = 0\n"
    "        for token_id in token_ids:\n"
)


def _fatal(msg):
    print(f"{LOG} FATAL {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    if not TARGET.exists():
        _fatal(f"missing target {TARGET}")
    src = TARGET.read_text(encoding="utf-8")

    if MARK_HELPER in src and MARK_CALL in src:
        print(f"{LOG} already applied (both markers present) — no-op")
        return 0
    if (MARK_HELPER in src) != (MARK_CALL in src):
        _fatal("half-applied: exactly one marker present, refusing to patch")

    n = src.count(ANCH_METHOD)
    if n != 1:
        _fatal(f"method anchor found {n}x, expected 1 — parser_engine.py shape changed")
    n = src.count(ANCH_STOCK_BODY)
    if n != 1:
        _fatal(
            f"stock depth-walk body found {n}x, expected 1 — "
            "count_reasoning_tokens has already been rewritten; refusing to graft"
        )
    if "from vllm.parser.engine.parser_engine_config import" not in src or (
        "ParserState" not in src
    ):
        _fatal("ParserState is not imported in the target — refusing to graft")

    src = src.replace(ANCH_METHOD, HELPER_SRC + REPL_METHOD, 1)

    try:
        compile(src, str(TARGET), "exec")
    except SyntaxError as exc:
        _fatal(f"patched file does not compile: {exc}")

    TARGET.write_text(src, encoding="utf-8")

    cache = TARGET.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(TARGET.stem + ".*.pyc"):
            try:
                pyc.unlink()
            except OSError:
                pass

    # Informational only — the AUTHORITATIVE state is the env at REQUEST time
    # (the injected helper re-reads it on every call, so a flag flip needs no
    # re-patch). This line reports the env as seen by THIS applier process.
    _armed = os.environ.get("GENESIS_ENABLE_BUG158_REASONING_TOKENS", "0") == "1"
    _state = (
        "ARMED: GENESIS_ENABLE_BUG158_REASONING_TOKENS=1 at apply time"
        if _armed
        else "dark: GENESIS_ENABLE_BUG158_REASONING_TOKENS unset/0 at apply time"
    )
    print(f"{LOG} applied to {TARGET} ({_state})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
