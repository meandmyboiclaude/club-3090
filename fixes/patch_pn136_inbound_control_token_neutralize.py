#!/usr/bin/env python3
"""PN136 / BUG-160 graft (2026-07-27) — INBOUND control-token neutraliser on the
chat-completions message path. Dark by default.

WHAT BUG-160 IS
---------------
A prompt that *quotes* a reasoning delimiter as prose forges that delimiter for
the request that carries it. `prod-099` of `prod_mixed_v3` shipped a retrieved
hindsight memory reading

    "...to enforce graceful `</think>` termination and avoid truncation."

The model echoed the quoted tag while planning. The qwen3 parser
(`vllm/parser/qwen3.py`) is a state machine whose `(REASONING, THINK_END)`
transition fires on the FIRST `</think>` **string** in the output; everything
after it becomes the ANSWER, and the model's own later `</think>` is silently
absorbed by the `(CONTENT, THINK_END)` no-op transition. The caller was served
the model's planning text beginning mid-clause with a bare backtick.

No counter caught it — `think_tag_in_answer=false`, `close_tags_in_answer=0`,
`finish_reason=stop`, `content_ok=true` — because the delimiter is consumed
before any counter runs.

WHY AN ENGINE-SIDE LEG EXISTS AT ALL
------------------------------------
The caller-side fix (hindsight `engine/prompt_utils.py::neutralize_control_tokens`,
applied at `llm_wrapper.py:955` / `:1110`) is committed and e2e-proven, and it is
the RIGHT place for the owner: only hindsight knows which bytes are untrusted
retrieved data. But it protects hindsight's mouth, not its hands. Verified live
2026-07-27: one ordinary `POST /v1/default/banks/default/memories/recall` on
`:8100` returned **5 raw `</think>` and 3 raw `</thinking>`** in the response
body. Any consumer that pastes recall output into a prompt — anubis
`memory_search`, a bench harness, a third-party caller — reproduces BUG-160 with
the hindsight fix fully deployed. The engine is the only layer that can defend
against callers we do not own.

(a) WHERE — `_parse_chat_message_content` in `entrypoints/chat_utils.py`
-----------------------------------------------------------------------
Placed immediately before the `_parse_chat_message_content_parts(role, content,
...)` hand-off, i.e. after the message content has been normalised to a parts
list and after Genesis PN91's developer->system rename, and BEFORE the chat
template renders anything.

That position is load-bearing for THREE reasons:

  * it is the only chokepoint both `parse_chat_messages` and
    `parse_chat_messages_async` funnel through, so chat-completions and the
    Responses API are covered by one hook;
  * `role` is in scope here (it is not inside
    `_parse_chat_message_content_part`), which the policy needs;
  * it is upstream of the template, so the sanitiser **cannot** touch template
    output. See (d).

vLLM already guards inbound text at this exact layer —
`_reject_reserved_placeholder_in_text` rejects a caller-supplied
`prompt_embeds` sentinel for structurally the same reason (caller text that
tokenises to a control id). This graft is that idea applied to the reasoning
delimiter, with rewriting instead of rejection.

(b) WHAT/HOW — square-bracket rewrite, not deletion, not entity escaping
------------------------------------------------------------------------
`</think>` -> `[/think]`. Deleting is lossy: a caller may legitimately be ASKING
ABOUT the tag, which is literally how this loop started (prod-011's text is a
bug report *about* `<think>` handling). Bracket form keeps the string readable
and self-describing for a human and for the model, while removing the exact byte
sequence the parser's terminal matches.

HTML-entity escaping was tried first and is WRONG on measured evidence
(BUG-160 record): `&lt;/think&gt;` corrupted 3/3 trials because the model
decodes the entity back to a live tag when quoting. Bracket form: 0/6.

The tradeoff, stated plainly: this MUTATES the caller's payload. A caller who
sends `</think>` and expects byte-exact round-trip does not get it. That is why
the graft is flag-gated and dark, and why the exemptions in (d) exist.

(c) SCOPE — the live stack's terminals, and nothing else
--------------------------------------------------------
Default literal set (3 items):

    <think>        -- qwen3.py THINK_START
    </think>       -- qwen3.py THINK_END, the actual corrupting terminal
    </thinking>    -- NOT a parser terminal, but Genesis PN71's
                      `Qwen3Parser._preprocess_feed` rewrites `</thinking>` ->
                      `</think>` on EVERY feed path before the state machine
                      sees it. On this deployment an echoed `</thinking>` is
                      exactly as corrupting as the closer.

Plus ChatML specials `<|...|>` -> `[|...|]` (default on, `PN136_CHATML=0` to
disable): these are real added tokens on the tokenizer, so `<|im_end|>` inside
user text becomes the genuine role terminator and truncates the turn. Same
class, different terminal.

DELIBERATE DIVERGENCE FROM THE HINDSIGHT SET. hindsight neutralises seven more
literals — `<thinking>`, `<thought>`, `</thought>`, `<reasoning>`,
`</reasoning>`, `|startthink|`, `|endthink|` — because it is provider-agnostic
and cannot know which parser will receive its prompt. The engine DOES know: the
loaded parser is qwen3 and nothing in it, in the template, or in PN71 keys on
any of those. Including them would be pure risk with no mechanism behind it, and
`<reasoning>` in particular is a common *requested output format* ("wrap your
rationale in <reasoning> tags") that this graft would then silently break.
Measured cost of the wider set on 465 real prompts is also zero, so this is
narrowed on "no mechanism => no benefit", not on observed harm.

TOOL-CALL TERMINALS ARE OFF BY DEFAULT (`PN136_TOOLCALL_TOKENS=1` to arm).
`<tool_call>`, `</tool_call>`, `<function=`, `</function>`, `<parameter=`,
`</parameter>` are the qwen3/hermes tool grammar and are the same forgery class
— but system prompts that TEACH the tool-call format contain them literally and
must reach the model intact. Neutralising them by default would break tool
calling to fix a defect never observed on this surface. `<function=` and
`<parameter=` additionally collide with ordinary XML/code. This is where the
boundary goes: wide enough to cover every terminal the live parser keys on for
*reasoning*, not so wide that ordinary angle-bracketed text is corrupted.

(d) THE ASSISTANT-PREFILL TRAP
------------------------------
The chat template PRE-FILLS the opening tag in prompt space
(`chat_template_v2json.jinja` tail: `'<think>\\n'` after `<|im_start|>assistant`,
and the full `'<think>\\n\\n</think>\\n\\n'` pair when `enable_thinking=false`).
Exactly one closer in a well-formed response is therefore CORRECT, not a leak.

This graft cannot break that, structurally: it runs on message content BEFORE
`apply_chat_template`, never on the rendered prompt string and never on
generated tokens. The template writes its own delimiters after the sanitiser has
already run and is never shown to it.

Two further exemptions protect legitimate reasoning round-trips:

  * ROLE `assistant` IS EXEMPT by default (`PN136_ROLES`). The live template
    reads `</think>` out of assistant content on purpose
    (`chat_template_v2json.jinja:94-96` splits `content` on `</think>` to
    recover `reasoning_content`); neutralising it would break a supported input
    shape. This is also the exact risk the BUG-160 record flagged for a
    serving-side leg: "would break a legitimate client replaying prior assistant
    turns that contain real reasoning blocks."
  * The `reasoning` / `reasoning_content` message field and the `thinking`
    content-part type are NEVER touched — they bypass the parts list entirely
    (`_parse_chat_message_content` assigns `result_msg["reasoning"]` directly),
    so interleaved thinking is untouched by construction.

Only `type: text` / `input_text` / `output_text` parts and bare string parts are
rewritten. `refusal`, `thinking`, `tool_reference` and every multimodal part
type pass through unread — the graft never overrides content it does not own.

(e) FALSE-POSITIVE BOUND — replayed over real traffic
------------------------------------------------------
Rule replayed over 465 real captured prompts (the BUG-077 precedent):

    prod_mixed_v3.jsonl   111 rows ->  2 rows altered, 3 occurrences
    prod_mixed_v1.jsonl   106 rows ->  2 rows altered, 3 occurrences (same two)
    gpqa_full.jsonl       198 rows ->  0
    gpqa_subset.jsonl     100 rows ->  0
    lcb_subset.jsonl       50 rows ->  0

Both altered rows are TRUE positives, and both are user-role retrieved/report
prose:

    prod-011  msg[1] user  `<think>`      @731   "...chat template opens <think>
                                                  in the PROMPT so start_thinking
                                                  is prompt-space..."
    prod-099  msg[1] user  `</think>`     @6019  the BUG-160 memory
    prod-099  msg[1] user  `</thinking>`  @6958  "vLLM uses the native
                                                  thinking_token_budget ... to
                                                  force </thinking> at N tokens"

The third occurrence is NEW relative to the BUG-160 record, which counted only
`</think>` and called prod-099 the single closer-carrying row. prod-099 carries
TWO independently corrupting literals from TWO different retrieved memories, and
the served corruption quoted in that record ("...termination, avoid truncation
trap, reliably emit an answer") splices language from BOTH. A fix covering only
`</think>` leaves the second one armed — which is why `</thinking>` is in the
default set.

ChatML specials: 0 occurrences in all 465 prompts. Zero measured cost.

STATUS: DARK. With `GENESIS_ENABLE_PN136_INBOUND_NEUTRALIZE` unset the helper
returns its argument object unchanged (identity, not a copy) and
`_parse_chat_message_content` behaves byte-identically. Flipping it ON mutates
caller payloads and wants a bench arm.

Idempotent by marker; anchor drift = FATAL exit 1, mirroring the other /fixes
appliers. The token set is additionally pinned to `vllm/parser/qwen3.py`: if
that file's THINK_START/THINK_END constants ever move, this applier refuses to
run rather than silently defending the wrong string.
"""
import os
import pathlib
import sys

LOG = "[patch_pn136_inbound_control_token_neutralize]"
BASE = pathlib.Path(
    os.environ.get("PN136_VLLM_BASE", "/usr/local/lib/python3.12/dist-packages/vllm")
)
CHAT_UTILS = BASE / "entrypoints/chat_utils.py"
QWEN3_PARSER = BASE / "parser/qwen3.py"

# ── The injected helper. Self-contained: imports inside, env read at CALL time
#    so the flag can be flipped without rebuilding the module. Uses the module's
#    existing `logger`. ────────────────────────────────────────────────────────
MARK_HELPER = "# [Genesis PN136] inbound control-token neutraliser"
HELPER_SRC = '''
# [Genesis PN136] inbound control-token neutraliser (BUG-160). Dark unless
# GENESIS_ENABLE_PN136_INBOUND_NEUTRALIZE=1. Rewrites reasoning/ChatML
# delimiters that arrive as CALLER DATA into a readable bracket form so the
# model cannot echo them back as live control tokens. Runs before the chat
# template, so the template's own prefilled <think> is never seen or touched.
# See /fixes/patch_pn136_inbound_control_token_neutralize.py for the design and
# the 465-prompt false-positive bound.

# literal -> replacement. Only terminals the LIVE stack keys on:
#   <think>/</think>  -> vllm/parser/qwen3.py THINK_START / THINK_END
#   </thinking>       -> Genesis PN71 _preprocess_feed rewrites it to </think>
#                        on every feed path, so an echo is equally corrupting.
_PN136_BASE_LITERALS = (
    ("<think>", "[think]"),
    ("</think>", "[/think]"),
    ("</thinking>", "[/thinking]"),
)
# qwen3/hermes tool grammar. Same forgery class, OFF by default: system prompts
# that teach the tool-call format contain these literally and must survive.
_PN136_TOOLCALL_LITERALS = (
    ("<tool_call>", "[tool_call]"),
    ("</tool_call>", "[/tool_call]"),
    ("<function=", "[function="),
    ("</function>", "[/function]"),
    ("<parameter=", "[parameter="),
    ("</parameter>", "[/parameter]"),
)
_PN136_CHATML_RE = None  # compiled lazily on first armed call
_PN136_POLICY_CACHE = {}
# Content-part types this graft OWNS. `thinking` and `refusal` are
# model-authored, `tool_reference` and every mm type are not text.
_PN136_TEXT_PART_TYPES = frozenset(("text", "input_text", "output_text"))
_PN136_DEFAULT_ROLES = "system,user,tool,function"


def _pn136_policy():
    """Read the env policy. Cached per distinct env tuple, re-read every call so
    a flag flip needs no restart of anything but the request."""
    import os as _os

    key = (
        _os.environ.get("GENESIS_ENABLE_PN136_INBOUND_NEUTRALIZE", "0"),
        _os.environ.get("PN136_ROLES", _PN136_DEFAULT_ROLES),
        _os.environ.get("PN136_CHATML", "1"),
        _os.environ.get("PN136_TOOLCALL_TOKENS", "0"),
        _os.environ.get("PN136_EXTRA", ""),
    )
    cached = _PN136_POLICY_CACHE.get(key)
    if cached is not None:
        return cached
    armed = key[0] == "1"
    roles = frozenset(
        r.strip().lower() for r in key[1].split(",") if r.strip()
    )
    literals = list(_PN136_BASE_LITERALS)
    if key[3] == "1":
        literals.extend(_PN136_TOOLCALL_LITERALS)
    for extra in key[4].split(","):
        extra = extra.strip()
        # An extra literal must be angle-bracketed; anything else would let an
        # operator neutralise arbitrary prose by accident.
        if len(extra) > 2 and extra.startswith("<") and extra.endswith(">"):
            literals.append((extra, "[" + extra[1:-1] + "]"))
    policy = (armed, roles, tuple(literals), key[2] == "1")
    _PN136_POLICY_CACHE[key] = policy
    return policy


def _pn136_neutralize_text(text, literals, chatml):
    """Bracket-rewrite control tokens in one string.

    Returns the ORIGINAL object when nothing matched, so callers can use
    identity to detect "unchanged". Idempotent: the output carries none of the
    angle-bracketed forms, so a second pass is a no-op.
    """
    if not text or not isinstance(text, str):
        return text
    if "<" not in text:  # every pattern needs it
        return text
    out = text
    for literal, replacement in literals:
        if literal in out:
            out = out.replace(literal, replacement)
    if chatml and "<|" in out:
        global _PN136_CHATML_RE
        if _PN136_CHATML_RE is None:
            import re as _re

            _PN136_CHATML_RE = _re.compile(r"<\\|([^|>]{1,64})\\|>")
        out = _PN136_CHATML_RE.sub(r"[|\\1|]", out)
    return out


def _pn136_neutralize_content(role, content):
    """Neutralise caller-supplied text parts of one message.

    `content` is the already-normalised parts list. Returns the original list
    object when nothing changed; otherwise a NEW list with new part dicts, so
    the caller's request objects are never mutated in place.
    """
    armed, roles, literals, chatml = _pn136_policy()
    if not armed or not content:
        return content
    if (role or "").lower() not in roles:
        # assistant is exempt by default: the chat template reads </think> out
        # of assistant content on purpose to recover reasoning_content.
        return content
    if not isinstance(content, list):
        return content

    changed = 0
    out = None
    for idx, part in enumerate(content):
        new_part = part
        if isinstance(part, str):
            new_text = _pn136_neutralize_text(part, literals, chatml)
            if new_text is not part:
                new_part = new_text
        elif isinstance(part, dict) and part.get("type") in _PN136_TEXT_PART_TYPES:
            text = part.get("text")
            new_text = _pn136_neutralize_text(text, literals, chatml)
            if new_text is not text:
                new_part = dict(part)
                new_part["text"] = new_text
        if new_part is not part:
            if out is None:
                out = list(content)
            out[idx] = new_part
            changed += 1
    if out is None:
        return content
    logger.warning(
        "[Genesis PN136] neutralised control tokens in %d inbound text part(s) "
        "of a %r message (BUG-160: a quoted reasoning delimiter forges the "
        "parser terminal and corrupts the answer)",
        changed,
        role,
    )
    return out

'''

ANCH_HELPER = "def _parse_chat_message_content(\n"
ANCH_CALL = (
    "    result = _parse_chat_message_content_parts(\n"
    "        role,\n"
    "        content,  # type: ignore\n"
)
MARK_CALL = "# [Genesis PN136] inbound neutralisation point"
REPL_CALL = (
    "    # [Genesis PN136] inbound neutralisation point (identity when the flag\n"
    "    # is off). Runs after the developer->system rename and after content is\n"
    "    # normalised to a parts list, and BEFORE the chat template renders —\n"
    "    # so the template's own prefilled <think> is never in scope.\n"
    "    content = _pn136_neutralize_content(role, content)\n"
    "    result = _parse_chat_message_content_parts(\n"
    "        role,\n"
    "        content,  # type: ignore\n"
)

# The token set is only correct while these hold in the live parser.
PARSER_PINS = (
    'THINK_START = "<think>"',
    'THINK_END = "</think>"',
)
PN71_MARKER = '"</thinking>", "</think>"'


def _fatal(msg):
    print(f"{LOG} FATAL {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    if not CHAT_UTILS.exists():
        _fatal(f"missing target {CHAT_UTILS}")
    src = CHAT_UTILS.read_text(encoding="utf-8")

    if MARK_HELPER in src and MARK_CALL in src:
        print(f"{LOG} already applied (both markers present) — no-op")
        return 0
    if (MARK_HELPER in src) != (MARK_CALL in src):
        _fatal("half-applied: exactly one marker present, refusing to patch")

    # ── the token set is pinned to the parser it defends ─────────────────────
    if not QWEN3_PARSER.exists():
        _fatal(
            f"missing {QWEN3_PARSER} — this graft's literal set is derived from "
            f"the qwen3 parser's terminals; on a build without it the set is "
            f"unverifiable and the patch must not guess"
        )
    parser_src = QWEN3_PARSER.read_text(encoding="utf-8")
    for pin in PARSER_PINS:
        if pin not in parser_src:
            _fatal(
                f"parser drift: {QWEN3_PARSER} no longer declares {pin!r}. The "
                f"neutralised literal set would be defending a string the "
                f"parser has stopped keying on — re-derive it before applying."
            )
    if PN71_MARKER not in parser_src:
        print(
            f"{LOG} note: Genesis PN71's </thinking> -> </think> normalizer is "
            f"not present in {QWEN3_PARSER}; '</thinking>' stays in the default "
            f"literal set but is defensive rather than load-bearing on this build"
        )

    n_helper = src.count(ANCH_HELPER)
    if n_helper != 1:
        _fatal(f"helper anchor {ANCH_HELPER!r} found {n_helper}x, expected 1")
    n_call = src.count(ANCH_CALL)
    if n_call != 1:
        _fatal(f"call anchor {ANCH_CALL!r} found {n_call}x, expected 1")
    if "logger" not in src:
        _fatal("target has no module-level `logger` — the helper logs through it")

    src = src.replace(ANCH_HELPER, HELPER_SRC.lstrip("\n") + "\n" + ANCH_HELPER, 1)
    src = src.replace(ANCH_CALL, REPL_CALL, 1)

    CHAT_UTILS.write_text(src, encoding="utf-8")
    print(
        f"{LOG} applied to {CHAT_UTILS} "
        f"(dark: GENESIS_ENABLE_PN136_INBOUND_NEUTRALIZE unset)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
