#!/usr/bin/env python3
"""PN101 — call-sites for answer rescue (PN100/PN71 companion, house-original).

Two anchored insertions in chat_completion/serving.py:
  1. hint site — sync `maybe_add_answer_hint(request)` at the top of
     `_create_chat_completion`, AFTER PN100's await hook and after genesis PN16,
     and BEFORE `chat_template_kwargs = self._effective_chat_template_kwargs(...)`
     is read. That window is not cosmetic: `maybe_add_answer_hint` returns early
     unless `request.thinking_token_budget > 0` (answer_rescue.py:579), which is
     what PN100 assigns, and it writes the banner INTO
     `request.chat_template_kwargs`, which the `_effective_...` call then reads.
     Land outside the window and the hint is a silent no-op.
  2. repair site — wraps the `create_chat_completion` return value with
     `await maybe_rescue_answer(self, request, result)` (non-streaming
     responses only; the module gates everything else).

Module: vllm/_genesis/middleware/answer_rescue.py (mounted Genesis tree).
Master env flag GENESIS_ENABLE_PN101_ANSWER_RESCUE is DEFAULT OFF — behavioral
patches never default-on (house rule). With the flag off both call sites are
inert passthroughs. Fail-open: every exception is swallowed to debug.

FAIL-OPEN AT BOOT (2026-07-26)
------------------------------
This patcher used to `return 1` on a missing or ambiguous anchor. The compose
entrypoint runs under `set -e`, so that exit code took the whole engine down —
no vLLM, not a degraded one. Measured on both live pins by replaying `apply_all`
plus the entrypoint prefix and rewording ONE upstream comment:

    dev1474cherrymax-1757 / dev1474cherry-1711, hint anchor absent
      -> "FATAL: anchor-not-found (hint)", container rc=1,
         PN71T and everything after it never ran.
    ... same pins, hint anchor duplicated
      -> "FATAL: ambiguous anchor (hint, 2 hits)", container rc=1.

And the anchor it staked the boot on was `        # Streaming response\n` — a
single upstream COMMENT line. Any upstream bump that rewords it kills the boot.

Unlike PN100 (BUG-141) there is NO conditional-anchor dependency here: the hint
anchor is upstream's own text, not another opt-in patch's insertion. Genesis
PN16 / PN40 / edge-guard / P68-69 all rewrite that region, but each one re-emits
the `# Streaming response` + `tokenizer = ...` pair verbatim, so the count is 1
whether those lanes are on or off. Flipping GENESIS_ENABLE_PN16_LAZY_REASONER
does NOT remove PN101's anchor. The exposure is pure upstream-comment drift —
smaller than PN100's, and just as fatal when it fires.

Three changes fix the class:

  1. A COUNTED VARIANT SET per site, most-guaranteed first, preferring upstream
     CODE over comment text. First variant with count == 1 wins; a variant with
     count > 1 is REFUSED, never guessed. The repair site is rewritten by
     parenthesis-balanced statement span rather than a byte-exact three-line
     match, so an upstream arg reflow no longer counts as drift.
  2. COMPILE-CHECK BEFORE WRITE. A bad insert leaves serving.py byte-identical
     instead of writing first and raising afterwards.
  3. NEVER a non-zero exit, on any path. Missing target, unresolvable anchor,
     failed compile-check all _shout() the CAPABILITY that is inert this boot
     and return 0. The shout has to be loud enough that nobody reads it as a
     clean boot — a silently-skipped patch shipped a no-op here on 2026-07-25.

The two sites resolve INDEPENDENTLY. They are separate features behind separate
sub-toggles (GENESIS_PN101_HINT / GENESIS_PN101_REPAIR), and the hint block is
also the anchor patch_pn114_seed_span.py's S4 site keys on — losing the repair
site is no reason to also drop the banner and take PN114-SEED down with it.

Anchor counts as the BOOT sees them: python3 fixes/verify_pn101_anchors.py
"""
from __future__ import annotations

import logging
import pathlib
import sys

LOG = "[pn101-answer-rescue]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/"
    "chat_completion/serving.py"
)
HINT_MARKER = "# PN101a:"
REPAIR_MARKER = "# PN101b:"

# ── Hint site ─────────────────────────────────────────────────────────────
# NB `            _pn101_hint(request)\n        except Exception:\n` is the
# anchor patch_pn114_seed_span.py's S4 site counts. Keep those two lines
# adjacent and byte-identical in any future edit of this block.
HINT_BLOCK = (
    "        # PN101a: bounded-envelope answer-first hint (fail-open; default-OFF\n"
    "        # master flag; see _genesis/middleware/answer_rescue.py).\n"
    "        try:\n"
    "            from vllm._genesis.middleware.answer_rescue import (\n"
    "                maybe_add_answer_hint as _pn101_hint,\n"
    "            )\n"
    "            _pn101_hint(request)\n"
    "        except Exception:\n"
    "            import logging as _pn101a_logging\n"
    "            _pn101a_logging.getLogger(\n"
    "                'genesis.middleware.answer_rescue'\n"
    "            ).debug('PN101 hint raised; ignored', exc_info=True)\n"
)

# Most-guaranteed first. All three land inside the legal window (after PN100's
# await hook, before the effective chat_template_kwargs read) —
# `_check_window()` proves that per boot rather than trusting this comment.
# mode "after" inserts below the anchor text, "before" above it.
#
# 1. Upstream's tokenizer fetch + its assert. CODE ONLY, so a reword of the
#    `# Streaming response` comment above it cannot invalidate it. count==1 in
#    the pristine image AND post-apply_all on both live pins (2026-07-26).
#    Inserted AFTER so the `# Streaming response` + tokenizer pair — the anchor
#    genesis PN16 / PN40 / edge-guard / P68-69 all key on — is left contiguous.
V_TOKENIZER = (
    "        tokenizer = self.renderer.tokenizer\n"
    "        assert tokenizer is not None\n"
)
# 2. The historical anchor: that same pair, which those genesis lanes re-emit
#    verbatim. Reproduces the pre-2026-07-26 insertion point byte-for-byte
#    (block goes ABOVE the comment) whenever the comment is intact.
V_STREAMING_PAIR = (
    "        # Streaming response\n"
    "        tokenizer = self.renderer.tokenizer\n"
)
# 3. Last resort: the read that closes the window. Survives BOTH a comment
#    reword and a renamed tokenizer fetch; still ahead of the point where the
#    banner must already be in request.chat_template_kwargs.
V_CTK_READ = (
    "        chat_template_kwargs = self._effective_chat_template_kwargs(request)\n"
)

HINT_VARIANTS = (
    ("tokenizer-fetch", V_TOKENIZER, "after"),
    ("streaming-pair", V_STREAMING_PAIR, "before"),
    ("ctk-read", V_CTK_READ, "before"),
)

# ── Repair site ───────────────────────────────────────────────────────────
# Matched by HEAD LINE and then extended over the balanced parentheses of the
# statement, so upstream reflowing the arguments is not drift. On both live
# pins variant 1 reproduces the pre-2026-07-26 replacement byte-for-byte.
R_WRAPPED = "        return await self._with_kv_transfer_rejection_cleanup(\n"
# The shape `create_chat_completion` had before upstream added the kv-transfer
# cleanup wrapper, and the shape it returns to if that wrapper is ever dropped.
R_DIRECT = "        return await self._create_chat_completion(request, raw_request)\n"

REPAIR_VARIANTS = (
    ("cleanup-wrapper", R_WRAPPED),
    ("direct-call", R_DIRECT),
)

REPAIR_TAIL = (
    "        # PN101b: answer-rescue post-pass (non-streaming, bounded-envelope,\n"
    "        # finish=length only; fail-open; default-OFF master flag).\n"
    "        try:\n"
    "            from vllm._genesis.middleware.answer_rescue import (\n"
    "                maybe_rescue_answer as _pn101_rescue,\n"
    "            )\n"
    "            _pn101_result = await _pn101_rescue(self, request, _pn101_result)\n"
    "        except Exception:\n"
    "            import logging as _pn101b_logging\n"
    "            _pn101b_logging.getLogger(\n"
    "                'genesis.middleware.answer_rescue'\n"
    "            ).debug('PN101 rescue raised; ignored', exc_info=True)\n"
    "        return _pn101_result\n"
)

# Window sniffs. PN100 assigns the thinking budget the hint gates on; the
# effective-kwargs read consumes the banner the hint writes.
PN100_CALL = "await _pn100_apply_hook(self, request)"
CTK_READ_CALL = "chat_template_kwargs = self._effective_chat_template_kwargs(request)"


def _shout(lines: list[str]) -> None:
    """PN101 sits on the path of every chat request. A soft skip must be unmissable."""
    bar = "=" * 72
    print(bar, file=sys.stderr)
    for ln in lines:
        print(ln, file=sys.stderr)
    print(bar, file=sys.stderr)
    logging.getLogger("vllm.pn101").error(" | ".join(lines))


def _stmt_end(src: str, start: int) -> int:
    """Index just past the newline ending the statement that begins at `start`.

    Walks parenthesis depth so a multi-line call is taken whole, and skips
    string literals and `#` comments so a bracket inside either cannot
    unbalance the scan. Returns -1 if the statement never closes (a truncated
    file) — which resolve_repair() reports as drift rather than guessing.
    """
    depth = 0
    i = start
    n = len(src)
    while i < n:
        c = src[i]
        if c in "\"'":
            quote = src[i:i + 3] if src[i:i + 3] in ('"""', "'''") else c
            i += len(quote)
            while i < n:
                if src[i] == "\\":
                    i += 2
                    continue
                if src.startswith(quote, i):
                    i += len(quote)
                    break
                i += 1
            continue
        if c == "#":
            nl = src.find("\n", i)
            if nl == -1:
                return -1
            i = nl  # leave the newline for the depth check below
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "\n" and depth <= 0:
            return i + 1
        i += 1
    return -1


def _counts(src: str, variants) -> list[tuple[str, int]]:
    return [(v[0], src.count(v[1])) for v in variants]


def _pick(src: str, variants) -> tuple[str | None, int, list[str]]:
    """First variant matching EXACTLY once -> (name, index, problems). Never guesses.

    `index` is the insertion offset: the start of the anchor for mode "before"
    (the default) and the end of it for mode "after".
    """
    problems: list[str] = []
    for variant in variants:
        name, text = variant[0], variant[1]
        mode = variant[2] if len(variant) > 2 else "before"
        n = src.count(text)
        if n == 0:
            problems.append(f"{name}: absent")
        elif n > 1:
            problems.append(f"{name}: ambiguous ({n} hits, need exactly 1)")
        else:
            i = src.index(text)
            return name, (i + len(text) if mode == "after" else i), problems
    return None, -1, problems


def resolve_hint(src: str):
    """(variant name, insertion offset, counts, problems) for the hint site."""
    name, idx, problems = _pick(src, HINT_VARIANTS)
    return name, idx, _counts(src, HINT_VARIANTS), problems


def resolve_repair(src: str):
    """(variant name, (start, end) of the return statement, counts, problems)."""
    name, idx, problems = _pick(src, REPAIR_VARIANTS)
    if name is None:
        return None, None, _counts(src, REPAIR_VARIANTS), problems
    end = _stmt_end(src, idx)
    if end == -1:
        problems.append(f"{name}: return statement never closes its parentheses")
        return None, None, _counts(src, REPAIR_VARIANTS), problems
    return name, (idx, end), _counts(src, REPAIR_VARIANTS), problems


def _repair_replacement(stmt: str) -> str:
    """Rewrite `return await <call>` as `_pn101_result = await <call>` + hook."""
    indent = stmt[:len(stmt) - len(stmt.lstrip(" "))]
    body = stmt.replace("return ", "_pn101_result = ", 1)
    if not body.endswith("\n"):
        body += "\n"
    tail = REPAIR_TAIL
    if indent != "        ":
        tail = "".join(
            (indent + ln[8:]) if ln.startswith("        ") else ln
            for ln in REPAIR_TAIL.splitlines(keepends=True)
        )
    return body + tail


def _check_window(src: str, off: int) -> list[str]:
    """Warnings if the hint offset falls outside its legal window."""
    out = []
    at = src.find(PN100_CALL)
    if at != -1 and off < at:
        out.append(
            "lands BEFORE PN100's await hook — request.thinking_token_budget is "
            "still 0 there, so maybe_add_answer_hint() returns early and the "
            "PN102 banner is never added")
    ctk = src.find(CTK_READ_CALL)
    if ctk != -1 and off > ctk:
        out.append(
            "lands AFTER the effective chat_template_kwargs read — the banner "
            "would be written too late to reach the rendered prompt")
    return out


def build(src: str):
    """(patched source, applied, missing, warnings). Pure — writes nothing.

    `missing` names sites with no usable anchor (that capability is inert);
    `warnings` names sites that WERE installed but at a suspect offset.
    """
    applied: list[str] = []
    missing: list[str] = []
    warnings: list[str] = []
    edits: list[tuple[int, int, str]] = []

    if HINT_MARKER in src:
        applied.append("hint already present")
    else:
        name, off, counts, probs = resolve_hint(src)
        cline = " ".join(f"{n}={c}" for n, c in counts)
        if name is None:
            missing.append(f"hint: no variant resolved ({cline}); " + "; ".join(probs))
        else:
            warnings += [f"hint[{name}]: {w}" for w in _check_window(src, off)]
            edits.append((off, off, HINT_BLOCK))
            applied.append(f"hint via '{name}' ({cline})")

    if REPAIR_MARKER in src:
        applied.append("repair already present")
    else:
        name, span, counts, probs = resolve_repair(src)
        cline = " ".join(f"{n}={c}" for n, c in counts)
        if name is None:
            missing.append(f"repair: no variant resolved ({cline}); " + "; ".join(probs))
        else:
            start, end = span
            edits.append((start, end, _repair_replacement(src[start:end])))
            applied.append(f"repair via '{name}' ({cline})")

    patched = src
    for start, end, text in sorted(edits, key=lambda e: e[0], reverse=True):
        patched = patched[:start] + text + patched[end:]
    return patched, applied, missing, warnings


HINT_DEAD = (
    "  The bounded-envelope answer-first HINT has no call site this boot:",
    "  GENESIS_ENABLE_PN102_CONTRACT is inert, no request gets the reply-window",
    "  banner, and patch_pn114_seed_span.py's S4 site (which anchors on this",
    "  block) will refuse too. Serving is UNHARMED — answers are just unshaped.",
)
REPAIR_DEAD = (
    "  The answer-RESCUE post-pass has no call site this boot: a bounded",
    "  request that comes back finish_reason=length with no parseable answer is",
    "  returned truncated instead of being completed by the 16-token",
    "  continuation. GENESIS_ENABLE_PN101_ANSWER_RESCUE's repair leg is inert.",
)


def main() -> int:
    if not TARGET.exists():
        _shout([
            f"{LOG} NOT APPLIED: {TARGET} absent on this pin.",
            *HINT_DEAD, *REPAIR_DEAD,
            "  Re-derive the target path.",
        ])
        return 0

    src = TARGET.read_text(encoding="utf-8")
    if HINT_MARKER in src and REPAIR_MARKER in src:
        print(f"{LOG} already applied — skipping")
        return 0

    patched, applied, missing, warnings = build(src)

    if patched != src:
        try:
            compile(patched, str(TARGET), "exec")
        except SyntaxError as e:
            _shout([
                f"{LOG} NOT APPLIED: the insertion would not compile: {e}",
                "  serving.py left byte-identical — nothing was written.",
                *HINT_DEAD, *REPAIR_DEAD,
                "  Re-anchor with: python3 fixes/verify_pn101_anchors.py",
            ])
            return 0
        TARGET.write_text(patched, encoding="utf-8")

    if missing:
        dead: list[str] = []
        if any(m.startswith("hint") for m in missing):
            dead += list(HINT_DEAD)
        if any(m.startswith("repair") for m in missing):
            dead += list(REPAIR_DEAD)
        _shout([
            f"{LOG} ERROR: {len(missing)} of 2 call sites NOT INSTALLED "
            f"— serving.py drifted.",
            *[f"  {m}" for m in missing],
            *dead,
            f"  Installed: {', '.join(applied) if applied else 'nothing'}.",
            "  Boot continues; re-anchor with: "
            "python3 fixes/verify_pn101_anchors.py",
        ])
        return 0

    if warnings:
        _shout([
            f"{LOG} INSTALLED AT A SUSPECT OFFSET — the call site exists but "
            f"may never fire.",
            *[f"  {w}" for w in warnings],
            "  Ordering is wrong, not broken: serving is unharmed, the feature",
            "  is silently degraded. Re-anchor with: "
            "python3 fixes/verify_pn101_anchors.py",
        ])

    print(f"{LOG} applied: {'; '.join(applied)} (master flag default OFF)")
    return 0


if __name__ == "__main__":
    # Importable: fixes/verify_pn101_anchors.py reuses build()/resolve_*()
    # against a replayed boot file. A bare `sys.exit(main())` at module level
    # would patch the verifier's own host filesystem on import.
    sys.exit(main())
