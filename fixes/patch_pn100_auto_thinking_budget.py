#!/usr/bin/env python3
"""PN100 — call-site for the auto thinking-budget router (house-original).

Inserts an awaited hook at the TOP of `_create_chat_completion`, ahead of the
PN16 lazy-reasoner block when that block is present. The router itself lives in
the mounted Genesis tree (vllm/_genesis/middleware/auto_budget.py): one
thinking-off self-call rates the request 0-3 and sets thinking_token_budget per
tier — the Qwen-side twin of the Claude Code effort router. PN16 stays the sync
heuristic pre-pass (its variant-1 trivial-prompt OFF decisions short-circuit
PN100 for free); PN100 needs its own call site because PN16's hook is sync and
the classify pass must await the engine.

Fail-open at RUNTIME: the inserted block swallows every exception — a router
failure serves the request exactly as today. Gate:
GENESIS_ENABLE_PN100_AUTO_BUDGET. Self-retires if upstream ever ships a native
per-request auto budget.

Fail-open at BOOT (2026-07-26, BUG-141). This patcher used to anchor on ONE
string — the `# [Genesis PN16 lazy-reasoner]` comment that genesis apply_all
writes into serving.py — and returned 1 ("FATAL: PN16 block drifted") when it
was absent. The entrypoint runs under `set -e`, so that exit code takes the
whole boot down. And the anchor is not a property of the image at all: PN16 is
OPT-IN env (`GENESIS_ENABLE_PN16_LAZY_REASONER=1`, set in the compose). Flip
that one variable to 0 — a legitimate A/B — and apply_all skips PN16, this
patch finds nothing, and the engine never starts. Measured 2026-07-26 by
replaying apply_all + the boot entrypoint prefix on both live pins with the
lane off (fixes/verify_pn100_anchors.py, PN16-OFF arm).

Two changes fix the class, not the instance:

  1. A COUNTED VARIANT SET, most-guaranteed first. The primary anchor is now
     upstream's own `_create_chat_completion` def header, which is present in
     the PRISTINE image on both live pins and is touched by nothing between
     `apply_all` and this patcher's line in the entrypoint (verified: the only
     writers to serving.py before PN100 are genesis PN16/P89/PN288/P107 — no
     /fixes sibling touches the file). The PN16 comment is kept as variant 2 so
     the byte-for-byte historical insertion point still wins when it IS there,
     and upstream's `# Streaming response` head as variant 3.
  2. NEVER a non-zero exit. Unresolvable anchors, a compile-check failure or a
     missing target all _shout() and return 0. A missing optimisation is not
     worth a dead engine — but the shout has to be loud enough that nobody
     mistakes it for a clean boot, because a silently-skipped patch shipped a
     no-op here on 2026-07-25.

Anchor counts as the BOOT sees them: python3 fixes/verify_pn100_anchors.py
"""
import logging
import pathlib
import sys

LOG = "[pn100-auto-thinking-budget]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/"
    "chat_completion/serving.py"
)
MARKER = "# PN100:"

# The awaited hook, indented for a method body. Inserted verbatim; the anchor
# only decides WHERE. MARKER must be the first line so re-runs are idempotent.
BLOCK = (
    # NB the wording deliberately avoids the UPSTREAM_ABSORBED sniff string
    # below — otherwise this patcher's own output reads as upstream absorption
    # to anything that checks the sniff before the marker.
    "        # PN100: automatic reasoning-budget router (house variant — see\n"
    "        # _genesis/middleware/auto_budget.py). MUST run BEFORE PN16: the\n"
    "        # LLM classifier owns the thinking on/off + budget decision; its\n"
    "        # explicit enable_thinking result makes PN16's regex variant-1\n"
    "        # defer via variant-3 (respect-explicit). PN16's heuristic would\n"
    "        # otherwise kill thinking on short-but-hard prompts (validated:\n"
    "        # knight-path prompt, 2026-07-18). Async — needs its own await\n"
    "        # site; PN16's sync hook can't run the classify pass. Fail-open.\n"
    "        try:\n"
    "            from vllm._genesis.middleware.auto_budget import (\n"
    "                apply_hook_async as _pn100_apply_hook,\n"
    "            )\n"
    "            await _pn100_apply_hook(self, request)\n"
    "        except Exception:\n"
    "            import logging as _pn100_logging\n"
    "            _pn100_logging.getLogger(\n"
    "                'genesis.middleware.auto_budget'\n"
    "            ).debug('PN100 hook raised; ignored', exc_info=True)\n"
)

# ── Anchor variants, most-guaranteed first ────────────────────────────────
# mode "after"  -> insert BLOCK immediately after the anchor text
# mode "before" -> insert BLOCK immediately before the anchor text
# Both land at the top of the `_create_chat_completion` body; variants 1 and 2
# resolve to the SAME offset whenever PN16 is applied.

# 1. Upstream's def header. Byte-identical in the pristine image on both live
#    pins (dev1474cherrymax-1757 / dev1474cherry-1711, 2026-07-26) and
#    unmodified by every patcher that runs before this one.
V_FN_HEAD = (
    "    async def _create_chat_completion(\n"
    "        self,\n"
    "        request: ChatCompletionRequest,\n"
    "        raw_request: Request | None = None,\n"
    "    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:\n"
)
# 2. The historical anchor: genesis PN16's block. Only exists when the PN16
#    lane actually ran (opt-in env), which is exactly why it can't be the only
#    one — but when it is there it reproduces the pre-2026-07-26 insertion
#    point byte-for-byte.
V_PN16 = "        # [Genesis PN16 lazy-reasoner] Per-request decision on whether\n"
# 3. Upstream's first two body statements, in case the signature is
#    re-annotated (return-type churn is the likeliest drift for variant 1).
V_STREAMING = (
    "        # Streaming response\n"
    "        tokenizer = self.renderer.tokenizer\n"
)

VARIANTS = (
    ("fn-head", V_FN_HEAD, "after"),
    ("pn16-block", V_PN16, "before"),
    ("streaming-head", V_STREAMING, "before"),
)

# Presence sniff for genesis PN16's hook CALL (not its comment): used to check
# the resolved offset really lands ahead of PN16, whichever variant won.
PN16_CALL = "_genesis_pN16_apply_hook(self, request)"

# Upstream self-retirement sniff, unchanged since 2026-07-18 and never yet
# fired: a prose string upstream would only carry if it shipped this feature
# itself. Checked strictly after MARKER, so the text this patcher inserts can
# never trip it. Kept as a courtesy retirement path, not as evidence about
# upstream — nothing here establishes that upstream has such a feature.
UPSTREAM_ABSORBED = "auto thinking-budget"


def _shout(lines: list[str]) -> None:
    """PN100 is on the path of every request. A soft skip must be unmissable."""
    bar = "=" * 72
    print(bar, file=sys.stderr)
    for ln in lines:
        print(ln, file=sys.stderr)
    print(bar, file=sys.stderr)
    logging.getLogger("vllm.pn100").error(" | ".join(lines))


def _skip_docstring(src: str, off: int) -> int:
    """Move past a function docstring if one follows the insertion point.

    `_create_chat_completion` has no docstring on either live pin, but if a
    future pin adds one, inserting BLOCK ahead of it would demote it to a bare
    string expression. Cheap to be right about.
    """
    i = off
    while i < len(src):
        line_end = src.find("\n", i)
        if line_end == -1:
            return off
        line = src[i:line_end]
        if not line.strip():
            i = line_end + 1
            continue
        quote_at = i + (len(line) - len(line.lstrip()))
        quote = src[quote_at:quote_at + 3]
        if quote not in ('"""', "'''"):
            return off
        end = src.find(quote, quote_at + 3)
        if end == -1:
            return off
        nl = src.find("\n", end + 3)
        return len(src) if nl == -1 else nl + 1
    return off


def resolve(src: str) -> tuple[str | None, int, list[tuple[str, int]], list[str]]:
    """Return (variant name, insertion offset, per-variant counts, problems).

    Counts every variant so callers can report the whole picture; picks the
    first that matches EXACTLY once. Never guesses on an ambiguous anchor.
    """
    counts = [(name, src.count(text)) for name, text, _mode in VARIANTS]
    problems: list[str] = []
    for name, text, mode in VARIANTS:
        n = src.count(text)
        if n == 0:
            problems.append(f"{name}: absent")
            continue
        if n > 1:
            problems.append(f"{name}: ambiguous ({n} hits, need exactly 1)")
            continue
        i = src.index(text)
        off = _skip_docstring(src, i + len(text)) if mode == "after" else i
        return name, off, counts, problems
    return None, -1, counts, problems


def main() -> int:
    if not TARGET.exists():
        _shout([
            f"{LOG} NOT APPLIED: {TARGET} absent on this pin.",
            "  The auto thinking-budget router has NO call site this boot —",
            "  GENESIS_ENABLE_PN100_AUTO_BUDGET is inert, every request serves",
            "  with the operator's static budget. Re-derive the target path.",
        ])
        return 0

    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"{LOG} already applied — skipping")
        return 0
    if UPSTREAM_ABSORBED in src:
        print(f"{LOG} upstream ships an auto budget "
              f"({UPSTREAM_ABSORBED!r} present) — patch self-retired, drop PN100")
        return 0

    name, off, counts, problems = resolve(src)
    count_line = " ".join(f"{n}={c}" for n, c in counts)
    if name is None:
        _shout([
            f"{LOG} NOT APPLIED: no anchor variant resolved ({count_line}).",
            *[f"  {p}" for p in problems],
            "  The auto thinking-budget router has NO call site this boot —",
            "  GENESIS_ENABLE_PN100_AUTO_BUDGET is inert and every request",
            "  serves with the operator's static budget. Serving is UNHARMED.",
            "  Re-anchor with: python3 fixes/verify_pn100_anchors.py",
        ])
        return 0

    patched = src[:off] + BLOCK + src[off:]
    try:
        compile(patched, str(TARGET), "exec")
    except SyntaxError as e:
        _shout([
            f"{LOG} NOT APPLIED: insertion via '{name}' would not compile: {e}",
            "  serving.py left byte-identical; PN100 is inert this boot.",
            "  Re-anchor with: python3 fixes/verify_pn100_anchors.py",
        ])
        return 0

    TARGET.write_text(patched, encoding="utf-8")

    note = ""
    pn16_at = src.find(PN16_CALL)
    if pn16_at != -1 and off > pn16_at:
        note = " [WARNING: lands AFTER genesis PN16's hook — PN16's variant-1 " \
               "may pre-empt the classifier on short prompts]"
        _shout([
            f"{LOG} applied via '{name}', but the offset is BEHIND genesis "
            f"PN16's hook call.",
            "  PN100 is supposed to decide first (PN16 then defers via",
            "  variant-3 respect-explicit). Ordering is wrong, not broken:",
            "  short-but-hard prompts can lose thinking. Re-anchor.",
        ])
    print(f"{LOG} applied: await _pn100_apply_hook at top of "
          f"_create_chat_completion via anchor '{name}' ({count_line}){note}")
    return 0


if __name__ == "__main__":
    # Importable: fixes/verify_pn100_anchors.py reuses resolve()/BLOCK against
    # a replayed boot file. A bare `sys.exit(main())` here would patch the
    # verifier's own host filesystem path on import.
    sys.exit(main())
