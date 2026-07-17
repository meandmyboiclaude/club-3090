#!/usr/bin/env python3
"""PN100 — call-site for the auto thinking-budget router (house-original).

Inserts an awaited hook right after the PN16 lazy-reasoner block in
_create_chat_completion. The router itself lives in the mounted Genesis tree
(vllm/_genesis/middleware/auto_budget.py): one thinking-off self-call rates
the request 0-3 and sets thinking_token_budget per tier — the Qwen-side twin
of the Claude Code effort router. PN16 stays the sync heuristic pre-pass
(its variant-1 trivial-prompt OFF decisions short-circuit PN100 for free);
PN100 needs its own call site because PN16's hook is sync and the classify
pass must await the engine.

Fail-open: the inserted block swallows every exception — a router failure
serves the request exactly as today. Gate: GENESIS_ENABLE_PN100_AUTO_BUDGET.
Self-retires if upstream ever ships a native per-request auto budget.
"""
import pathlib
import sys

LOG = "[pn100-auto-thinking-budget]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/"
    "chat_completion/serving.py"
)
MARKER = "# PN100:"

ANCHOR = (
    "        # [Genesis PN16 lazy-reasoner] Per-request decision on whether\n"
)
REPLACEMENT = (
    "        # PN100: auto thinking-budget router (house variant — see\n"
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
    "        # [Genesis PN16 lazy-reasoner] Per-request decision on whether\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"{LOG} already applied — skipping")
        return 0
    if "auto thinking-budget" in src:
        print(f"{LOG} upstream ships an auto budget — patch self-retired, drop PN100")
        return 0
    n = src.count(ANCHOR)
    if n == 0:
        print(
            f"{LOG} FATAL: anchor-not-found — PN16 block drifted; re-derive "
            "anchors from serving.py",
            file=sys.stderr,
        )
        return 1
    if n > 1:
        print(f"{LOG} FATAL: ambiguous anchor ({n} hits)", file=sys.stderr)
        return 1
    TARGET.write_text(src.replace(ANCHOR, REPLACEMENT, 1), encoding="utf-8")
    import py_compile

    py_compile.compile(str(TARGET), doraise=True)
    print(f"{LOG} applied: await _pn100_apply_hook after PN16 block")
    return 0


sys.exit(main())
