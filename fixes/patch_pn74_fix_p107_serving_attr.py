"""PN74 — fix Genesis P107's stale attribute reference on the new base.

Genesis P107 (MTP truncation detector, vllm#41467) injects a hook into
chat_completion/serving.py's stream generator. Its injected code references
`self.reasoning_parser_cls`, which existed on OpenAIServingChat in the old vLLM
but on b4c80ec0 moved to `self.parser_cls.reasoning_parser_cls`. P107's anchor
still matches (so it applies), but at stream time the hook raises:
    AttributeError: 'OpenAIServingChat' object has no attribute 'reasoning_parser_cls'
(logged as "Error in chat completion stream generator"). Streaming still returns
content via the error handler, but it's a logged error — not acceptable.

Fix: after apply_all has injected P107, repoint its attribute access to the new
location. Only P107's `self.reasoning_parser_cls` matches (the base code already
uses `self.parser_cls.reasoning_parser_cls`, which is not a substring of the
search term), so the replace is precise.

Runs AFTER apply_all in the entrypoint. Idempotent; set -e safe; no-op if absent.
"""
import sys
import pathlib

LOG = "[pn74-fix-p107-attr]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py"
)
OLD = "self.reasoning_parser_cls"
NEW = "self.parser_cls.reasoning_parser_cls"


def main():
    if not TARGET.exists():
        print(f"{LOG} SKIP: serving.py not present; no-op.", file=sys.stderr)
        return
    text = TARGET.read_text()
    if OLD not in text:
        print(f"{LOG} no stale attr present (P107 not injected, or already fixed/re-anchored); no-op.")
        return
    n = text.count(OLD)
    TARGET.write_text(text.replace(OLD, NEW))
    print(f"{LOG} applied: repointed {n} P107 attribute ref(s) -> {NEW}")


main()
