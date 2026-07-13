#!/usr/bin/env python3
"""PN85 — RETIRED 2026-07-13: vllm#46066 is in pin dev1060+ (373eb314).

Was: backport of vllm#46066 — gate the async-scheduling spec-decode
rejection accounting in vllm/v1/core/sched/scheduler.py::update_from_output
on `request.async_tokens_to_discard == 0`, so a stale spec frame still
pending discard is skipped (BUG-042 EngineCore fatal: "sample_tokens() must
be called after execute_model() returns None").

#46066 merged upstream inside our pin range (373eb314, present in
nightly-9e57de71 / dev1060 at scheduler.py::update_from_output — verified
line `and request.async_tokens_to_discard == 0`). This file is kept as a
self-retire no-op so entrypoints that list it keep working; it verifies the
upstream fix is actually present and exits 0 with a log line. It also
accepts an image previously patched by the pre-retirement PN85 (marker
"# PN85:"), e.g. a rollback to the validated dev799 image.

FAIL-LOUD: if neither the upstream guard nor a prior PN85 application is
found, exit 1 — that image would re-expose the BUG-042 engine crash
(prompt_logprobs x async-scheduling x spec-decode), and the pre-retirement
patcher from git history (commit d9e6652e) must be restored.
"""
import pathlib
import sys

LOG = "[pn85-async-spec-discard-guard]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)
MARKER = "# PN85:"
UPSTREAM_FIX = "and request.async_tokens_to_discard == 0"


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    text = TARGET.read_text()
    if UPSTREAM_FIX in text:
        print(f"{LOG} retired 2026-07-13: vllm#46066 in pin dev1060+ — "
              f"upstream discard guard verified present (no-op)")
        return 0
    if MARKER in text:
        print(f"{LOG} retired 2026-07-13: pre-retirement PN85 patch already "
              f"applied in this image (no-op)")
        return 0
    print(f"{LOG} FATAL: upstream #46066 discard guard ABSENT from this image "
          f"and no prior PN85 application found — BUG-042 engine crash "
          f"(prompt_logprobs x async-scheduling x spec-decode) is live here. "
          f"Restore the pre-retirement PN85 patcher from git history.",
          file=sys.stderr)
    return 1


sys.exit(main())
