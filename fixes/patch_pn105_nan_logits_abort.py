#!/usr/bin/env python3
"""PN105 — abort requests whose logits contain NaNs (BUG-076 damage isolation).

Upstream ships NaN detection (VLLM_COMPUTE_NANS_IN_LOGITS=1 -> per-request NaN
counts on ModelRunnerOutput) but only REPORTS the count on EngineCoreOutput.
A NaN logits row otherwise still samples (argmax over all-NaN = token id 0 ->
"!" completions, grammar FSM rejections, silently-corrupt text for
unconstrained requests) and the request keeps running on poisoned GDN state.

PN105 turns detection into action: when a request's step produced NaN logits,
terminate it with FINISHED_ERROR BEFORE the stop-finalization block (so
finish_reason capture, _handle_stopped_request and request freeing all run
normally) — the caller gets a clean error to retry instead of garbage, and the
poisoned per-request state is freed instead of feeding later steps.

Inert unless VLLM_COMPUTE_NANS_IN_LOGITS=1 (counts are None otherwise).
Anchor: the `finish_reason = None` stop-finalization header in
Scheduler.update_from_output (the nan dict is in scope from the function
prologue; the storage site further down remains upstream-untouched).
"""
import pathlib
import sys

LOG = "[pn105-nan-logits-abort]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)
MARKER = "# PN105:"

SITE_OLD = (
    "            finish_reason = None\n"
    "            if stopped:\n"
    "                # Capture finish_reason BEFORE _handle_stopped_request, which may\n"
)

SITE_NEW = (
    "            # PN105: NaN logits mean this request's forward pass is\n"
    "            # corrupt (BUG-076: async x MTP x hybrid-GDN state skew).\n"
    "            # Upstream only reports the count; sampling from an all-NaN\n"
    '            # row yields token id 0 ("!" completions / grammar\n'
    "            # rejections / silent garbage) and the poisoned state feeds\n"
    "            # every later step. Terminate the request instead — callers\n"
    "            # get a clean error to retry. Runs BEFORE the finalization\n"
    "            # block below so the normal stop path handles it.\n"
    "            if (\n"
    "                not stopped\n"
    "                and num_nans_in_logits is not None\n"
    "                and num_nans_in_logits.get(req_id)\n"
    "            ):\n"
    "                logger.error(\n"
    '                    "PN105: %d NaN logits for request %s — terminating "\n'
    '                    "request (forward-pass corruption).",\n'
    "                    num_nans_in_logits[req_id],\n"
    "                    req_id,\n"
    "                )\n"
    "                request.status = RequestStatus.FINISHED_ERROR\n"
    "                request.resumable = False\n"
    "                stopped = True\n"
    "                new_token_ids = []\n"
    "\n"
    "            finish_reason = None\n"
    "            if stopped:\n"
    "                # Capture finish_reason BEFORE _handle_stopped_request, which may\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"{LOG} already applied — skip")
        return 0
    n = src.count(SITE_OLD)
    if n != 1:
        print(f"{LOG} FATAL: anchor hits={n} — upstream drifted", file=sys.stderr)
        return 1
    TARGET.write_text(src.replace(SITE_OLD, SITE_NEW, 1), encoding="utf-8")
    print(f"{LOG} applied: NaN-logits requests now abort cleanly pre-finalization")
    return 0


sys.exit(main())
