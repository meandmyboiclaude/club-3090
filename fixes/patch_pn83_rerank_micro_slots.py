#!/usr/bin/env python3
"""PN83 — rerank micro-slots: give rerank requests their own seats.

USER 2026-07-07: "give rerank its own slots" (ik_llama rerank-slots precedent).

Problem: prefill is token-budget scheduled (max_num_batched_tokens=4128/step)
but seat-capped at max_num_seqs. Rerank items are ~150-600 tokens, so at 5
seats a step packs only ~1-3K rerank tokens — the budget runs 30% full and a
136-doc rerank takes ~8s despite an idle GPU.

Design (inversion keeps upstream sizing machinery correct): the compose RAISES
--max-num-seqs (5 → 29) so the model runner sizes input batch, GDN/mamba state
cache, and cudagraph dummies for 29 seats; THIS patch caps CHAT-class
admission at GENESIS_PN83_CHAT_SEQS (5, the proven KV-limited chat ceiling)
inside the v1 scheduler. generative-scoring requests (PN81 rerank: tiny
prefill + 1-token decode, request_id prefix "generative-scoring-") fill the
remaining seats. FCFS order is preserved — an over-cap chat request at the
head of the queue still ends admission for the step (no queue-jumping/chat
starvation).

GENESIS_PN83_CHAT_SEQS=0 (or env absent) = patch inert → pure upstream
behavior at whatever --max-num-seqs says. FAIL-LOUD on anchor drift: without
the cap, 29 all-class seats would let 29 concurrent chats thrash the KV pool
(the original reason max_num_seqs was 5).
"""
import pathlib
import sys

LOG = "[pn83-rerank-micro-slots]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)
MARKER = "# PN83:"

HDR_OLD = "logger = init_logger(__name__)\n"
HDR_NEW = (
    "logger = init_logger(__name__)\n"
    "\n"
    "# PN83: chat-class seat cap (rerank micro-slots). 0/absent = inert.\n"
    "import os as _pn83_os\n"
    "\n"
    '_PN83_CHAT_SEQS = int(_pn83_os.environ.get("GENESIS_PN83_CHAT_SEQS", "0") or 0)\n'
)

ADMIT_OLD = (
    "                num_running = len(self.running) + self.num_waiting_for_streaming_input\n"
    "                if num_running >= self.max_num_running_reqs:\n"
    "                    break\n"
    "\n"
    "                request_queue = self._select_waiting_queue_for_scheduling()\n"
    "                assert request_queue is not None\n"
    "\n"
    "                request = request_queue.peek_request()\n"
    "                request_id = request.request_id\n"
)
ADMIT_NEW = (
    "                num_running = len(self.running) + self.num_waiting_for_streaming_input\n"
    "                if num_running >= self.max_num_running_reqs:\n"
    "                    break\n"
    "\n"
    "                request_queue = self._select_waiting_queue_for_scheduling()\n"
    "                assert request_queue is not None\n"
    "\n"
    "                request = request_queue.peek_request()\n"
    "                request_id = request.request_id\n"
    "\n"
    "                # PN83: rerank micro-slots — CHAT-class requests stay capped at\n"
    "                # _PN83_CHAT_SEQS (the proven KV-limited chat ceiling) while\n"
    "                # generative-scoring requests fill the raised max_num_seqs.\n"
    "                # FCFS preserved: an over-cap chat at the queue head ends the\n"
    "                # admission loop for this step.\n"
    "                if _PN83_CHAT_SEQS > 0 and not request_id.startswith(\n"
    "                    \"generative-scoring-\"\n"
    "                ):\n"
    "                    _pn83_chat_running = (\n"
    "                        sum(\n"
    "                            1\n"
    "                            for _pn83_r in self.running\n"
    "                            if not _pn83_r.request_id.startswith(\n"
    "                                \"generative-scoring-\"\n"
    "                            )\n"
    "                        )\n"
    "                        + self.num_waiting_for_streaming_input\n"
    "                    )\n"
    "                    if _pn83_chat_running >= _PN83_CHAT_SEQS:\n"
    "                        break\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    text = TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} already applied (idempotent)")
        return 0
    for name, anchor in (("header", HDR_OLD), ("admission", ADMIT_OLD)):
        if anchor not in text:
            print(f"{LOG} FATAL: anchor-not-found ({name}) — scheduler refactored. "
                  f"Do NOT boot with raised --max-num-seqs until re-derived "
                  f"(chat would get all seats and thrash KV).", file=sys.stderr)
            return 1
        if text.count(anchor) != 1:
            print(f"{LOG} FATAL: ambiguous anchor ({name})", file=sys.stderr)
            return 1
    text = text.replace(HDR_OLD, HDR_NEW, 1).replace(ADMIT_OLD, ADMIT_NEW, 1)
    TARGET.write_text(text)
    print(f"{LOG} applied: chat-class admission cap wired "
          f"(GENESIS_PN83_CHAT_SEQS gates; rerank fills remaining seats)")
    return 0


sys.exit(main())
