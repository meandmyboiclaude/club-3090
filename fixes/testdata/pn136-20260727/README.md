# PN136 applier fixtures — copies of the REAL in-container files

Extracted read-only 2026-07-27 from the running `vllm-tcbench-8021` container
(image `localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725`) via its
overlay merged dir — no exec, no restart, no write to the container.

    chat_utils.py   <- /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/chat_utils.py
    qwen3.py        <- /usr/local/lib/python3.12/dist-packages/vllm/parser/qwen3.py

`test_pn136_inbound_neutralize.py` dry-runs the applier against a temp COPY of
these, never against the live tree. Refresh them when the image pin moves.
