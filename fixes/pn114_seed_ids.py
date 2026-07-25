#!/usr/bin/env python3
"""PN114-SEED boot-time token-id derivation (2026-07-25).

The PN102 "think-seed" (``Budget: ~N short steps.\\nStep 1:``) is rendered by
the chat template INSIDE ``<think>`` (chat_template_v2json.jinja:147-151), so
it is the one half of the PN102 treatment that can be moved out of the prompt
and into forced OUTPUT without changing either its position or its voice.
This script is the tokenizer half of that move: the EngineCore process has no
tokenizer, so every seed the forcer may ever need is tokenized HERE, at boot,
and written to ``/tmp/genesis_pn114_seed_ids.json``.

THE INVARIANT THIS FILE ENFORCES (it does not assume it)
--------------------------------------------------------
For a seed ``S`` to be forceable it must be true that

    encode(BASE + S) == encode(BASE) + encode(S)

where ``BASE`` is the prompt tail the template emits immediately before the
seed (``<|im_start|>assistant\\n<think>\\n``). If BPE merged across that
boundary, the forced span would land DIFFERENT ids at the same positions and
the whole equivalence claim would be false. Every candidate seed is checked
here and a seed that fails is REJECTED — it never enters the table, so both
the API-side strip and the engine-side arm decline it and the request keeps
its prompt-rendered seed. Fail-closed, symmetric on both sides, by
construction.

(Measured on the boot pin dev1474cherrymax-1757-20260725 with the
thinkingcap-gptq-pro-v2 tokenizer: ``<think>`` is a real special token
[248068] followed by ``\\n`` [198], so the seed text is a tokenizer chunk of
its own and every candidate splits exactly. The check stays because a future
pin/tokenizer is not obliged to keep that property.)

Fail-open: on ANY error the table is absent, PN114-SEED reports itself
disabled, and serving is untouched. Skips fast (no tokenizer load) when
GENESIS_ENABLE_PN114_SEED_SPAN is not set.
"""
import json
import os
import sys

OUT = "/tmp/genesis_pn114_seed_ids.json"
MODEL = os.environ.get("GENESIS_PN114_TOKENIZER_PATH",
                       "/root/.cache/huggingface/thinkingcap-gptq-pro-v2")

# The prompt tail the chat template emits right before pn_env_seed. Keep this
# byte-identical to chat_template_v2json.jinja:144-151.
BASE = "<|im_start|>assistant\n<think>\n"
THINK_END = "</think>"

# Every seed string reachable from answer_rescue.maybe_add_answer_hint's
# dispatch chain (_genesis/middleware/answer_rescue.py):
#   v3 sized      -> f"{label}: ~{steps} short steps.\nStep 1:"          (:468)
#   v3 + B2-S1    -> f"{label}: ~{steps} short steps.\nStep 1 — what ..."(:465)
#   v5            -> f"Budget: ~{steps} short steps.\nStep 1:"           (:528)
#   v4 / v8 / v6  -> "Step 1:"                              (:231/:273/:557)
# label is "Budget" or "Plan" (:411/:423).
SEED_LABELS = ("Budget", "Plan")
ECHO_TAIL = "Step 1 — what exactly is being asked:"
PLAIN_TAIL = "Step 1:"


def _on(name: str) -> bool:
    return (os.environ.get(name, "").strip().lower()
            in ("1", "true", "yes", "on"))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def candidates(max_n: int) -> list[tuple[str, str, int, str]]:
    """(text, label, steps, tail) for every seed the dispatch can produce."""
    out: list[tuple[str, str, int, str]] = [(PLAIN_TAIL, "", 0, "plain")]
    for label in SEED_LABELS:
        for n in range(1, max_n + 1):
            head = f"{label}: ~{n} short steps.\n"
            out.append((head + PLAIN_TAIL, label, n, "plain"))
            out.append((head + ECHO_TAIL, label, n, "echo"))
    return out


def main() -> int:
    if not _on("GENESIS_ENABLE_PN114_SEED_SPAN"):
        print("[pn114_seed_ids] GENESIS_ENABLE_PN114_SEED_SPAN not set — "
              "skipping (no tokenizer load)", flush=True)
        return 0
    max_n = max(1, _env_int("GENESIS_PN114_SEED_MAX_N", 64))
    try:
        from transformers import AutoTokenizer
        # local quant (never on the hub) — hard-forbid any hub lookup
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True,
                                            local_files_only=True)
        base_ids = tok.encode(BASE, add_special_tokens=False)
        end_ids = tok.encode(THINK_END, add_special_tokens=False)
        if not base_ids or not end_ids:
            print("[pn114_seed_ids] WARN: empty BASE/think-end encoding — "
                  "refusing to write a table", flush=True)
            return 0
        by_text: dict[str, list[int]] = {}
        by_steps: dict[str, str] = {}   # "label|tail|n" -> text
        rejected: list[str] = []
        for text, label, steps, tail in candidates(max_n):
            ids = tok.encode(text, add_special_tokens=False)
            joint = tok.encode(BASE + text, add_special_tokens=False)
            if not ids or joint != base_ids + ids:
                # BPE merged across the <think>\n boundary: forcing this span
                # would NOT reproduce the prompt-rendered ids. Drop it.
                rejected.append(text)
                continue
            if any(e in ids for e in end_ids):
                # A seed containing </think> would close the block from inside
                # the forced span. Structurally impossible for these strings,
                # but never ship a forcer that can emit its own terminator.
                rejected.append(text)
                continue
            by_text[text] = ids
            if steps:
                by_steps[f"{label}|{tail}|{steps}"] = text
        if not by_text:
            print("[pn114_seed_ids] WARN: no candidate seed survived the "
                  "split-equivalence check — PN114-SEED stays inert",
                  flush=True)
            return 0
        table = {
            "version": 1,
            "base": base_ids,
            "base_text": BASE,
            "think_end": end_ids,
            "by_text": by_text,
            "by_steps": by_steps,
            "max_n": max_n,
            "rejected": len(rejected),
        }
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(table, f)
        print(f"[pn114_seed_ids] wrote {OUT}: seeds={len(by_text)} "
              f"rejected={len(rejected)} base={len(base_ids)}tok "
              f"max_n={max_n}", flush=True)
        if rejected:
            print(f"[pn114_seed_ids] NOTE: {len(rejected)} seed(s) failed the "
                  f"split check (e.g. {rejected[0]!r}) — those requests keep "
                  f"their prompt-rendered seed", flush=True)
        return 0
    except Exception as exc:  # fail-open — a boot is never blocked by this
        print(f"[pn114_seed_ids] WARN: {type(exc).__name__}: {exc} — "
              f"PN114-SEED will be inert", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
