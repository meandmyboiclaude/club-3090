#!/usr/bin/env python3
"""PN114 boot-time token-id derivation (2026-07-23).

The holder (EngineCore process) has no tokenizer; PN114's forced spans need
exact token ids for the probe / newline / wrap-up strings. This runs in the
compose boot sequence (before serve), tokenizes with the served model's
tokenizer, and writes /tmp/genesis_pn114_ids.json for pn114.py to read.

Fail-open: on any error the ids file is absent and PN114 logs itself disabled;
serving is never blocked. Skips fast when no PN114/PN112_WRAPUP/CONFIRM flag
is set (no tokenizer load cost on ordinary boots).
"""
import json
import os
import sys

OUT = "/tmp/genesis_pn114_ids.json"
MODEL = os.environ.get("GENESIS_PN114_TOKENIZER_PATH",
                       "/root/.cache/huggingface/thinkingcap-gptq-pro-v2")
PROBE_STR = "\nMy current answer: "
WRAPUP_STR = "\nConsidering the limited time, I'll provide the final answer now.\n"
THINK_END = "</think>"


def _on(name):
    return (os.environ.get(name, "").strip().lower()
            in ("1", "true", "yes", "on"))


def main() -> int:
    try:
        ppen_on = float(os.environ.get("GENESIS_PPEN_LAMBDA", "0") or 0) > 0
    except ValueError:
        ppen_on = False
    if not (_on("GENESIS_ENABLE_PN114_PROBE") or _on("GENESIS_PN112_WRAPUP")
            or _on("GENESIS_PN112_CONFIRM")
            or _on("GENESIS_PN112_WRAPUP_AT_CAP") or ppen_on):
        print("[pn114_boot_ids] no PN114-family/P-pen flag set — skipping",
              flush=True)
        return 0
    try:
        from transformers import AutoTokenizer
        # local quant (never on the hub) — hard-forbid any hub lookup
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True,
                                            local_files_only=True)
        probe = tok.encode(PROBE_STR, add_special_tokens=False)
        newline = tok.encode("\n", add_special_tokens=False)
        wrap = tok.encode(WRAPUP_STR, add_special_tokens=False)
        end = tok.encode(THINK_END, add_special_tokens=False)
        # P-pen hesitation markers (single-token only, leading-space + bare +
        # capitalized variants; 'so'/'So' excluded per pre-registration —
        # too load-bearing in math prose). Bank = our measured thrash regex
        # (doom-sep, AUC 0.79-0.95) + FAIR-cited examples.
        words = ["Wait", "wait", "But", "but", "However", "however",
                 "Alternatively", "alternatively", "Hmm", "hmm",
                 "Actually", "actually", "Instead", "instead",
                 "Reconsider", "reconsider", "Alternately"]
        ppen: list[int] = []
        for w in words:
            for form in (w, " " + w):
                t = tok.encode(form, add_special_tokens=False)
                if len(t) == 1 and t[0] not in ppen:
                    ppen.append(t[0])
        ids = {"probe": probe, "newline": newline,
               "wrapup_close": wrap + end, "ppen": ppen}
        with open(OUT, "w") as f:
            json.dump(ids, f)
        print(f"[pn114_boot_ids] wrote {OUT}: probe={len(probe)} "
              f"newline={len(newline)} wrapup_close={len(ids['wrapup_close'])}",
              flush=True)
        return 0
    except Exception as exc:  # fail-open
        print(f"[pn114_boot_ids] WARN: {type(exc).__name__}: {exc} — "
              f"PN114 will be inert", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
