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
# Parenthesis-constrained (2026-07-23 v2): after "answer: (" the next token
# is the letter itself — C then measures ANSWER uncertainty, not format
# spread (unconstrained probes read 4.9-7.0 even when settled; confounded).
PROBE_STR = "\nMy current answer: ("
WRAPUP_STR = "\nConsidering the limited time, I'll provide the final answer now.\n"
THINK_END = "</think>"
# PN121 soft-landing close (2026-07-26). Same transition sentence as the
# R1b wrap-up, but terminated with "</think>\n\n" per the P7 research doc §4
# ("newline first, then </think>", answer-mode scaffolding after it). Kept as
# a SEPARATE key so PN121 can ship without moving the existing wrapup_close
# bytes that GENESIS_PN112_WRAPUP / the killed WRAPUP_AT_CAP arm emit.
SOFTLAND_TAIL = "\n\n"
# Qwen3 tool-call opener — the implicit reasoning end of upstream #44676.
# PN121 refuses to inject anywhere after this appears in the think slice.
TOOL_CALL_STR = "<tool_call>"

# PN117 deep-band rescue arm texts (plan 2.3, s12-P17). First-person own-voice
# continuations injected as REAL tokens at a sentence boundary — NEVER a numeric
# banner, NEVER a bracketed note, NEVER anything resembling </think>. Tokenized
# WITH a leading space so they attach as a natural continuation after a
# sentence-end token. Arms 1-3 VERBATIM from plan 2.3; arm 5 = converge cue.
PN117_ARMS = {
    "arm1": " Well — I actually have plenty of room to work this properly, so "
            "let me slow down and go through the remaining cases carefully.",
    "arm2": " Well, let me check — I'm only about a fifth of the way through my "
            "budget, so there's plenty left; let me keep working the remaining "
            "cases.",
    "arm3": " Well, let me keep going…",
    "arm5": " Alright — I have a decent picture now; let me pull this together "
            "and settle on the answer that fits best.",
}
# Tokens that mark a clean sentence boundary (the previous landed token id must
# be one of these for PN117 to inject). Bare + newline-suffixed variants.
PN117_SENTENCE_STRS = (".", "\n", "?", "!", ".\n", "?\n", "!\n")


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
            or _on("GENESIS_PN112_WRAPUP_AT_CAP") or ppen_on
            or _on("GENESIS_ENABLE_PN117_RESCUE")
            or _on("GENESIS_ENABLE_PN121_SOFTLAND")):
        print("[pn114_boot_ids] no PN114-family/P-pen/PN117 flag set — "
              "skipping", flush=True)
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
        close_paren = tok.encode(")\n", add_special_tokens=False)
        ids = {"probe": probe, "newline": newline,
               "close_paren": close_paren,
               "wrapup_close": wrap + end, "ppen": ppen}
        # PN121 soft landing (only when enabled — the vocab scan below is the
        # one non-trivial cost in this script).
        if _on("GENESIS_ENABLE_PN121_SOFTLAND"):
            ids["softland_close"] = (
                wrap + end + tok.encode(SOFTLAND_TAIL, add_special_tokens=False)
            )
            ids["tool_call"] = tok.encode(TOOL_CALL_STR,
                                          add_special_tokens=False)
            # Boundary set: every id whose surface form ENDS in a newline.
            # "land at the next newline token" cannot be done with the single
            # "\n" id alone — Qwen merges ".\n", ")\n", ":\n", "\n\n" etc. into
            # single tokens, and those are exactly the sentence/paragraph
            # boundaries the Nemotron rule targets.
            nl_end: list[int] = []
            try:
                vocab_n = len(tok)
                for _tid in range(vocab_n):
                    _s = tok.convert_ids_to_tokens(_tid)
                    if _s is None:
                        continue
                    _d = tok.convert_tokens_to_string([_s])
                    if _d.endswith("\n"):
                        nl_end.append(_tid)
            except Exception as _exc:
                print(f"[pn114_boot_ids] WARN: newline scan failed "
                      f"({type(_exc).__name__}: {_exc}) — falling back to the "
                      f"bare \\n id", flush=True)
                nl_end = []
            for _t in newline:
                if _t not in nl_end:
                    nl_end.append(_t)
            ids["nl_end"] = nl_end
            print(f"[pn114_boot_ids] PN121: softland_close="
                  f"{len(ids['softland_close'])} tool_call={ids['tool_call']} "
                  f"nl_end={len(nl_end)} nl_bare={newline}", flush=True)
        # PN117 arm texts + sentence-end id set (only when PN117 is enabled).
        if _on("GENESIS_ENABLE_PN117_RESCUE"):
            for _k, _s in PN117_ARMS.items():
                _t = tok.encode(_s, add_special_tokens=False)
                # never inject anything that resembles </think>: hard guard.
                if any(_e in _t for _e in end):
                    print(f"[pn114_boot_ids] WARN: {_k} contains a think-end "
                          f"id — dropping", flush=True)
                    continue
                ids[_k] = _t
            send: list[int] = []
            for _s in PN117_SENTENCE_STRS:
                _t = tok.encode(_s, add_special_tokens=False)
                # a sentence-end marker is the LAST landed token; take the
                # trailing token of each rendering (covers ".", ".\n", etc.).
                if _t and _t[-1] not in send:
                    send.append(_t[-1])
            ids["sentence_end"] = send
        with open(OUT, "w") as f:
            json.dump(ids, f)
        _p117 = (f" pn117_arms={sum(1 for k in ids if k.startswith('arm'))}"
                 f" sentence_end={len(ids.get('sentence_end', []))}"
                 if _on("GENESIS_ENABLE_PN117_RESCUE") else "")
        print(f"[pn114_boot_ids] wrote {OUT}: probe={len(probe)} "
              f"newline={len(newline)} wrapup_close={len(ids['wrapup_close'])}"
              f"{_p117}", flush=True)
        return 0
    except Exception as exc:  # fail-open
        print(f"[pn114_boot_ids] WARN: {type(exc).__name__}: {exc} — "
              f"PN114 will be inert", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
