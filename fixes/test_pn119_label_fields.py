#!/usr/bin/env python3
"""PN119-v2 `_label_fields` thinking-label coverage over every PN102 banner.

Run: ~/shared/needfit/lens-venv/bin/python fixes/test_pn119_label_fields.py

Why this test exists
--------------------
`_label_fields` decides `thinking` by scanning the tail of `prompt_token_ids`
for the think markers. It used a fixed 8-token window. PN102 seeds text INSIDE
the think region, so the distance from `<think>` to end-of-prompt is
1 + len(seed tokens). Under the v5-family seed ("Step 1:", ~3 tokens) 8 was
enough; under the v3 seed ("Budget: ~11 short steps.\nStep 1:") it is not — so
the marker fell out of the window, `thinking` came back None, and the refit
loop would have trained on rows with no label. v3 is the validated prod path.

The seeds are NOT hardcoded here: each PN102 contract builder in
`_genesis/middleware/answer_rescue.py` is driven directly and the seed it
actually writes to `ctk["pn_env_seed"]` is read back. A new banner variant is
covered by adding one row to VARIANTS.

Token counts use a deliberately pessimistic surrogate (ceil(chars/2)) — real
BPE averages >=3 chars/token on English, so every case here is harder than the
live tokenizer makes it.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
AR_PATH = os.path.join(
    REPO, "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis"
    "/middleware/answer_rescue.py")

THINK_START = 248068
THINK_END = 248069

_FAILURES: list[str] = []


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        _FAILURES.append(name)


# ── seed harvesting ────────────────────────────────────────────────────────

# (label, env overrides, builder attr, extra ctk) — one row per PN102 banner
# variant reachable from `maybe_add_answer_hint`'s dispatch chain.
VARIANTS = [
    ("v3 sized (prod path, Budget label)", {}, "_contract_v3_sized", {}, 2048),
    # Plan branch needs budget >= PERMISSION_MIN AND steps*tps < 0.7*budget.
    ("v3 sized (Plan label, generous budget)", {}, "_contract_v3_sized",
     {"pn100_steps": 7}, 8192),
    ("v3 sized (planner steps)", {}, "_contract_v3_sized", {"pn100_steps": 7}, 2048),
    ("v3 sized + STEP1_ECHO",
     {"GENESIS_PN102_V3_STEP1_ECHO": "1"}, "_contract_v3_sized", {}, 2048),
    ("v3 sized + static-first + range",
     {"GENESIS_PN102_BANNER_STATIC_FIRST": "1", "GENESIS_PN102_V3_RANGE": "1.5"},
     "_contract_v3_sized", {}, 4096),
    ("v4 static", {}, "_contract_v4_static", {}, 2048),
    ("v5 settled", {}, "_contract_v5_settled", {}, 2048),
    ("v5 settled + answer clause",
     {"GENESIS_PN102_V5_ANSWER_CLAUSE": "1"}, "_contract_v5_settled", {}, 2048),
    ("v6a prove-it", {}, "_contract_v6a_proveit", {}, 2048),
    ("v6b named-unresolved", {}, "_contract_v6b_named", {}, 2048),
    ("v7 state-answer-early", {}, "_contract_v7_stateanswer", {}, 2048),
    ("v8 hybrid", {}, "_contract_v8_hybrid", {}, 2048),
    ("v8b lean-anchor",
     {"GENESIS_PN102_V8_LEAN_ANCHOR": "1"}, "_contract_v8_hybrid", {}, 2048),
    ("v8b lean-anchor + planner steps",
     {"GENESIS_PN102_V8_LEAN_ANCHOR": "1"}, "_contract_v8_hybrid",
     {"pn100_steps": 13}, 2048),
]


def harvest_seeds(ar) -> list[tuple[str, str]]:
    """Drive every banner builder and read back the seed it emits."""
    out = []
    for label, env, fn_name, ctk_extra, budget in VARIANTS:
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            ctk = dict(ctk_extra)
            getattr(ar, fn_name)(ctk, budget)
            seed = ctk.get("pn_env_seed")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        if not seed:
            _FAILURES.append(f"{label}: builder emitted no pn_env_seed")
            print(f"FAIL  seed harvest {label} — no pn_env_seed")
            continue
        out.append((label, seed))
    return out


def seed_token_len(seed: str) -> int:
    """Pessimistic upper bound on the BPE token count of `seed`."""
    return max(1, math.ceil(len(seed) / 2))


# ── the actual assertions ──────────────────────────────────────────────────

class _SP:
    max_tokens = 512


class _Req:
    def __init__(self, prompt_token_ids, output_token_ids=(), max_tokens=512):
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids = list(output_token_ids)
        sp = _SP()
        sp.max_tokens = max_tokens
        self.sampling_params = sp


def _prompt_thinking_on(seed_tokens: int, filler: int = 400) -> list[int]:
    """chat template, thinking ON: ... <think> <seed tokens>"""
    return [7] * filler + [THINK_START] + [11] * seed_tokens


def _prompt_thinking_off(filler: int = 400) -> list[int]:
    """chat template, thinking OFF: ... <think> </think>"""
    return [7] * filler + [THINK_START, THINK_END]


def main() -> int:
    ar = _load("pn102_answer_rescue", AR_PATH)
    router_mod = _load("pn119_router_under_test",
                       os.path.join(HERE, "pn119_router.py"))

    r = object.__new__(router_mod.PN119Router)
    # Marker SEQUENCES now (the holder's shape) — single-element here because
    # <think>/</think> are single tokens in the served tokenizer.json.
    r._think_start_ids = [THINK_START]
    r._think_end_ids = [THINK_END]
    r._tail_window = 64  # the shipped default
    r._censor_slack = len(r._think_end_ids) + 12   # BUG-139: covers gap 5 AND 13
    r._h119_applied = {}
    r._h119_forced = {}

    seeds = harvest_seeds(ar)
    print(f"\n-- harvested {len(seeds)} PN102 banner seeds "
          f"(window={r._tail_window}) --\n")

    widest = 0
    for label, seed in seeds:
        n = seed_token_len(seed)
        widest = max(widest, n)
        req = _Req(_prompt_thinking_on(n))
        thinking, rtok, cap_hit, censored, _src = r._label_fields(req, generated=64)
        check(f"thinking=True under {label}", thinking is True,
              f"seed={seed!r} tok<={n} got={thinking}")

    check("window clears the widest seed with >=2x headroom",
          r._tail_window >= 2 * (widest + 1),
          f"widest seed <= {widest} tok, +1 for <think>")

    # thinking-OFF must still read False (both markers present, </think> last).
    req = _Req(_prompt_thinking_off())
    thinking, _, _, _, _ = r._label_fields(req, generated=64)
    check("thinking=False on a pre-closed think region", thinking is False,
          f"got={thinking}")

    # raw/completion prompt: neither marker -> None (refit treats as ineligible)
    thinking, _, _, _, _ = r._label_fields(_Req([7] * 400), generated=64)
    check("thinking=None with no markers", thinking is None, f"got={thinking}")

    # multi-turn: an EARLIER turn's closed region must not outrank this turn's
    # open <think>. Last-marker-wins is what makes widening safe.
    multi = ([7] * 380 + [THINK_START] + [11] * 6 + [THINK_END]
             + [7] * 4 + [THINK_START] + [11] * 12)
    thinking, _, _, _, _ = r._label_fields(_Req(multi), generated=64)
    check("thinking=True with a prior closed region in-window",
          thinking is True, f"got={thinking}")

    # rtok / cap_hit still work off the widened label.
    longest = max(seed_token_len(s) for _, s in seeds)
    req = _Req(_prompt_thinking_on(longest), output_token_ids=[5, 5, 5, THINK_END, 9])
    thinking, rtok, cap_hit, censored, _src = r._label_fields(req, generated=5)
    check("rtok counts tokens before </think>",
          thinking is True and rtok == 3 and cap_hit is False,
          f"thinking={thinking} rtok={rtok} cap_hit={cap_hit}")

    req = _Req(_prompt_thinking_on(longest), output_token_ids=[5] * 512)
    thinking, rtok, cap_hit, censored, _src = r._label_fields(req, generated=512)
    check("cap_hit when </think> never emitted",
          thinking is True and rtok == 512 and cap_hit is True,
          f"thinking={thinking} rtok={rtok} cap_hit={cap_hit}")

    # A LATER </think> inside the answer must not be mistaken for the end of
    # the spend: the FIRST occurrence is the boundary.
    req = _Req(_prompt_thinking_on(longest),
               output_token_ids=[5, 5, THINK_END, 9, 9, THINK_END])
    thinking, rtok, _, _, _ = r._label_fields(req, generated=6)
    check("rtok takes the FIRST </think>, not the last", rtok == 2, f"rtok={rtok}")

    # ── BUG-139: censoring is derived from the BUDGET, not from max_tokens ──
    # The exact live signature — 43 of 79 thinking rows sat at grant-5 while
    # only 4 flagged cap_hit, because cap_hit never looked at the budget.
    req = _Req(_prompt_thinking_on(longest),
               output_token_ids=[5] * 1295 + [THINK_END] + [9] * 40,
               max_tokens=4096)
    thinking, rtok, cap_hit, censored, _src = r._label_fields(
        req, generated=1336, budget=1300)
    check("censored at grant-5 (the BUG-139 signature)",
          rtok == 1295 and censored is True and cap_hit is False,
          f"rtok={rtok} censored={censored} cap_hit={cap_hit}")

    req = _Req(_prompt_thinking_on(longest),
               output_token_ids=[5] * 300 + [THINK_END], max_tokens=4096)
    _, rtok, _, censored, _src = r._label_fields(req, generated=301, budget=1300)
    check("a natural stop well below the grant is not censored",
          rtok == 300 and censored is False, f"rtok={rtok} censored={censored}")

    _, _, _, censored, _src = r._label_fields(
        _Req(_prompt_thinking_on(longest),
             output_token_ids=[5] * 1295 + [THINK_END], max_tokens=4096),
        generated=1296, budget=None)
    check("no budget => nothing could have truncated it here",
          censored is False, f"censored={censored}")

    # An empty output never closed the think region either, but there is no
    # evidence in it at all — labelling it a cap hit put a y=1 censoring label
    # on a request that produced nothing (an abort, a disconnect).
    _, rtok, cap_hit, _, _ = r._label_fields(
        _Req(_prompt_thinking_on(longest), output_token_ids=[]), generated=0)
    check("an EMPTY output is not a cap hit",
          rtok == 0 and cap_hit is False, f"rtok={rtok} cap_hit={cap_hit}")

    # Regression witness: the OLD 8-token window loses the v3-family seeds.
    r._tail_window = 8
    lost = [lbl for lbl, seed in seeds
            if r._label_fields(_Req(_prompt_thinking_on(seed_token_len(seed))),
                               generated=64)[0] is None]
    check("regression witness: old 8-token window drops >=1 variant",
          bool(lost), f"would have labelled None: {lost}")

    print()
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)}): " + "; ".join(_FAILURES))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
