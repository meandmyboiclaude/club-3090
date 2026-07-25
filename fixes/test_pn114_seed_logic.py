#!/usr/bin/env python3
"""Offline logic tests for PN114-SEED (fixes/pn114_seed.py). No vLLM, no GPU.

Gate M2 (fixes/test_pn114_seed_equivalence.py) proves the MECHANISM against
the real holder in a container. This file covers the DECISIONS around it — the
guard matrix that decides when a span may be armed at all — cheaply enough to
run on every edit:

  * flag / mode resolution and the fail-closed table lookup
  * the serving-side strip: what it refuses to touch and what it carries
  * the arm guard matrix (late, closing, provisional, unbudgeted, once-only)
  * the countdown arithmetic that charges the span like prompt tokens
  * completion ownership (never claim a span this module did not arm)

    python3 fixes/test_pn114_seed_logic.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail
                                                     else ""))
    if not ok:
        FAILURES.append(name)


def load_module(table_path: str):
    spec = importlib.util.spec_from_file_location(
        "pn114_seed_under_test", HERE / "pn114_seed.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pn114_seed_under_test"] = mod
    spec.loader.exec_module(mod)
    mod._TABLE_PATH = table_path
    mod._TABLE = None
    mod._TABLE_TRIED = False
    return mod


SEED3 = "Budget: ~3 short steps.\nStep 1:"
SEED9 = "Budget: ~9 short steps.\nStep 1:"
IDS3 = [601, 602, 603, 604]
IDS9 = [601, 602, 605, 604]

TABLE = {
    "version": 1,
    "base": [1, 2, 3, 900, 901],
    "base_text": "<|im_start|>assistant\n<think>\n",
    "think_end": [999],
    "by_text": {SEED3: IDS3, SEED9: IDS9},
    "by_steps": {"Budget|plain|3": SEED3, "Budget|plain|9": SEED9},
    "max_n": 64,
    "rejected": 0,
}


def fresh_state(**over):
    st = {
        "in_think": True,
        "in_end": False,
        "thinking_token_budget": 400,
        "output_tok_ids": [],
        "prompt_tok_ids": [1, 2, 3, 900, 901],
        "start_thinking": 4,
        "continue_thinking": True,
        "check_count_down": 400,
        "end_count": 0,
        "force_index": [],
    }
    st.update(over)
    return st


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pn114seed-"))
    tpath = tmp / "table.json"
    tpath.write_text(json.dumps(TABLE), encoding="utf-8")
    m = load_module(str(tpath))
    os.environ["GENESIS_ENABLE_PN114_SEED_SPAN"] = "1"
    os.environ["GENESIS_PN114_SEED_MODE"] = "mirror"
    os.environ["GENESIS_PN102_TOKENS_PER_STEP"] = "193"

    print("\n[1] flag / mode / table")
    check("enabled() follows the master flag", m.enabled())
    os.environ["GENESIS_ENABLE_PN114_SEED_SPAN"] = "0"
    check("flag off disables the module", not m.enabled())
    os.environ["GENESIS_ENABLE_PN114_SEED_SPAN"] = "1"
    check("mode defaults to mirror", m.mode() == "mirror")
    os.environ["GENESIS_PN114_SEED_MODE"] = "nonsense"
    check("an unknown mode falls back to mirror", m.mode() == "mirror")
    os.environ["GENESIS_PN114_SEED_MODE"] = "routed"
    check("routed mode is selectable", m.mode() == "routed")
    os.environ["GENESIS_PN114_SEED_MODE"] = "mirror"
    check("known seed resolves to its ids", m.seed_ids_for_text(SEED3) == IDS3)
    check("unknown seed resolves to nothing (fail-closed)",
          m.seed_ids_for_text("Budget: ~7 short steps.\nStep 1:") is None)

    print("\n[2] N derivation mirrors PN102")
    check("steps_for_budget floors at 3", m.steps_for_budget(1) == 3)
    check("steps_for_budget = round(budget/tps)",
          m.steps_for_budget(1737) == 9, str(m.steps_for_budget(1737)))
    check("routed_seed_text keeps the label/tail family",
          m.routed_seed_text(1737, SEED3) == SEED9)
    check("routed_seed_text returns None when N is off the table",
          m.routed_seed_text(193 * 40, SEED3) is None)

    print("\n[3] serving-side strip")
    r = SimpleNamespace(chat_template_kwargs={"pn_env_seed": SEED3},
                        vllm_xargs=None)
    ok = m.strip_prompt_seed(r)
    check("a known seed is stripped and carried in vllm_xargs",
          ok and "pn_env_seed" not in r.chat_template_kwargs
          and r.vllm_xargs["pn114_seed_text"] == SEED3)
    r = SimpleNamespace(chat_template_kwargs={"pn_env_seed": "unknown"},
                       vllm_xargs=None)
    check("an unknown seed is left in the prompt",
          not m.strip_prompt_seed(r)
          and r.chat_template_kwargs["pn_env_seed"] == "unknown")
    r = SimpleNamespace(chat_template_kwargs={"pn_env_seed": SEED3,
                                              "enable_thinking": False},
                       vllm_xargs=None)
    check("thinking-off requests are never touched",
          not m.strip_prompt_seed(r)
          and r.chat_template_kwargs["pn_env_seed"] == SEED3)
    r = SimpleNamespace(chat_template_kwargs={}, vllm_xargs=None)
    check("no seed, nothing to strip", not m.strip_prompt_seed(r))
    r = SimpleNamespace(chat_template_kwargs=None, vllm_xargs=None)
    check("a request without chat_template_kwargs is safe",
          not m.strip_prompt_seed(r))
    os.environ["GENESIS_ENABLE_PN114_SEED_SPAN"] = "0"
    r = SimpleNamespace(chat_template_kwargs={"pn_env_seed": SEED3},
                        vllm_xargs=None)
    check("flag off strips nothing", not m.strip_prompt_seed(r))
    os.environ["GENESIS_ENABLE_PN114_SEED_SPAN"] = "1"

    print("\n[4] note_params")
    st = fresh_state()
    m.note_params(st, SimpleNamespace(extra_args={"pn114_seed_text": SEED3}))
    check("the stripped seed reaches the state entry",
          st[m.K_TEXT] == SEED3)
    st2 = fresh_state()
    m.note_params(st2, SimpleNamespace(extra_args=None))
    check("no xargs, no stash", m.K_TEXT not in st2)

    print("\n[5] the arm guard matrix")
    def armed(**over):
        st = fresh_state(**over)
        st[m.K_TEXT] = over.pop("_seed", SEED3)
        return m.maybe_arm(st, 2, req_id="r"), st

    ok, st = armed()
    check("a clean request arms", ok and st["force_seq"] == IDS3
          and st["force_seq_base"] == 0 and st["in_end"] and st["force_index"]
          == [0])
    check("arming is one-shot", not m.maybe_arm(st, 2, req_id="r"))
    check("a token already landed -> declined",
          not armed(output_tok_ids=[7])[0])
    check("closing already -> declined", not armed(in_end=True)[0])
    check("not in think -> declined", not armed(in_think=False)[0])
    check("unbudgeted -> declined",
          not armed(thinking_token_budget=-1)[0])
    check("another span owns the forcer -> declined",
          not armed(force_seq=[1, 2])[0])
    st = fresh_state()
    check("nothing stripped -> declined", not m.maybe_arm(st, 2))
    os.environ["GENESIS_PN114_SEED_MODE"] = "routed"
    check("routed: an unresolved H119 route waits",
          not armed(_h119_provisional=True)[0])
    ok, st = armed(thinking_token_budget=1737)
    check("routed: the span is the ROUTED N, not the pre-prefill one",
          ok and st["force_seq"] == IDS9)
    os.environ["GENESIS_PN114_SEED_MODE"] = "mirror"
    ok, st = armed(thinking_token_budget=1737)
    check("mirror: the span is the pre-prefill seed",
          ok and st["force_seq"] == IDS3)

    print("\n[6] the span is charged like prompt tokens")
    st = fresh_state(output_tok_ids=[1] * 4, thinking_token_budget=400)
    check("countdown = budget - (prompt-think + span)",
          m._countdown_after_span(st) == 400 - 4,
          str(m._countdown_after_span(st)))
    st = fresh_state(output_tok_ids=[1] * 4, thinking_token_budget=3)
    check("a seed longer than the budget yields a non-positive countdown",
          m._countdown_after_span(st) == -1)
    st = fresh_state(output_tok_ids=[1] * 4, prompt_tok_ids=[0] * 20,
                     start_thinking=9, thinking_token_budget=100)
    check("prompt-side think tokens are counted too",
          m._countdown_after_span(st) == 100 - (10 + 4),
          str(m._countdown_after_span(st)))

    print("\n[7] completion ownership")
    _ok, st = armed()
    st["output_tok_ids"] = list(IDS3)
    check("our span is claimed", m.on_force_complete(st))
    check("and returns to think mode",
          st["in_think"] and not st["in_end"] and st["force_seq"] is None
          and st["end_count"] == 0)
    check("with the span charged",
          st["check_count_down"] == 400 - len(IDS3))
    check("completion is not re-claimed", not m.on_force_complete(st))
    check("a span we did not arm is never claimed",
          not m.on_force_complete(fresh_state(force_seq=[1, 2],
                                              in_end=True)))
    _ok, st = armed(thinking_token_budget=3)
    st["output_tok_ids"] = list(IDS3)
    m.on_force_complete(st)
    check("an over-budget seed closes immediately instead of freeing a token",
          st["in_end"] and st["force_index"] == [0]
          and st["force_seq"] is None)

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("PASS: PN114-SEED logic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
