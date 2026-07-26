"""PN121 soft-landing tests (2026-07-26).

Runs INSIDE a THROWAWAY container off the pinned image, against the LIVE
patched holder — i.e. the bytes the next boot will execute:

  sudo podman run --rm --entrypoint /bin/bash \\
    -v $REPO/fixes:/fixes:ro \\
    -v $REPO/models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis:\\
       /usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \\
    localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725 -lc '
      python3 /fixes/patch_pr44812_tool_guard.py &&
      python3 /fixes/patch_pn114_forced_span.py &&
      python3 /fixes/patch_pn121_softland.py &&
      python3 -m pytest -q --noconftest /fixes/test_pn121_softland.py'

Nothing is asserted against a constant here: every case runs a real decode
loop through ThinkingBudgetStateHolder.update_state / apply_to_logits until
the budget is genuinely overrun, and reads what the holder forced.

Cases
  T1  baseline (flag OFF) — bare </think> at the cap, byte-for-byte stock
  T2  no fire before cap - PN121_PARK_AT  (the killed -512 arm's error)
  T3  park + land on the next newline; the close is LATER than stock, and
      the forced span is the transition phrase, newline first
  T4  no newline ever -> hard force at cap + PN121_HARD_MARGIN
  T5  <tool_call> in the think slice -> PN121 stands down (#44676)
  T6  structured-output grammar row -> PN121 stands down
  T7  MTP: spec tokens in flight at the trigger step, span still exact
  T8  soft-phase nudge reaches the logits; no nudge outside the phase
  T9  release() hands the parked budget back
"""
import importlib
import importlib.util
import json
import os
import sys

import pytest
import torch

HOLDER_PATH = ("/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/"
               "thinking_budget_state.py")

# token ids for the synthetic vocab
V = 3000
START, END = 999, 998
NL = 100          # bare "\n"
NL_DOT = 101      # ".\n" — a merged boundary token
TOOL = 77093 % V  # tool-call opener (id value is irrelevant to the logic)
NAT = 555         # the model's "natural" next token
W1, W2, W3 = 200, 201, 202   # transition-phrase body

SOFTLAND_CLOSE = [NL, W1, W2, W3, NL, END, NL, NL]
WRAPUP_CLOSE = [NL, W1, W2, W3, NL, END]

IDS = {
    "probe": [301, 302],
    "newline": [NL],
    "nl_end": [NL, NL_DOT],
    "close_paren": [303],
    "wrapup_close": WRAPUP_CLOSE,
    "softland_close": SOFTLAND_CLOSE,
    "tool_call": [TOOL],
    "ppen": [],
}

BUDGET = 2048
PARK_AT = 16
SOFT_RESERVE = 320
HARD_MARGIN = 384


def _env(**kw):
    """Set PN121 env and drop every sibling flag, then re-import the genesis
    modules so nothing caches a stale gate."""
    for v in ("GENESIS_ENABLE_PN114_PROBE", "GENESIS_PN112_WRAPUP",
              "GENESIS_PN112_CONFIRM", "GENESIS_PN112_WRAPUP_AT_CAP",
              "GENESIS_ENABLE_PN117_RESCUE", "GENESIS_PPEN_LAMBDA",
              "GENESIS_ENABLE_PN108_PLATEAU_CAP",
              "GENESIS_ENABLE_PN112_SETTLED_STOP",
              "GENESIS_ENABLE_PR44812_TOOL_GUARD"):
        os.environ.pop(v, None)
    os.environ["GENESIS_ENABLE_PN121_SOFTLAND"] = "0"
    os.environ["PN121_PARK_AT"] = str(PARK_AT)
    os.environ["PN121_SOFT_RESERVE"] = str(SOFT_RESERVE)
    os.environ["PN121_HARD_MARGIN"] = str(HARD_MARGIN)
    os.environ["PN121_MIN_BUDGET"] = "512"
    for k, v in kw.items():
        os.environ[k] = str(v)
    with open("/tmp/genesis_pn114_ids.json", "w") as f:
        json.dump(IDS, f)
    from vllm._genesis.plateau import pn114, pn121_softland
    importlib.reload(pn114)
    importlib.reload(pn121_softland)
    pn114._IDS = None
    return pn114, pn121_softland


def _load_holder():
    spec = importlib.util.spec_from_file_location("tbs_live", HOLDER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # deterministic CPU h2d (no pinned memory in a CPU-only test container)
    mod.async_tensor_h2d = lambda data, dtype, device: torch.tensor(
        data, dtype=dtype, device=device)
    return mod


class _RC:
    """Minimal ReasoningConfig stand-in (the holder only reads these two)."""
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END]


class Run:
    """One request decoded step by step through the real holder.

    The prompt already opens <think> — that is this deployment's chat
    template (continue_thinking=True), and it is the shape in which
    think_count freezes and the live slice is the only usable depth.
    """

    def __init__(self, tbs, num_spec=0, budget=BUDGET, script=None,
                 grammar=False, tool_at=None):
        self.holder = tbs.ThinkingBudgetStateHolder(
            _RC(), max_num_seqs=4, num_spec_tokens=num_spec,
            device=torch.device("cpu"), is_pin_memory=False,
        )
        self.num_spec = num_spec
        prompt = [1, 2, 3, START]
        self.state = self.holder._init_state_entry(prompt, budget)
        self.out: list[int] = []
        self.state["output_tok_ids"] = self.out
        self.state["spec_token_ids"] = []
        self.holder._state[0] = self.state
        self.script = script or {}
        self.grammar = grammar
        self.tool_at = tool_at
        self.forced: list[tuple[int, int]] = []   # (depth, token)
        self.nudge_seen: list[tuple[int, float]] = []

    def step(self, i, spec=None, accept=None):
        """One engine step.

        `spec` = draft tokens in flight (MTP). The holder owns len(spec) rows
        this step and graft S masks EVERY one of them — whole-window masking,
        with position authority taken from the LANDED output only. `accept`
        limits how many of those rows are committed, which is how a mid-span
        rejection is simulated.
        """
        if self.grammar:
            self.state["_pn121_grammar_rows"] = {0}
        spec = list(spec or [])
        self.state["spec_token_ids"] = list(spec)
        rows = len(spec) if (self.num_spec and spec) else 1
        logits = torch.zeros(rows, V)
        nat = self.script.get(i, NAT)
        if self.tool_at is not None and i == self.tool_at:
            nat = TOOL
        logits[:, nat] = 5.0
        base_nl = float(logits[0, NL])
        self.holder.apply_to_logits(logits, False, [spec] if spec else [[]])
        if float(logits[0, NL]) != base_nl:
            self.nudge_seen.append((len(self.out), float(logits[0, NL])))
        n_commit = rows if accept is None else min(accept, rows)
        last = None
        for r in range(n_commit):
            tok = int(torch.argmax(logits[r]))
            if float(logits[r, tok]) >= 1e8:
                self.forced.append((len(self.out), tok))
            self.out.append(tok)
            last = tok
        self.holder.update_state([self.out], [spec] if spec else [[]])
        return last

    def span_done(self):
        return len([t for _d, t in self.forced]) >= len(SOFTLAND_CLOSE)

    def run(self, n, spec=None, accept=None):
        for i in range(n):
            self.step(i, spec=spec, accept=accept)
            if self.span_done():
                break
        return self


@pytest.fixture
def tbs():
    return _load_holder()


def test_t1_baseline_bare_close_at_cap(tbs):
    """Flag OFF: stock behaviour — a BARE </think> forced at the cap."""
    _env(GENESIS_ENABLE_PN121_SOFTLAND=0)
    r = Run(tbs).run(BUDGET + 200)
    assert r.forced, "stock holder never forced a close"
    depth, tok = r.forced[0]
    assert tok == END, f"stock close should be the bare end token, got {tok}"
    assert BUDGET - 32 <= depth <= BUDGET, (
        f"stock close landed at {depth}, expected ~cap {BUDGET}")
    print(f"\nT1 stock: bare </think> at depth={depth} (cap={BUDGET})")


def test_t2_never_fires_before_park_at(tbs):
    """The error that killed GENESIS_PN112_WRAPUP_AT_CAP: it fired at
    budget-512 and shortened deep items. PN121 must touch nothing until
    budget - PN121_PARK_AT."""
    _env(GENESIS_ENABLE_PN121_SOFTLAND=1)
    r = Run(tbs)
    for i in range(BUDGET - PARK_AT - 1):
        r.step(i)
        assert not r.forced, f"PN121 forced at depth {len(r.out)} — too early"
        assert r.state["thinking_token_budget"] == BUDGET, (
            f"budget mutated at depth {len(r.out)}")
    print(f"\nT2 no fire and no budget change through depth "
          f"{BUDGET - PARK_AT - 1} (cap-{PARK_AT + 1})")


def test_t3_land_on_next_newline(tbs):
    """Park at the cap, land on the next newline. The close must be LATER
    than stock's (extension, not pre-emption) and must be the transition
    phrase with a newline first."""
    _env(GENESIS_ENABLE_PN121_SOFTLAND=1)
    nl_at = BUDGET + 40
    r = Run(tbs, script={nl_at: NL_DOT}).run(BUDGET + 500)
    assert r.forced, "PN121 never closed"
    first_depth = r.forced[0][0]
    span = [t for _d, t in r.forced]
    assert span[:len(SOFTLAND_CLOSE)] == SOFTLAND_CLOSE, (
        f"forced span {span[:len(SOFTLAND_CLOSE)]} != {SOFTLAND_CLOSE}")
    assert span[0] == NL, "span must start with a newline (Mueller's rule)"
    assert END in span, "span must contain </think>"
    assert first_depth > BUDGET, (
        f"landed at {first_depth}, must be at/after the cap {BUDGET} — "
        "PN121 extends, it never pre-empts")
    print(f"\nT3 landed at depth={first_depth} (cap+{first_depth - BUDGET}) "
          f"on newline at {nl_at}; span={span[:len(SOFTLAND_CLOSE)]}")


def test_t4_hard_force_at_cap_plus_margin(tbs):
    """No newline ever arrives -> unconditional force at cap+HARD_MARGIN
    (Nemotron's rule, our margin)."""
    _env(GENESIS_ENABLE_PN121_SOFTLAND=1)
    r = Run(tbs).run(BUDGET + HARD_MARGIN + 200)
    assert r.forced, "PN121 never hard-forced"
    depth = r.forced[0][0]
    assert depth >= BUDGET + HARD_MARGIN, (
        f"hard force at {depth}, expected >= {BUDGET + HARD_MARGIN}")
    assert depth <= BUDGET + HARD_MARGIN + 8, (
        f"hard force overran to {depth}")
    span = [t for _d, t in r.forced]
    assert span[:len(SOFTLAND_CLOSE)] == SOFTLAND_CLOSE
    print(f"\nT4 hard force at depth={depth} "
          f"(cap+{depth - BUDGET}, margin={HARD_MARGIN})")


def test_t5_tool_call_suppresses(tbs):
    """#44676: a <tool_call> opener in the think slice is an implicit
    reasoning end. PN121 must not inject its phrase into the JSON."""
    _env(GENESIS_ENABLE_PN121_SOFTLAND=1)
    r = Run(tbs, tool_at=BUDGET - 100).run(BUDGET + 200)
    assert TOOL in r.out, "the tool-call opener never landed"
    span = [t for _d, t in r.forced]
    assert span[:len(SOFTLAND_CLOSE)] != SOFTLAND_CLOSE, (
        "PN121 injected its transition phrase after a tool-call opener")
    from vllm._genesis.plateau import pn121_softland as p
    assert p._STATS["tool_suppressed"] > 0, "the tool guard never fired"
    print(f"\nT5 suppressed after <tool_call>; forced={span[:4]} "
          f"stats={p.stats_line()}")


def test_t6_grammar_suppresses(tbs):
    """A row under constrained decoding is stamped by graft X; PN121 must
    stand down there (structured output is where overflow corruption is
    measured at ~30% without it)."""
    _env(GENESIS_ENABLE_PN121_SOFTLAND=1)
    r = Run(tbs, grammar=True).run(BUDGET + 200)
    span = [t for _d, t in r.forced]
    assert span[:len(SOFTLAND_CLOSE)] != SOFTLAND_CLOSE, (
        "PN121 injected into a grammar-constrained row")
    assert span and span[0] == END, (
        "stock bare close should still happen on the grammar row")
    from vllm._genesis.plateau import pn121_softland as p
    assert p._STATS["grammar_suppressed"] > 0
    print(f"\nT6 grammar row: stock bare close at depth={r.forced[0][0]}")


def test_t7_mtp_spec_tokens_in_flight(tbs):
    """MTP: drafts in flight at the trigger step must be counted (the base
    carries PR #34668 — apply_to_logits takes predict_bonus_token and
    spec_token_ids) and must not desync the span."""
    _env(GENESIS_ENABLE_PN121_SOFTLAND=1)
    # 3 tokens land per step, so step 700 sits at depth ~2100 — past the cap
    r = Run(tbs, num_spec=3, script={700: NL_DOT})
    assert r.holder.in_spec_mode, "holder not in spec mode with num_spec=3"
    # three drafts pending every step, as under mtp/3, all accepted
    r.run(BUDGET + HARD_MARGIN + 100, spec=[7, 8, 9])
    assert r.forced[0][0] < BUDGET + HARD_MARGIN, (
        f"landed by hard force at {r.forced[0][0]}, expected the newline path")
    span = [t for _d, t in r.forced]
    assert span[:len(SOFTLAND_CLOSE)] == SOFTLAND_CLOSE, (
        f"MTP span desync: {span[:len(SOFTLAND_CLOSE)]}")
    assert r.forced[0][0] >= BUDGET - PARK_AT, (
        f"MTP close pre-empted at {r.forced[0][0]}")
    print(f"\nT7 mtp/3 all-accept: landed at depth={r.forced[0][0]}, "
          f"span exact ({len(span)} forced rows)")


def test_t10_mid_span_rejection(tbs):
    """Mid-span rejection is the COMMON path, not the rare one: at mtp/3
    with measured acceptance 0.92/0.77/0.62 the all-accept probability for a
    3-token chunk is ~0.44, and our span is 8 tokens. Only ONE draft row is
    committed per step here, i.e. a rejection at every single step — the
    span must still come out exact, because graft S masks the whole window
    and graft B takes position authority from the landed output alone."""
    _env(GENESIS_ENABLE_PN121_SOFTLAND=1)
    r = Run(tbs, num_spec=3)
    r.run(BUDGET + HARD_MARGIN + 400, spec=[7, 8, 9], accept=1)
    span = [t for _d, t in r.forced]
    assert span[:len(SOFTLAND_CLOSE)] == SOFTLAND_CLOSE, (
        f"span broke at the rejection boundary: {span[:len(SOFTLAND_CLOSE)]}")
    tail = r.out[-len(SOFTLAND_CLOSE):]
    assert tail == SOFTLAND_CLOSE, f"landed tail wrong: {tail}"
    print(f"\nT10 rejection at EVERY step (accept=1 of 3): span exact, "
          f"landed tail={tail}")


def test_t8_soft_phase_nudge(tbs):
    """Graft N: the newline logit is bumped inside the soft phase and
    untouched outside it."""
    _env(GENESIS_ENABLE_PN121_SOFTLAND=1)
    r = Run(tbs)
    for i in range(BUDGET - SOFT_RESERVE - 5):
        r.step(i)
    assert not r.nudge_seen, (
        f"nudge applied before the soft phase at {r.nudge_seen[:3]}")
    for i in range(20):
        r.step(i)
    assert r.nudge_seen, "no nudge inside the soft phase"
    depth, val = r.nudge_seen[0]
    assert depth >= BUDGET - SOFT_RESERVE - 2, (
        f"nudge started at depth {depth}, before cap-{SOFT_RESERVE}")
    assert 0 < val < 1e8, f"nudge must be an additive bump, got {val}"
    print(f"\nT8 nudge first seen at depth={depth} value={val} "
          f"(soft phase starts at {BUDGET - SOFT_RESERVE})")


def test_t9_release_restores_budget(tbs):
    """A parked budget must be handed back when the block ends elsewhere —
    otherwise a later think block inherits the sentinel and is uncapped."""
    pn114, pn121 = _env(GENESIS_ENABLE_PN121_SOFTLAND=1)
    r = Run(tbs)
    for i in range(BUDGET - PARK_AT + 2):
        r.step(i)
    assert r.state["thinking_token_budget"] == pn121._PARK_SENTINEL, (
        "PN121 never parked")
    pn121.release(r.state)
    assert r.state["thinking_token_budget"] == BUDGET, (
        f"budget not restored: {r.state['thinking_token_budget']}")
    print(f"\nT9 parked then released; budget back to "
          f"{r.state['thinking_token_budget']}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s", "--noconftest"]))
