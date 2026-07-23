"""PN114 forced-span holder logic tests (span-soundness redesign, 2026-07-23).

Run IN the vllm-tcbench-8021 container after building /tmp/pn114/ via
fixes/pn114_build_holders.py:
  /tmp/pn114/thinking_budget_state.py  — NEW holder (reworked grafts)
  /tmp/pn114/holder_pre.py             — PRE-fix holder (live text) for T-A

Scenarios (BUILD-SPEC-forced-span-mtp-soundness-20260723):
  T1   stock budget-force entry unchanged (legacy)
  T-A  stock parity: NO force_seq, spec + non-spec — pre/new holders must
       produce IDENTICAL states and identical mask writes every step
  T-B  non-spec span: A,B,C forced in order, index 0 NOT skipped,
       completion fires once, divert honored
  T-C  spec acceptance: drafts match — span completes without desync,
       and NOT before the tokens actually landed
  T-D  spec REJECTION mid-span: emitted stays put, next step re-forces,
       final tail exact, no early completion (also with trailing frees)
  T-E  bonus-pass duality: bonus + target calls force disjoint rows,
       apply never mutates end_count, repeat calls idempotent
  T-F  wrapup at parked budget: normal reset restores budget exactly,
       force_seq/force_seq_base cleaned up
"""
import importlib.util
import os
import sys
import types

for _v in ("GENESIS_ENABLE_PN112_SETTLED_STOP", "GENESIS_PPEN_LAMBDA",
           "GENESIS_ENABLE_PN114_PROBE", "GENESIS_PN112_WRAPUP",
           "GENESIS_PN112_CONFIRM", "GENESIS_PN112_WRAPUP_AT_CAP",
           "GENESIS_ENABLE_PN108_PLATEAU_CAP",
           "GENESIS_ENABLE_PN112_SETTLED_STOP"):
    os.environ.pop(_v, None)

import torch  # noqa: E402  (container has it)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # deterministic CPU mask writes (no pinned-memory h2d in tests)
    mod.async_tensor_h2d = lambda data, dtype, device: torch.tensor(
        data, dtype=dtype, device=device)
    return mod


tbs_new = load("tbs_new", "/tmp/pn114/thinking_budget_state.py")
tbs_pre = load("tbs_pre", "/tmp/pn114/holder_pre.py")

START, END = 999, 998
A, B, C, X, NAT = 700, 701, 702, 888, 555
V = 2000

# ---- stub _genesis so graft G/observe seats are inert & divert is ours ----
calls = {"complete": 0}


def _ofc(state):
    p = state.get("_pn114") or {}
    ph = p.get("phase")
    calls["complete"] += 1
    if ph == "probe_force":
        p["phase"] = "probe_free"
        state["in_end"] = False
        state["in_think"] = True
        state["end_count"] = 0
        state["force_index"] = []
        state["force_seq"] = None
        state.pop("force_seq_base", None)
        return True
    if ph == "wrapup":
        p["phase"] = None
        state["force_seq"] = None
        state.pop("force_seq_base", None)
        saved = p.get("saved_budget")
        if saved is not None:
            state["thinking_token_budget"] = saved
        p["saved_budget"] = None
        return False
    return False


stub114 = types.ModuleType("pn114")
stub114.on_force_complete = _ofc
stub114.any_enabled = lambda: False
stub108 = types.ModuleType("pn108")
stub108.observe_state = lambda *a, **k: None
stub112 = types.ModuleType("pn112")
stub112.observe_state = lambda *a, **k: None
pkg = types.ModuleType("vllm._genesis.plateau")
pkg.pn114, pkg.pn108, pkg.pn112 = stub114, stub108, stub112
sys.modules["vllm._genesis.plateau"] = pkg
sys.modules["vllm._genesis.plateau.pn114"] = stub114
sys.modules["vllm._genesis.plateau.pn108"] = stub108
sys.modules["vllm._genesis.plateau.pn112"] = stub112


# ---- harness ----------------------------------------------------------
def mk(tbs, nspec):
    h = tbs.ThinkingBudgetStateHolder.__new__(tbs.ThinkingBudgetStateHolder)
    h.think_start_token_ids = [START]
    h.think_end_token_ids = [END]
    h.implicit_think_end_token_ids = []
    h.is_enabled = True
    h.in_spec_mode = nspec > 0
    h.num_spec_tokens = nspec
    h._state = {}
    h.cu_num_tokens = {}
    h._mask_capacity = 8 * (nspec + 1)
    h.device = torch.device("cpu")
    return h


def st(out, budget=3):
    return {"thinking_token_budget": budget, "in_think": True,
            "in_end": False, "think_count": 0, "check_count_down": budget,
            "force_index": [], "start_thinking": -1, "end_thinking": -1,
            "in_spec_mode": False, "bonus_token_forced": False,
            "continue_thinking": False, "scan_offset": 0,
            "output_tok_ids": list(out), "prev_output_length": 0,
            "spec_token_ids": [], "end_count": 0, "prompt_tok_ids": None}


def fresh_logits(rows):
    lg = torch.zeros((rows, V), dtype=torch.float32)
    lg[:, NAT] = 5.0
    return lg


def masked(lg):
    return sorted((r, t) for r, t in (lg == 1e9).nonzero(as_tuple=False)
                  .tolist())


def spec_step(h, s, out, drafts, drop_recovery=False):
    """One MTP engine step: update_state, bonus pass, target pass, greedy
    rejection sampling. Returns (landed, bonus_masks, target_masks)."""
    h._state = {0: s}
    h.update_state([out + list(drafts)], [list(drafts)])
    lg_b = fresh_logits(1)
    h.apply_to_logits(lg_b, True, [list(drafts)])
    lg_t = fresh_logits(max(len(drafts), 1))
    h.apply_to_logits(lg_t, False, [list(drafts)])
    landed, rejected = [], False
    for k, d in enumerate(drafts):
        tgt = int(lg_t[k].argmax())
        if d == tgt:
            landed.append(d)
        else:
            rejected = True
            if not drop_recovery:
                landed.append(tgt)
            break
    if not rejected and drafts:
        landed.append(int(lg_b[0].argmax()))
    if not drafts:
        landed.append(int(lg_b[0].argmax()))
    out.extend(landed)
    return landed, masked(lg_b), masked(lg_t)


def ns_step(h, s, out):
    """One non-spec engine step (single row, single call)."""
    h._state = {0: s}
    h.update_state([list(out)], None)
    lg = fresh_logits(1)
    h.apply_to_logits(lg, False, [[]])
    tok = int(lg[0].argmax())
    out.append(tok)
    return tok, masked(lg)


def arm(s, seq, out, phase="probe_force", saved=None):
    """Mirror pn114._arm (post-redesign: no prepend, base recorded)."""
    s["_pn114"] = {"phase": phase, "saved_budget": saved}
    s["force_seq"] = list(seq)
    s["force_seq_base"] = len(out)
    s["in_think"] = False
    s["in_end"] = True
    s["end_count"] = 0
    s["bonus_token_forced"] = False
    s["force_index"] = [0]
    if saved is not None:
        s["thinking_token_budget"] = 10_000_000
        s["check_count_down"] = 10_000_000


def snap(s):
    keys = ("in_think", "in_end", "end_count", "think_count",
            "check_count_down", "force_index", "start_thinking",
            "end_thinking", "continue_thinking", "scan_offset",
            "prev_output_length", "bonus_token_forced",
            "thinking_token_budget")
    return {k: s.get(k) for k in keys}


# ---- T1: stock budget-force entry (legacy) ----------------------------
h = mk(tbs_new, 0)
s = st([START, 10, 11, 12, 13])
h._update_think_state(s)
assert s["in_end"] and s["force_index"] == [0], ("T1", s)
assert "force_seq_base" not in s, ("T1 stray key", s)
print("T1 OK: stock budget-force unchanged")

# ---- T-A: stock parity, non-spec + spec, pre vs new -------------------
for label, nspec, plan in (
    ("non-spec", 0, [None, None, None, None]),
    ("spec", 2, [[12, 13], [14, 15], [16, 17], [NAT, NAT]]),
):
    traces = []
    for tbs in (tbs_pre, tbs_new):
        h = mk(tbs, nspec)
        s = st([START, 10, 11], budget=3)
        out = [START, 10, 11]
        trace = []
        for drafts in plan:
            if drafts is None:
                tok, mk_ = ns_step(h, s, out)
                trace.append((tok, mk_, snap(s)))
            else:
                landed, mb, mt = spec_step(h, s, out, drafts)
                trace.append((landed, mb, mt, snap(s)))
        trace.append(("final_out", list(out)))
        traces.append(trace)
    assert traces[0] == traces[1], (
        "T-A %s DIVERGED\npre: %r\nnew: %r" % (label, traces[0], traces[1]))
    assert "force_seq_base" not in s, ("T-A stray key", label, s)
    print("T-A OK (%s): pre/new holders byte-equal behavior, %d steps"
          % (label, len(plan)))

# ---- T-B: non-spec span, order A,B,C, index 0 NOT skipped -------------
calls["complete"] = 0
h = mk(tbs_new, 0)
out = [START, 7, 7, 7, 7, 7]
s = st(out, budget=100)
s["prev_output_length"] = len(out)
s["start_thinking"] = 0
s["think_count"] = 5
arm(s, [A, B, C], out, saved=100)
mask_seq = []
for _ in range(4):
    tok, mk_ = ns_step(h, s, out)
    mask_seq.append(mk_)
assert mask_seq[:3] == [[(0, A)], [(0, B)], [(0, C)]], ("T-B order", mask_seq)
assert mask_seq[3] == [], ("T-B post-completion mask", mask_seq)
assert out[6:9] == [A, B, C], ("T-B tail", out)
assert calls["complete"] == 1, ("T-B completion count", calls)
assert s["in_think"] and not s["in_end"], ("T-B divert", s)
assert s["_pn114"]["phase"] == "probe_free", ("T-B phase", s["_pn114"])
assert s.get("force_seq") is None and "force_seq_base" not in s, ("T-B", s)
print("T-B OK: non-spec span forces A,B,C in order (index 0 intact), "
      "completes once, diverts")

# ---- T-C: spec acceptance — no desync, no EARLY completion ------------
calls["complete"] = 0
h = mk(tbs_new, 2)
out = [START, 7, 7, 7, 7, 7]
s = st(out, budget=100)
s["prev_output_length"] = len(out)
s["start_thinking"] = 0
s["think_count"] = 5
arm(s, [A, B, C], out, saved=100)
landed, mb, mt = spec_step(h, s, out, [A, B])
assert mt == [(0, A), (1, B)], ("T-C target masks", mt)
assert mb == [(0, C)], ("T-C bonus mask", mb)
assert landed == [A, B, C], ("T-C landed", landed)
assert calls["complete"] == 0, ("T-C EARLY completion", calls)
landed2, _, _ = spec_step(h, s, out, [NAT, NAT])
assert calls["complete"] == 1, ("T-C completion count", calls)
assert out[6:9] == [A, B, C], ("T-C tail", out)
assert s["_pn114"]["phase"] == "probe_free", ("T-C divert", s["_pn114"])
print("T-C OK: accepted drafts land whole span in one step, completion "
      "only after tokens actually landed")

# ---- T-D: spec rejection mid-span (the bug's scenario) ----------------
calls["complete"] = 0
h = mk(tbs_new, 2)
out = [START, 7, 7, 7, 7, 7]
s = st(out, budget=100)
s["prev_output_length"] = len(out)
s["start_thinking"] = 0
s["think_count"] = 5
arm(s, [A, B, C], out, saved=100)
landed1, mb1, mt1 = spec_step(h, s, out, [A, X], drop_recovery=True)
assert mt1 == [(0, A), (1, B)], ("T-D step1 target masks", mt1)
assert landed1 == [A], ("T-D step1 landed", landed1)
landed2, mb2, mt2 = spec_step(h, s, out, [NAT, NAT])
assert s["end_count"] >= 1, ("T-D emitted", s["end_count"])
assert mt2 == [(0, B), (1, C)], ("T-D step2 re-force from emitted=1", mt2)
assert mb2 == [], ("T-D step2 bonus clamped past span end", mb2)
assert landed2 == [B], ("T-D step2 landed (recovery = forced)", landed2)
assert calls["complete"] == 0, ("T-D early completion", calls)
landed3, mb3, mt3 = spec_step(h, s, out, [C, NAT])
assert landed3 == [C, NAT, NAT], ("T-D step3 span end + frees", landed3)
assert calls["complete"] == 0, ("T-D completion before observed", calls)
landed4, _, _ = spec_step(h, s, out, [NAT, NAT])
assert calls["complete"] == 1, ("T-D completion count", calls)
assert out[6:9] == [A, B, C], ("T-D final tail", out)
assert s["_pn114"]["phase"] == "probe_free", ("T-D divert", s["_pn114"])
print("T-D OK: mid-span rejection re-forces from landed position, no "
      "early completion, trailing frees don't break completion")

# ---- T-E: bonus-pass duality ------------------------------------------
h = mk(tbs_new, 2)
out = [START, 7, 7, 7, 7, 7]
s = st(out, budget=100)
s["prev_output_length"] = len(out)
s["start_thinking"] = 0
arm(s, [A, B, C], out, saved=100)
h._state = {0: s}
h.update_state([out + [A, B]], [[A, B]])
ec0 = s["end_count"]
lg_b1 = fresh_logits(1)
h.apply_to_logits(lg_b1, True, [[A, B]])
lg_b2 = fresh_logits(1)
h.apply_to_logits(lg_b2, True, [[A, B]])
lg_t = fresh_logits(2)
h.apply_to_logits(lg_t, False, [[A, B]])
assert masked(lg_b1) == masked(lg_b2) == [(0, C)], (
    "T-E bonus idempotent", masked(lg_b1), masked(lg_b2))
assert masked(lg_t) == [(0, A), (1, B)], ("T-E target", masked(lg_t))
assert s["end_count"] == ec0, ("T-E apply mutated end_count", s)
print("T-E OK: bonus/target passes force disjoint rows, apply is "
      "stateless (no double-advance, no double-force)")

# ---- T-F: wrapup at parked budget — exact restore + cleanup -----------
calls["complete"] = 0
h = mk(tbs_new, 0)
out = [START, 7, 7, 7, 7, 7]
s = st(out, budget=100)
s["prev_output_length"] = len(out)
s["start_thinking"] = 0
s["think_count"] = 5
arm(s, [A, END], out, phase="wrapup", saved=100)
for _ in range(3):
    ns_step(h, s, out)
assert out[6:8] == [A, END], ("T-F tail", out)
assert calls["complete"] == 1, ("T-F completion count", calls)
assert not s["in_end"] and s["end_count"] == 0, ("T-F reset", s)
assert s["thinking_token_budget"] == 100, ("T-F budget restore", s)
assert s["check_count_down"] == 100, ("T-F countdown", s)
assert s.get("force_seq") is None and "force_seq_base" not in s, ("T-F", s)
print("T-F OK: wrapup span completes into normal reset, parked budget "
      "restored exactly, span keys cleaned")

# ---- T-G: hole-in-span (unified probe): hole row free, close anchored ---
calls["complete"] = 0
h = mk(tbs_new, 2)
out = [START, 7, 7, 7, 7, 7]
s = st(out, budget=100)
s["prev_output_length"] = len(out)
s["start_thinking"] = 0
s["think_count"] = 5
arm(s, [A, None, B], out, saved=100)
landed1, mb1, mt1 = spec_step(h, s, out, [A, X])
assert mt1 == [(0, A)], ("T-G hole row must NOT be masked", mt1)
assert mb1 == [(0, B)], ("T-G bonus = close, positionally anchored", mb1)
assert landed1 == [A, NAT], ("T-G step1 landed (hole free-samples)", landed1)
landed2, mb2, mt2 = spec_step(h, s, out, [B, NAT])
assert s["end_count"] >= 2, ("T-G wildcard advance through hole", s)
assert mt2 == [(0, B)], ("T-G close re-forced at exact position", mt2)
assert landed2 == [B, NAT, NAT], ("T-G step2 landed", landed2)
landed3, _, _ = spec_step(h, s, out, [NAT, NAT])
assert calls["complete"] == 1, ("T-G completion count", calls)
assert out[6:9] == [A, NAT, B], ("T-G span layout probe/hole/close", out)
print("T-G OK: hole row free-samples, close is position-anchored "
      "(no overrun possible), wildcard walk completes exactly once")

print("ALL PN114 LOGIC TESTS PASSED")
