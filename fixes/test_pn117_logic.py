"""PN117 deep-band rescue detector + injection logic tests (2026-07-23, M9).

Pure-python engine-sim (no GPU); clones test_pn114_module.py's harness. Loads
pn117_rescue + pn114 from the G tree under a stubbed vllm._genesis.plateau
package (pn108 stubbed so the live think-slice falls back to think_count), a
stubbed vllm.logger, and an isolated boot-ids file.

Covers the M9 kill criteria:
  (i)   no fire outside the [WIN_LO, WIN_HI] think window
  (ii)  fire only when c_mean < CTHRESH
  (iii) injection only at a sentence boundary (prev landed token in send set)
  (iv)  the span lands intact and thinking resumes with the budget UNCHARGED
  (v)   shadow mode changes NOTHING (no pending, no force_seq)
  (vi)  the converge trigger fires at CONV_FRAC without any conf condition
  (vii) never fires twice per think block
  (viii)never arms while pn114.phase_active() (a span already in flight)
  (+)   tiny-budget requests are skipped (min-budget floor)
"""
import importlib.util
import json
import os
import sys
import types

# ---- clean env ------------------------------------------------------------
for _v in ("GENESIS_ENABLE_PN117_RESCUE", "GENESIS_PN117_MODE",
           "GENESIS_PN117_ARM", "GENESIS_PN117_CTHRESH", "GENESIS_PN117_WIN_LO",
           "GENESIS_PN117_WIN_HI", "GENESIS_PN117_MIN_SAMPLES",
           "GENESIS_PN117_MIN_BUDGET", "GENESIS_PN117_CONVERGE",
           "GENESIS_PN117_CONV_FRAC"):
    os.environ.pop(_v, None)

# ---- stub vllm.logger + vllm._genesis.plateau.pn108 -----------------------
import logging  # noqa: E402

_vllm = types.ModuleType("vllm")
_vllm_logger = types.ModuleType("vllm.logger")
_vllm_logger.init_logger = lambda name: logging.getLogger(name)
sys.modules.setdefault("vllm", _vllm)
sys.modules["vllm.logger"] = _vllm_logger

_pn108 = types.ModuleType("vllm._genesis.plateau.pn108")
# return None -> pn114._live_think_len falls back to state["think_count"]
_pn108._think_token_slice = lambda state, tsl: None

_pkg = types.ModuleType("vllm._genesis.plateau")
_pkg.__path__ = []  # mark as package
sys.modules["vllm._genesis"] = types.ModuleType("vllm._genesis")
sys.modules["vllm._genesis.plateau"] = _pkg
sys.modules["vllm._genesis.plateau.pn108"] = _pn108
_pkg.pn108 = _pn108

_HERE = os.path.dirname(os.path.abspath(__file__))
_G = os.path.join(_HERE, "..", "models", "qwen3.6-27b", "vllm", "patches",
                  "genesis", "vllm", "_genesis", "plateau")


def _load(modname, filename):
    path = os.path.abspath(os.path.join(_G, filename))
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


pn114 = _load("vllm._genesis.plateau.pn114", "pn114.py")
pn117 = _load("vllm._genesis.plateau.pn117_rescue", "pn117_rescue.py")
_pkg.pn114 = pn114
_pkg.pn117_rescue = pn117

# ---- isolated boot-ids file -----------------------------------------------
DOT, NL = 46, 10           # sentence-end token ids
ARM1 = [701, 702, 703]
ARM3 = [710]
ARM5 = [720, 721]
_IDS = "/tmp/pn117_test_ids.json"
json.dump({"arm1": ARM1, "arm3": ARM3, "arm5": ARM5,
           "sentence_end": [DOT, NL]}, open(_IDS, "w"))
pn114._IDS_PATH = _IDS
pn114._IDS = None  # force reload from the isolated path

NAT = 555  # a natural (non-boundary) think token


def mkstate(budget=4000, out_len=100):
    return {"thinking_token_budget": budget, "start_thinking": 0,
            "end_thinking": -1, "think_count": out_len,
            "output_tok_ids": [NAT] * out_len, "in_think": True,
            "in_end": False, "end_count": 0, "force_index": [],
            "check_count_down": budget}


def feed(s, think_len, conf, last=NAT, req="t"):
    """One observe step at the given think depth / conf, with `last` as the
    most-recently-landed output token (controls the sentence-boundary check)."""
    if s["output_tok_ids"]:
        s["output_tok_ids"][-1] = last
    pn117.observe(s, think_len, conf, req)


# ==========================================================================
# (i) no fire outside the window; (ii) fire only under CTHRESH; (vii) once
# ==========================================================================
os.environ["GENESIS_ENABLE_PN117_RESCUE"] = "1"
os.environ["GENESIS_PN117_MODE"] = "enforce"
os.environ["GENESIS_PN117_ARM"] = "1"
os.environ["GENESIS_PN117_CTHRESH"] = "13.0"

# low conf BELOW the window -> no fire
s = mkstate()
for tl in range(400, 560, 8):
    feed(s, tl, 8.0)
assert not s["_pn117"]["fired"], ("(i) fired below window", s["_pn117"])
assert "force_seq" not in s or s.get("force_seq") is None, "(i) armed below win"

# HIGH conf inside the window -> no fire (settled grinder is not rescued)
s = mkstate()
for tl in range(600, 800, 8):
    feed(s, tl, 20.0)
assert not s["_pn117"]["fired"], ("(ii) fired on high conf", s["_pn117"])

# low conf ABOVE the window (converge off) -> no fire
s = mkstate()
for tl in range(820, 980, 8):
    feed(s, tl, 8.0)
assert not s["_pn117"]["fired"], ("(i) fired above window", s["_pn117"])
print("T1 OK: no fire below/above window, no fire on high conf")

# low conf INSIDE the window -> FIRE (pending queued, not yet armed off-boundary)
s = mkstate()
for tl in range(600, 720, 8):
    feed(s, tl, 8.0, last=NAT)  # last token non-boundary
st = s["_pn117"]
assert st["fired"] and st["reason"] == "cband", ("(ii) cband fire", st)
assert st["pending"] == "arm1", ("(iii) pending queued", st)
assert st["c_mean"] is not None and st["c_mean"] < 13.0, ("c_mean", st)
assert s.get("force_seq") in (None,), ("(iii) armed off-boundary", s.get("force_seq"))
# (vii) more low-conf steps must NOT re-fire / change the pending arm
_before = dict(st)
feed(s, 728, 8.0, last=NAT)
assert st["fired"] and st["pending"] == "arm1", ("(vii) refired", st)
print("T2 OK: conf-band fires under CTHRESH once, injection pends off-boundary")

# ==========================================================================
# (iii) injection only at a sentence boundary; (iv) span lands, budget uncharged
# ==========================================================================
ORIG = s["thinking_token_budget"]
# still off-boundary -> no arm
feed(s, 736, 8.0, last=NAT)
assert s.get("force_seq") in (None,), ("(iii) armed off-boundary again", s)
# now the previous landed token IS a sentence end -> arm this step
feed(s, 744, 8.0, last=DOT)
assert s.get("force_seq") == ARM1, ("(iii) not armed at boundary", s.get("force_seq"))
assert s["in_end"] and s["force_index"] == [0], ("(iii) forcer not primed", s)
assert s["thinking_token_budget"] == 10_000_000, ("(iv) budget not parked", s)
assert s["_pn114"]["saved_budget"] == ORIG, ("(iv) saved budget", s["_pn114"])
assert st["armed"] and st["pending"] is None, ("(iii) arm flags", st)
# --- emulate the holder landing the span, then completion divert ---
s["output_tok_ids"].extend(ARM1)
s["end_count"] = len(ARM1)
handled = pn114.on_force_complete(s)
assert handled is True, "(iv) pn117 completion must skip answer-mode reset"
assert s.get("force_seq") is None and "force_seq_base" not in s, ("(iv) span keys leak", s)
assert s["in_think"] and not s["in_end"], ("(iv) did not resume think", s)
assert s["thinking_token_budget"] == ORIG + len(ARM1), (
    "(iv) budget NOT uncharged", s["thinking_token_budget"], ORIG, len(ARM1))
assert s["_pn114"]["phase"] is None, ("(iv) phase not cleared", s["_pn114"])
print("T3 OK: injection arms only at sentence boundary; span lands intact, "
      "thinking resumes with budget uncharged")

# ==========================================================================
# (v) shadow mode changes NOTHING
# ==========================================================================
os.environ["GENESIS_PN117_MODE"] = "shadow"
s = mkstate()
for tl in range(600, 760, 8):
    feed(s, tl, 8.0, last=DOT)  # boundary present every step, yet shadow
st = s["_pn117"]
assert st["fired"], ("(v) shadow should still detect", st)
assert st["pending"] is None, ("(v) shadow queued an injection", st)
assert s.get("force_seq") in (None,), ("(v) shadow armed", s.get("force_seq"))
assert s["thinking_token_budget"] == 4000, ("(v) shadow touched budget", s)
assert "_pn114" not in s, ("(v) shadow created forcer state", s.keys())
print("T4 OK: shadow mode logs would-fire but changes nothing")

# ==========================================================================
# (vi) converge trigger fires at CONV_FRAC without any conf condition
# ==========================================================================
os.environ["GENESIS_PN117_MODE"] = "enforce"
os.environ["GENESIS_PN117_CONVERGE"] = "1"
os.environ["GENESIS_PN117_CONV_FRAC"] = "0.7"
s = mkstate(budget=4000)
# well inside the budget, HIGH conf -> conf-band never fires; converge waits
feed(s, 1000, 25.0, last=NAT)
assert not s["_pn117"]["fired"], ("(vi) converge fired early", s["_pn117"])
# cross 0.7 * 4000 = 2800 with HIGH conf and a boundary -> converge fires+arms
feed(s, 2808, 25.0, last=DOT)
st = s["_pn117"]
assert st["fired"] and st["reason"] == "converge", ("(vi) converge reason", st)
assert s.get("force_seq") == ARM5, ("(vi) converge arm != arm5", s.get("force_seq"))
print("T5 OK: converge trigger fires at CONV_FRAC with no conf gate, injects arm5")
os.environ["GENESIS_PN117_CONVERGE"] = "0"

# ==========================================================================
# (viii) never arms while pn114.phase_active() (a span already in flight)
# ==========================================================================
s = mkstate()
# simulate a pn114 probe span already in flight
s["_pn114"] = {"phase": "probe_force", "saved_budget": 4000}
s["force_seq"] = [601, 602]
for tl in range(600, 760, 8):
    feed(s, tl, 8.0, last=DOT)
assert "_pn117" not in s or not s["_pn117"].get("fired"), (
    "(viii) fired while pn114 phase active", s.get("_pn117"))
assert s["force_seq"] == [601, 602], ("(viii) clobbered pn114 span", s["force_seq"])
print("T6 OK: never observes/arms while a pn114 span is in flight")

# ==========================================================================
# (+) tiny-budget requests are skipped
# ==========================================================================
s = mkstate(budget=500)  # below default min-budget 1024
for tl in range(600, 760, 8):
    feed(s, tl, 8.0, last=DOT)
assert not s["_pn117"]["fired"], ("(+) fired on tiny budget", s["_pn117"])
assert s.get("force_seq") in (None,), ("(+) armed on tiny budget", s)
print("T7 OK: tiny-budget requests skipped (min-budget floor)")

print("ALL PN117 LOGIC TESTS PASSED")
