# In-container logic test for patch_pr44812_tool_guard.py: copy a PATCHED
# thinking_budget_state.py to /tmp/pr44812/ inside the container, then run
# this file with the container python (needs torch importable).
# Validated 2026-07-23: T1 implicit close / T2 slice fix / T3 dark=stock.
import importlib.util
spec = importlib.util.spec_from_file_location("tbs", "/tmp/pr44812/thinking_budget_state.py")
tbs = importlib.util.module_from_spec(spec); spec.loader.exec_module(tbs)
H = tbs.ThinkingBudgetStateHolder
START, END, TOOL = 999, 998, 997

def mk(implicit):
    h = H.__new__(H)
    h.think_start_token_ids = [START]; h.think_end_token_ids = [END]
    h.implicit_think_end_token_ids = implicit
    return h

def st(toks, offset=0):
    return {"thinking_token_budget": 3, "in_think": True, "in_end": False,
            "think_count": 0, "check_count_down": 3, "force_index": [],
            "start_thinking": -1, "end_thinking": -1, "in_spec_mode": False,
            "bonus_token_forced": False, "continue_thinking": False,
            "scan_offset": offset, "output_tok_ids": toks,
            "prev_output_length": 0, "spec_token_ids": []}

# 1. implicit end inside think closes thinking
h = mk([[TOOL]]); s = st([START, 10, TOOL, 20, 21])
h._update_think_state(s)
assert not s["in_think"] and s["start_thinking"] == -1, ("T1 FAIL", s)
print("T1 OK: implicit <tool_call> closes thinking (in_think False)")

# 2. slice fix: stale pre-offset TOOL must NOT register as end
h = mk([[TOOL]]); s = st([TOOL, 1, 2, 3, 4, START, 10, 20], offset=5)
h._update_think_state(s)
assert s["end_thinking"] == -1, ("T2 FAIL: stale tool_call detected", s["end_thinking"])
assert s["start_thinking"] == 5, ("T2 FAIL start", s["start_thinking"])
print("T2 OK: stale pre-think <tool_call> ignored (slice fix)")

# 3. dark mode: empty implicit list == stock behavior
h = mk([]); s = st([START, 10, TOOL, 20, 21])
h._update_think_state(s)
assert s["end_thinking"] == -1, ("T3 FAIL: implicit detected while dark", s)
assert s["in_end"], ("T3 FAIL: stock budget-force should be active", s)
print("T3 OK: flag-dark = no implicit detection; stock budget-force intact")
