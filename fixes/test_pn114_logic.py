import importlib.util, sys, types
spec = importlib.util.spec_from_file_location("tbs", "/tmp/pn114/thinking_budget_state.py")
tbs = importlib.util.module_from_spec(spec); spec.loader.exec_module(tbs)
H = tbs.ThinkingBudgetStateHolder
START, END, A, B, C = 999, 998, 700, 701, 702

def mk():
    h = H.__new__(H)
    h.think_start_token_ids=[START]; h.think_end_token_ids=[END]
    h.implicit_think_end_token_ids=[]
    return h

def st(toks, budget=3):
    return {"thinking_token_budget":budget,"in_think":True,"in_end":False,
            "think_count":0,"check_count_down":budget,"force_index":[],
            "start_thinking":-1,"end_thinking":-1,"in_spec_mode":False,
            "bonus_token_forced":False,"continue_thinking":False,
            "scan_offset":0,"output_tok_ids":toks,"prev_output_length":0,
            "spec_token_ids":[],"end_count":0}

# T1 stock equivalence: budget exceed -> in_end forcing (no force_seq)
h=mk(); s=st([START,10,11,12,13])
h._update_think_state(s)
assert s["in_end"] and s["force_index"]==[0], ("T1",s)
print("T1 OK: stock budget-force unchanged")

# T2 wrapup-style: force_seq completes -> normal answer-mode reset + force_seq popped
h=mk(); s=st([START,10,11], budget=100)
s.update({"in_think":False,"in_end":True,"end_count":0,"force_index":[0],
          "force_seq":[A,B,END],"start_thinking":0,"prev_output_length":3,
          "_pn114":{"phase":"wrapup"}})
# stub pn114: wrapup returns False (fall through to normal reset)
stub=types.ModuleType("pn114")
def ofc(state):
    p=state.get("_pn114") or {}
    if p.get("phase")=="wrapup": p["phase"]=None; state["force_seq"]=None; return False
    if p.get("phase")=="probe_force":
        p["phase"]="probe_free"; state["in_end"]=False; state["in_think"]=True
        state["end_count"]=0; state["force_index"]=[]; state["force_seq"]=None; return True
    return False
stub.on_force_complete=ofc
pkg=types.ModuleType("vllm._genesis.plateau"); pkg.pn114=stub
sys.modules["vllm._genesis.plateau"]=pkg; sys.modules["vllm._genesis.plateau.pn114"]=stub
# simulate: forced tokens A,B,END landed one per step
for tok in (A,B,END):
    s["output_tok_ids"].append(tok)
    s["prev_output_length"]=len(s["output_tok_ids"])-1
    h._update_think_state(s)
assert not s["in_end"] and s.get("force_seq") is None and s["think_count"]==0, ("T2",s)
print("T2 OK: wrapup span forces through + normal reset (force_seq cleared)")

# T3 probe divert: phase probe_force -> resume think, NO answer-mode reset
h=mk(); s=st([START,10,11], budget=100)
s.update({"in_think":False,"in_end":True,"end_count":0,"force_index":[0],
          "force_seq":[A,B],"start_thinking":0,"prev_output_length":3,
          "_pn114":{"phase":"probe_force"}})
for tok in (A,B):
    s["output_tok_ids"].append(tok)
    s["prev_output_length"]=len(s["output_tok_ids"])-1
    h._update_think_state(s)
assert s["in_think"] and not s["in_end"] and s["start_thinking"]==0, ("T3",s)
assert s["_pn114"]["phase"]=="probe_free", ("T3 phase",s["_pn114"])
print("T3 OK: probe span diverts to free window, think state preserved")

# T4 natural-abort guard: armed span survives the first pass (no span token yet)
h=mk(); s=st([START,10,11], budget=100)
s.update({"in_think":False,"in_end":True,"end_count":0,"force_index":[0],
          "force_seq":[A,B],"start_thinking":0,"prev_output_length":3,
          "_pn114":{"phase":"probe_force"}})
h._update_think_state(s)   # no new tokens landed yet
assert s["in_end"], ("T4: span was cancelled by natural-abort", s)
print("T4 OK: armed span survives pre-force pass")
