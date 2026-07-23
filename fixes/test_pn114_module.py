import importlib.util, json, os
# ultra-review (cut-list): NEVER write the production ids path — a live
# engine in the same container would lazy-load garbage ids. Isolated path.
_IDS="/tmp/pn114_test_ids.json"
json.dump({"probe":[601,602,603],"newline":[604],"close_paren":[604],
           "wrapup_close":[605,606,998]}, open(_IDS,"w"))
os.environ["GENESIS_ENABLE_PN114_PROBE"]="1"
os.environ["GENESIS_PN114_FREE_LEN"]="3"  # test the 3-token contract explicitly
os.environ["GENESIS_PN114_MODE"]="enforce"
os.environ["GENESIS_PN114_DEPTHS"]="100,200"
os.environ["GENESIS_PN114_STABLE_K"]="2"
os.environ["GENESIS_PN114_CMIN"]="13.0"
spec=importlib.util.spec_from_file_location("pn114","/tmp/pn114/pn114.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m._IDS_PATH=_IDS  # isolate from production ids file

SPAN=[601,602,603,None,None,None,604]  # probe + free_len holes + close

# start_thinking/end_thinking present: depth arming measures the LIVE pn108
# think-slice (state["think_count"] is frozen while under budget — the
# 2026-07-23 canary finding), so the slice must be derivable.
s={"thinking_token_budget":5000,"think_count":120,"output_tok_ids":[0]*120,
   "in_think":True,"in_end":False,"end_count":0,"force_index":[],
   "check_count_down":4880,"spec_token_ids":[],
   "start_thinking":0,"end_thinking":-1}
# depth 100 passed -> arm
m.observe_state(s, 1, 0, conf=14.0, req_id="t")
st=s["_pn114"]
# unified span redesign: NO prepend, ONE span with a free hole, base recorded
assert st["phase"]=="probe_force" and s["force_seq"]==SPAN and s["in_end"], st
assert s["force_seq_base"]==120, ("base not recorded at arm", s.get("force_seq_base"))
assert s["thinking_token_budget"]==10_000_000, "budget trigger not parked"
# span tokens LAND (probe text, letter 777 + 2 free, close) — the holder's
# B-site walk advances end_count positionally; emulate it for conf capture.
s["output_tok_ids"]+=[601,602,603,777,55,56,604]
s["end_count"]=4
m.observe_state(s,1,0,conf=14.2,req_id="t")   # in-flight hole-region confs
m.observe_state(s,1,0,conf=14.1,req_id="t")
assert st["free_confs"]==[14.2,14.1], ("conf capture", st["free_confs"])
assert m.on_force_complete(s) is True and st["phase"] is None
assert st["free_start"]==123, ("letter position", st["free_start"])
assert st["probes"][-1]["letter"]==777 and s["thinking_token_budget"]==5000+3+3+1, (st["probes"],s["thinking_token_budget"])
assert s.get("force_seq") is None and "force_seq_base" not in s, ("span keys leak", s.keys())
assert not s["in_end"] and s["in_think"], ("resume think", s)
print("M1 OK: unified probe span (arm/force+hole/close/resume, budget compensated, no prepend)")

# second probe at depth 200, same letter -> stability close (enforce)
s["think_count"]=210
s["output_tok_ids"]+=[0]*90   # live slice (216) must cross depth 200
m.observe_state(s,1,0,conf=14.0,req_id="t")
assert st["phase"]=="probe_force" and s["force_seq"]==SPAN
assert s["force_seq_base"]==217, ("M2 base", s.get("force_seq_base"))
s["output_tok_ids"]+=[601,602,603,777,55,56,604]
s["end_count"]=4
m.observe_state(s,1,0,conf=14.3,req_id="t")
m.observe_state(s,1,0,conf=14.3,req_id="t")
m.on_force_complete(s)
assert st["phase"] is None
assert st["probes"][-1]["letter"]==777, ("M2 letter", st["probes"])
assert s["thinking_token_budget"]==s["think_count"]+384, ("close not applied",s["thinking_token_budget"],s["think_count"])
print("M2 OK: stable x2 same letter -> enforce close (budget=think+grace)")

# M3 confirm path: weak conf cancels + resets pn112 streak
os.environ["GENESIS_PN112_CONFIRM"]="1"
s2={"thinking_token_budget":5000,"think_count":900,"output_tok_ids":[0]*900,
    "in_think":True,"in_end":False,"end_count":0,"force_index":[],
    "check_count_down":4100,"spec_token_ids":[],
    "start_thinking":0,"end_thinking":-1,
    "_pn112":{"streak":3,"fired":True}}
assert m.request_confirm(s2,"t2") is True
st2=s2["_pn114"]; assert st2["phase"]=="probe_force" and st2["reason"]=="confirm"
assert s2["force_seq"]==SPAN and s2["force_seq_base"]==900, (s2.get("force_seq"),s2.get("force_seq_base"))
s2["output_tok_ids"]+=[601,602,603,888,1,2,604]
s2["end_count"]=4
m.observe_state(s2,1,0,conf=10.0,req_id="t2")
m.observe_state(s2,1,0,conf=10.0,req_id="t2")
m.on_force_complete(s2)
assert st2["probes"][-1]["letter"]==888, ("letter capture", st2["probes"])
assert s2["_pn112"]["streak"]==0 and s2["_pn112"]["fired"] is False, s2["_pn112"]
assert s2["thinking_token_budget"]!=s2["think_count"]+384, "weak confirm must NOT cut"
print("M3 OK: weak confirm cancels fire + resets PN112 streak")
