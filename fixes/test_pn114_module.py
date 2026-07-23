import importlib.util, json, os
json.dump({"probe":[601,602,603],"newline":[604],"wrapup_close":[605,606,998]},
          open("/tmp/genesis_pn114_ids.json","w"))
os.environ["GENESIS_ENABLE_PN114_PROBE"]="1"
os.environ["GENESIS_PN114_MODE"]="enforce"
os.environ["GENESIS_PN114_DEPTHS"]="100,200"
os.environ["GENESIS_PN114_STABLE_K"]="2"
os.environ["GENESIS_PN114_CMIN"]="13.0"
spec=importlib.util.spec_from_file_location("pn114","/tmp/pn114/pn114.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

s={"thinking_token_budget":5000,"think_count":120,"output_tok_ids":[0]*120,
   "in_think":True,"in_end":False,"end_count":0,"force_index":[],
   "check_count_down":4880,"spec_token_ids":[]}
# depth 100 passed -> arm
m.observe_state(s, 1, 0, conf=14.0, req_id="t")
st=s["_pn114"]
assert st["phase"]=="probe_force" and s["force_seq"]==[601,602,603] and s["in_end"], st
assert s["thinking_token_budget"]==10_000_000, "budget trigger not parked"
# span completes
assert m.on_force_complete(s) is True and st["phase"]=="probe_free"
# 3 free tokens (letter=777) with confs
s["output_tok_ids"]+= [777]
m.observe_state(s,1,0,conf=14.2,req_id="t")
s["output_tok_ids"]+= [55,56]
m.observe_state(s,1,0,conf=14.1,req_id="t")
assert st["phase"]=="probe_nl" and s["force_seq"]==[604], (st["phase"],s.get("force_seq"))
# newline lands
assert m.on_force_complete(s) is True and st["phase"] is None
assert st["probes"][-1]["letter"]==777 and s["thinking_token_budget"]==5000+3+3+1, (st["probes"],s["thinking_token_budget"])
print("M1 OK: full probe cycle (arm/force/free/nl/resume, budget compensated)")

# second probe at depth 200, same letter -> stability close (enforce)
s["think_count"]=210
m.observe_state(s,1,0,conf=14.0,req_id="t")
assert st["phase"]=="probe_force"
m.on_force_complete(s)
s["output_tok_ids"]+= [777]
m.observe_state(s,1,0,conf=14.3,req_id="t")
s["output_tok_ids"]+= [55,56]
m.observe_state(s,1,0,conf=14.3,req_id="t")
m.on_force_complete(s)
assert st["phase"] is None
assert s["thinking_token_budget"]==s["think_count"]+384, ("close not applied",s["thinking_token_budget"],s["think_count"])
print("M2 OK: stable x2 same letter -> enforce close (budget=think+grace)")

# M3 confirm path: weak conf cancels + resets pn112 streak
os.environ["GENESIS_PN112_CONFIRM"]="1"
s2={"thinking_token_budget":5000,"think_count":900,"output_tok_ids":[0]*900,
    "in_think":True,"in_end":False,"end_count":0,"force_index":[],
    "check_count_down":4100,"spec_token_ids":[],
    "_pn112":{"streak":3,"fired":True}}
assert m.request_confirm(s2,"t2") is True
st2=s2["_pn114"]; assert st2["phase"]=="probe_force" and st2["reason"]=="confirm"
m.on_force_complete(s2)
s2["output_tok_ids"]+=[888]; m.observe_state(s2,1,0,conf=10.0,req_id="t2")
s2["output_tok_ids"]+=[1,2]; m.observe_state(s2,1,0,conf=10.0,req_id="t2")
m.on_force_complete(s2)
assert s2["_pn112"]["streak"]==0 and s2["_pn112"]["fired"] is False, s2["_pn112"]
assert s2["thinking_token_budget"]!=s2["think_count"]+384, "weak confirm must NOT cut"
print("M3 OK: weak confirm cancels fire + resets PN112 streak")
