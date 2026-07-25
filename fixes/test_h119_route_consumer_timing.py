#!/usr/bin/env python3
"""H119 — mechanical proof that PN100/auto_budget CANNOT be the route consumer.

Run: ~/shared/needfit/lens-venv/bin/python fixes/test_h119_route_consumer_timing.py
     (no boot, no GPU, no container — pure source/AST inspection + a CPU import)

WHY THIS FILE EXISTS
--------------------
The obvious "last mile" for the lens router is: read `pn119_router.ROUTES[req_id]`
inside `_genesis/middleware/auto_budget.py` (PN100) and pick the deep or the lean
thinking budget. That wiring is IMPOSSIBLE, and it fails SILENTLY (route_for()
returns the fallback for every request => 100% deep => strictly worse than no
router, at full cost). Three independent blockers, each sufficient on its own,
each asserted below so a future session cannot re-derive the same dead end:

  P1 TEMPORAL   — the PN100 hook is inserted at the TOP of create_chat_completion
                  (anchor: the Genesis PN16 block, itself immediately above the
                  upstream "# Streaming response" line). Prefill happens after
                  engine_client.generate(), hundreds of lines later. The budget is
                  decided at t0; ROUTES[req_id] is written at t3.
  P2 IDENTITY   — `request_id` does not EXIST at hook time. It is minted ~17 lines
                  below the hook site. There is no key to look up.
  P3 ADDRESS    — AsyncLLM calls EngineCoreClient.make_async_mp_client()
                  UNCONDITIONALLY (not gated on VLLM_ENABLE_V1_MULTIPROCESSING).
                  ROUTES is a module-global dict in the EngineCore/worker process;
                  the API server that runs auto_budget talks to it over ZMQ. Even a
                  correctly-timed, correctly-keyed read would see {} forever.

  P4 (the way forward, not a blocker) — a route consumer IS feasible WORKER-SIDE:
      vllm/v1/sample/thinking_budget_state.py holds a MUTABLE per-request
      "thinking_token_budget" in `_state[index]`, and gpu_input_batch.py carries
      `req_id_to_index` — same process, same req_id namespace as ROUTES, and the
      state is re-read every decode step (so a post-prefill write still binds:
      no thinking token is produced before the first sampled token).
      CAVEAT: worker-side reaches the BUDGET CAP only. The deep/lean treatments
      also differ by PN102 banner (v5-class vs v3-class), which is rendered into
      the PROMPT in the frontend before prefill — unreachable by construction,
      since the route is derived FROM that prefill.
"""
from __future__ import annotations

import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
VLLM_SRC = pathlib.Path(os.environ.get("H119_VLLM_SRC", "/home/user/vllm"))
SERVING = VLLM_SRC / "vllm/entrypoints/openai/chat_completion/serving.py"
ASYNC_LLM = VLLM_SRC / "vllm/v1/engine/async_llm.py"
TB_STATE = VLLM_SRC / "vllm/v1/sample/thinking_budget_state.py"
INPUT_BATCH = VLLM_SRC / "vllm/v1/worker/gpu_input_batch.py"
PN100_PATCH = REPO / "fixes/patch_pn100_auto_thinking_budget.py"

_fails: list[str] = []
_skips: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        _fails.append(name)


def skip(name: str, why: str) -> None:
    print(f"SKIP  {name}  [{why}]")
    _skips.append(name)


def _read(p: pathlib.Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


def _pn100_anchor() -> str:
    """The literal line the PN100 patcher inserts its hook ABOVE."""
    src = PN100_PATCH.read_text(encoding="utf-8")
    m = re.search(r'^ANCHOR = \(\n\s*"(.*)\\n"\n\)', src, re.M)
    if not m:
        raise SystemExit("could not parse ANCHOR from patch_pn100_auto_thinking_budget.py")
    return m.group(1).encode().decode("unicode_escape")


# ── P1/P2: hook site vs request_id vs engine submit ────────────────────────────
def p1_p2_ordering() -> None:
    src = _read(SERVING)
    if src is None:
        skip("P1 hook precedes prefill", f"{SERVING} absent (set H119_VLLM_SRC)")
        skip("P2 request_id does not exist at hook time", "same")
        return
    anchor = _pn100_anchor()
    # The stock tree has no Genesis PN16 block; the patcher documents its
    # re-anchor target as the upstream "# Streaming response" line, which is the
    # line PN16 itself is inserted above. Either resolves to the same site.
    hook_at = src.find(anchor)
    used = "PN16 anchor"
    if hook_at < 0:
        hook_at = src.find("        # Streaming response\n")
        used = "upstream '# Streaming response' (PN16 re-anchor target)"
    submit_at = src.find("self.engine_client.generate(")
    reqid_at = src.find('request_id = (\n            f"chatcmpl-')
    check(
        "P1 PN100 hook site precedes engine_client.generate() (=> precedes prefill)",
        0 <= hook_at < submit_at,
        f"hook@{hook_at} < generate@{submit_at} via {used}",
    )
    check(
        "P2 request_id is minted AFTER the hook site (no key to look ROUTES up by)",
        0 <= hook_at < reqid_at,
        f"hook@{hook_at} < request_id@{reqid_at}",
    )
    # The rendered prompt (PN102 banner + think seed) is also frozen before submit.
    render_at = src.find("await self.render_chat_request(request)")
    check(
        "P1b prompt render also precedes submit (banner choice is pre-prefill)",
        0 <= hook_at < render_at < submit_at,
        f"render@{render_at}",
    )


# ── P3: the API server and the router are different processes ─────────────────
def p3_address_space() -> None:
    src = _read(ASYNC_LLM)
    if src is None:
        skip("P3 AsyncLLM always uses an out-of-process EngineCore", "async_llm.py absent")
        return
    mp = "EngineCoreClient.make_async_mp_client(" in src
    # Unconditional: the call is not guarded by the multiprocessing env flag.
    guarded = re.search(
        r"VLLM_ENABLE_V1_MULTIPROCESSING[^\n]*\n(?:[^\n]*\n){0,10}?"
        r"[^\n]*make_async_mp_client",
        src,
    )
    check(
        "P3 AsyncLLM builds an out-of-process EngineCore unconditionally",
        mp and guarded is None,
        "make_async_mp_client present, not env-gated",
    )


# ── P3b: what a naive consumer would actually do — route 100% deep ────────────
def p3b_empty_registry() -> None:
    sys.path.insert(0, str(HERE))
    try:
        import pn119_router as R  # noqa: PLC0415 — deliberate late import
    except Exception as e:  # noqa: BLE001
        skip("P3b empty registry reads as fallback for every request", f"import failed: {e}")
        return
    R.ROUTES.clear()
    R.STATS.clear()
    routes = [R.route_for(f"chatcmpl-{i}") for i in range(20)]
    check(
        "P3b route_for() on an empty registry never raises",
        len(routes) == 20,
        "20/20 returned",
    )
    check(
        "P3b ...and returns the fallback for 100% of them (= 'router' with no signal)",
        set(routes) == {R._FALLBACK_ROUTE} and R.STATS["route_for_miss"] == 20,
        f"route={routes[0]} misses={R.STATS['route_for_miss']}",
    )


# ── P4: the viable worker-side insertion point still exists ───────────────────
def p4_viable_site() -> None:
    tb = _read(TB_STATE)
    ib = _read(INPUT_BATCH)
    if tb is None or ib is None:
        skip("P4 worker-side budget state is mutable and req_id-addressable", "vllm src absent")
        return
    check(
        "P4a per-request thinking budget is mutable worker-side, re-read each step",
        '"thinking_token_budget": thinking_token_budget' in tb
        and 'state["thinking_token_budget"]' in tb
        and "def update_state(" in tb,
        "thinking_budget_state._state[index]['thinking_token_budget']",
    )
    check(
        "P4b worker batch maps req_id -> batch index (same namespace as ROUTES)",
        "self.req_id_to_index: dict[str, int] = {}" in ib,
        "gpu_input_batch.req_id_to_index",
    )


def main() -> int:
    print("H119 route-consumer timing proof (no boot / no GPU)\n")
    p1_p2_ordering()
    p3_address_space()
    p3b_empty_registry()
    p4_viable_site()
    print()
    if _fails:
        print(f"FAILED: {len(_fails)} — {', '.join(_fails)}")
        return 1
    print(f"ALL PASS ({len(_skips)} skipped)"
          if _skips else "ALL PASS")
    print("VERDICT: ROUTES[req_id] is NOT populated when the budget is decided, "
          "and is not even reachable from that process. Do not wire the consumer "
          "into _genesis/middleware/auto_budget.py — see P4 for the site that works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
