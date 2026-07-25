#!/usr/bin/env python3
"""H119 enforce route consumer — behaviour + dual-pin anchors (no boot, no GPU).

Run: ~/shared/needfit/lens-venv/bin/python fixes/test_h119_budget_consumer.py

Nothing here starts a service, touches the GPU or runs a benchmark. The REAL
vllm/v1/sample/thinking_budget_state.py is extracted from each pinned image
with `podman run --rm --entrypoint cat`, patched by the real
fixes/patch_h119_lens_router.py anchors, exec'd against stubbed vllm modules,
and driven directly.

What is pinned
--------------
P1  ANCHORS: sites E/F/G are count==1 on BOTH pins
    (dev1060cherry-20260713 and dev1474cherry-1711-20260725) and the patched
    file byte-compiles on both. If the file is absent on a pin, the patcher
    soft-skips instead of failing.
P2  FLAG OFF == STOCK. With GENESIS_ENABLE_H119_ROUTE_BUDGET unset, the patched
    holder's `_state` is deep-equal to the STOCK holder's `_state` after the
    same batch add + decode steps, for both the budgeted and the unbudgeted
    request. Same for flag-on-but-router-in-shadow and flag-on-but-no-router:
    the consumer refuses to act unless all three conditions hold.
P3  ROUTE DEEP: an unbudgeted request whose route is "deep" ends up capped at
    H119_DEEP_BUDGET, counter h119_routed_deep.
P4  ROUTE LEAN: ... at H119_LEAN_BUDGET, counter h119_routed_lean, and
    check_count_down is shifted by the same delta as the budget (the entry's
    `budget - think_count` invariant survives the rewrite).
P5  ROUTE MISSING is COUNTED, never silent: a request that has begun generating
    with no decision on record commits to route_for()'s defined fallback and
    bumps both h119_route_missing and the router's own route_for_miss. A
    request that is merely still prefilling is NOT a miss and burns no
    fallback.
P6  CALLER EXPLICIT WINS: params.thinking_token_budget is never overridden even
    when a contradicting route is on record; it is only counted.
P7  THE CAP IS REAL. Driven to the lean budget with a tiny cap, the holder
    raises in_end / force_index at the routed count — i.e. the rewrite actually
    reaches the forcing machinery, it is not a cosmetic dict edit.
P8  TIMING: resolution happens on the first update_state after the route is
    published, while output_tok_ids is still EMPTY — no thinking token has
    escaped the cap.
"""
from __future__ import annotations

import copy
import importlib.util
import os
import py_compile
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHER = os.path.join(HERE, "patch_h119_lens_router.py")
SIDECAR = os.path.join(HERE, "pn119_router.py")
TBS_REL = "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/thinking_budget_state.py"
PINS = ("dev1060cherry-20260713", "dev1474cherry-1711-20260725")

THINK_START = [900]
THINK_END = [901]

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ── stub the vllm modules thinking_budget_state.py imports ────────────────
class MoveDirectionality:
    UNIDIRECTIONAL = 0
    SWAP = 1


class BatchUpdate:
    def __init__(self, added=(), removed=(), moved=()):
        self.added = list(added)
        self.removed = list(removed)
        self.moved = list(moved)

    def __bool__(self):
        return bool(self.added or self.removed or self.moved)


def _install_vllm_stubs() -> None:
    import torch  # noqa: F401 — the real thing; the module imports it

    for name in ("vllm", "vllm.platforms", "vllm.utils", "vllm.utils.torch_utils",
                 "vllm.v1", "vllm.v1.sample", "vllm.v1.sample.logits_processor",
                 "vllm.v1.sample.logits_processor.interface", "vllm.config",
                 "vllm.config.reasoning"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vllm.platforms"].current_platform = types.SimpleNamespace(
        is_rocm=lambda: False, is_cuda=lambda: True)
    sys.modules["vllm.utils.torch_utils"].async_tensor_h2d = lambda *a, **k: None
    iface = sys.modules["vllm.v1.sample.logits_processor.interface"]
    iface.BatchUpdate = BatchUpdate
    iface.MoveDirectionality = MoveDirectionality
    sys.modules["vllm.config.reasoning"].ReasoningConfig = object
    # Package attribute wiring so `from vllm import _genesis_pn119` resolves.
    sys.modules["vllm"].platforms = sys.modules["vllm.platforms"]


def _load_sidecar():
    """Import fixes/pn119_router.py under the name the shims look for."""
    spec = importlib.util.spec_from_file_location("vllm._genesis_pn119", SIDECAR)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vllm._genesis_pn119"] = mod
    sys.modules["vllm"]._genesis_pn119 = mod
    spec.loader.exec_module(mod)
    return mod


def _load_patcher_ns():
    src = open(PATCHER, encoding="utf-8").read().replace("sys.exit(main())", "")
    ns = {"__name__": "h119_patcher_under_test"}
    exec(compile(src, PATCHER, "exec"), ns)
    return ns


def _extract(tag: str) -> str | None:
    try:
        out = subprocess.run(
            ["sudo", "podman", "run", "--rm", "--entrypoint", "cat",
             f"localhost/vllm-qwen36-endgame:{tag}", TBS_REL],
            capture_output=True, timeout=180)
    except Exception:
        return None
    return out.stdout.decode("utf-8") if out.returncode == 0 else None


def _exec_module(source: str, name: str):
    mod = types.ModuleType(name)
    mod.__dict__["__name__"] = name
    exec(compile(source, name, "exec"), mod.__dict__)
    return mod


# ── driving the holder ────────────────────────────────────────────────────
class _Params:
    def __init__(self, budget=None):
        self.thinking_token_budget = budget


class _FakeRouter:
    """Only the two attributes the consumer reads off a live router."""

    def __init__(self, mode: str, req_ids: list[str]):
        self.mode = mode
        self.runner = types.SimpleNamespace(
            input_batch=types.SimpleNamespace(req_ids=req_ids))


def _holder(mod):
    rc = types.SimpleNamespace(reasoning_start_token_ids=THINK_START,
                               reasoning_end_token_ids=THINK_END)
    import torch
    return mod.ThinkingBudgetStateHolder(rc, 8, 0, torch.device("cpu"), False)


def _add(holder, index, budget, prompt, out_ids):
    holder.sync_batch(BatchUpdate(added=[(index, _Params(budget), prompt, out_ids)]))


def _reset(side, *, flag, mode, router_present, deep=None, lean=None,
           req_ids=("r0",)):
    """Put the sidecar into a defined consumer configuration."""
    side.STATS.clear()
    side.ROUTES.clear()
    side.SCORES.clear()
    for k, v in (("GENESIS_ENABLE_H119_ROUTE_BUDGET", "1" if flag else None),
                 ("H119_DEEP_BUDGET", deep), ("H119_LEAN_BUDGET", lean)):
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    side.reset_consumer_cache()
    side.ROUTER = _FakeRouter(mode, list(req_ids)) if router_present else None


# ── the per-pin behavioural suite ─────────────────────────────────────────
def run_pin(tag: str, stock_src: str, ns) -> None:
    print(f"\n─── pin {tag} ({len(stock_src.splitlines())} lines) ───")

    # P1 — anchors + byte-compile
    text = stock_src
    for name, old, new in (("E", ns["E_OLD"], ns["E_NEW"]),
                           ("F", ns["F_OLD"], ns["F_NEW"]),
                           ("G", ns["G_OLD"], ns["G_NEW"])):
        c = text.count(old)
        check(f"{tag} P1 site {name} anchor count == 1", c == 1, f"count={c}")
        if c != 1:
            return
        text = text.replace(old, new, 1)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(text)
        patched_path = fh.name
    try:
        py_compile.compile(patched_path, doraise=True, cfile=patched_path + "c")
        check(f"{tag} P1 patched file byte-compiles", True)
    except Exception as e:  # noqa: BLE001
        check(f"{tag} P1 patched file byte-compiles", False, str(e))
        return
    finally:
        os.unlink(patched_path)

    stock = _exec_module(stock_src, f"tbs_stock_{tag}")
    patched = _exec_module(text, f"tbs_patched_{tag}")
    side = _load_sidecar()

    prompt = [1, 2, 3, THINK_START[0]]  # thinking-on template shape
    DEEP, LEAN = 10240, 800

    # ── P2 flag OFF == stock ────────────────────────────────────────────
    for label, kw in (("flag-off", dict(flag=False, mode="enforce", router_present=True)),
                      ("router-shadow", dict(flag=True, mode="shadow", router_present=True)),
                      ("no-router", dict(flag=True, mode="enforce", router_present=False))):
        _reset(side, **kw)
        side.ROUTES["r0"] = "lean"
        hs, hp = _holder(stock), _holder(patched)
        for h in (hs, hp):
            _add(h, 0, None, prompt, [])       # unbudgeted
            _add(h, 1, 4096, prompt, [])       # caller-budgeted
            h.update_state([[], []], None)
        check(f"{tag} P2 {label}: _state identical to stock",
              hs._state == hp._state,
              f"stock keys={sorted(hs._state)} patched keys={sorted(hp._state)}")
        check(f"{tag} P2 {label}: unbudgeted row still untracked",
              0 not in hp._state)
        check(f"{tag} P2 {label}: no routing counters fired",
              not any(k.startswith("h119_routed") for k in side.STATS),
              side.stats_line())

    # ── P3 route deep ───────────────────────────────────────────────────
    _reset(side, flag=True, mode="enforce", router_present=True)
    side.ROUTES["r0"] = "deep"
    h = _holder(patched)
    _add(h, 0, None, prompt, [])
    check(f"{tag} P3 provisional entry created at add", 0 in h._state)
    check(f"{tag} P3 provisional carries the deep default",
          h._state[0]["thinking_token_budget"] == DEEP)
    check(f"{tag} P3 has_tracked_requests() true before the route lands",
          h.has_tracked_requests())
    h.update_state([[]], None)
    check(f"{tag} P3 deep route -> deep budget",
          h._state[0]["thinking_token_budget"] == DEEP)
    check(f"{tag} P3 counter h119_routed_deep", side.STATS.get("h119_routed_deep") == 1,
          side.stats_line())

    # ── P4 route lean + P8 timing ───────────────────────────────────────
    _reset(side, flag=True, mode="enforce", router_present=True)
    side.ROUTES["r0"] = "lean"
    h = _holder(patched)
    _add(h, 0, None, prompt, [])
    pre_countdown = h._state[0]["check_count_down"]
    h.update_state([[]], None)                 # first sampler call, no tokens yet
    st = h._state[0]
    check(f"{tag} P4 lean route -> lean budget", st["thinking_token_budget"] == LEAN)
    check(f"{tag} P4 check_count_down shifted by the same delta",
          st["check_count_down"] == pre_countdown + (LEAN - DEEP),
          f"{pre_countdown} -> {st['check_count_down']}")
    check(f"{tag} P4 counter h119_routed_lean", side.STATS.get("h119_routed_lean") == 1,
          side.stats_line())
    check(f"{tag} P8 resolved before any token was sampled",
          st["output_tok_ids"] == [] and st["think_count"] == 0)
    check(f"{tag} P8 entry is no longer provisional",
          st.get(side.H119_PROVISIONAL) is False)

    # ── P5 route missing ────────────────────────────────────────────────
    _reset(side, flag=True, mode="enforce", router_present=True)
    h = _holder(patched)                        # ROUTES deliberately empty
    _add(h, 0, None, prompt, [])
    h.update_state([[]], None)                  # still prefilling: NOT a miss
    check(f"{tag} P5 prefilling request is not counted as a miss",
          "h119_route_missing" not in side.STATS and "route_for_miss" not in side.STATS,
          side.stats_line())
    check(f"{tag} P5 it stays provisional at the fail-safe deep budget",
          h._state[0].get(side.H119_PROVISIONAL) is True
          and h._state[0]["thinking_token_budget"] == DEEP)
    h._state[0]["output_tok_ids"] = [42]        # generation started, still no route
    h.update_state([[42]], None)
    check(f"{tag} P5 generating-with-no-route IS counted",
          side.STATS.get("h119_route_missing") == 1
          and side.STATS.get("route_for_miss") == 1, side.stats_line())
    check(f"{tag} P5 it lands on the defined fallback route (deep)",
          h._state[0]["thinking_token_budget"] == DEEP)
    check(f"{tag} P5 the miss is committed once, not re-counted",
          (h.update_state([[42]], None) or side.STATS.get("h119_route_missing")) == 1)

    # ── P6 caller explicit wins ─────────────────────────────────────────
    _reset(side, flag=True, mode="enforce", router_present=True)
    side.ROUTES["r0"] = "lean"                  # contradicting route on record
    h = _holder(patched)
    _add(h, 0, 4096, prompt, [])
    h.update_state([[]], None)
    check(f"{tag} P6 explicit caller budget survives a lean route",
          h._state[0]["thinking_token_budget"] == 4096)
    check(f"{tag} P6 counted as caller-explicit, not routed",
          side.STATS.get("h119_caller_explicit") == 1
          and not any(k.startswith("h119_routed") for k in side.STATS),
          side.stats_line())

    # ── P7 the cap actually forces the end token ────────────────────────
    _reset(side, flag=True, mode="enforce", router_present=True, lean=4)
    side.ROUTES["r0"] = "lean"
    h = _holder(patched)
    out: list[int] = []
    _add(h, 0, None, prompt, out)
    forced_at = None
    for i in range(8):
        h.update_state([out], None)
        if h._state[0]["force_index"] or h._state[0]["in_end"]:
            forced_at = len(out)
            break
        out.append(500 + i)                     # a plain thinking token
    check(f"{tag} P7 lean cap=4 forces the end token at the routed count",
          forced_at is not None and forced_at <= 5,
          f"forced_at={forced_at} budget={h._state[0]['thinking_token_budget']}")
    stock_h = _holder(stock)
    stock_out: list[int] = []
    _add(stock_h, 0, None, prompt, stock_out)
    for i in range(8):
        stock_h.update_state([stock_out], None)
        stock_out.append(500 + i)
    check(f"{tag} P7 stock never forces (nothing tracked it)", not stock_h._state)


def main() -> int:
    _install_vllm_stubs()
    ns = _load_patcher_ns()
    ran = 0
    for tag in PINS:
        src = _extract(tag)
        if src is None:
            check(f"{tag} image extractable", False,
                  "podman extraction failed — per-pin verification SKIPPED")
            continue
        ran += 1
        run_pin(tag, src, ns)
    check("at least one pin exercised", ran > 0)

    # The patcher must survive a pin that simply has no such file.
    missing = ns["_patch_thinking_budget_state"]
    ns["TBS"] = ns["pathlib"].Path("/nonexistent/thinking_budget_state.py")
    msg = missing()
    check("absent-file pin soft-skips cleanly", msg.startswith("soft-skip E-G"), msg)

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: " + ", ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
