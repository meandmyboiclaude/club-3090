#!/usr/bin/env python3
"""GATE M2 — PN114-SEED mechanism equivalence. This test should be BORING.

THE CLAIM UNDER TEST
--------------------
Rendering the PN102 think-seed into the PROMPT and forcing the same seed as
OUTPUT produce the SAME TOKEN IDS AT THE SAME ABSOLUTE POSITIONS, under the
same thinking-token cap — so the seed can be chosen AFTER prefill (i.e. from
the H119 route) without changing anything else about the request.

If that claim is false anywhere, the escape is unsound and the fallback is a
static per-boot seed, not a cleverer arm.

WHAT IS ACTUALLY DRIVEN
-----------------------
The REAL ``ThinkingBudgetStateHolder`` from the boot image, with the REAL
entrypoint patch stack replayed onto it (pn108 -> pn112 -> pr44812 ->
syncbatch -> pn114 forced-span -> h119 lens router -> pn114-seed), and the
REAL tokenizer for the seed ids. Nothing about the holder is mocked: the test
drives ``sync_batch`` / ``update_state`` / ``apply_to_logits`` in the order
``vllm/v1/sample/sampler.py`` drives them and samples argmax, which is exactly
what a forced row produces (forcing writes 1e9 into the row's logit).

  A) prompt arm  — prompt = ...<think>\\n + SEED, no forcing
  B) forced arm  — prompt = ...<think>\\n,        seed forced as output
  C) routed arm  — prompt = ...<think>\\n, seed for the ROUTED N forced,
                   compared against a prompt arm rendered with that same N
  D) MTP arm     — B under spec-decode with every acceptance depth 0..n_spec,
                   asserting the span lands byte-exact regardless of where the
                   rejection falls (drafts must never be credited)

Usage (host):   python3 fixes/test_pn114_seed_equivalence.py
It re-executes itself inside a throwaway container from the boot pin; nothing
outside that container is written, no GPU is used, the serving container is
not touched.
"""
from __future__ import annotations

import base64
import os
import pathlib
import subprocess
import sys

PIN = "localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725"
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
GENESIS = REPO / "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis"
HFCACHE = pathlib.Path.home() / ".cache/huggingface"
DIST = "/usr/local/lib/python3.12/dist-packages"

PREFIX = (
    "patch_pn108_plateau_cap.py",
    "patch_pn112_conf_tap.py",
    "patch_pr44812_tool_guard.py",
    "patch_holder_syncbatch_fix.py",
    "patch_pn114_forced_span.py",
    "patch_h119_lens_router.py",
    "patch_pn74_fix_p107_serving_attr.py",
    "patch_pn100_auto_thinking_budget.py",
    "patch_pn101_answer_rescue.py",
    "pn114_seed_ids.py",
    "patch_pn114_seed_span.py",
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail
                                                     else ""))
    if not ok:
        FAILURES.append(name)


# ═══════════════════════════════════════════════════════════════════════════
# In-container body
# ═══════════════════════════════════════════════════════════════════════════
def body() -> int:
    import json
    from types import SimpleNamespace

    import torch

    sys.path.insert(0, "/fixes")
    # CPU-only container: async_tensor_h2d pins host memory through CUDA.
    # Test-side only — this file never runs in a serving process.
    import vllm.utils.torch_utils as _tu
    _tu.PIN_MEMORY = False
    from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

    table = json.load(open("/tmp/genesis_pn114_seed_ids.json",
                           encoding="utf-8"))
    THINK_START = table["base"][-2:]          # <think> , \n
    THINK_END = table["think_end"]
    VOCAB = max(max(THINK_END), max(THINK_START)) + 64
    PREFIX_IDS = [11, 12, 13]                 # stand-in for the rendered chat
    NAT_LO, NAT_HI = 1000, 1050               # "natural" continuation tokens

    def natural(pos: int) -> int:
        """The token the model would sample at ABSOLUTE position `pos`.

        Position-keyed on purpose: both arms must see the same continuation at
        the same place, otherwise the comparison tests the script, not the
        mechanism.
        """
        return NAT_LO + (pos * 7919) % (NAT_HI - NAT_LO)

    def holder_for(spec: int) -> ThinkingBudgetStateHolder:
        rc = SimpleNamespace(
            reasoning_start_token_ids=list(THINK_START),
            reasoning_end_token_ids=list(THINK_END),
            implicit_reasoning_end_token_ids=[],
        )
        return ThinkingBudgetStateHolder(rc, 1, spec, torch.device("cpu"),
                                         False, False)

    def batch_update(params, prompt_ids, out):
        return SimpleNamespace(removed=(), moved=(),
                               added=[(0, params, list(prompt_ids), out)])

    def run(prompt_ids, seed_text_xarg, budget, max_new=4096, spec=0,
            accept_pattern=None):
        """Drive the holder until </think> lands (or max_new). Returns output."""
        holder = holder_for(spec)
        params = SimpleNamespace(
            thinking_token_budget=budget,
            extra_args=({"pn114_seed_text": seed_text_xarg}
                        if seed_text_xarg else None),
        )
        out: list[int] = []
        holder.sync_batch(batch_update(params, prompt_ids, out))
        base = len(prompt_ids)
        step = 0
        while len(out) < max_new:
            drafts = ([natural(base + len(out) + 1 + k) for k in range(spec)]
                      if spec else [])
            holder.update_state([out], [drafts] if spec else None)
            rows = max(1, spec)
            logits = torch.zeros((rows, VOCAB), dtype=torch.float32)
            for k in range(rows):
                logits[k, natural(base + len(out) + k)] = 1.0
            logits = holder.apply_to_logits(logits, False,
                                            [drafts] if spec else None)
            targets = [int(logits[k].argmax()) for k in range(rows)]
            if spec:
                # forced-vs-forced verification: accept `j` drafts, the
                # rejection's recovery token IS the forced token at row j.
                j = (accept_pattern[step % len(accept_pattern)]
                     if accept_pattern else 0)
                j = min(j, spec - 1)
                landed = targets[:j + 1]
            else:
                landed = targets[:1]
            out.extend(landed)
            step += 1
            if any(t in THINK_END for t in landed):
                break
        return out

    def seq(prompt_ids, out):
        return list(prompt_ids) + list(out)

    def seed_for(label, tail, n):
        return table["by_steps"].get(f"{label}|{tail}|{n}")

    base_prompt = PREFIX_IDS + list(THINK_START)

    # ── A vs B: same cap, prompt-rendered seed vs forced-span seed ─────────
    print("\n[M2-1] prompt-rendered vs forced-span, identical cap")
    os.environ["GENESIS_ENABLE_PN114_SEED_SPAN"] = "1"
    os.environ["GENESIS_PN114_SEED_MODE"] = "mirror"
    for n in (3, 5, 13, 40):
        for label, tail in (("Budget", "plain"), ("Plan", "plain"),
                            ("Budget", "echo")):
            text = seed_for(label, tail, n)
            if text is None:
                check(f"seed table has {label}|{tail}|{n}", False)
                continue
            ids = table["by_text"][text]
            for budget in (8, 40, 400):
                a_prompt = base_prompt + ids
                a_out = run(a_prompt, None, budget)
                b_out = run(base_prompt, text, budget)
                sa, sb = seq(a_prompt, a_out), seq(base_prompt, b_out)
                same = sa == sb
                if same:
                    detail = ""
                else:
                    d = next((i for i in range(min(len(sa), len(sb)))
                              if sa[i] != sb[i]), min(len(sa), len(sb)))
                    detail = (f"first divergence at abs pos {d}: "
                              f"A={sa[d:d + 4]} B={sb[d:d + 4]} "
                              f"(len A={len(sa)} B={len(sb)})")
                check(f"n={n} {label}/{tail} budget={budget}: identical ids "
                      f"at identical positions", same, detail)

    # ── the span itself must be byte-exact at output positions 0..len-1 ────
    print("\n[M2-2] the forced span lands exactly, at position 0")
    text = seed_for("Budget", "plain", 13)
    ids = table["by_text"][text]
    b_out = run(base_prompt, text, 400)
    check("forced span == the seed's token ids", b_out[:len(ids)] == ids,
          f"got {b_out[:len(ids)]}")
    check("no duplicated/skipped span token",
          b_out[len(ids)] == natural(len(base_prompt) + len(ids)),
          f"first free token {b_out[len(ids)]}")

    # ── the cap must bind at the same absolute position ────────────────────
    print("\n[M2-3] the cap binds at the same absolute position")
    for budget in (8, 40, 100, 400):
        a_prompt = base_prompt + ids
        a_out = run(a_prompt, None, budget)
        b_out = run(base_prompt, text, budget)
        pa = seq(a_prompt, a_out).index(THINK_END[0]) \
            if THINK_END[0] in seq(a_prompt, a_out) else -1
        pb = seq(base_prompt, b_out).index(THINK_END[0]) \
            if THINK_END[0] in seq(base_prompt, b_out) else -1
        check(f"budget={budget}: </think> at the same index", pa == pb,
              f"prompt-arm={pa} forced-arm={pb}")

    # ── C: routed N — the actual escape from cap-only ──────────────────────
    print("\n[M2-4] routed mode == the prompt arm you could not have rendered")
    os.environ["GENESIS_PN114_SEED_MODE"] = "routed"
    os.environ["GENESIS_PN102_TOKENS_PER_STEP"] = "193"
    ok_any = False
    for budget in (900, 1200, 2400):
        n_routed = max(3, round(budget / 193))
        routed_text = seed_for("Budget", "plain", n_routed)
        if routed_text is None:
            continue
        ok_any = True
        r_ids = table["by_text"][routed_text]
        # what PN102 would have rendered pre-prefill (a DIFFERENT, lean N)
        prior = seed_for("Budget", "plain", 3)
        b_out = run(base_prompt, prior, budget)
        a_prompt = base_prompt + r_ids
        a_out = run(a_prompt, None, budget)
        check(f"budget={budget} -> N={n_routed}: forced-routed == "
              f"prompt-rendered(N={n_routed})",
              seq(base_prompt, b_out) == seq(a_prompt, a_out))
        if n_routed != 3:
            check(f"budget={budget}: routed span differs from the "
                  f"pre-prefill seed (the escape is real)",
                  routed_text != prior)
    check("routed cases were exercised", ok_any)
    os.environ["GENESIS_PN114_SEED_MODE"] = "mirror"

    # ── D: MTP — drafts never credited, rejection recovery IS the span ─────
    print("\n[M2-5] MTP: the span is exact at every acceptance depth")
    for pattern in ([0], [1], [2], [0, 1, 2], [2, 0, 1, 2, 1]):
        b_out = run(base_prompt, text, 400, spec=3, accept_pattern=pattern)
        check(f"accept-pattern {pattern}: span byte-exact",
              b_out[:len(ids)] == ids, f"got {b_out[:len(ids)]}")

    # ── E: fail-closed — an unknown seed is never stripped nor forced ──────
    print("\n[M2-6] fail-closed on a seed the boot table does not know")
    unknown = "Budget: ~9999 short steps.\nStep 1:"
    check("unknown seed absent from the table",
          unknown not in table["by_text"])
    b_out = run(base_prompt, unknown, 400)
    check("no span is armed for an unknown seed",
          b_out[0] == natural(len(base_prompt)), f"got {b_out[0]}")

    from vllm import _genesis_pn114_seed as seedmod  # installed by the patch
    req = SimpleNamespace(
        chat_template_kwargs={"pn_env_seed": unknown}, vllm_xargs=None)
    seedmod.strip_prompt_seed(req)
    check("serving side leaves an unknown seed in the prompt",
          req.chat_template_kwargs.get("pn_env_seed") == unknown)
    req2 = SimpleNamespace(
        chat_template_kwargs={"pn_env_seed": text}, vllm_xargs=None)
    stripped = seedmod.strip_prompt_seed(req2)
    check("serving side strips a known seed and carries it in vllm_xargs",
          stripped and "pn_env_seed" not in req2.chat_template_kwargs
          and req2.vllm_xargs.get("pn114_seed_text") == text)
    os.environ["GENESIS_ENABLE_PN114_SEED_SPAN"] = "0"
    req3 = SimpleNamespace(
        chat_template_kwargs={"pn_env_seed": text}, vllm_xargs=None)
    check("flag OFF strips nothing",
          not seedmod.strip_prompt_seed(req3)
          and req3.chat_template_kwargs.get("pn_env_seed") == text)

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} check(s) failed: {FAILURES[:6]}")
        return 1
    print("PASS: gate M2 — the forced-span seed is mechanism-equivalent to "
          "the prompt-rendered seed")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Host side: replay the entrypoint prefix in a throwaway container, then run
# this same file inside it.
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    if os.environ.get("PN114_SEED_M2_IN_CONTAINER") == "1":
        return body()
    steps = [f"python3 -m vllm._genesis.patches.apply_all >/tmp/aa.log 2>&1 "
             f"|| true"]
    steps += [f"python3 /fixes/{n} >/tmp/p.log 2>&1 "
              f"|| echo 'NOTE {n} rc!=0: '$(tail -1 /tmp/p.log)"
              for n in PREFIX]
    steps.append("PN114_SEED_M2_IN_CONTAINER=1 python3 "
                 "/fixes/test_pn114_seed_equivalence.py")
    blob = base64.b64encode("\n".join(steps).encode()).decode()
    cmd = ["sudo", "podman", "run", "--rm", "--network", "none",
           "-e", "GENESIS_ENABLE_PN114_SEED_SPAN=1",
           "-e", "PN114_SEED_M2_IN_CONTAINER",
           "-v", f"{HERE}:/fixes:ro",
           "-v", f"{GENESIS}:{DIST}/vllm/_genesis:ro"]
    if HFCACHE.is_dir():
        cmd += ["-v", f"{HFCACHE}:/root/.cache/huggingface:ro"]
    cmd += ["--entrypoint", "/bin/bash", PIN, "-c",
            f"echo {blob} | base64 -d > /tmp/run.sh && bash /tmp/run.sh"]
    r = subprocess.run(cmd, text=True)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
