#!/usr/bin/env python3
"""H119 router — partial-prefill / prefix-cache-hit correctness (CPU only).

Run: ~/shared/needfit/lens-venv/bin/python fixes/test_pn119_partial_prefill.py

No GPU, no engine, no service touched: the accumulator is driven directly
with synthetic aux hidden states whose value depends only on (layer, token
id, position) — the same property a prefix cache relies on, which is what
lets a cached prefix be reconstructed exactly.

What is being pinned
--------------------
P1  full-recompute path is BYTE-IDENTICAL to the pre-fix module (the module
    at git HEAD is imported alongside and must produce the same float score),
    single-step and chunked.
P2  a prefix-cache hit (num_computed_tokens > 0 at first sight) is NEVER
    silently unrouted under enforce: it is counted, logged and published to
    ROUTES/SCORES as the configured fallback route.  The same input on the
    HEAD module produces no score and no registry entry at all — the
    regression witness for the defect being fixed.
P3  PN119_FALLBACK_ROUTE selects the fallback deterministically.
P4  with PN119_PREFIX_MEMO=1 a cache-hit request whose cached length lands on
    a stored checkpoint is scored EXACTLY like the full recompute (that is
    option (c): reconstruct, don't approximate); a non-checkpoint length
    misses and falls back rather than scoring on partial features.
P5  route_for() never returns None.
P6  shadow mode still acts on nothing (no registry writes).
P7  unscoreable requests write NO feature row, and the sink row indices stay
    aligned for the rows that do exist (refit_pn119_probe.load_sink joins the
    scored request and ignores the fallback one).
P8  a request first seen in decode is flagged, not ignored.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NEEDFIT = os.path.expanduser("~/shared/needfit")
PROBE = os.path.join(NEEDFIT, "pn119-live/probe.npz")

D_MODEL = 5120
LAYERS = (42, 47, 51)
PROMPT_LEN = 40
UNIT = 8

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── stubs mirroring the gpu_model_runner surfaces the tap reads ────────────
class _SP:
    max_tokens = 512


class StubState:
    """CachedRequestState surface used by the router."""

    def __init__(self, prompt_ids, num_computed=0):
        self.prompt_token_ids = list(prompt_ids)
        self.num_prompt_tokens = len(prompt_ids)
        # PRE-step engine progress, exactly as _update_states leaves it
        # before the forward (== APC-cached prefix on a first prefill step).
        self.num_computed_tokens = num_computed
        self.output_token_ids: list[int] = []
        self.sampling_params = _SP()


class StubBatch:
    def __init__(self):
        self.req_ids: list[str] = []


class StubRunner:
    device = "cpu"

    def __init__(self):
        self.input_batch = StubBatch()
        self.requests: dict[str, StubState] = {}


class Sched:
    def __init__(self, d):
        self.num_scheduled_tokens = d


_IDX = torch.arange(D_MODEL, dtype=torch.float32)


def aux_rows(layer_i: int, token_ids, positions) -> torch.Tensor:
    """Deterministic stand-in for the layer-42/47/51 residual states.

    Depends ONLY on (layer, token id, position) — the property that makes a
    cached prefix reconstructible at all.
    """
    t = torch.tensor(token_ids, dtype=torch.float32).unsqueeze(1)
    p = torch.tensor(positions, dtype=torch.float32).unsqueeze(1)
    return 0.05 * torch.sin(0.0013 * (_IDX + 1) * (t + 1)
                            + 0.37 * layer_i + 0.011 * p)


def prompt_ids(seed: int, n: int = PROMPT_LEN) -> list[int]:
    return [(seed * 7919 + i * 104729) % 150000 + 1 for i in range(n)]


def run_prefill(router, runner, req_id, ids, chunks, cached=0):
    """Drive router.observe over `chunks` scheduled-token counts."""
    state = StubState(ids, num_computed=cached)
    runner.requests[req_id] = state
    runner.input_batch.req_ids = [req_id]
    pos = cached
    for n in chunks:
        state.num_computed_tokens = pos          # pre-step value
        toks = ids[pos:pos + n]
        aux = [aux_rows(li, toks, range(pos, pos + n)) for li in range(len(LAYERS))]
        router.observe(Sched({req_id: n}), aux)
        pos += n
    return state


def new_router(mod, **env):
    for k in ("PN119_MODE", "PN119_TDEEP", "PN119_SINK", "PN119_EXPLORE",
              "PN119_FALLBACK_ROUTE", "PN119_PREFIX_MEMO", "PN119_MEMO_UNIT",
              "PN119_MEMO_MAX", "PN119_STATS_EVERY"):
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    mod.SCORES.clear()
    mod.EXPLORE.clear()
    if hasattr(mod, "ROUTES"):
        mod.ROUTES.clear()
        mod.STATS.clear()
    runner = StubRunner()
    return mod.PN119Router(runner, PROBE), runner


def score_of(router, runner, req_id, ids, chunks, cached=0):
    st = run_prefill(router, runner, req_id, ids, chunks, cached)
    return router.scored.get(req_id), st


# ── P1: full-recompute path byte-identical to HEAD ─────────────────────────
def p1_byte_identical(new_mod, old_mod):
    ids = prompt_ids(11)
    for label, chunks in (("single-step", [PROMPT_LEN]),
                          ("chunked 17/13/10", [17, 13, 10])):
        rn, run_n = new_router(new_mod, PN119_MODE="enforce", PN119_TDEEP="0.495")
        ro, run_o = new_router(old_mod, PN119_MODE="enforce", PN119_TDEEP="0.495")
        sn, _ = score_of(rn, run_n, "r1", ids, chunks)
        so, _ = score_of(ro, run_o, "r1", ids, chunks)
        check(f"P1 {label}: score identical to HEAD",
              sn is not None and so is not None and sn == so,
              f"new={sn!r} head={so!r}")
        check(f"P1 {label}: observe raised nothing (guard silent)",
              rn._warned == 0 and ro._warned == 0,
              f"new_warned={rn._warned} head_warned={ro._warned}")
    # the two chunkings must also agree with each other (fused vs split adds)
    rn, run_n = new_router(new_mod, PN119_MODE="enforce")
    s_single, _ = score_of(rn, run_n, "a", ids, [PROMPT_LEN])
    rn2, run_n2 = new_router(new_mod, PN119_MODE="enforce")
    s_chunk, _ = score_of(rn2, run_n2, "a", ids, [17, 13, 10])
    check("P1 chunked == single-step (same summation order per slice)",
          s_single == s_chunk, f"{s_single!r} vs {s_chunk!r}")
    return s_single


# ── P2/P3: cache hit is explicit, counted, routed ──────────────────────────
def p2_cache_hit_not_silent(new_mod, old_mod):
    ids = prompt_ids(23)
    cached = 16

    # regression witness: the HEAD module drops it on the floor
    ro, run_o = new_router(old_mod, PN119_MODE="enforce", PN119_TDEEP="0.495")
    so, _ = score_of(ro, run_o, "cachehit", ids, [PROMPT_LEN - cached], cached=cached)
    check("P2 witness: HEAD leaves a prefix-cache hit unscored AND unpublished",
          so is None and not old_mod.SCORES,
          f"score={so!r} SCORES={dict(old_mod.SCORES)}")

    rn, run_n = new_router(new_mod, PN119_MODE="enforce", PN119_TDEEP="0.495",
                           PN119_FALLBACK_ROUTE="deep")
    sn, _ = score_of(rn, run_n, "cachehit", ids, [PROMPT_LEN - cached], cached=cached)
    check("P2 no partial-feature score is invented", sn is None, f"score={sn!r}")
    check("P2 request is flagged unscoreable with the right reason",
          rn.unscored.get("cachehit") == "partial_prefill",
          f"unscored={rn.unscored}")
    check("P2 counter visible",
          new_mod.STATS["unscoreable"] == 1
          and new_mod.STATS["unscoreable_partial_prefill"] == 1
          and new_mod.STATS["prefill_partial_cached"] == 1,
          new_mod.stats_line())
    check("P2 enforce publishes an explicit route",
          new_mod.ROUTES.get("cachehit") == "deep"
          and new_mod.route_for("cachehit") == "deep",
          f"ROUTES={dict(new_mod.ROUTES)}")
    check("P2 legacy SCORES reader lands on the same side of TDEEP",
          new_mod.SCORES.get("cachehit", -9e9) >= rn.tdeep,
          f"SCORES={dict(new_mod.SCORES)} tdeep={rn.tdeep}")

    rl, run_l = new_router(new_mod, PN119_MODE="enforce", PN119_TDEEP="0.495",
                           PN119_FALLBACK_ROUTE="lean")
    score_of(rl, run_l, "cachehit", ids, [PROMPT_LEN - cached], cached=cached)
    check("P3 PN119_FALLBACK_ROUTE=lean routes lean",
          new_mod.ROUTES.get("cachehit") == "lean"
          and new_mod.SCORES.get("cachehit", 9e9) < rl.tdeep
          and new_mod.STATS["fallback_lean"] == 1,
          f"ROUTES={dict(new_mod.ROUTES)} {new_mod.stats_line()}")

    rb, run_b = new_router(new_mod, PN119_MODE="enforce",
                           PN119_FALLBACK_ROUTE="sideways")
    check("P3 invalid fallback falls back to deep", rb.fallback_route == "deep")


# ── P4: memo reconstructs a cached prefix EXACTLY ──────────────────────────
def p4_memo(new_mod, truth_score):
    ids = prompt_ids(11)          # same prompt P1 measured the truth on
    cached = 2 * UNIT             # 16 — a checkpoint boundary

    r, run = new_router(new_mod, PN119_MODE="enforce", PN119_TDEEP="0.495",
                        PN119_PREFIX_MEMO="1", PN119_MEMO_UNIT=UNIT)
    # request A: cold, full recompute -> stores checkpoints
    sa, _ = score_of(r, run, "A", ids, [PROMPT_LEN])
    check("P4 memo stores checkpoints on a full recompute",
          new_mod.STATS["memo_store"] == PROMPT_LEN // UNIT,
          f"{new_mod.stats_line()}")
    # request B: same prompt, APC serves the first 16 tokens
    sb, _ = score_of(r, run, "B", ids, [PROMPT_LEN - cached], cached=cached)
    check("P4 memo HIT scores the request (no fallback)",
          sb is not None and new_mod.STATS["memo_hit"] == 1
          and new_mod.STATS.get("unscoreable", 0) == 0,
          f"score={sb!r} {new_mod.stats_line()}")
    check("P4 memo-hit score == full-recompute score",
          sb is not None and math.isclose(sb, sa, rel_tol=0, abs_tol=1e-4),
          f"cached={sb!r} full={sa!r} delta={None if sb is None else sb - sa:.3e}")
    check("P4 memo-hit score == the memo-OFF ground truth",
          sb is not None and math.isclose(sb, truth_score, rel_tol=0, abs_tol=1e-4),
          f"cached={sb!r} truth={truth_score!r}")
    check("P4 memo-hit route matches the full-recompute route",
          new_mod.ROUTES.get("B") == new_mod.ROUTES.get("A"),
          f"A={new_mod.ROUTES.get('A')} B={new_mod.ROUTES.get('B')}")

    # unaligned cached length -> miss -> fallback, never a partial score
    r2, run2 = new_router(new_mod, PN119_MODE="enforce",
                          PN119_PREFIX_MEMO="1", PN119_MEMO_UNIT=UNIT)
    score_of(r2, run2, "A", ids, [PROMPT_LEN])
    sc, _ = score_of(r2, run2, "C", ids, [PROMPT_LEN - 13], cached=13)
    check("P4 unaligned cached length misses and falls back (no partial score)",
          sc is None and new_mod.STATS["memo_miss"] == 1
          and new_mod.STATS["unscoreable_partial_prefill"] == 1
          and new_mod.ROUTES.get("C") == "deep",
          new_mod.stats_line())

    # a DIFFERENT prompt with the same cached length must not collide
    r3, run3 = new_router(new_mod, PN119_MODE="enforce",
                          PN119_PREFIX_MEMO="1", PN119_MEMO_UNIT=UNIT)
    score_of(r3, run3, "A", ids, [PROMPT_LEN])
    other = prompt_ids(77)
    sd, _ = score_of(r3, run3, "D", other, [PROMPT_LEN - cached], cached=cached)
    check("P4 different prompt, same offset -> miss (hash is over the tokens)",
          sd is None and new_mod.STATS["memo_miss"] == 1, new_mod.stats_line())

    # LRU bound holds
    r4, run4 = new_router(new_mod, PN119_MODE="enforce", PN119_PREFIX_MEMO="1",
                          PN119_MEMO_UNIT=UNIT, PN119_MEMO_MAX=3)
    for i in range(4):
        score_of(r4, run4, f"L{i}", prompt_ids(100 + i), [PROMPT_LEN])
    check("P4 memo respects PN119_MEMO_MAX", len(r4._memo) == 3,
          f"entries={len(r4._memo)} {new_mod.stats_line()}")


# ── P5/P6/P8 ───────────────────────────────────────────────────────────────
def p5_route_for(new_mod):
    r, _ = new_router(new_mod, PN119_MODE="enforce", PN119_FALLBACK_ROUTE="deep")
    check("P5 route_for on an unknown req_id returns the fallback, never None",
          new_mod.route_for("never-seen") == "deep"
          and new_mod.STATS["route_for_miss"] == 1, new_mod.stats_line())


def p6_shadow(new_mod):
    ids = prompt_ids(31)
    r, run = new_router(new_mod, PN119_MODE="shadow")
    score_of(r, run, "s1", ids, [PROMPT_LEN - 16], cached=16)
    check("P6 shadow counts + logs but writes no registry entry",
          not new_mod.ROUTES and not new_mod.SCORES
          and new_mod.STATS["unscoreable_partial_prefill"] == 1,
          f"ROUTES={dict(new_mod.ROUTES)} SCORES={dict(new_mod.SCORES)}")
    score_of(r, run, "s2", ids, [PROMPT_LEN])
    check("P6 shadow still scores a full recompute",
          r.scored.get("s2") is not None and not new_mod.SCORES)


def p8_decode_only(new_mod):
    ids = prompt_ids(41)
    r, run = new_router(new_mod, PN119_MODE="enforce")
    st = StubState(ids, num_computed=PROMPT_LEN)   # first sight is a decode step
    run.requests["d1"] = st
    run.input_batch.req_ids = ["d1"]
    aux = [aux_rows(li, [ids[-1]], [PROMPT_LEN]) for li in range(len(LAYERS))]
    r.observe(Sched({"d1": 1}), aux)
    check("P8 request first seen in decode is flagged, not ignored",
          r.unscored.get("d1") == "prefill_not_observed"
          and new_mod.ROUTES.get("d1") == "deep"
          and new_mod.STATS["unscoreable_prefill_not_observed"] == 1,
          new_mod.stats_line())


# ── P7: sink stays aligned; fallback rows can never become training data ───
def p7_sink_alignment(new_mod):
    ids_ok = prompt_ids(51)
    ids_hit = prompt_ids(52)
    with tempfile.TemporaryDirectory(dir=NEEDFIT) as d:
        r, run = new_router(new_mod, PN119_MODE="enforce", PN119_SINK=d,
                            PN119_TDEEP="0.495")
        st_ok = run_prefill(r, run, "ok", ids_ok, [PROMPT_LEN])
        st_hit = run_prefill(r, run, "miss", ids_hit, [PROMPT_LEN - 16], cached=16)
        st_ok.output_token_ids = [5, 5, r._think_end, 9]
        st_ok.prompt_token_ids = list(ids_ok) + [r._think_start]
        r.on_finish("ok", st_ok)
        r.on_finish("miss", st_hit)
        r._sink_feat.flush()
        r._sink_meta.flush()

        feat = [f for f in os.listdir(d) if f.startswith("feats-")][0]
        n_rows = os.path.getsize(os.path.join(d, feat)) // (30720 * 2)
        check("P7 exactly one feature row written (the scoreable request)",
              n_rows == 1, f"rows={n_rows}")

        meta_p = os.path.join(d, [f for f in os.listdir(d)
                                  if f.startswith("meta-")][0])
        lines = [json.loads(x) for x in open(meta_p, encoding="utf-8") if x.strip()]
        uns = [m for m in lines if m.get("unscoreable")]
        check("P7 fallback is recorded in the sink with no 'row' key",
              len(uns) == 2 and all("row" not in m for m in uns),
              f"lines={len(lines)} unscoreable={len(uns)}")

        sys.path.insert(0, HERE)
        from refit_pn119_probe import load_sink
        counts: dict = {}
        rows = load_sink(d, counts)
        check("P7 refit joins only the scoreable request",
              len(rows) == 1 and rows[0].req_id == "ok",
              f"rows={[x.req_id for x in rows]} counts={counts}")


def main() -> int:
    if not os.path.isfile(PROBE):
        print(f"probe missing: {PROBE}")
        return 1
    with tempfile.TemporaryDirectory() as td:
        head = os.path.join(td, "pn119_router_head.py")
        with open(head, "wb") as f:
            f.write(subprocess.run(["git", "-C", REPO, "show",
                                    "HEAD:fixes/pn119_router.py"],
                                   check=True, capture_output=True).stdout)
        old_mod = _load("pn119_router_head", head)
    new_mod = _load("pn119_router_new", os.path.join(HERE, "pn119_router.py"))

    print("== P1 full-recompute path unchanged ==")
    truth = p1_byte_identical(new_mod, old_mod)
    print("== P2/P3 prefix-cache hit is explicit ==")
    p2_cache_hit_not_silent(new_mod, old_mod)
    print("== P4 exact reconstruction via the prefix memo ==")
    p4_memo(new_mod, truth)
    print("== P5 route_for total function ==")
    p5_route_for(new_mod)
    print("== P6 shadow mode unchanged ==")
    p6_shadow(new_mod)
    print("== P7 sink alignment ==")
    p7_sink_alignment(new_mod)
    print("== P8 decode-only first sight ==")
    p8_decode_only(new_mod)

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
