#!/usr/bin/env python3
"""PN119-v2 offline validation (no GPU, no service touched).

Run: ~/shared/needfit/lens-venv/bin/python fixes/test_pn119_refit.py

  T1 atomic swap under SIGKILL — writer killed at random moments; the
     target must ALWAYS load as exactly one complete, coherent version.
  T2 real-sink parse + censoring-guard accounting.
  T3 refit end-to-end (dry-run) on the real sink + seed; AUC + rho gates.
  T4 router hot-reload on mtime (real PN119Router on CPU): picks up an
     atomic swap, survives a corrupt swap with old weights intact.
  T5 explore knob: deterministic per req_id, ~rate over many ids.
  T6 censoring guards on synthetic enforce-era rows.
  T7 pipeline baseline: CURRENT live probe scores on seed features
     rank-correlate vs router_loo_scores.json (validates feature-order +
     scoring equivalence, and anchors the --min-rho default).
"""
from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NEEDFIT = os.path.expanduser("~/shared/needfit")
PY = os.path.join(NEEDFIT, "lens-venv/bin/python")

from pn119_atomic import atomic_write_bytes, atomic_write_npz  # noqa: E402
from refit_pn119_probe import (FEAT_DIM, apply_guards, auc, bf16_rows,  # noqa: E402
                               load_sink, Row, score_with, select_feat_dim,
                               spearman)

# G6's threshold. apply_guards takes it as a required positional now, so the
# value has to be supplied here; 32 is the refit CLI's own --min-generated
# default (the parser builds it inside main(), so it cannot be imported).
MIN_GENERATED = 32

FAILURES = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ── T1: atomic swap kill-test ───────────────────────────────────────────────
def t1_kill_test(kills=25):
    with tempfile.TemporaryDirectory(dir=NEEDFIT) as d:
        target = os.path.join(d, "probe.npz")
        atomic_write_npz(target, {"mu": np.zeros(FEAT_DIM, np.float32),
                                  "sd": np.ones(FEAT_DIM, np.float32),
                                  "Vt10": np.zeros((10, FEAT_DIM), np.float32),
                                  "w": np.zeros(11, np.float32),
                                  "version": np.array([0])})
        torn = 0
        for k in range(kills):
            p = subprocess.Popen([PY, os.path.join(HERE, "pn119_atomic.py"), target])
            time.sleep(random.uniform(0.02, 0.25))
            p.send_signal(signal.SIGKILL)
            p.wait()
            try:
                z = np.load(target, allow_pickle=True)
                mu, vt, w = z["mu"], z["Vt10"], z["w"]
                val = mu[0]
                coherent = (np.all(mu == val) and np.all(vt == val) and np.all(w == val)
                            and mu.shape == (FEAT_DIM,) and vt.shape == (10, FEAT_DIM))
                if not coherent:
                    torn += 1
            except Exception as e:  # noqa: BLE001 — any load failure = torn file
                torn += 1
                print(f"    kill {k}: LOAD FAILED: {e}")
        leftovers = [f for f in os.listdir(d) if f.startswith(".pn119-swap-")]
        check("T1 atomic swap never torn under SIGKILL",
              torn == 0, f"{kills} kills, {torn} torn, {len(leftovers)} orphan tmp "
              f"(orphans are hidden dotfiles, never the target; refit prunes stale ones)")


# ── T2: real sink parse + guards ────────────────────────────────────────────
def t2_real_sink():
    counts = {}
    # load_sink returns (rows, {tag: meta bytes}) and the rows come back
    # WITHOUT features — .x stays None and the bytes are materialised per row
    # index by bf16_rows. Unpacking it as a bare list made `len(rows)` report 2
    # (the tuple) and every later attribute access blow up.
    rows, sizes = load_sink(os.path.join(NEEDFIT, "pn119-sink"), counts)
    check("T2 sink parses", len(rows) > 0,
          f"{counts.get('scored', 0)} scored, {len(rows)} joined w/ finish, "
          f"{len(sizes)} windows, counts={counts}")

    # Feature width is a per-WINDOW property now: the sink spans two eras
    # (15360 last-only and 30720 last+mean) and select_feat_dim picks the
    # newest, so a flat `r.x.shape == (FEAT_DIM,)` is checking a constant the
    # live windows no longer use. Check each row against ITS OWN window width,
    # and pin that the era split is the one select_feat_dim reports.
    dim, kept = select_feat_dim(rows, {})
    bad = []
    for r in rows[:200]:
        x = bf16_rows(r.feat_path, [r.row_idx], r.feat_dim)[0]
        if x.shape != (r.feat_dim,) or not np.isfinite(x).all():
            bad.append(r.req_id)
    check("T2 features well-formed at each window's own width, finite",
          not bad and all(r.feat_dim in (15360, 30720) for r in rows),
          f"newest era={dim} ({len(kept)}/{len(rows)} rows), "
          f"widths={sorted({r.feat_dim for r in rows})}, bad={bad[:5]}")

    c2, c3 = {}, {}
    strict = apply_guards(rows, False, MIN_GENERATED, c2)
    legacy = apply_guards(rows, True, MIN_GENERATED, c3)
    # This used to assert `len(strict) == 0` — "the sink predates the thinking
    # flag". It no longer does: the router has been writing `thinking` on the
    # finish line since 730b6833, and today's sink carries 1125 thinking=True
    # finish lines against 948 pre-flag ones. Strict is the REAL training set
    # now, and asserting it is empty would have kept passing only for as long
    # as the flag stayed unshipped. What is actually invariant is the
    # containment: legacy-ok accepts everything strict accepts PLUS exactly the
    # rows G1 booked as g1_legacy_accepted.
    s_ids, l_ids = {r.req_id for r in strict}, {r.req_id for r in legacy}
    check("T2 G1 strict keeps the post-flag rows the sink now carries",
          len(strict) > 0,
          f"strict eligible={len(strict)}, legacy-ok eligible={len(legacy)}, "
          f"legacy accepted={c3.get('g1_legacy_accepted', 0)}, "
          f"legacy dropped by strict={c2.get('g1_legacy_dropped', 0)}")
    # Containment, not equality of counts: G1 admits g1_legacy_accepted rows
    # and G6/G8 then thin them, so the surviving extras are a SUBSET of what
    # G1 let through — every one of them a `legacy` row and never a real one.
    extra = [r for r in legacy if r.req_id not in s_ids]
    check("T2 G1 legacy-ok is strict plus legacy rows only",
          s_ids <= l_ids and extra
          and all(r.thinking == "legacy" for r in extra)
          and len(extra) <= c3.get("g1_legacy_accepted", 0),
          f"extra={len(extra)} (all legacy: "
          f"{all(r.thinking == 'legacy' for r in extra)}) <= g1_legacy_accepted="
          f"{c3.get('g1_legacy_accepted', 0)}; missing={len(s_ids - l_ids)}")
    check("T2 G1 drops thinking-off and thinking-unknown under both settings",
          c2.get("g1_thinking_off") == c3.get("g1_thinking_off")
          and c2.get("g1_thinking_unknown") == c3.get("g1_thinking_unknown"),
          f"strict={c2.get('g1_thinking_off')}/{c2.get('g1_thinking_unknown')} "
          f"legacy={c3.get('g1_thinking_off')}/{c3.get('g1_thinking_unknown')}")
    return legacy


# ── T3: refit end-to-end on real data ───────────────────────────────────────
def t3_refit():
    import hashlib
    live = os.path.join(NEEDFIT, "pn119-live/probe.npz")
    md5_before = hashlib.md5(open(live, "rb").read()).hexdigest()

    # 3a — default gates on today's LEGACY-labeled sink (generated≈spend
    # proxy, thinking-off rows polluting the negatives): expect the rho gate
    # to evaluate; on this data it REJECTS (exit 2) — that is the gate doing
    # its job, and the live probe must be untouched either way.
    r = subprocess.run(
        [PY, os.path.join(HERE, "refit_pn119_probe.py"), "--dry-run", "--force",
         "--legacy-thinking-ok"],
        capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    print("    " + "\n    ".join(out.strip().splitlines()))
    check("T3a gates evaluated on real data (exit 0=pass or 2=reject)",
          r.returncode in (0, 2), f"exit={r.returncode}")
    check("T3a rho gate rejects legacy-labeled candidate (expected on this sink)",
          r.returncode == 2 and "rho" in out)

    # 3b — full REAL refit+swap into a TEMP target (live probe untouched),
    # gate relaxed below the measured legacy rho to exercise the swap path.
    with tempfile.TemporaryDirectory(dir=NEEDFIT) as d:
        tmp_out = os.path.join(d, "probe.npz")
        tmp_state = os.path.join(d, "state")
        args = [PY, os.path.join(HERE, "refit_pn119_probe.py"), "--force",
                "--legacy-thinking-ok", "--out", tmp_out, "--state", tmp_state,
                "--min-rho", "0.70"]
        r2 = subprocess.run(args, capture_output=True, text=True, timeout=600)
        out2 = r2.stdout + r2.stderr
        print("    " + "\n    ".join(out2.strip().splitlines()[:1]))
        check("T3b real refit SWAPPED to temp target", r2.returncode == 0
              and "SWAPPED" in out2 and os.path.isfile(tmp_out))
        z = np.load(tmp_out, allow_pickle=True)
        check("T3b swapped npz complete + router-shaped",
              z["mu"].shape == (FEAT_DIM,) and z["Vt10"].shape[1] == FEAT_DIM
              and z["w"].shape == (z["Vt10"].shape[0] + 1,)
              and float(z["refit_auc"]) > 0.9,
              f"auc={float(z['refit_auc']):.4f} rho={float(z['refit_rho_vs_loo']):.4f} "
              f"n_sink={int(z['refit_n_sink'])} n_seed={int(z['refit_n_seed'])}")
        check("T3b state written (cursor + reservoir + report)",
              all(os.path.isfile(os.path.join(tmp_state, f)) for f in
                  ("cursor.json", "reservoir.npz", "refit-report.json")))
        # sink is LIVE (benchmark writing) — don't hardcode a row count;
        # below the 2000 cap the reservoir must hold every row it has seen.
        res = np.load(os.path.join(tmp_state, "reservoir.npz"), allow_pickle=True)
        n_res, n_seen = len(res["X"]), int(res["seen_total"])
        check("T3b reservoir holds all eligible rows seen (cap not yet hit)",
              n_res == min(n_seen, 2000) and n_res >= 400
              and res["X"].shape[1] == FEAT_DIM,
              f"reservoir={n_res} seen_total={n_seen}")
        # idempotence: immediately rerun WITHOUT --force -> min-new skip
        r3 = subprocess.run([a for a in args if a != "--force"],
                            capture_output=True, text=True, timeout=600)
        check("T3b rerun skips via min-new cursor", r3.returncode == 0
              and "skip" in (r3.stdout + r3.stderr))
    md5_after = hashlib.md5(open(live, "rb").read()).hexdigest()
    check("T3 live probe untouched throughout", md5_before == md5_after)
    return out


# ── T4: router hot-reload (real class, CPU) ─────────────────────────────────
def t4_hot_reload():
    import torch  # noqa: F401

    for v in ("PN119_SINK", "PN119_EXPLORE", "GENESIS_ENABLE_PN119_ROUTER"):
        os.environ.pop(v, None)
    os.environ["PN119_RELOAD_S"] = "1"
    sys.path.insert(0, HERE)
    import importlib
    prm = importlib.import_module("pn119_router")

    class StubRunner:
        device = "cpu"

    with tempfile.TemporaryDirectory(dir=NEEDFIT) as d:
        probe = os.path.join(d, "probe.npz")
        base = np.load(os.path.join(NEEDFIT, "pn119-live/probe.npz"), allow_pickle=True)
        arrays = {k: base[k] for k in ("mu", "sd", "Vt10", "w")}
        atomic_write_npz(probe, arrays)
        router = prm.PN119Router(StubRunner(), probe)
        w0 = router.w.clone()
        # swap in a probe with shifted weights
        arrays2 = dict(arrays)
        arrays2["w"] = arrays["w"] + 1.0
        atomic_write_npz(probe, arrays2)
        router._next_reload_check = 0.0
        router._maybe_reload()
        check("T4 hot-reload picks up atomic swap",
              bool(torch_allclose(router.w, w0 + 1.0)),
              f"w[-1] {float(w0[-1]):.4f} -> {float(router.w[-1]):.4f}")
        # corrupt swap: atomic rename of a non-npz payload
        w_good = router.w.clone()
        atomic_write_bytes(probe, b"this is not an npz file")
        router._next_reload_check = 0.0
        router._maybe_reload()
        check("T4 corrupt swap keeps old weights, no raise",
              bool(torch_allclose(router.w, w_good)) and router._failed_sig is not None)
        # recovery: next good swap loads despite earlier failure
        atomic_write_npz(probe, arrays)
        router._next_reload_check = 0.0
        router._maybe_reload()
        check("T4 recovers on next good swap", bool(torch_allclose(router.w, w0)))
    return prm


def torch_allclose(a, b):
    import torch
    return torch.allclose(a, b, atol=1e-6)


# ── T5: explore knob ────────────────────────────────────────────────────────
def t5_explore(prm):
    class StubRunner:
        device = "cpu"

    os.environ["PN119_EXPLORE"] = "0.03"
    router = prm.PN119Router(StubRunner(), os.path.join(NEEDFIT, "pn119-live/probe.npz"))
    ids = [f"chatcmpl-{i:08x}-{i*2654435761 % 2**32:08x}" for i in range(20000)]
    flags = [router._is_explore(r) for r in ids]
    rate = sum(flags) / len(flags)
    check("T5 explore rate ~3%", 0.02 < rate < 0.04, f"measured {rate:.4f}")
    check("T5 explore deterministic per req_id",
          all(router._is_explore(r) == f for r, f in zip(ids[:500], flags[:500])))
    os.environ["PN119_EXPLORE"] = "0"
    r0 = prm.PN119Router(StubRunner(), os.path.join(NEEDFIT, "pn119-live/probe.npz"))
    check("T5 explore=0 flags nothing", not any(r0._is_explore(r) for r in ids[:2000]))
    os.environ.pop("PN119_EXPLORE", None)


# ── T6: censoring guards on synthetic enforce-era rows ──────────────────────
def t6_guards():
    x = np.zeros(FEAT_DIM, np.float32)

    def row(req, mode="enforce", route="lean", explore=False, cap=False,
            thinking=True, rtok=100):
        return Row(req_id=req, x=x, rtok=rtok, cap_hit=cap, mode=mode,
                   route=route, explore=explore, thinking=thinking, ts=0.0)

    rows = [
        row("censored-lean"),                     # enforce+lean → DROP
        row("lean-caphit", cap=True),             # enforce+lean+cap → KEEP (y=1)
        row("explore-lean", explore=True),        # explore → KEEP
        row("deep-routed", route="deep"),         # deep got full budget → KEEP
        row("shadow-lean", mode="shadow"),        # shadow uncensored → KEEP
        row("thinking-off", mode="shadow", thinking=False),   # G1 → DROP
        row("thinking-unknown", mode="shadow", thinking=None),  # G1 → DROP
    ]
    c = {}
    kept = {r.req_id for r in apply_guards(rows, legacy_thinking_ok=False, counts=c)}
    expect = {"lean-caphit", "explore-lean", "deep-routed", "shadow-lean"}
    check("T6 censoring guards keep exactly the uncensored set",
          kept == expect, f"kept={sorted(kept)} counts={c}")
    caphit = [r for r in rows if r.req_id == "lean-caphit"][0]
    y = 1.0 if (caphit.cap_hit or caphit.rtok >= 2000) else 0.0
    check("T6 lean cap-hit labels positive despite censored rtok", y == 1.0,
          f"rtok={caphit.rtok} cap_hit={caphit.cap_hit} -> y={y}")


# ── T7: current-probe baseline rho (feature-order + scoring equivalence) ────
def t7_baseline_rho():
    from refit_pn119_probe import load_seed
    seed_ids, X_seed, _ = load_seed(2000)
    z = np.load(os.path.join(NEEDFIT, "pn119-live/probe.npz"), allow_pickle=True)
    scores = score_with(z["mu"], z["sd"], z["Vt10"], z["w"], X_seed)
    loo = json.load(open(os.path.join(NEEDFIT, "router_loo_scores.json"),
                         encoding="utf-8"))
    loo_map = dict(zip(loo["ids"], loo["scores"]))
    overlap = [i for i in seed_ids if i in loo_map]
    idx = {i: k for k, i in enumerate(seed_ids)}
    rho = spearman(np.array([scores[idx[i]] for i in overlap]),
                   np.array([loo_map[i] for i in overlap]))
    top25_loo = set(sorted(overlap, key=lambda i: -loo_map[i])[:25])
    top25_new = set(sorted(overlap, key=lambda i: -scores[idx[i]])[:25])
    inter = len(top25_loo & top25_new)
    check("T7 current probe vs LOO reference rho >= 0.85",
          rho >= 0.85, f"rho={rho:.4f} n={len(overlap)} top25 overlap {inter}/25")
    return rho


def main():
    random.seed(119)
    print("== T1 atomic swap kill-test ==")
    t1_kill_test()
    print("== T2 real sink parse + guards ==")
    t2_real_sink()
    print("== T3 refit on real data ==")
    t3_refit()
    print("== T4 router hot-reload ==")
    prm = t4_hot_reload()
    print("== T5 explore knob ==")
    t5_explore(prm)
    print("== T6 censoring guards ==")
    t6_guards()
    print("== T7 baseline rho ==")
    t7_baseline_rho()
    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
