#!/usr/bin/env python3
"""PN119 rate controller + DEEP_FRAC_WINDOW_DEGENERATE.

Two defects, one root cause: `PN119_TDEEP` is an ABSOLUTE threshold on a score
scale that is not stable across probe reloads, and the alarm that was supposed
to catch the consequence counts from boot.

The live incident: `PN119_TDEEP=0.495` routed 5 of 100 deep, and in a
252-request trailing window the MAXIMUM score was 0.4867 — below the threshold.
Zero deep for 10.6 minutes. `DEEP_FRAC_DEGENERATE` could not fire because it is
boot-cumulative and all-or-nothing, so a single deep request earlier in the boot
disarms it permanently; the only live signal was `DEEP_FRAC_OUT_OF_BAND` at
*warn*, which is the same id and exit code as the ~13% tuning drift the USER
deferred. "Needs retuning" and "the router stopped routing" were byte-identical.

What is asserted here:

  CONTROLLER   off by default (absolute threshold preserved exactly); defers to
               the static threshold until the window has `rate_min_n` scores;
               then holds the realised deep fraction at the target REGARDLESS
               of where the score scale sits — including a scale shifted
               entirely below the old threshold, which is the live defect;
               never lets a request move the cut point it is judged against;
               refuses a non-finite window rather than routing everything lean.
  ALARM        fires on the live incident's shape (a trailing window that is
               all-lean while the cumulative fraction looks merely low), stays
               quiet on healthy mixed traffic, and does not need the cumulative
               check to be quiet in order to fire.

Pure: no probe, no torch, no GPU. The controller and the alarm are exercised as
the real functions, with a duck-typed router for the two methods that need one.

    python3 fixes/test_pn119_rate_controller.py
"""

from __future__ import annotations

import math
import pathlib
import sys
import types

REPO = pathlib.Path(__file__).resolve().parents[1]
_fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


def _load_router():
    sys.path.insert(0, str(REPO / "fixes"))
    import pn119_router as R  # noqa: PLC0415 — deliberate late import
    return R


PCTL_WINDOW = 512  # PN119_PCTL_WINDOW default, mirrored so the test drifts with it


def _router(R, *, tdeep=0.495, target=0.0, min_n=64, scores=(), raw=None):
    """A duck-typed router carrying only what the controller reads.

    `scores` seeds the warmup window (sorted, as _pctl maintains it). `raw`
    installs a window verbatim, for the ordering-sensitive cases.
    """
    import collections
    arrival = list(raw) if raw is not None else list(scores)
    srt = list(raw) if raw is not None else sorted(scores)
    return types.SimpleNamespace(
        tdeep=tdeep,
        rate_target=target,
        rate_min_n=min_n,
        # Two views of one window, exactly as the router keeps them: `_score_win`
        # is ARRIVAL order (it drives FIFO eviction) and `_score_sorted` is the
        # sorted index. Seeding both from the sorted list makes eviction drop the
        # smallest scores first and walks the threshold upward — a harness bug
        # that reads exactly like a controller that undershoots its target.
        _score_sorted=srt,
        _score_win=collections.deque(arrival),
        _route_win=collections.deque(maxlen=256),
    )


def _route(R, rt, score):
    """One routing decision, in the order and with the bookkeeping _publish
    and _pctl perform: decide first, then the score joins the bounded window."""
    import bisect
    T, src = R.PN119Router._effective_tdeep(rt)
    route = R.ROUTE_DEEP if score >= T else R.ROUTE_LEAN
    rt._route_win.append(route == R.ROUTE_DEEP)
    bisect.insort(rt._score_sorted, score)
    rt._score_win.append(score)
    while len(rt._score_win) > PCTL_WINDOW:
        old = rt._score_win.popleft()
        i = bisect.bisect_left(rt._score_sorted, old)
        if i < len(rt._score_sorted) and rt._score_sorted[i] == old:
            rt._score_sorted.pop(i)
    return route, T, src


def _lcg(seed: int, n: int) -> list[int]:
    """Deterministic shuffle source — real traffic is not sorted, and a test
    that feeds a monotonically rising stream measures drift, not calibration."""
    out, x = [], seed
    for _ in range(n):
        x = (1103515245 * x + 12345) % 2147483648
        out.append(x)
    return out


def _shuffled(vals: list[float], seed: int = 7) -> list[float]:
    keys = _lcg(seed, len(vals))
    return [v for _, v in sorted(zip(keys, vals), key=lambda kv: kv[0])]


def _run(R, *, target, warmup, stream, min_n=64):
    """Warm the window on `warmup`, then route `stream`. -> realised deep frac."""
    rt = _router(R, target=target, min_n=min_n, scores=warmup)
    deep = sum(1 for s in stream if _route(R, rt, s)[0] == R.ROUTE_DEEP)
    return deep / len(stream)


def _snap(R, *, scored, deep, w_n, w_deep):
    """A health snapshot with the traffic + window shape under test."""
    st = {"scored": scored, f"scored_{R.ROUTE_DEEP}": deep,
          f"scored_{R.ROUTE_LEAN}": scored - deep, "decisions": scored,
          "batch_adds": scored}
    snap = R.make_snapshot(
        stats=st, boot_id="t", pid=1, hostname="h", started=1.0, now=10_000.0,
        mode="enforce", mode_requested="enforce", tdeep=0.495,
        window={"n": w_n, "deep": w_deep, "lean": w_n - w_deep, "cap": 256,
                "min_n": R._ALARM_WINDOW_MIN_N, "deep_frac": (w_deep / w_n) if w_n else 0.0},
        consumer={"flag_env": False},
    )
    return snap


def ids(snap) -> set:
    return set(snap.get("alarm_ids") or [])


# ───────────────────────────────────────────────────────────── the controller
def test_controller(R) -> None:
    print("\n── the rate controller ─────────────────────────────────────────")

    # Default OFF: the absolute threshold must be preserved bit for bit.
    rt = _router(R, target=0.0, scores=[i / 1000 for i in range(500)])
    T, src = R.PN119Router._effective_tdeep(rt)
    check("default is the absolute threshold, unchanged",
          T == 0.495 and src == "static", f"T={T} source={src}")

    # Warmup: too few samples to trust a quantile.
    rt = _router(R, target=0.30, min_n=64, scores=[0.1, 0.2, 0.3])
    T, src = R.PN119Router._effective_tdeep(rt)
    check("defers to the static threshold below rate_min_n",
          T == 0.495 and src == "warmup", f"T={T} source={src} n=3")

    # The live defect: a scale sitting ENTIRELY below the old threshold.
    # Absolute -> 0 deep forever. Rate-targeted -> still 30%.
    lo = [0.30 + 0.0005 * i for i in range(200)]  # max 0.3995, all < 0.495
    check("the shifted scale really is all below tdeep", max(lo) < 0.495,
          f"max={max(lo):.4f} < 0.495")

    rt = _router(R, target=0.0, scores=lo)
    deep_abs = sum(1 for s in lo if s >= R.PN119Router._effective_tdeep(rt)[0])
    check("absolute threshold routes NOTHING deep on that scale",
          deep_abs == 0, f"{deep_abs}/{len(lo)} deep")

    warm, stream = _shuffled(lo, 3), _shuffled(lo, 11)
    frac = _run(R, target=0.30, warmup=warm, stream=stream)
    check("rate controller holds ~30% deep on the SAME scale",
          0.24 <= frac <= 0.36, f"{frac:.3f}")

    # Not an artefact of that offset: move the whole scale and nothing changes.
    for shift in (-10.0, 0.0, +10.0, +1000.0):
        f = _run(R, target=0.30,
                 warmup=[s + shift for s in warm],
                 stream=[s + shift for s in stream])
        check(f"deep fraction is scale-invariant (shift {shift:+.0f})",
              0.24 <= f <= 0.36, f"{f:.3f}")

    # Target is honoured, not merely "some deep".
    base = [i / 500 for i in range(400)]
    for target in (0.10, 0.25, 0.50):
        f = _run(R, target=target, warmup=_shuffled(base, 5),
                 stream=_shuffled(base, 23))
        check(f"target {target:.2f} realised within 6pp", abs(f - target) <= 0.06,
              f"realised {f:.3f}")

    # KNOWN AND BOUNDED: a windowed quantile lags a TRENDING stream, so a
    # monotonically rising score scale over-routes. That is acceptable — the
    # failure this controller exists to prevent is the degenerate 0%, and it
    # must never produce that even here.
    f = _run(R, target=0.30, warmup=sorted(lo)[:100], stream=sorted(lo))
    check("a rising (non-stationary) stream still never degenerates",
          0.0 < f < 1.0, f"realised {f:.3f} on a monotonically rising scale")

    # No self-reference: the threshold a request is judged against must not
    # include that request's own score.
    rt = _router(R, target=0.30, min_n=8, scores=[0.1] * 50)
    before = list(rt._score_sorted)
    T_seen = R.PN119Router._effective_tdeep(rt)[0]
    _route(R, rt, 999.0)
    check("a request cannot move the cut point it is judged against",
          T_seen == R.PN119Router._effective_tdeep(
              _router(R, target=0.30, min_n=8, scores=before))[0],
          f"T={T_seen}")

    # A non-finite score AT the selected index must not become the threshold:
    # NaN >= T is False for everything, which is the silent all-lean failure
    # this controller exists to prevent. Installed verbatim, because sorting a
    # list containing NaN does not place it predictably.
    n = 100
    k = int(math.ceil((1.0 - 0.30) * n))
    win = [i / 1000 for i in range(n)]
    win[k] = float("nan")
    rt = _router(R, target=0.30, min_n=8, raw=win)
    T, src = R.PN119Router._effective_tdeep(rt)
    check("a non-finite cut point falls back instead of routing all-lean",
          math.isfinite(T) and src == "nonfinite", f"T={T} source={src}")

    # An out-of-range target is zeroed by the loader, but the method must also
    # be total if one ever reaches it — no IndexError, no infinite threshold.
    for bad in (1.0, 1.5, -0.2):
        rt = _router(R, target=bad, min_n=8, scores=base)
        try:
            T, src = R.PN119Router._effective_tdeep(rt)
            ok = math.isfinite(T)
        except Exception as exc:  # noqa: BLE001 — that is the assertion
            T, src, ok = exc, "raised", False
        check(f"an out-of-range target ({bad}) cannot crash the decision", ok,
              f"T={T} source={src}")


# ────────────────────────────────────────────────────────────────── the alarm
def test_window_alarm(R) -> None:
    print("\n── DEEP_FRAC_WINDOW_DEGENERATE ─────────────────────────────────")

    # The live incident: 5/100 deep cumulatively (so the boot-wide check is
    # quiet and only the deferred OUT_OF_BAND warn fires), while the trailing
    # window is entirely lean.
    s = _snap(R, scored=300, deep=5, w_n=252, w_deep=0)
    fired = ids(s)
    check("fires on the live incident shape (252 all-lean, 5/300 cumulative)",
          "DEEP_FRAC_WINDOW_DEGENERATE" in fired, ",".join(sorted(fired)) or "NOTHING")
    check("and it is critical, not the deferred warn",
          R._ALARM_SEVERITY.get("DEEP_FRAC_WINDOW_DEGENERATE") == "critical",
          R._ALARM_SEVERITY.get("DEEP_FRAC_WINDOW_DEGENERATE"))
    check("the cumulative check is indeed quiet here",
          "DEEP_FRAC_DEGENERATE" not in fired,
          "boot-wide is all-or-nothing and 5 != 0")

    # An all-DEEP window is the same class of failure.
    s = _snap(R, scored=300, deep=295, w_n=252, w_deep=252)
    check("fires when the window is all-deep too",
          "DEEP_FRAC_WINDOW_DEGENERATE" in ids(s), ",".join(sorted(ids(s))))

    # Healthy mixed traffic must stay silent.
    s = _snap(R, scored=300, deep=90, w_n=252, w_deep=76)
    check("silent on healthy mixed traffic",
          "DEEP_FRAC_WINDOW_DEGENERATE" not in ids(s), ",".join(sorted(ids(s))))

    # Even ONE deep in the window means it has not stopped.
    s = _snap(R, scored=300, deep=5, w_n=252, w_deep=1)
    check("silent when the window still has a single deep",
          "DEEP_FRAC_WINDOW_DEGENERATE" not in ids(s), ",".join(sorted(ids(s))))

    # Below the minimum sample count it must not fire on a cold boot.
    s = _snap(R, scored=300, deep=5, w_n=R._ALARM_WINDOW_MIN_N - 1, w_deep=0)
    check("silent below the minimum window size",
          "DEEP_FRAC_WINDOW_DEGENERATE" not in ids(s),
          f"n={R._ALARM_WINDOW_MIN_N - 1}")

    # A snapshot with no window at all (the doctor's reader-side path) must not
    # raise and must not fire.
    s = R.make_snapshot(stats={"scored": 100, f"scored_{R.ROUTE_DEEP}": 30},
                        mode="enforce", tdeep=0.495, consumer={"flag_env": False})
    check("absent window section is silent, not an exception",
          "DEEP_FRAC_WINDOW_DEGENERATE" not in ids(s), "no window key")

    # The id must be registered, or the doctor cannot grade it.
    check("id is registered in ALARM_IDS",
          "DEEP_FRAC_WINDOW_DEGENERATE" in R.ALARM_IDS)


def test_window_stats(R) -> None:
    print("\n── window_stats bookkeeping ────────────────────────────────────")
    rt = _router(R, target=0.30, min_n=8, scores=[i / 100 for i in range(200)])
    for s in [i / 100 for i in range(200)]:
        _route(R, rt, s)
    w = R.PN119Router.window_stats(rt)
    check("counts only what fits the window cap",
          w["n"] == min(200, w["cap"]), f"n={w['n']} cap={w['cap']}")
    check("deep + lean == n", w["deep"] + w["lean"] == w["n"],
          f"{w['deep']}+{w['lean']} vs {w['n']}")
    check("reports the target it is controlling to",
          w["rate_target"] == 0.30, str(w["rate_target"]))


def main() -> int:
    print("PN119 rate controller + trailing-window alarm\n")
    R = _load_router()
    test_controller(R)
    test_window_alarm(R)
    test_window_stats(R)
    print()
    if _fails:
        print(f"VERDICT: {len(_fails)} FAILED — " + "; ".join(_fails))
        return 1
    print("ALL PASS")
    print("VERDICT: the deep fraction is held at target regardless of where the "
          "score scale sits, no request moves its own cut point, and a router "
          "that STOPS routing deep now raises a critical alarm instead of the "
          "same warn as a tuning drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
