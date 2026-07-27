#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PN162 — closed-loop thinking-budget calibrator, HOST/LEARNING half.

Reads the PN119 sink, converts each scored generation's BUDGET OUTCOME into a
per-steps-bucket multiplier, and writes the ledger the in-engine consumer
(`_genesis/middleware/pn162_budget_cal.py`) hot-reloads.

    /usr/bin/python3 ~/shared/needfit/pn162-ledger-update.py            # one pass
    /usr/bin/python3 ~/shared/needfit/pn162-ledger-update.py --dry-run --json

Stdlib only — no numpy, no vllm, no network, no GPU. Idempotent (a per-file
byte cursor), safe to run every minute from the pn119-cont refit cycle:
add ONE line, no other coupling.

THE SIGNAL
----------
Not correctness, not an extra LLM call. The engine already records, per
request, whether its own thinking budget bound (BUG-139 censor schema 2):

    censor_forced / censored          -> BOUND  : the cap stopped the reasoning
    grant - rtok > SLACK_FRAC * grant -> SLACK  : the cap was far too loose
    otherwise                         -> OK     : the cap was about right

    k[bucket] *= BUMP  on BOUND   (default 1.15, clamp 3.0)
    k[bucket] *= DECAY on SLACK   (default 0.97, floor 0.7)
    k[bucket] unchanged on OK

Causal grounding (banked 2026-07-26): force-closed items gain +9.6pt when given
more; natural stops lose nothing. Bumping a bound bucket is the paying side of
the trade; decaying a slack bucket is free.

WHAT CARRIES THE STEPS ESTIMATE — IT IS NOT IN THE SINK
-------------------------------------------------------
Verified 2026-07-27 over the live sink: the meta jsonl rows carry
req_id / row / score / pctl / route / prompt_tok / budget_grant / budget_source
/ lane_key / rtok / generated / cap_hit / censored / censor_forced / censor_src
/ thinking / caller / suite / cached_prefix / ts — and NO steps field, and no
prompt hash. PN100's step estimate never leaves the API-server process (it goes
into `chat_template_kwargs["pn100_steps"]`, which the PN102 banner pops).

So steps is INVERTED from the grant. PN100's continuous grid is
`round100(steps x TOK_PER_STEP)`, which is injective over the live range
(1..39 steps at 260 tok/step; verified in `test_pn162_ledger_update.py`), so
`steps ~= round(grant / TOK_PER_STEP)` recovers it exactly.

That inversion has ONE trap, and it is PN162's own feedback: once k != 1 the
grant is `round100(steps x TOK * k[steps])`, so the naive quotient recovers
`steps*k`, not `steps`, and the update would be credited to the wrong bucket —
a self-reinforcing drift. `invert_steps()` therefore inverts THROUGH the
current bucket map: it searches for the b whose calibrated grant equals the
observed one, and falls back to the naive quotient only when no b matches
(a ledger changed mid-window). With k all 1.0 it is exactly the quotient.

Assumes the two other multipliers on that path are at their defaults
(`GENESIS_PN100_HIGHSTEP_MULT=1.0`, `GENESIS_PN100_STEP_BUDGET_MAP` unset).
Pass --highstep-mult / --highstep-min if the boot sets them; the map is
mutually exclusive with PN162 and the consumer refuses to multiply under it.

TELEMETRY (recorded, NEVER steers)
----------------------------------
Control is purely bound/slack/ok. Alongside it the ledger records, per bucket:
realized rtok median/p80, implied realized steps (rtok / TOK_PER_STEP), the
outcome counts, and the LEAN-lane anchor-adherence rate — so "estimated 5,
realized ~8" is visible in the JSON and a bucket's drift from k=1 is
explainable. The estimate's only consumer is the budget (the v5 banner pops
`pn100_steps` and announces nothing), so bucket-k absorbs estimator bias by
construction; this telemetry exists to judge the estimator PROMPT, not to
steer.

There is no realized step-marker count anywhere in the sink (searched: no
`steps`, `step_markers`, `pn100_*` field exists on any row), so realized steps
is `rtok / TOK_PER_STEP` and is labelled `steps_real_*` to keep that visible.

THE ANNOUNCEMENT BLIND SPOT + THE ORACLE
-----------------------------------------
The LEAN lane announces N into the prompt (`_contract_v3_sized` renders "about
{steps} short reasoning steps"; the DEEP lane runs v5, which pops the estimate
and announces nothing). Announced N is a behavioural anchor, so an
under-announced N SELF-FULFILS: the model paces to N, stops clean, nothing
binds. The bound/slack loop is structurally blind to it.

The oracle is cross-lane: within a lens-score band (the `score` field on every
sink score line), deep-lane rows are unanchored, so their realized steps are
ground truth for comparable difficulty. `deep_realized_med - lean_announced_mean`
per band estimates the announcement bias, and each bucket inherits the
band-weighted bias of its lean rows. Computed only over NON-BOUND deep rows —
a bound deep row measures its cap, not its need.

EXPLORATION ARMS (dark, USER RULING PENDING)
--------------------------------------------
When the engine runs with GENESIS_ENABLE_PN162_EXPLORE=1 it stamps an arm
label into `vllm_xargs["x_caller"]`, which the router surfaces as the sink's
`caller` column (pn119_router.py:2440) — no router change needed. Arms are
`pn162:c` (control) and `pn162:e1` / `pn162:e2` (announced N+delta). This pass
compares them per bucket on the proxies that are ACTUALLY derivable from sink
rows:

    derivable : bound rate, rtok median, answer tokens (= generated - rtok),
                empty-answer rate (answer tokens <= --empty-answer-tok),
                cap_hit rate
    NOT derivable from the sink : PN101/PN102 rescue fires, format/parse
                failures, correctness. Those live in the server log and the
                qbench results JSON; joining them is the refit's
                `--results-dir accuracy_monitor` path and is deliberately out
                of scope here (it needs a completed results dir).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

SINK_DEFAULT = "/home/user/shared/needfit/pn119-sink"
LEDGER_DEFAULT = "/home/user/shared/needfit/pn119-live/pn162-ledger.json"
CURSOR_DEFAULT = "/home/user/shared/needfit/pn162-cursor.json"

LEDGER_SCHEMA = 1
CURSOR_SCHEMA = 1
MAX_BUCKET = 16           # must equal pn162_budget_cal.MAX_BUCKET
MAX_STEPS_SEARCH = 64     # ceil 10240 / 260 / k_min -> ~56

DEFAULTS = dict(
    window=1000,
    tok_per_step=260,
    bump=1.15,
    decay=0.97,
    slack_frac=0.40,
    k_min=0.7,
    k_max=3.0,
    max_step=1.5,          # per-pass cap on |k_new / k_old|
    min_generated=32,
    censor_slack=13,
    empty_answer_tok=8,
    adherence_frac=0.15,
    score_bands=5,
    highstep_mult=1.0,
    highstep_min=10,
    budget_floor=128,
    budget_ceil=10240,
    key_schema="steps",     # "steps" | "steps_ptok"  (see bucket_keys)
    min_cell=30,            # composite cells below this are not written
    # BUG-169 orphan bounds — how long an unpaired score line may pin the
    # cursor before it is dropped. See read_sink_since for what each one
    # bounds; whichever trips first retires the line.
    orphan_bytes=1048576,   # newer bytes in the SAME file after it
    orphan_age_s=1800.0,    # sink-clock span to the file's newest ts
    orphan_stale_s=300.0,   # wall-clock since the file was last appended to
    # Which `budget_source` values are trustworthy grants to invert.
    # "h119" is included because the shipped H119_LEAN_MULT/DEEP_MULT are 1.0
    # (exact passthrough — `_h119_route_budget` returns int(prior) unchanged),
    # so an h119-sourced grant IS still PN100's number; the router just owns
    # the row. If a boot ever moves those multipliers off 1.0, run with
    # --sources pn100 or the steps inversion reads a doubly-multiplied grant.
    # "caller" is never included: a client-pinned budget is not ours to learn
    # from, and PN100 stamps h119_overridable=0 on exactly those.
    sources="pn100,h119",
)


# ── small stats helpers (stdlib only) ───────────────────────────────────────
def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def _r2(x):
    return None if x is None else round(float(x), 2)


def round100(x) -> int:
    return int(round(float(x) / 100.0)) * 100


def bucket_of(steps) -> int:
    try:
        s = int(round(float(steps)))
    except (TypeError, ValueError):
        return 1
    return max(1, min(MAX_BUCKET, s))


# ── THE KEY SCHEMA — mirrors pn162_budget_cal.bucket_keys EXACTLY ───────────
# The ledger is meant to grow into an extensive SHAPE-keyed table (steps today;
# steps x prompt-length band next), so key derivation lives in one function on
# each side and the two are cross-checked in test_pn162_ledger_update.py.
# Never prompt-identity — shape features only.
#
# Occupancy is the constraint, not the code: a composite cell is WRITTEN only
# when it has >= --min-cell observations in the window. The marginal steps
# bucket is always written, so the consumer's most-specific-first walk lands
# on it whenever a cell is missing or too thin.
KEY_SCHEMA_STEPS = "steps"
KEY_SCHEMA_STEPS_PTOK = "steps_ptok"
KEY_SCHEMAS = (KEY_SCHEMA_STEPS, KEY_SCHEMA_STEPS_PTOK)
DEFAULT_PTOK_BANDS = (256, 1024, 4096)


def band_index(value, edges) -> int:
    if value is None:
        return -1
    try:
        v = float(value)
    except (TypeError, ValueError):
        return -1
    for i, e in enumerate(edges):
        if v < e:
            return i
    return len(edges)


def bucket_keys(steps, ptok=None, schema=KEY_SCHEMA_STEPS, bands=None) -> list:
    """Lookup chain, most specific first, marginal steps bucket last."""
    b = bucket_of(steps)
    keys = []
    if schema == KEY_SCHEMA_STEPS_PTOK:
        idx = band_index(ptok, bands or DEFAULT_PTOK_BANDS)
        if idx >= 0:
            keys.append(f"{b}|p{idx}")
    keys.append(str(b))
    return keys


def lookup_k(kmap: dict, keys: list, cfg: dict) -> float:
    for key in keys:
        if key in kmap:
            return _clamp_k(kmap[key], cfg)
    return 1.0


# ── the PN100 grant grid, and its inverse ───────────────────────────────────
def pn100_grant(steps: int, k: float, cfg: dict) -> int:
    """Reproduce `_continuous_budget`'s k-path exactly (auto_budget.py)."""
    raw = steps * cfg["tok_per_step"]
    if cfg["highstep_mult"] > 1.0 and steps >= cfg["highstep_min"]:
        raw *= cfg["highstep_mult"]
    raw *= k
    return max(cfg["budget_floor"], min(cfg["budget_ceil"], round100(raw)))


def invert_steps(grant: int, kmap: dict, cfg: dict, ptok=None) -> int:
    """Observed grant -> the steps estimate that produced it (bucket domain).

    Inverts THROUGH the live key map so PN162's own multiplier cannot make the
    updater credit the wrong bucket. Ambiguity (two b sharing a grant, or a
    saturated clamp) resolves to the candidate nearest the naive quotient; no
    candidate at all falls back to the naive quotient outright.
    """
    naive = max(1, int(round(grant / float(cfg["tok_per_step"]))))
    schema = cfg.get("key_schema", KEY_SCHEMA_STEPS)
    cands = [
        b for b in range(1, MAX_STEPS_SEARCH + 1)
        if pn100_grant(
            b, lookup_k(kmap, bucket_keys(b, ptok, schema), cfg), cfg) == grant
    ]
    if not cands:
        return bucket_of(naive)
    return bucket_of(min(cands, key=lambda b: (abs(b - naive), b)))


def _clamp_k(k, cfg) -> float:
    try:
        k = float(k)
    except (TypeError, ValueError):
        return 1.0
    if k != k or k in (float("inf"), float("-inf")):
        return 1.0
    return max(cfg["k_min"], min(cfg["k_max"], k))


# ── outcome classification (mirrors refit_pn119_probe.censoring_of order) ───
def classify(row: dict, cfg: dict) -> str:
    """'bound' | 'slack' | 'ok' | 'drop:<why>' for one paired sink row.

    Trust order, same as the refit's `censoring_of`:
      1. censor_forced — the engine tap OBSERVED the holder force `</think>`
      2. censored      — the router's own verdict
      3. the arithmetic — rtok within `censor_slack` of the grant

    Deliberately NOT the refit's rule in one place: a bare `cap_hit` (max_tokens
    truncation) is dropped rather than counted as bound. `cap_hit` means the
    COMPLETION cap stopped the row; the thinking budget may have been nowhere
    near binding, and rtok is not a clean slack measurement either. Counting it
    would let a client's small max_tokens inflate every bucket's k.
    """
    grant = row.get("budget_grant")
    if not grant or grant <= 0:
        return "drop:no_grant"
    if row.get("thinking") is False:
        return "drop:thinking_off"
    srcs = [s for s in str(cfg.get("sources", "")).split(",") if s]
    if srcs and (row.get("budget_source") or "") not in srcs:
        return "drop:budget_source"
    gen = row.get("generated")
    if gen is not None and gen < cfg["min_generated"]:
        return "drop:no_generation"       # refit G6: max_tokens=1 diagnostics
    rtok = int(row.get("rtok") or 0)
    if row.get("censor_forced") or row.get("censored"):
        return "bound"
    if rtok >= grant - cfg["censor_slack"]:
        return "bound"
    if row.get("cap_hit"):
        return "drop:cap_hit"
    if (grant - rtok) > cfg["slack_frac"] * grant:
        return "slack"
    return "ok"


# ── sink reading ────────────────────────────────────────────────────────────
def load_markers(sink: str) -> set:
    """req_ids fenced off by a diagnostic tool (`.synthetic-*.json`).

    Same convention `pn119_b3_numerics.mark_synthetic` writes and the refit
    reads: synthetic max_tokens=1 traffic must never train anything.
    """
    ids = set()
    for p in sorted(glob.glob(os.path.join(sink, ".synthetic-*.json"))):
        try:
            with open(p, "r", encoding="utf-8") as f:
                ids.update(json.load(f).get("req_ids") or [])
        except (OSError, ValueError):
            continue
    return ids


def _orphaned(sm: dict, s_end: int, raw_end: int, max_ts: float,
              mtime: float, now: float, cfg: dict) -> bool:
    """Has this unpaired score line waited long enough to be given up on?

    Any ONE of the three bounds retires it. They are deliberately different in
    kind because the ways a finish can fail to arrive are different in kind:
    a burst of traffic behind it (bytes), a slow trickle behind it (sink
    clock), or a dead boot whose file will never grow again (wall clock).
    """
    if (raw_end - s_end) > cfg["orphan_bytes"]:
        return True
    try:
        s_ts = float(sm.get("ts") or 0.0)
    except (TypeError, ValueError):
        s_ts = 0.0
    if s_ts > 0.0 and (max_ts - s_ts) > cfg["orphan_age_s"]:
        return True
    return (now - mtime) > cfg["orphan_stale_s"]


def read_sink_since(sink: str, offsets: dict, cfg: dict,
                    counts: dict) -> tuple[list, dict]:
    """New paired rows since the cursor, plus the updated offsets.

    A meta file is append-only, so a byte offset is a sound cursor. A file
    that SHRANK was rotated/truncated under us — reread it from zero.

    Score and finish lines are paired inside one file (the router writes both
    for a request into whichever file its boot owns). The cursor stops at the
    START of the EARLIEST score line whose finish has not landed yet, and only
    pairs lying ENTIRELY behind that point are emitted. A request in flight at
    a pass boundary is therefore DELAYED to the next pass, never skipped.

    BUG-169: this used to advance to max(finish offset) over the pass's
    complete pairs, which seeks past an in-flight score line sitting before
    that maximum — and when its finish lands, the next pass no longer has the
    score line, so the row is dropped in BOTH passes. At seqs=6 there is
    almost always a long thinker straddling a pass, and the rows lost that way
    are the slow ones, i.e. exactly the `bound` population that drives k
    upward. Emitting only fully-behind-the-pin pairs is what keeps the delay
    idempotent: the region the cursor covers contains complete pairs only, so
    re-reading can never double-credit a row.

    A score line whose finish will NEVER land must not pin the cursor for
    ever. `_orphaned` retires one on whichever of these trips first:

      * --orphan-bytes    (1 MiB) newer bytes appended to the SAME file after
        it — bounds both the per-pass re-read and the lag under traffic.
      * --orphan-age-s    (1800 s) sink-clock span between it and the newest
        ts in the file — bounds the lag under a slow trickle, where 1 MiB
        would take hours to accumulate. No live request thinks for 30 min
        (the budget ceiling is 10240 tokens).
      * --orphan-stale-s  (300 s) wall clock since the file was last appended
        to — a boot that died mid-request leaves a file that never grows
        again, so neither of the other two bounds can ever trip on it.

    Retiring a line counts `orphan_abandoned`; it is the ONLY path on which a
    score line is dropped without its finish.
    """
    markers = load_markers(sink)
    new_offsets = dict(offsets)
    rows = []
    now = time.time()
    for path in sorted(glob.glob(os.path.join(sink, "meta-*.jsonl"))):
        name = os.path.basename(path)
        try:
            size = os.path.getsize(path)
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        start = int(new_offsets.get(name, 0) or 0)
        if start > size:
            counts["files_truncated"] = counts.get("files_truncated", 0) + 1
            start = 0
        if start == size:
            continue
        counts["files_read"] = counts.get("files_read", 0) + 1
        scores, finishes = {}, {}
        max_ts = 0.0
        raw_end = start
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(start)
                pos = start
                for raw in f:
                    line_start = pos
                    pos += len(raw.encode("utf-8"))
                    # Only a newline-terminated line is whole. A torn tail must
                    # stay UNCONSUMED — the next append concatenates onto it,
                    # and a cursor past it would lose both rows.
                    if raw.endswith("\n"):
                        raw_end = pos
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        m = json.loads(line)
                    except ValueError:
                        counts["bad_json"] = counts.get("bad_json", 0) + 1
                        continue
                    if m.get("pn119_header"):
                        continue
                    rid = m.get("req_id")
                    if not rid:
                        continue
                    try:
                        max_ts = max(max_ts, float(m.get("ts") or 0.0))
                    except (TypeError, ValueError):
                        pass
                    if m.get("finish"):
                        finishes[rid] = (m, pos)
                    elif "row" in m or "budget_grant" in m:
                        scores.setdefault(rid, (m, line_start, pos))
        except OSError:
            continue

        # Where the cursor is allowed to stop: the earliest still-live
        # unpaired score line, or the end of the parsed content if there is
        # none. `start` is the floor, so a pin on the first line means no
        # advance at all.
        pin = None
        for rid, (sm, s_start, s_end) in scores.items():
            if rid in finishes:
                continue
            counts["unfinished"] = counts.get("unfinished", 0) + 1
            if _orphaned(sm, s_end, raw_end, max_ts, mtime, now, cfg):
                counts["orphan_abandoned"] = counts.get(
                    "orphan_abandoned", 0) + 1
                continue
            pin = s_start if pin is None else min(pin, s_start)
        complete_end = raw_end if pin is None else pin

        for rid, (sm, _s_start, s_end) in scores.items():
            got = finishes.get(rid)
            if got is None:
                continue
            fm, f_off = got
            if max(s_end, f_off) > complete_end:
                # Behind the pin: delayed to the next pass, NOT dropped, and
                # not counted here either — the next pass counts it once.
                continue
            if rid in markers:
                counts["marker_excluded"] = counts.get("marker_excluded", 0) + 1
                continue
            rows.append(_pair(rid, sm, fm))
        new_offsets[name] = complete_end
    rows.sort(key=lambda r: r["ts"])
    return rows, new_offsets


def _pair(rid: str, sm: dict, fm: dict) -> dict:
    """One flat row from the (score, finish) pair. Wire data: never coerce
    a missing field into a fake value — None stays None and `classify` drops."""
    def pick(key, default=None):
        v = fm.get(key, sm.get(key, default))
        return default if v in (None, "") else v

    gen = fm.get("generated")
    rtok = fm.get("rtok")
    rtok = gen if rtok is None else rtok
    return {
        "req_id": rid,
        "ts": float(fm.get("ts", sm.get("ts", 0.0)) or 0.0),
        "route": str(sm.get("route", "") or ""),
        "score": (None if sm.get("score") is None else float(sm["score"])),
        "prompt_tok": sm.get("prompt_tok"),
        "budget_grant": (None if pick("budget_grant") is None
                         else int(pick("budget_grant"))),
        "budget_source": str(pick("budget_source", "") or ""),
        "rtok": (None if rtok is None else int(rtok)),
        "generated": (None if gen is None else int(gen)),
        "cap_hit": bool(fm.get("cap_hit", False)),
        "censored": fm.get("censored", sm.get("censored")),
        "censor_forced": fm.get("censor_forced"),
        "thinking": fm.get("thinking"),
        # PN162 exploration arm, if the engine is armed. `caller` is the
        # router's surface for vllm_xargs caller/x_caller (pn119_router:2440).
        "arm": (str(sm.get("caller")) if sm.get("caller") else None),
    }


# ── the update ──────────────────────────────────────────────────────────────
def writable_keys(window: list, cfg: dict) -> set:
    """Keys the ledger may carry: every marginal steps bucket, plus composite
    cells that clear `min_cell` occupancy IN THE WINDOW.

    This is the whole occupancy rule. A thin cell is simply never written, and
    the consumer's most-specific-first walk then lands on the marginal bucket.
    """
    marg = {str(r["_bucket"]) for r in window if r.get("_bucket")}
    if cfg.get("key_schema", KEY_SCHEMA_STEPS) == KEY_SCHEMA_STEPS:
        return marg
    occ: dict = {}
    for r in window:
        for key in (r.get("_keys") or [])[:-1]:      # composite parts only
            occ[key] = occ.get(key, 0) + 1
    return marg | {k for k, n in occ.items() if n >= cfg["min_cell"]}


def _row_key(row: dict, allowed: set) -> str:
    for key in (row.get("_keys") or []):
        if key in allowed:
            return key
    return str(row.get("_bucket") or 1)


def update_buckets(rows: list, kmap: dict, cfg: dict,
                   allowed: set | None = None) -> tuple[dict, dict]:
    """(new kmap, per-key outcome counts) from NEW rows only.

    Per-row multiplicative, exactly as specified — but the whole pass's change
    for one key is then capped at `max_step` in either direction. Without that
    cap a single unlucky window (say 40 bound rows in one bucket) slams k
    straight to the clamp, which is a control move no evidence supports; with
    it the loop still converges, just over a couple of passes.
    """
    out = {key: _clamp_k(k, cfg) for key, k in kmap.items()}
    per: dict = {}
    for r in rows:
        if r.get("_outcome") not in ("bound", "slack", "ok"):
            continue
        key = _row_key(r, allowed) if allowed is not None else str(r["_bucket"])
        marg = str(r["_bucket"])
        # Credit the APPLIED key and, when they differ, the marginal bucket
        # too. Without the second credit a marginal bucket stops seeing any
        # traffic the moment its composite cells open, and it is precisely the
        # fallback that new and thin cells land on. The cost is a confound —
        # the marginal then learns partly from rows that were sized by a
        # composite k — which is one reason a composite schema needs its own
        # screened boot rather than a flag flip.
        for kk in ({key, marg} if key != marg else {key}):
            c = per.setdefault(kk, {"bound": 0, "slack": 0, "ok": 0})
            c[r["_outcome"]] += 1
    for key, c in per.items():
        k0 = _clamp_k(out.get(key, 1.0), cfg)
        k = k0 * (cfg["bump"] ** c["bound"]) * (cfg["decay"] ** c["slack"])
        k = max(k0 / cfg["max_step"], min(k0 * cfg["max_step"], k))
        out[key] = _clamp_k(k, cfg)
    return out, per


def score_bands(window: list, cfg: dict) -> list:
    """Lens-score band edges from the window (equal-count bins)."""
    vals = sorted(r["score"] for r in window if r.get("score") is not None)
    n = cfg["score_bands"]
    if len(vals) < n * 4:
        return []
    return [_pct(vals, i / n) for i in range(1, n)]


def band_of(score, edges) -> int:
    if score is None or not edges:
        return -1
    for i, e in enumerate(edges):
        if score < e:
            return i
    return len(edges)


def announcement_oracle(window: list, cfg: dict) -> dict:
    """Per-band (lean announced N) vs (deep free-run realized steps).

    Deep rows have no announced N (the v5 banner pops the estimate and prints
    nothing), so within a difficulty band their realizations are unanchored
    ground truth. Bound deep rows are excluded — they measured their cap.
    """
    edges = score_bands(window, cfg)
    out = {"edges": [_r2(e) for e in edges], "bands": {}}
    if not edges:
        out["note"] = "too few scored rows for bands"
        return out
    tps = float(cfg["tok_per_step"])
    acc: dict = {}
    for r in window:
        b = band_of(r.get("score"), edges)
        if b < 0:
            continue
        a = acc.setdefault(b, {"lean_n": [], "deep_steps": []})
        if r["route"] == "lean":
            a["lean_n"].append(r["_bucket"])
        elif r["route"] == "deep" and r["_outcome"] != "bound":
            a["deep_steps"].append((r["rtok"] or 0) / tps)
    for b, a in sorted(acc.items()):
        lean_mean = (sum(a["lean_n"]) / len(a["lean_n"])) if a["lean_n"] else None
        deep_med = _pct(a["deep_steps"], 0.5)
        out["bands"][str(b)] = {
            "n_lean": len(a["lean_n"]), "n_deep_free": len(a["deep_steps"]),
            "lean_announced_mean": _r2(lean_mean),
            "deep_realized_steps_med": _r2(deep_med),
            "announce_bias": (None if (lean_mean is None or deep_med is None)
                              else _r2(deep_med - lean_mean)),
        }
    return out


def telemetry(window: list, kmap: dict, kprev: dict, per: dict,
              oracle: dict, cfg: dict) -> dict:
    """Per-bucket estimator-error + anchor telemetry. Records, never steers."""
    tps = float(cfg["tok_per_step"])
    edges = oracle.get("edges") or []
    by: dict = {}
    for r in window:
        by.setdefault(r["_bucket"], []).append(r)
    # Telemetry reports on the marginal steps axis. `update_buckets` always
    # credits the marginal key as well as the applied one, so no rollup is
    # needed — and summing composite cells in here would double-count.
    per_marg = {k: v for k, v in per.items() if "|" not in k}
    out = {}
    for b in sorted(by):
        rs = by[b]
        rtoks = [r["rtok"] for r in rs if r["rtok"] is not None]
        grants = [r["budget_grant"] for r in rs if r["budget_grant"]]
        ans = [max(0, (r["generated"] or 0) - (r["rtok"] or 0))
               for r in rs if r["generated"] is not None]
        lean = [r for r in rs if r["route"] == "lean" and r["rtok"] is not None]
        adher = [r for r in lean
                 if abs((r["rtok"] / tps) - b) <= cfg["adherence_frac"] * b]
        biases = []
        for r in lean:
            bd = oracle.get("bands", {}).get(str(band_of(r.get("score"), edges)))
            if bd and bd.get("announce_bias") is not None:
                biases.append(bd["announce_bias"])
        pc = per_marg.get(str(b), {})
        out[str(b)] = {
            "n": len(rs),
            "bound": sum(1 for r in rs if r["_outcome"] == "bound"),
            "slack": sum(1 for r in rs if r["_outcome"] == "slack"),
            "ok": sum(1 for r in rs if r["_outcome"] == "ok"),
            "new_bound": pc.get("bound", 0),
            "new_slack": pc.get("slack", 0),
            "new_ok": pc.get("ok", 0),
            "k": round(_clamp_k(kmap.get(str(b), 1.0), cfg), 4),
            "k_prev": round(_clamp_k(kprev.get(str(b), 1.0), cfg), 4),
            "grant_med": _r2(_pct(grants, 0.5)),
            "rtok_med": _r2(_pct(rtoks, 0.5)),
            "rtok_p80": _r2(_pct(rtoks, 0.8)),
            # No realized step-marker count exists anywhere in the sink, so
            # this is rtok / TOK_PER_STEP. Named to keep that visible.
            "steps_real_med": _r2((_pct(rtoks, 0.5) or 0) / tps) if rtoks else None,
            "steps_real_p80": _r2((_pct(rtoks, 0.8) or 0) / tps) if rtoks else None,
            "answer_tok_med": _r2(_pct(ans, 0.5)),
            "n_lean": len(lean),
            "anchor_adherence": (_r2(len(adher) / len(lean)) if lean else None),
            "announce_bias_est": (_r2(sum(biases) / len(biases))
                                  if biases else None),
        }
    return out


def explore_report(window: list, cfg: dict) -> dict:
    """Per-bucket arm comparison on sink-derivable proxies only.

    Everything here is computable from the sink. PN101 rescue fires, format
    failures and correctness are NOT — see the module docstring.
    """
    armed = [r for r in window
             if str(r.get("arm") or "").startswith("pn162:")]
    if not armed:
        return {"armed": False,
                "note": "no pn162:* arm labels in the window — exploration off"}
    out: dict = {"armed": True, "n": len(armed), "buckets": {}}
    by: dict = {}
    for r in armed:
        by.setdefault(r["_bucket"], {}).setdefault(r["arm"], []).append(r)
    for b in sorted(by):
        arms = {}
        for arm, rs in sorted(by[b].items()):
            ans = [max(0, (r["generated"] or 0) - (r["rtok"] or 0))
                   for r in rs if r["generated"] is not None]
            arms[arm] = {
                "n": len(rs),
                "bound_rate": _r2(sum(1 for r in rs
                                      if r["_outcome"] == "bound") / len(rs)),
                "cap_hit_rate": _r2(sum(1 for r in rs
                                        if r["cap_hit"]) / len(rs)),
                "rtok_med": _r2(_pct([r["rtok"] for r in rs
                                      if r["rtok"] is not None], 0.5)),
                "answer_tok_med": _r2(_pct(ans, 0.5)),
                "empty_answer_rate": (
                    _r2(sum(1 for a in ans if a <= cfg["empty_answer_tok"])
                        / len(ans)) if ans else None),
            }
        out["buckets"][str(b)] = arms
    return out


# ── ledger + cursor io ──────────────────────────────────────────────────────
def read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def atomic_write_json(path: str, obj: dict) -> None:
    """tmp + rename in the SAME directory — the consumer's mtime-cached read
    never observes a partial file, and every real write moves the mtime."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_kmap(ledger: dict, cfg: dict) -> dict:
    """String-keyed k map. Mirrors the consumer's `_normalise` validation."""
    out = {}
    for key, val in (ledger.get("bucket") or {}).items():
        if not isinstance(key, str):
            continue
        try:
            b = int(key.split("|", 1)[0])
        except (TypeError, ValueError):
            continue
        if 1 <= b <= MAX_BUCKET:
            out[key] = _clamp_k(val, cfg)
    return out


def run_pass(cfg: dict, sink: str, ledger_path: str, cursor_path: str,
             dry_run: bool = False) -> dict:
    """One idempotent pass. Returns the ledger it wrote (or would write)."""
    counts: dict = {}
    ledger = read_json(ledger_path, {}) or {}
    cursor = read_json(cursor_path, {}) or {}
    if int(cursor.get("schema", CURSOR_SCHEMA)) != CURSOR_SCHEMA:
        cursor = {}
    kprev = load_kmap(ledger, cfg)

    schema = cfg.get("key_schema", KEY_SCHEMA_STEPS)
    if schema not in KEY_SCHEMAS:
        raise SystemExit(f"[pn162] unknown --key-schema {schema!r}; "
                         f"expected one of {KEY_SCHEMAS}")

    new_rows, offsets = read_sink_since(
        sink, cursor.get("offsets") or {}, cfg, counts)
    for r in new_rows:
        r["_outcome"] = classify(r, cfg)
        if r["_outcome"].startswith("drop:"):
            counts[r["_outcome"]] = counts.get(r["_outcome"], 0) + 1
            r["_bucket"] = None
            r["_keys"] = None
        else:
            r["_bucket"] = invert_steps(r["budget_grant"], kprev, cfg,
                                        r.get("prompt_tok"))
            r["_keys"] = bucket_keys(r["_bucket"], r.get("prompt_tok"), schema)
    scored = [r for r in new_rows if r["_bucket"] is not None]
    counts["new_rows"] = len(new_rows)
    counts["new_scored"] = len(scored)

    # Rolling window: last N SCORED rows, carried in the cursor so a pass never
    # rereads consumed sink bytes. Trimmed to the fields telemetry needs.
    keep = ("ts", "route", "score", "rtok", "generated", "cap_hit",
            "budget_grant", "prompt_tok", "arm", "_outcome", "_bucket", "_keys")
    window = list(cursor.get("window") or []) + [
        {k: r.get(k) for k in keep} for r in scored]
    window = window[-cfg["window"]:]

    # Occupancy is judged on the WHOLE window, not just this pass's rows.
    allowed = writable_keys(window, cfg)
    kmap, per = update_buckets(scored, kprev, cfg, allowed)
    kmap = {k: v for k, v in kmap.items() if k in allowed or k in kprev}

    oracle = announcement_oracle(window, cfg)
    tel = telemetry(window, kmap, kprev, per, oracle, cfg)
    ledger_out = {
        "schema": LEDGER_SCHEMA,
        "updated_ts": time.time(),
        "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Versioned so the table can grow into composite shape keys without a
        # rewrite: the consumer walks `bucket_keys(...)` most-specific-first
        # and always has the marginal steps bucket to fall back to.
        "key_schema": schema,
        "key_bands": {"ptok": list(DEFAULT_PTOK_BANDS)},
        "bucket": {key: round(k, 4) for key, k in sorted(kmap.items())},
        "cells": {key: {"n": sum(1 for r in window
                                 if _row_key(r, allowed) == key)}
                  for key in sorted(kmap) if "|" in key},
        # Nothing populates this. The sink carries no prompt hash; see
        # PN162-BUDGET-CAL.md "EXACT". Written empty, never faked.
        "exact": {},
        "exact_note": ("not implemented: the PN119 sink carries no prompt "
                       "hash, so no entry can be keyed the consumer would "
                       "recognise. Consumer read path is live + tested."),
        "params": {k: cfg[k] for k in sorted(cfg)},
        "window": {
            "n": len(window),
            "bound": sum(1 for r in window if r["_outcome"] == "bound"),
            "slack": sum(1 for r in window if r["_outcome"] == "slack"),
            "ok": sum(1 for r in window if r["_outcome"] == "ok"),
            "new_scored": len(scored),
            "counts": counts,
        },
        "telemetry": tel,
        "oracle": oracle,
        "explore": explore_report(window, cfg),
    }
    if not dry_run:
        atomic_write_json(ledger_path, ledger_out)
        atomic_write_json(cursor_path, {
            "schema": CURSOR_SCHEMA,
            "updated_ts": time.time(),
            "offsets": offsets,
            "window": window,
        })
    return ledger_out


def build_cfg(args) -> dict:
    cfg = dict(DEFAULTS)
    for k in cfg:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="PN162 ledger updater — closed-loop thinking-budget "
                    "calibration from PN119 sink budget outcomes.")
    p.add_argument("--sink", default=SINK_DEFAULT)
    p.add_argument("--ledger", default=LEDGER_DEFAULT)
    p.add_argument("--cursor", default=CURSOR_DEFAULT)
    p.add_argument("--dry-run", action="store_true",
                   help="compute and print; write nothing")
    p.add_argument("--json", action="store_true",
                   help="print the whole ledger instead of a summary")
    for name, val in sorted(DEFAULTS.items()):
        p.add_argument(f"--{name.replace('_', '-')}", dest=name,
                       type=type(val), default=None,
                       help=f"(default {val})")
    args = p.parse_args(argv)
    cfg = build_cfg(args)
    led = run_pass(cfg, args.sink, args.ledger, args.cursor, args.dry_run)
    if args.json:
        print(json.dumps(led, indent=1, sort_keys=True))
        return 0
    w = led["window"]
    print(f"[pn162] {'DRY-RUN ' if args.dry_run else ''}"
          f"new={w['new_scored']} window={w['n']} "
          f"(bound={w['bound']} slack={w['slack']} ok={w['ok']}) "
          f"-> {args.ledger}")
    print(f"[pn162] key_schema={led['key_schema']}"
          + (f" cells={len(led['cells'])}" if led["cells"] else ""))
    for b, k in sorted(led["bucket"].items(),
                       key=lambda kv: (int(kv[0].split('|')[0]), kv[0])):
        if "|" in b:
            continue                       # composite cells print below
        t = led["telemetry"].get(b, {})
        if k == 1.0 and not t.get("n"):
            continue
        print(f"  bucket {b:>3}  k={k:<6} n={t.get('n', 0):<4} "
              f"bound={t.get('bound', 0):<4} slack={t.get('slack', 0):<4} "
              f"rtok_med={t.get('rtok_med')} "
              f"steps_real_med={t.get('steps_real_med')} "
              f"adherence={t.get('anchor_adherence')} "
              f"bias={t.get('announce_bias_est')}")
    for key, meta in sorted(led["cells"].items()):
        print(f"    cell {key:>8}  k={led['bucket'][key]:<6} n={meta['n']}")
    drops = {k: v for k, v in w["counts"].items() if k.startswith("drop:")}
    if drops:
        print(f"  drops: {drops}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
