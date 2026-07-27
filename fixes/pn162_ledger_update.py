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

PER-ROW EVIDENCE, NOT PER-PASS (2026-07-27 rewrite)
---------------------------------------------------
The update is EXPONENTIAL IN THE ROW COUNT, so one pass's k move depends on how
much evidence that pass carried and on nothing else:

    k[key] *= (1 + BETA)  ** n_bound_rows      (BETA  default 0.007)
    k[key] *= (1 - GAMMA) ** n_slack_rows      (GAMMA default 0.004)
    k[key] unchanged on OK
    floor 1.0 (= NEUTRAL), ceiling K_MAX, and a per-pass |k'/k| <= MAX_STEP rail

The rule it replaced was per-PASS multiplicative: ANY pass containing a bound
row bumped the whole bucket by 1.15. At the 60 s timer cadence a pass often
carries one or two rows, so ONE outlier request moved a bucket 15% — and two
manual passes drove live buckets to 2.25-3.0 on starved-era evidence. Counting
rows instead of passes makes the pass boundary irrelevant: 20 bound rows move k
+15% whether they arrive in one pass or twenty, and a lone outlier moves it
+0.7%, which the majority erases with three slack rows.

Counts, not magnitudes, is the deliberate choice. The goal is to fit the ~90%
majority of (usually repeating) requests, so each request gets exactly one vote;
a single request that needed 8x its grant cannot outvote the 99 that did not.
See PN162-BUDGET-CAL.md §2.1 for the calibration table (rows -> % move) and the
1000-row-window outlier bound.

Causal grounding (banked 2026-07-26): force-closed items gain +9.6pt when given
more; natural stops lose nothing. Bumping a bound bucket is the paying side of
the trade; an unused grant costs nothing (cost = E[min(need, grant)]), so decay
exists only to hand back bumps this loop granted. It stops at 1.0 — identity
with PN100's validated flat grid — because a sub-neutral k starves lean items
(-6pt on a live full-100). Reclaiming below the flat grid is out of scope, and
the k_min clamp IS the "decay only while k > 1.0" rule.

THE ERA-CONSISTENCY FILTER
--------------------------
A sink row is evidence about the grant IT WAS SERVED, not about the grant the
bucket hands out now. Rows generated minutes ago under a smaller k kept voting
as if current, so a bucket that had already been raised was raised again on the
starvation its own raise had fixed — the observed 2.25-3.0 runaway.

The sink row carries the granted budget, and `invert_steps` already inverts a
grant THROUGH the live key map, so the row's era is recoverable exactly:

    era_grant = row.budget_grant
    now_grant = pn100_grant(bucket, k_current[bucket])

    BOUND row with era_grant < now_grant  -> STALE: the current k already
                                             answered this starvation. Weight 0.
    SLACK row with era_grant > now_grant  -> STALE (symmetric): the current k
                                             already handed that fat back.

Stale rows still count in telemetry and in `n_evidence_total` — they are real
observations, just not observations about the CURRENT cap. Rows served at
exactly the current grant (the overwhelming majority, since the ledger only
moves once a minute) always vote.

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
a self-reinforcing drift. `invert_candidates()` therefore inverts THROUGH the
current bucket map: it collects every b whose calibrated grant equals the
observed one, and falls back to the naive quotient only when no b matches
(a ledger changed mid-window — the only path `era_stale` can fire on). With k
all 1.0 it is exactly the quotient.

`round100` is many-to-one, so two buckets can genuinely share a grant. Control
does NOT guess between them: an ambiguous row splits its single vote evenly
across the candidates. Winner-take-all LOCKS the loop (see `invert_candidates`);
`invert_steps()` still returns the single best candidate, for telemetry, key
derivation and the era filter.

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
    # PER-ROW evidence steps (2026-07-27). See the module docstring and
    # PN162-BUDGET-CAL.md §2.1. beta is calibrated so ~20 consistent bound rows
    # reproduce the retired per-pass 1.15 step (1.007**20 = 1.1498) and one
    # outlier row moves k by +0.7%; gamma is deliberately slower so an
    # over-granted bucket relaxes to neutral over hundreds of slack rows.
    beta=0.007,
    gamma=0.004,
    slack_frac=0.40,
    # 2026-07-27: floor raised 0.7 -> 1.0 — slack is free on this stack (grant
    # invisible, cost=E[min(need,grant)]); sub-neutral k starved lean items
    # -6pt on the all-armed full-100. Decay now converges to neutral, not below.
    k_min=1.0,
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


def invert_candidates(grant: int, kmap: dict, cfg: dict,
                      ptok=None) -> list:
    """Every bucket whose CURRENT calibrated grant is `grant`, with vote weights.

    Inverts THROUGH the live key map so PN162's own multiplier cannot make the
    updater credit the wrong bucket. Returns `[(bucket, keys, weight)]`, best
    candidate first, weights summing to 1.

    AMBIGUITY IS REAL AND MUST BE SPLIT, NOT GUESSED. `round100` is many-to-one,
    so two buckets can genuinely share a grant — bucket 8 at k=1.258 (raw 2616)
    and bucket 10 at k=1.0 (raw 2600) both serve 2600, and NOTHING in the sink
    row distinguishes them. Awarding the whole vote to one of them is what
    locks the loop: under the per-ROW step sizes a moving bucket now visits
    every 100-token grant on its way (instead of stepping over most of them, as
    the retired 15%-per-pass rule did by luck), so it meets these coincidences
    constantly. Each winner-take-all decision hands ALL of a bucket's evidence
    to a neighbour that does not size its requests; the real bucket's k freezes
    and never reaches its need. Both directions were reproduced: a permanent
    10/100 force-close climbing, and an over-granted bucket stuck at k=1.258
    decaying.

    So an ambiguous row splits its single vote evenly across the candidates.
    That is honest (the posterior over buckets really is flat given only the
    grant), it keeps every candidate moving, and it composes exactly with the
    exponential rule, which takes fractional exponents without a special case.
    The cost is a documented smear: a bucket with no traffic of its own can
    pick up a fraction of a neighbour's votes. Its own traffic, when any
    arrives, decays it back.

    No candidate at all -> the naive quotient, weight 1. That is the ONLY path
    on which `era_stale` can fire; see it for why.
    """
    tps = float(cfg["tok_per_step"])
    naive = max(1, int(round(grant / tps)))
    schema = cfg.get("key_schema", KEY_SCHEMA_STEPS)
    cands = []
    for b in range(1, MAX_STEPS_SEARCH + 1):
        keys = bucket_keys(b, ptok, schema)
        k = lookup_k(kmap, keys, cfg)
        if pn100_grant(b, k, cfg) != grant:
            continue
        raw = b * tps * (cfg["highstep_mult"]
                         if (cfg["highstep_mult"] > 1.0
                             and b >= cfg["highstep_min"]) else 1.0) * k
        cands.append((abs(raw - grant), abs(b - naive), b, keys))
    if not cands:
        b = bucket_of(naive)
        return [(b, bucket_keys(b, ptok, schema), 1.0)]
    cands.sort()
    w = 1.0 / len(cands)
    return [(b, keys, w) for _d, _n, b, keys in cands]


def invert_steps(grant: int, kmap: dict, cfg: dict, ptok=None) -> int:
    """The single best bucket for `grant` — telemetry, keys and the era filter.

    Control does NOT go through this: an ambiguous grant splits its vote across
    `invert_candidates`. Ordering is nearest UNROUNDED calibrated grant, then
    nearest naive quotient, then lowest bucket.
    """
    return bucket_of(invert_candidates(grant, kmap, cfg, ptok)[0][0])


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


def _pick_key(keys: list, bucket, allowed: set) -> str:
    for key in (keys or []):
        if key in allowed:
            return key
    return str(bucket or 1)


def _row_key(row: dict, allowed: set) -> str:
    return _pick_key(row.get("_keys"), row.get("_bucket"), allowed)


def era_stale(row: dict, kmap: dict, cfg: dict) -> bool:
    """Is this row evidence about a grant the CURRENT k no longer hands out?

    The sink row carries `budget_grant` — the number the engine actually served
    it — and `pn100_grant` reproduces what the bucket would grant under the
    ledger's current k. A bound row served LESS than today's grant is telling
    us about a starvation today's k has already addressed; a slack row served
    MORE than today's grant is telling us about fat today's k has already
    handed back. Both are true observations about a dead regime, so they are
    weight 0 for control and still counted everywhere else.

    Equal grants — the normal case, since the ledger moves at most once a
    minute — are never stale. OK rows have no weight to zero out.

    MECHANICALLY this fires only on `invert_candidates`' naive-quotient
    fallback, and that is exactly right: an exact candidate is by construction
    a bucket whose CURRENT grant equals the row's, i.e. a row still on the live
    grid. A row served under an old k regime whose grant has since fallen off
    the grid entirely matches nothing, gets credited by quotient to a bucket
    that never sized it, and is precisely the "starved-era outcome voting as if
    current" the 2026-07-27 runaway was made of. Rows from an old regime that
    happen to ALIAS onto some current bucket's grant are not detectable — no
    field in the sink dates a grant against a ledger revision.
    """
    b = row.get("_bucket")
    outcome = row.get("_outcome")
    grant = row.get("budget_grant")
    if not b or not grant or outcome not in ("bound", "slack"):
        return False
    keys = row.get("_keys") or bucket_keys(b, row.get("prompt_tok"),
                                           cfg.get("key_schema",
                                                   KEY_SCHEMA_STEPS))
    now = pn100_grant(int(b), lookup_k(kmap, keys, cfg), cfg)
    return grant < now if outcome == "bound" else grant > now


_ZERO = ("bound", "slack", "ok", "bound_eff", "slack_eff",
         "bound_stale", "slack_stale")


def update_buckets(rows: list, kmap: dict, cfg: dict,
                   allowed: set | None = None) -> tuple[dict, dict]:
    """(new kmap, per-key outcome counts) from NEW rows only.

    Per-ROW exponential: the pass's whole move for one key is
    `(1+beta)**bound_eff * (1-gamma)**slack_eff`, where `_eff` excludes the
    era-stale rows (`row["_stale"]`, set by `run_pass`). A pass with no rows
    for a key does nothing to it — the cadence of the timer is not evidence.

    `max_step` survives as an absolute safety rail, not as the control law:
    at the shipped beta it takes 58 bound rows in ONE pass to reach 1.5, so it
    now binds only on a genuine flood. Each key's realised step lands in
    `per[key]["step"]` and is written to the ledger as the audit trail.
    """
    out = {key: _clamp_k(k, cfg) for key, k in kmap.items()}
    per: dict = {}
    for r in rows:
        if r.get("_outcome") not in ("bound", "slack", "ok"):
            continue
        stale = bool(r.get("_stale"))
        # One row is ONE vote, split across the buckets its grant is ambiguous
        # between (`invert_candidates`). Unambiguous rows — the overwhelming
        # majority — are a single (bucket, keys, 1.0).
        alias = r.get("_alias") or [(r["_bucket"], r.get("_keys"), 1.0)]
        for b, keys, w in alias:
            key = _pick_key(keys, b, allowed) if allowed is not None else str(b)
            marg = str(b)
            # Credit the APPLIED key and, when they differ, the marginal bucket
            # too. Without the second credit a marginal bucket stops seeing any
            # traffic the moment its composite cells open, and it is precisely
            # the fallback that new and thin cells land on. The cost is a
            # confound — the marginal then learns partly from rows that were
            # sized by a composite k — which is one reason a composite schema
            # needs its own screened boot rather than a flag flip.
            for kk in ({key, marg} if key != marg else {key}):
                c = per.setdefault(kk, {z: 0.0 for z in _ZERO})
                c[r["_outcome"]] += w
                if r["_outcome"] in ("bound", "slack"):
                    c[r["_outcome"] + ("_stale" if stale else "_eff")] += w
    beta, gamma = float(cfg["beta"]), float(cfg["gamma"])
    for key, c in per.items():
        k0 = _clamp_k(out.get(key, 1.0), cfg)
        k = k0 * ((1.0 + beta) ** c["bound_eff"]) \
               * ((1.0 - gamma) ** c["slack_eff"])
        k = max(k0 / cfg["max_step"], min(k0 * cfg["max_step"], k))
        out[key] = _clamp_k(k, cfg)
        # The audit trail the ledger publishes: the step ACTUALLY applied,
        # after the rail and both clamps — not the step the counts asked for.
        c["step"] = round(out[key] / k0, 6) if k0 else 1.0
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


def roll_audit(prev: dict, per: dict, kmap: dict, kprev: dict,
               cfg: dict) -> dict:
    """The per-key audit trail, carried forward across passes.

    `n_evidence_total` is CUMULATIVE over the calibrator's whole life, not over
    the rolling window: it is the denominator that says how much a bucket's k
    has actually been paid for. `step` is what this pass applied. Keys that saw
    no rows this pass keep their total and report step 1.0, so a bucket sitting
    still is visibly sitting still rather than absent.
    """
    out: dict = {}
    for key in set(prev) | set(per) | set(kmap):
        p = prev.get(key) or {}
        c = per.get(key) or {}
        try:
            total = float(p.get("n_evidence_total") or 0)
        except (TypeError, ValueError):
            total = 0.0
        # Vote counts are fractional when a grant was ambiguous (one row, one
        # vote, split across candidates), so they round for publication.
        n_pass = c.get("bound", 0) + c.get("slack", 0) + c.get("ok", 0)
        out[key] = {
            "n_evidence_total": round(total + n_pass, 3),
            "n_evidence_pass": round(n_pass, 3),
            "bound_eff": round(c.get("bound_eff", 0), 3),
            "slack_eff": round(c.get("slack_eff", 0), 3),
            "bound_stale": round(c.get("bound_stale", 0), 3),
            "slack_stale": round(c.get("slack_stale", 0), 3),
            "step": round(float(c.get("step", 1.0)), 6),
            "k_prev": round(_clamp_k(kprev.get(key, 1.0), cfg), 4),
            "k": round(_clamp_k(kmap.get(key, 1.0), cfg), 4),
        }
    return out


def telemetry(window: list, kmap: dict, kprev: dict, per: dict,
              oracle: dict, cfg: dict, audit: dict | None = None) -> dict:
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
        au = (audit or {}).get(str(b), {})
        out[str(b)] = {
            "n": len(rs),
            "bound": sum(1 for r in rs if r["_outcome"] == "bound"),
            "slack": sum(1 for r in rs if r["_outcome"] == "slack"),
            "ok": sum(1 for r in rs if r["_outcome"] == "ok"),
            "new_bound": round(pc.get("bound", 0), 3),
            "new_slack": round(pc.get("slack", 0), 3),
            "new_ok": round(pc.get("ok", 0), 3),
            # era-consistency filter: how many of this pass's rows actually
            # voted, and how many were evidence about a dead k regime.
            "new_bound_stale": round(pc.get("bound_stale", 0), 3),
            "new_slack_stale": round(pc.get("slack_stale", 0), 3),
            # cumulative evidence behind this bucket's k, and the step this
            # pass actually applied (post-rail, post-clamp). Audit trail.
            "n_evidence_total": au.get("n_evidence_total", 0),
            "step_applied": au.get("step", 1.0),
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
            r["_alias"] = invert_candidates(r["budget_grant"], kprev, cfg,
                                            r.get("prompt_tok"))
            r["_bucket"] = r["_alias"][0][0]
            r["_keys"] = r["_alias"][0][1]
            # Era-consistency: judged against the k the ledger carries NOW
            # (kprev — the same map the inversion used), never against the k
            # this pass is about to write.
            r["_stale"] = era_stale(r, kprev, cfg)
    scored = [r for r in new_rows if r["_bucket"] is not None]
    counts["new_rows"] = len(new_rows)
    counts["new_scored"] = len(scored)
    counts["era_stale"] = sum(1 for r in scored if r.get("_stale"))

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

    audit = roll_audit(ledger.get("audit") or {}, per, kmap, kprev, cfg)
    audit = {k: v for k, v in audit.items() if k in kmap or k in kprev}
    oracle = announcement_oracle(window, cfg)
    tel = telemetry(window, kmap, kprev, per, oracle, cfg, audit)
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
        # Per-key audit trail for the per-ROW update: cumulative evidence
        # behind each k, this pass's effective vs era-stale row counts, and the
        # step actually applied. This is what makes a k move explainable
        # without re-reading the sink.
        "audit": audit,
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


ENV_PREFIX = "PN162_"


def env_overrides(env=None) -> dict:
    """`PN162_<KNOB>` for every DEFAULTS key, e.g. PN162_BETA / PN162_GAMMA.

    The timer runs the updater with no arguments, so env is the only way to
    retune it without editing the unit — and a knob that cannot be set where it
    runs is a knob that does not exist. Precedence is CLI > env > DEFAULTS.
    An unparseable value is IGNORED (with a stderr note) rather than fatal: the
    learning pass must not stop because someone typo'd a float.
    """
    env = os.environ if env is None else env
    out = {}
    for name, val in DEFAULTS.items():
        raw = env.get(ENV_PREFIX + name.upper())
        if raw is None or raw == "":
            continue
        try:
            out[name] = type(val)(raw)
        except (TypeError, ValueError):
            print(f"[pn162] ignoring {ENV_PREFIX}{name.upper()}={raw!r} "
                  f"(expected {type(val).__name__})", file=sys.stderr)
    return out


def build_cfg(args, env=None) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(env_overrides(env))
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
                       help=f"(default {val}; env {ENV_PREFIX}{name.upper()})")
    args = p.parse_args(argv)
    cfg = build_cfg(args)
    led = run_pass(cfg, args.sink, args.ledger, args.cursor, args.dry_run)
    if args.json:
        print(json.dumps(led, indent=1, sort_keys=True))
        return 0
    w = led["window"]
    print(f"[pn162] {'DRY-RUN ' if args.dry_run else ''}"
          f"new={w['new_scored']} era_stale={w['counts'].get('era_stale', 0)} "
          f"window={w['n']} "
          f"(bound={w['bound']} slack={w['slack']} ok={w['ok']}) "
          f"beta={cfg['beta']} gamma={cfg['gamma']} -> {args.ledger}")
    print(f"[pn162] key_schema={led['key_schema']}"
          + (f" cells={len(led['cells'])}" if led["cells"] else ""))
    for b, k in sorted(led["bucket"].items(),
                       key=lambda kv: (int(kv[0].split('|')[0]), kv[0])):
        if "|" in b:
            continue                       # composite cells print below
        t = led["telemetry"].get(b, {})
        if k == 1.0 and not t.get("n"):
            continue
        print(f"  bucket {b:>3}  k={k:<6} step={t.get('step_applied', 1.0):<8} "
              f"n={t.get('n', 0):<4} ev={t.get('n_evidence_total', 0):<6} "
              f"bound={t.get('bound', 0):<4} slack={t.get('slack', 0):<4} "
              f"stale={t.get('new_bound_stale', 0)}/{t.get('new_slack_stale', 0)} "
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
