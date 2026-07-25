"""PN119 lens-router sidecar (installed as vllm/_genesis_pn119.py at boot).

Per-request deep/mass thinking router on the serving model's OWN prefill
hidden states — no extra model, no extra VRAM. Basis: needfit lens probe
(seed LOO AUC 0.9459 last-only; PN119-BUILD-PACK.md is the spec).

Capture: vLLM's native EAGLE3 aux-hidden-state mechanism
(model.set_aux_hidden_state_layers((42, 47, 51)) — cudagraph/compile-safe,
no python hooks). Feature vector per request (order = the npz `blocks` key,
[3, 5120] row-major):
  L42-last, L47-last, L51-last
Score: xs = (x-mu)/sd; p = xs @ Vt10.T; score = concat(p,[1]) @ w.

LAST-ONLY (2026-07-25). The mean-pooled half of the feature vector is GONE.
The evaluation that decided it is in probe-v2/pn119-probe-lastonly-report.json:
seed LOO AUC 0.9459 vs the incumbent recipe's 0.9358 (shuffled p95 floor
0.7174), best of every configuration measured. The sink-side deltas are NOT
individually significant — on the full 79-row OOD window last-only wins rho
(0.6231 vs 0.5845) and AUC@2500 (0.8601 vs 0.8304) and LOSES AUC@2000
(0.8983 vs 0.9022); on the 62-row replication window it wins all three. The
case for shipping it is STRUCTURAL, not accuracy:
  * FEAT_DIM 30720 -> 15360. Sink bytes halve, 61,440 -> 30,720 per request.
  * The target stays BINARY (rtok >= 2000), so PN119_TDEEP=0.495 remains
    valid (the npz's own suggestion is 0.4897, deep frac 0.35).
  * 100% of requests become scoreable. vLLM ALWAYS recomputes the last prompt
    token even on a total prefix-cache hit — verified in this image at
    v1/core/kv_cache_manager.py: `max_cache_hit_length = request.num_tokens - 1`
    with the comment "When all tokens hit the cache, we must recompute the
    last token to obtain logits". The mean-pool was the ONLY reason a partial
    prefill could not be scored, so the exact-reconstruction memo
    (PN119_PREFIX_MEMO / _MEMO_UNIT), the accumulator, the partial_prefill
    refusal and the 15-blocking-sync memo D2H are all deleted rather than
    maintained. What survives is the ONE case that is still real: a request
    whose last prompt token this process never forwarded at all (the router
    attached mid-flight) — `prefill_not_observed`, still an explicit
    counted fallback, never a silent pass-through.

Modes (PN119_MODE): shadow (default) = log + sink only, act on nothing;
enforce = additionally publish the decision to the ROUTES/SCORES registries
(the route-action consumer wiring is a follow-up — v1 enforce publishes,
never mutates requests itself).

WHERE THE CONSUMER CANNOT LIVE (proved 2026-07-25, no boot needed —
fixes/test_h119_route_consumer_timing.py asserts all four points)
--------------------------------------------------------------------
NOT in _genesis/middleware/auto_budget.py (PN100), and NOT anywhere else on
the frontend request path. Three independent blockers, each fatal on its own:
  1. TEMPORAL. PN100's hook is inserted at the TOP of create_chat_completion
     (patch_pn100_auto_thinking_budget.py anchors on the Genesis PN16 block,
     which sits immediately above upstream's "# Streaming response").
     engine_client.generate() — and therefore prefill, and therefore the
     observe() call that writes ROUTES — is ~4000 chars further down. The
     budget is decided at t0; the route exists at t3.
  2. IDENTITY. `request_id` is minted ~17 lines BELOW the hook site. At
     budget-decision time there is no key to index ROUTES with.
  3. ADDRESS SPACE. AsyncLLM calls EngineCoreClient.make_async_mp_client()
     unconditionally (not gated on VLLM_ENABLE_V1_MULTIPROCESSING), so the
     API server that runs PN100 is a DIFFERENT PROCESS from the EngineCore /
     worker that owns this module's ROUTES dict. A correctly-timed,
     correctly-keyed read would still see {} forever.
Failure shape if wired there anyway: route_for() returns _FALLBACK_ROUTE for
100% of requests — i.e. everything routes "deep", the champion cost with none
of the saving. Strictly worse than leaving the router in shadow.

WHERE IT DOES LIVE (BUILT 2026-07-25 — see the H119 ENFORCE ROUTE CONSUMER
section at the bottom of this file). Worker-side, same process, same req_id
namespace: vllm/v1/sample/thinking_budget_state.py keeps a MUTABLE per-request
"thinking_token_budget" in _state[batch_index], re-read by update_state() on
every decode step, and gpu_input_batch.py carries req_ids / req_id_to_index.
The ordering is TIGHTER than first assumed: within ONE engine step,
_update_states -> refresh_metadata -> sync_batch runs first, then
execute_model's forward and the observe() that writes ROUTES, and only THEN
sample_tokens -> Sampler.forward -> holder.update_state(). So the route is on
record before the sampler that emits the request's FIRST token, not merely
before the second one.
LIMIT: that site reaches the budget CAP only. The deep/lean treatments also
differ by PN102 banner (v5-class vs v3-class), which is rendered into the
PROMPT before prefill. The route is derived FROM that prefill, so banner
selection from the route is circular and cannot be done in one pass — it needs
a separate cheap prefill-only probe request, or nothing. The shipped consumer
is therefore the budget half of deep/lean, NOT the full treatment: do not read
it as reproducing the 25-to-v5 / 75-to-v3 result.

THINKING-ON GATE (2026-07-25) — the router's contract population
-----------------------------------------------------------------
The router can only move a request that HAS a thinking budget to move. A
thinking-OFF prompt (the Qwen3.6 template pre-closes `<think></think>`)
spends zero reasoning tokens no matter what the probe says, so scoring it
buys nothing and costs a 30,720-byte sink row and a matvec. Worse, it
poisoned every rate: measured on one live window, 30 of 61 scored rows were
thinking-OFF, so "deep fraction", "fallback rate" and every percentile were
computed over a population that is roughly HALF requests the router cannot
affect.

`_prompt_thinking(state)` now runs at the TOP of `_finalize`:
  * True  -> score, sink row, `"routable": true`. The only rows in the
             deep-fraction denominator and the only rows the refit may learn
             from.
  * False -> publish ROUTE_LEAN with reason "thinking_off", SKIP the matvec
             and the feature-row write, `_bump("skip_thinking_off")`, meta
             line with `"routable": false`.
  * None  -> raw/completion prompt, no marker either way. Still scored
             (the treatment might matter) but tagged `"routable": null`,
             counted as `scored_unknown`, and kept OUT of every rate
             denominator.

MARKER IDENTITY (boot assertion). This module used to scan for the SINGLE
ids PN119_THINK_START_ID / PN119_THINK_END_ID while the holder that actually
forces `</think>` uses `reasoning_config.reasoning_start_token_ids` /
`…end_token_ids`, which are SEQUENCES from `tokenizer.encode()`. A silent
divergence mislabels the whole sink and nothing downstream would notice, so
`_resolve_think_markers()` compares them at init and ADOPTS the holder's
sequences on disagreement (ERROR log + `think_marker_divergence` counter +
the THINK_MARKERS_DIVERGED alarm). The scan itself is now a subsequence
search, identical in shape to the holder's `_find_last_sequence_index`.

v2 self-training sink (PN119-BUILD-PACK §v2): every finalized prefill
appends (bf16 features row, meta line w/ score+mode+explore) to PN119_SINK;
request finish appends a label line (generated, thinking flag, true rtok =
tokens before </think>, cap_hit, censored, budget_grant) keyed by req_id.
Shadow traffic is uncensored → doubles as the v2 training bootstrap.

CENSORING IS THE BINDING CONSTRAINT (BUG-139, 2026-07-25)
---------------------------------------------------------
Measured over all 79 thinking-finish rows in the sink: 43 of them (54%) have
rtok exactly equal to a PN100 100-token-grid grant MINUS FIVE — 1295 x20
(grant 1300), 3095 x12, 2095 x7, 3895 x4. That is the signature of the
holder forcing `</think>` when the budget ran out, i.e. a LOWER BOUND on the
request's need. Only FOUR rows logged `cap_hit=True`, because `_label_fields`
derived cap_hit from `max_tokens` alone and never looked at the thinking
budget at all. So over half the training corpus was a truncation recorded as
a natural stop, which caps rho and AUC for ANY probe fit on it (two
independent evaluations hit that ceiling and named it), and — once enforce
acts — makes lean an ABSORBING STATE: a lean-routed row that truncates is
indistinguishable from one that genuinely needed little, so the router
reinforces its own decision forever.
The fix is to record it, not to guess it. Each finish line now carries:
  censored     = thinking and budget and rtok >= budget - SLACK,
                 SLACK = len(think_end_ids) + 8 (the measured offset is
                 exactly 5; the slack absorbs a multi-token end marker and
                 the spec-decode lookahead without reaching a natural stop).
  budget_grant = the EFFECTIVE thinking budget — H119's own number when the
                 consumer rewrote it, otherwise sampling_params
                 .thinking_token_budget. On BOTH the score and finish lines,
                 because the score line is written before H119 resolves and
                 the pair is what makes the rewrite auditable.
  budget_source= caller | pn100 | h119 | none, from the PN100 ownership
                 stamp plus H119's own application record.
`cap_hit` keeps its old meaning for compatibility with sinks already on disk.
The sink is BUFFERED IN RAM and drained by a daemon thread, so the request
path holds no disk I/O at all (PN119_SINK_BUF_ROWS / _BUF_SECS / _BUF_MAX;
0 rows = legacy synchronous mode). Clean shutdown always drains via atexit;
a hard kill loses at most one buffer. On-disk bytes are unchanged.

v2 loop (fixes/refit_pn119_probe.py + pn119-refit.timer): refits the probe
from the sink on CPU and ATOMICALLY swaps the npz (pn119_atomic.py); this
router hot-reloads it on a CONTENT HASH change — PN119_RELOAD_S throttle, no
restart. The signature used to be (mtime_ns, size), which is blind in
exactly the case that matters: the tap-trained and the offline-trained
probes are byte-IDENTICAL in size, so `cp -p`, `rsync -a` and a snapshot
rollback all leave the pair unchanged and the router keeps serving the old
weights while every operator believes the swap landed. Hashing 1.4 MB once
per PN119_RELOAD_S (60 s) is not a cost worth a class of invisible failure.
PN119_EXPLORE=<frac> flags a deterministic ~frac of requests for generous
caps in enforce mode so labels stay uncensored (EXPLORE set).

LATENCY HYGIENE (2026-07-25). `float(torch.dot(...))` is an `.item()` in
disguise — a full device sync in the middle of a step, on a stack vLLM went
to some trouble to keep sync-free. Under PN119_ASYNC_SCORE=1 the score (and,
when the sink is on, the bf16 feature row) is copied into PINNED host memory
with `non_blocking=True`, a `cuda.Event` is recorded, and the readback is
resolved at the last responsible moment — inside `h119_resolve_routes`,
which runs LATER IN THE SAME STEP (sampler time) than observe() does
(post-forward). Semantics are preserved exactly: the route is still on
record before the sampler that emits the request's first token. Paired with
H119_ROUTE_GRACE_TOKENS (default 8): a provisional row is allowed a few
output tokens of slack before it is committed to the fallback, which against
an 800/10240 cap is arithmetically identical and gives the async path room.
Any failure to resolve asynchronously falls back to the blocking readback
and bumps `sync_fallback_used`. Two further step-level cuts: steps with no
`scheduled_new_reqs` and an empty accumulator early-out entirely (with a
full scan every PN119_FULLSCAN_EVERY=256 steps so a stuck request is still
found), and the two per-request `logger.info` calls are DEBUG — only the
periodic rollup stays at INFO.

HEALTH SURFACE (PN119_HEALTH, added 2026-07-25 — read this before touching it)
------------------------------------------------------------------------------
Every failure this module has actually had was SILENT. Five boots on
2026-07-25 ran degenerate — two pinned at 100% deep, three at 0% deep — and
none of them was noticed for hours; they were found by reading the sink
offline the next evening. The consumer's day-long no-op (it deferred to PN100
on 100% of requests) presented as a perfectly healthy boot with every counter
incrementing. STATS was already a good counter dict; the gap was that nothing
outside this process could read it.

So the router now publishes `health.json` (PN119_HEALTH, default
<PN119_SINK>/health.json) containing the counters, derived rates, probe and
consumer identity, and — the actual product — a list of ALARMS with a
designed trigger condition each (see `_ALARM_*` constants and `pn119_alarms`).
Three properties are load-bearing:
  * It is written ATOMICALLY (temp + os.replace), so a reader never sees a
    torn file and never needs a lock.
  * It is written FROM THE EXISTING SINK FLUSHER THREAD, which already wakes
    every PN119_SINK_BUF_SECS. The request path gains nothing at all — not a
    syscall, not a timestamp, not a branch. Timestamps that would otherwise
    cost the request path (first/last scored) are DERIVED by the flusher from
    watching the counters move, and are therefore quantised to buf_secs.
  * A failure to write it can never touch a request: the writer is guarded
    and self-disables after repeated failure, exactly like the sink.
The sink also gets a HEADER LINE at open ({"pn119_header": 1, boot_id, pid,
mode, ...}). Until now a 0-byte meta-*.jsonl was ambiguous between "the tap
never fired", "no traffic arrived" and "the router never started"; a
header-only file now means unambiguously "the router started and nothing was
ever scored". refit_pn119_probe.load_sink ignores the line (it keys on "row"
and "finish"), and skips such files anyway on the 0-byte feats pair.

`fixes/pn119_doctor.py` (installed as `~/bin/pn119-doctor`, symlink) renders
health.json, adds the alarms only a reader can raise (file missing, stale,
from a previous boot) and exits non-zero when any alarm is live. It is stdlib
only and imports nothing from this module: it has to run on a box with no
torch, and it must never be the reason a diagnosis cannot be made.

Never raises into serving: every entry point is fully guarded.
"""
from __future__ import annotations

import atexit
import collections
import hashlib
import itertools
import json
import logging
import math
import os
import socket
import sys
import threading
import time
import uuid
import weakref

import torch

# vLLM's DEFAULT_LOGGING_CONFIG attaches a handler to the "vllm" logger ONLY, so
# a name outside that tree gets no handler and an effective level of WARNING in
# the EngineCore process. Measured 2026-07-25: 61 scored requests produced 61
# logger.info calls and ZERO lines in a 2,637-line container log. Every INFO the
# router emitted about its own health -- including the mode/threshold banner --
# was discarded, which is why two degenerate boots (0% and 100% deep) were only
# noticed hours later by reading the sink offline. Sitting under "vllm" inherits
# the handler and the level.
logger = logging.getLogger("vllm.h119")

LAYERS = (42, 47, 51)
D_MODEL = 5120
# LAST-ONLY: one pooled block per layer (the last prompt token's residual),
# no mean-pool. See §LAST-ONLY in the module docstring — this halves the sink
# and, far more importantly, makes every request scoreable because vLLM always
# recomputes the last prompt token.
POOLS = ("last",)
FEAT_DIM = len(LAYERS) * len(POOLS) * D_MODEL  # 15360
# The npz `blocks` key must equal this, in this order.
FEAT_BLOCKS = tuple(f"L{li}-{p}" for li in LAYERS for p in POOLS)

# Live sinks, so a clean interpreter exit always drains their RAM buffers.
# WeakSet: a router that is garbage collected takes its buffer with it and
# has nothing left to flush.
_SINKS: weakref.WeakSet = weakref.WeakSet()


def _flush_all_sinks() -> None:
    for r in list(_SINKS):
        try:
            r._sink_close()
        except Exception:  # noqa: BLE001 — atexit must never raise
            pass


atexit.register(_flush_all_sinks)

# Sentinel for "the attribute is not on this scheduler_output at all", which is
# a different fact from "it is empty" — the first must force a full scan, the
# second is the whole point of the early-out.
_MISSING = object()

ROUTE_DEEP = "deep"
ROUTE_LEAN = "lean"
ROUTE_CHOICES = (ROUTE_DEEP, ROUTE_LEAN)

# Enforce-mode consumers (PN100/PN102 holder side) read this: req_id -> score.
# Kept for compatibility, but it is NOT the decision of record — an absent key
# reads as "no opinion", which is exactly how unrouted requests went silent.
SCORES: dict[str, float] = {}
# The decision of record: req_id -> "deep"|"lean". Populated under enforce for
# EVERY request the router sees a prefill for, scoreable or not.
ROUTES: dict[str, str] = {}
# PN119_EXPLORE (BUILD-PACK §v2 censoring guard): req_ids selected for
# exploration. Enforce-mode consumers MUST give these generous caps
# regardless of score — that is what keeps the self-training labels honest.
EXPLORE: set[str] = set()

# Route used when a request cannot be scored. Set from PN119_FALLBACK_ROUTE at
# router init; "deep" (champion treatment) is the fail-safe direction — it can
# only cost wall clock, never the accuracy line the router was shipped to hold.
_FALLBACK_ROUTE = ROUTE_DEEP

# Observability (the rate of unscoreable requests must be VISIBLE, not
# inferred). Monotonic counters; `stats_line()` renders them.
STATS: dict[str, int] = collections.defaultdict(int)


def _bump(key: str, n: int = 1) -> None:
    STATS[key] += n


def stats_line() -> str:
    """One-line counter snapshot; safe to call from anywhere."""
    if not STATS:
        return "(no PN119 activity)"
    return " ".join(f"{k}={v}" for k, v in sorted(STATS.items()))


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH SURFACE + ALARMS
# ═══════════════════════════════════════════════════════════════════════════
# Everything below is PURE (no torch, no router instance, no I/O) so that it
# can be replayed over historical sink data on a CPU-only box —
# fixes/test_pn119_alarms.py does exactly that against the 2026-07-25 boots,
# including the five degenerate ones. An alarm whose trigger cannot be
# re-derived offline from recorded data is an alarm nobody can trust.

HEALTH_SCHEMA = "pn119.health/1"

# ── the intended operating band ────────────────────────────────────────────
# The lens probe was trained and the deep/lean split was sized for roughly a
# quarter to a third of traffic taking the deep path (PN119-BUILD-PACK: the
# 25-to-v5 / 75-to-v3 result). A live shadow boot at tdeep=0.495 measured
# 31/100 deep, dead centre. The band is therefore a DESIGN band, not a
# measurement tolerance: leaving it means the score distribution has moved
# relative to the threshold, and the saving the router exists to produce is
# either not being taken (too much deep) or is being taken at the accuracy
# line's expense (too much lean).
_ALARM_BAND_LO = 0.25
_ALARM_BAND_HI = 0.35

# DEEP_FRAC_OUT_OF_BAND uses a 95% Wilson interval rather than the point
# estimate, so a boot is only accused of drifting once the data can carry the
# accusation. This is what keeps a fresh boot quiet: at n=4 with 0 deep the
# interval is [0.00, 0.49] and overlaps the band, so nothing fires; at n=61
# with 8 deep it is [0.07, 0.22], entirely below 0.25, and it fires. The
# floor below is belt-and-braces for pathological tiny-n cases.
_ALARM_Z = 1.96
_ALARM_BAND_MIN_N = 20

# DEEP_FRAC_DEGENERATE is the stronger, blunter claim: EVERY scored request
# went the same way. That is not a calibration question, it is the signature
# of a broken probe / threshold / feature vector, and it is what four of the
# five 2026-07-25 degenerate boots looked like (the fifth was 12-for-12 deep).
# Minimum sample 12: at the band's own lean-side rate (~0.70) the chance of 12
# consecutive lean routes on a HEALTHY router is 0.70^12 = 1.4%, and such a
# false positive self-clears on the very next deep request. The cost of the
# other error — missing it — was measured at one working day. 12 also happens
# to be the smallest degenerate boot in the history, which is not a
# coincidence: a threshold that cannot catch the smallest real instance of the
# failure it is named after is decoration.
_ALARM_DEGENERATE_MIN_N = 12

# CONSUMER_* need "meaningful traffic" before they can distinguish "never
# applied" from "not applied yet". 20 decisions is ~2 minutes of bench traffic
# and is far beyond the longest legitimate warm-up (a request is decided
# within its own prefill).
_ALARM_CONSUMER_MIN_N = 20

# FALLBACK_STORM. Every historical boot ran at a 0% unscoreable rate (prefix
# caching is off — BUG-131), so any sustained double-digit fallback rate is a
# regime change, not noise. It matters because the fallback route is "deep":
# a storm means the router is quietly buying the champion's cost for every
# request while reporting itself alive.
_ALARM_FALLBACK_RATE = 0.20
_ALARM_FALLBACK_MIN_N = 25

# UNROUTABLE_TRAFFIC counts requests the router never reached a decision for
# at all (no req state, no prompt length, or a consumer that had to invent one
# via route_for()). Unlike a fallback this is structural — the router is
# attached to a request lifecycle it does not understand. 1% is deliberately
# tight: the expected value is exactly zero, and the only reason not to fire
# at the first occurrence is that a single racing request during shutdown is
# not worth waking anyone for.
_ALARM_UNROUTABLE_RATE = 0.01
_ALARM_UNROUTABLE_MIN_N = 25

# INDEX_DESYNC. h119_index_desync means the batch slot changed hands between
# marking and resolving and the consumer REFUSED to rewrite it — the guard
# working. It is nonetheless an alarm at the first occurrence, because the
# guard is new and its rate is the only evidence for whether batch-index
# identity holds at all. h119_index_out_of_batch is benign in ones and twos
# (a row mid-move resolves on a later step) but a row that never returns sits
# at the provisional DEEP budget forever, so a sustained rate is real.
_ALARM_OUT_OF_BATCH_RATE = 0.05
_ALARM_OUT_OF_BATCH_MIN_N = 25

# TAP_NEVER_FIRED grace. A router that has observed nothing is either idle or
# blind, and from inside the engine those look identical (this is precisely
# why the 0-byte meta files were ambiguous). Two triggers: an immediate one
# when the CONSUMER saw batch rows the tap never saw — which is proof, not
# inference, since h119_on_batch_add runs off sync_batch and cannot be
# reached by a request the engine did not admit — and a grace-timed one at 10
# minutes for boots where the consumer is off. The timed form can fire on a
# genuinely idle box; that is the correct trade, it self-clears on the first
# request and it says "nothing has been routed for 10 minutes", which is true
# and worth knowing on a bench container that was booted to serve.
_ALARM_TAP_GRACE_S = 600.0

# CENSORED_STORM (BUG-139). A finish whose rtok lands within SLACK of its own
# thinking budget did not stop, it was STOPPED — the label is a lower bound.
# The 2026-07-25 sink measured 43 of 79 thinking finishes (54%) at exactly
# grant-5 while only 4 rows flagged cap_hit, which is why the ceiling on rho
# and AUC was invisible. Under enforce the same fact is worse than a data
# problem: a lean-routed row that truncates looks identical to one that needed
# little, so the router trains on its own decision. 0.35 sits above the noise a
# healthy shadow boot produces on runaway requests and well below the 0.54 that
# actually happened. Warn, not critical: the fix is a budget policy change, not
# a broken component.
_ALARM_CENSORED_RATE = 0.35
_ALARM_CENSORED_MIN_N = 25

# THINK_MARKERS_DIVERGED. The sidecar's PN119_THINK_*_ID single ids and the
# holder's reasoning_config token-id SEQUENCES must describe the same markers.
# When they do not, every `thinking` label and every rtok in the sink is wrong
# and nothing else in the system can tell. The router adopts the holder's
# sequences (it is the thing that actually forces `</think>`) and alarms so the
# env that lied gets fixed rather than silently overridden forever.

# Severity: "critical" = the router is not doing its job (or is doing it to
# the wrong requests); "warn" = it is working but outside its design point.
# pn119-doctor exits 2 on any critical, 1 on warnings only.
_ALARM_SEVERITY = {
    "ROUTER_ABSENT": "critical",
    "TAP_NEVER_FIRED": "critical",
    "DEEP_FRAC_DEGENERATE": "critical",
    "DEEP_FRAC_OUT_OF_BAND": "warn",
    "CONSUMER_NOT_WIRED": "critical",
    "CONSUMER_NEVER_APPLIED": "critical",
    "PROBE_CANARY_FAIL": "critical",
    "FALLBACK_STORM": "warn",
    "UNROUTABLE_TRAFFIC": "warn",
    "MODE_INVALID": "critical",
    "INDEX_DESYNC": "critical",
    "THINK_MARKERS_DIVERGED": "critical",
    "CENSORED_STORM": "warn",
}
ALARM_IDS = tuple(_ALARM_SEVERITY)


def _wilson(k: int, n: int, z: float = _ALARM_Z) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials.

    Closed form on purpose: the flusher recomputes this every buf_secs and an
    exact binomial tail would be a sum over n terms of arbitrary-precision
    integers, which is not something to run every two seconds forever. Wilson
    also behaves at k=0 and k=n, where the normal approximation collapses to a
    zero-width interval and would fire on every fresh boot.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _rate(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def make_snapshot(*, stats, boot_id="", pid=0, hostname="", started=0.0,
                  now=None, mode="shadow", mode_requested="",
                  router_present=True, router_enabled=True, tdeep=0.0,
                  fallback_route=ROUTE_DEEP, fallback_requested="",
                  explore_rate=0.0, probe=None, sink=None, consumer=None,
                  first_scored_ts=None, last_scored_ts=None,
                  last_decision_ts=None, extra=None) -> dict:
    """Build the health document. Pure: dict in, dict out.

    Kept separate from the router instance so the alarm logic can be exercised
    against recorded history (and so an alarm's trigger can be argued about
    without booting a 27B model).
    """
    now = time.time() if now is None else now
    st = dict(stats or {})
    # `scored` counts ROUTABLE rows only (thinking-ON prompts). Everything
    # keyed off it — deep_frac, the design band, the degenerate check — is
    # therefore computed over the population the router can actually move.
    # Thinking-OFF prompts spend zero reasoning tokens whatever the probe
    # says, and folding them in is how a "31% deep" reading was produced from
    # a window that was half out of contract.
    scored = int(st.get("scored", 0))
    deep = int(st.get("scored_deep", 0))
    lean = int(st.get("scored_lean", 0))
    unscoreable = int(st.get("unscoreable", 0))
    # Scored, but the prompt carried no think marker either way (raw /
    # completion). Published like any other route; kept out of every rate.
    scored_unknown = int(st.get("scored_unknown", 0))
    thinking_off = int(st.get("skip_thinking_off", 0))
    censored = int(st.get("finish_censored", 0))
    finished_thinking = int(st.get("finish_thinking", 0))
    decisions = scored + scored_unknown + unscoreable + thinking_off
    # Requests that reached NO probe decision. route_for_miss and
    # h119_route_missing are the consumer inventing a route; skip_* are the
    # observer failing to attach to a request at all.
    unroutable = (int(st.get("route_for_miss", 0))
                  + int(st.get("h119_route_missing", 0))
                  + int(st.get("h119_route_missing_kept_pn100", 0))
                  + int(st.get("skip_no_req_state", 0))
                  + int(st.get("skip_no_prompt_len", 0)))
    # Independent witness that the engine admitted requests at all: these all
    # bump from h119_on_batch_add, which the aux tap cannot influence.
    batch_adds = sum(int(st.get(k, 0)) for k in (
        "h119_provisional_added", "h119_caller_explicit", "h119_pn100_override",
        "h119_tier0_respected", "h119_pn100_entry_missing", "h119_no_router",
        "h119_router_not_enforce"))
    cons = dict(consumer or {})
    cons.setdefault("flag_env", False)
    cons.setdefault("checked", False)
    cons.setdefault("on", False)
    cons.update({
        "pn100_override": int(st.get("h119_pn100_override", 0)),
        "provisional_added": int(st.get("h119_provisional_added", 0)),
        "caller_explicit": int(st.get("h119_caller_explicit", 0)),
        "tier0_respected": int(st.get("h119_tier0_respected", 0)),
        "routed_deep": int(st.get("h119_routed_deep", 0)),
        "routed_lean": int(st.get("h119_routed_lean", 0)),
        "route_missing": int(st.get("h119_route_missing", 0)),
        "route_missing_kept_pn100": int(
            st.get("h119_route_missing_kept_pn100", 0)),
        "index_desync": int(st.get("h119_index_desync", 0)),
        "index_out_of_batch": int(st.get("h119_index_out_of_batch", 0)),
        "no_router": int(st.get("h119_no_router", 0)),
        "router_not_enforce": int(st.get("h119_router_not_enforce", 0)),
        "batch_adds": batch_adds,
        "applied": (int(st.get("h119_pn100_override", 0))
                    + int(st.get("h119_provisional_added", 0))),
    })
    snap = {
        "schema": HEALTH_SCHEMA,
        "boot_id": boot_id,
        "pid": int(pid),
        "hostname": hostname,
        "ts": now,
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + "Z",
        "started": started,
        "uptime_s": max(now - started, 0.0) if started else 0.0,
        "router": {
            "present": bool(router_present),
            "enabled": bool(router_enabled),
            "mode": mode,
            "mode_requested": mode_requested or mode,
            "tdeep": tdeep,
            "fallback_route": fallback_route,
            "fallback_requested": fallback_requested or fallback_route,
            "explore_rate": explore_rate,
        },
        "probe": dict(probe or {}),
        "sink": dict(sink or {}),
        "consumer": cons,
        "traffic": {
            "scored": scored,
            "deep": deep,
            "lean": lean,
            "unscoreable": unscoreable,
            "scored_unknown": scored_unknown,
            "thinking_off": thinking_off,
            "censored": censored,
            "finished_thinking": finished_thinking,
            "decisions": decisions,
            "unroutable": unroutable,
            "batch_adds": batch_adds,
            "first_scored_ts": first_scored_ts,
            "last_scored_ts": last_scored_ts,
            "last_decision_ts": last_decision_ts,
            "idle_s": (max(now - last_decision_ts, 0.0)
                       if last_decision_ts else None),
        },
        "rates": {
            # Over SCORED requests only: a fallback is not a route the probe
            # chose, and folding it in would let a fallback storm masquerade
            # as a healthy deep fraction.
            "deep_frac": _rate(deep, scored),
            "deep_frac_n": scored,
            # Everything that got a default instead of a decision.
            "fallback_rate": _rate(unscoreable + unroutable, decisions),
            # The narrower "the probe refused to score this" rate.
            "unscoreable_rate": _rate(unscoreable, decisions),
            "unroutable_rate": _rate(unroutable, decisions),
            # Share of everything the router saw that it can never move.
            # Not a failure — but if it is most of the traffic, the saving
            # the router can produce is bounded by (1 - this).
            "thinking_off_rate": _rate(thinking_off, decisions),
            "unknown_routable_rate": _rate(scored_unknown, decisions),
            # BUG-139: share of finished thinking requests whose rtok is a
            # lower bound, not a stop. Over ROUTABLE finishes only.
            "censored_rate": _rate(censored, finished_thinking),
            "censored_rate_n": finished_thinking,
        },
        "stats": {k: int(v) for k, v in sorted(st.items())},
    }
    if extra:
        snap.update(extra)
    snap["alarms"] = pn119_alarms(snap)
    snap["alarm_ids"] = [a["id"] for a in snap["alarms"]]
    snap["ok"] = not snap["alarms"]
    return snap


def _alarm(out, aid, detail, **fields):
    out.append({"id": aid, "severity": _ALARM_SEVERITY.get(aid, "warn"),
                "detail": detail, **fields})


def pn119_alarms(snap: dict) -> list[dict]:
    """The product. Every alarm below exists because of a specific way this
    router has failed, or can fail, while looking healthy from outside.

    Pure and total: it must never raise (it runs inside the flusher), so an
    unexpected snapshot shape degrades to "fewer alarms", never to an
    exception that takes the health file down with it.
    """
    out: list[dict] = []
    try:
        r = snap.get("router") or {}
        t = snap.get("traffic") or {}
        rt = snap.get("rates") or {}
        c = snap.get("consumer") or {}
        p = snap.get("probe") or {}
        st = snap.get("stats") or {}
        scored = int(t.get("scored", 0))
        deep = int(t.get("deep", 0))
        decisions = int(t.get("decisions", 0))
        uptime = float(snap.get("uptime_s", 0.0) or 0.0)

        # ── ROUTER_ABSENT ──────────────────────────────────────────────────
        # The router was asked for and is not there. In-process this means the
        # module global ROUTER is not this instance — i.e. maybe_create never
        # completed, or a second instance took the slot, either of which makes
        # `_consumer_active()` false forever and the consumer a no-op.
        # pn119-doctor raises the same id for the reader-side version of the
        # same fact: the flag is on and there is no health surface at all.
        if r.get("enabled") and not r.get("present"):
            _alarm(out, "ROUTER_ABSENT",
                   "router enabled but no live instance is registered "
                   "(ROUTER is None or another instance owns the global) — "
                   "the enforce consumer can never act")

        # ── MODE_INVALID ───────────────────────────────────────────────────
        # PN119_MODE="enforced" is not a hypothetical: everything downstream
        # tests `mode != "enforce"`, so a typo degrades to shadow and the boot
        # scores, logs and sinks exactly like a working enforce boot. Same
        # class for a coerced fallback route. The router already corrects both
        # and logs an ERROR — but a discarded log is how this survived once
        # already, so the coercion is also a standing alarm.
        want_mode = str(r.get("mode_requested", "") or "")
        if want_mode.strip().lower() != str(r.get("mode", "")).lower():
            _alarm(out, "MODE_INVALID",
                   f"PN119_MODE={want_mode!r} was coerced to "
                   f"{r.get('mode')!r} — nothing downstream will enforce")
        want_fb = str(r.get("fallback_requested", "") or "")
        if want_fb.strip().lower() != str(r.get("fallback_route", "")).lower():
            _alarm(out, "MODE_INVALID",
                   f"PN119_FALLBACK_ROUTE={want_fb!r} was coerced to "
                   f"{r.get('fallback_route')!r}")
        tdeep = r.get("tdeep")
        if tdeep is None or not math.isfinite(float(tdeep)):
            _alarm(out, "MODE_INVALID",
                   f"tdeep={tdeep!r} is not a finite threshold — every "
                   "comparison against it is False, i.e. everything routes lean")

        # ── PROBE_CANARY_FAIL ──────────────────────────────────────────────
        # The probe is the only thing standing between "a routing decision"
        # and "a constant". _load_probe proves the fold against the staged
        # form on every load and refuses a disagreement; this surfaces the
        # cases where that refusal happened at RELOAD time, because then the
        # router keeps serving the OLD weights indefinitely while the refit
        # timer believes it shipped new ones — no error, no restart, just a
        # probe that silently stopped tracking the traffic it was refit on.
        # The npz vanishing is the same class: the next reload cannot happen.
        if int(st.get("probe_reload_failed", 0)) > 0:
            _alarm(out, "PROBE_CANARY_FAIL",
                   f"{st.get('probe_reload_failed')} probe hot-reload(s) were "
                   "REFUSED — serving stale weights while the refit timer "
                   "thinks the swap landed",
                   count=int(st.get("probe_reload_failed", 0)))
        if p and p.get("readable") is False:
            _alarm(out, "PROBE_CANARY_FAIL",
                   f"probe file {p.get('path')!r} is no longer readable — the "
                   "loaded weights are now the only copy")
        resid = p.get("fold_resid")
        if resid is not None and (not math.isfinite(float(resid))
                                  or float(resid) > 1e-6):
            _alarm(out, "PROBE_CANARY_FAIL",
                   f"probe fold residual {resid!r} exceeds 1e-6 — a fold error "
                   "shifts every score by a constant, which is indistinguishable "
                   "from a threshold that needs recalibrating")

        # ── TAP_NEVER_FIRED ────────────────────────────────────────────────
        # The aux-hidden-state tap is set up at load and is never confirmed
        # again. If set_aux_hidden_state_layers stops taking effect (a model
        # refactor, an unpatched unpack site, a runner variant), observe() is
        # called with aux=None and returns — no counter, no log, no sink row.
        # From outside that is byte-identical to an idle box, which is why
        # twenty-odd 0-byte sink files exist and none of them told anyone
        # anything.
        observations = (scored + int(t.get("unscoreable", 0))
                        + int(t.get("scored_unknown", 0))
                        + int(t.get("thinking_off", 0))
                        + int(st.get("skip_no_req_state", 0))
                        + int(st.get("skip_no_prompt_len", 0)))
        batch_adds = int(t.get("batch_adds", 0))
        if observations == 0 and batch_adds > 0:
            _alarm(out, "TAP_NEVER_FIRED",
                   f"the consumer saw {batch_adds} batch row(s) and the tap "
                   "observed 0 prefills — traffic is arriving and the aux "
                   "hidden states are not",
                   batch_adds=batch_adds)
        elif observations == 0 and uptime > _ALARM_TAP_GRACE_S:
            _alarm(out, "TAP_NEVER_FIRED",
                   f"nothing observed in {uptime / 60.0:.1f} min of uptime — "
                   "either no traffic reached the engine or the tap is not "
                   "firing; both are worth knowing on a serving container",
                   uptime_s=round(uptime, 1))

        # ── DEEP_FRAC_DEGENERATE ───────────────────────────────────────────
        # Every scored request went the same way. This is the exact shape of
        # all five 2026-07-25 degenerate boots (12/12 and 24/24 deep;
        # 0/81, 0/81 and 0/61 deep) and of the NaN-score failure mode the
        # sd>=1e-4 gate now refuses at load — NaN >= tdeep is False, so a
        # degenerate probe routes 100% lean at full accuracy cost.
        if scored >= _ALARM_DEGENERATE_MIN_N and deep in (0, scored):
            side = "deep" if deep else "lean"
            _alarm(out, "DEEP_FRAC_DEGENERATE",
                   f"all {scored} scored requests routed {side} — the probe, "
                   "the threshold or the feature vector is broken, not "
                   "mis-calibrated",
                   n=scored, deep=deep, deep_frac=_rate(deep, scored))
        elif scored >= _ALARM_BAND_MIN_N:
            # ── DEEP_FRAC_OUT_OF_BAND ──────────────────────────────────────
            # Not degenerate, but the 95% interval sits entirely outside the
            # 25-35% design band, so the split the router was shipped to make
            # is not the split it is making. Reported as a warning: this is
            # the one alarm that can legitimately be answered with "retune
            # tdeep" rather than "something is broken".
            lo, hi = _wilson(deep, scored)
            if hi < _ALARM_BAND_LO or lo > _ALARM_BAND_HI:
                _alarm(out, "DEEP_FRAC_OUT_OF_BAND",
                       f"deep fraction {_rate(deep, scored):.3f} "
                       f"(95% CI {lo:.3f}-{hi:.3f}, n={scored}) is outside the "
                       f"{_ALARM_BAND_LO:.2f}-{_ALARM_BAND_HI:.2f} design band",
                       n=scored, deep=deep, ci=[round(lo, 4), round(hi, 4)],
                       deep_frac=_rate(deep, scored))

        # ── CONSUMER_NOT_WIRED / CONSUMER_NEVER_APPLIED ────────────────────
        # These two split the "the flag is on and nothing happens" space in
        # half, because the two halves have completely different fixes.
        if c.get("flag_env"):
            if int(c.get("no_router", 0)) or int(c.get("router_not_enforce", 0)):
                # The consumer ran and bailed on its own preconditions.
                _alarm(out, "CONSUMER_NOT_WIRED",
                       "consumer flag is on but it declined: "
                       f"no_router={c.get('no_router')} "
                       f"router_not_enforce={c.get('router_not_enforce')} "
                       "(shadow mode acts on nothing — enforce is required)")
            elif decisions >= _ALARM_CONSUMER_MIN_N and not c.get("checked"):
                # h119_on_batch_add was never called at all across meaningful
                # traffic => site F is not installed in this image. The boot
                # log will still have printed every patch site it "installed".
                _alarm(out, "CONSUMER_NOT_WIRED",
                       f"consumer flag is on and h119_on_batch_add never ran "
                       f"across {decisions} decisions — the sync_batch patch "
                       "site is not live in this image",
                       decisions=decisions)
            elif decisions >= _ALARM_CONSUMER_MIN_N and c.get("batch_adds", 0) == 0:
                _alarm(out, "CONSUMER_NOT_WIRED",
                       f"consumer saw 0 batch rows across {decisions} routed "
                       "decisions — it is attached to a holder that never "
                       "syncs, or to the wrong process",
                       decisions=decisions)
            elif (decisions >= _ALARM_CONSUMER_MIN_N
                  and int(c.get("applied", 0)) == 0):
                # THE 2026-07-25 DAY-LONG BUG, exactly. Every site installed,
                # every counter moving, GPQA-30 byte-identical to control: the
                # consumer deferred to PN100's budget on 100% of requests
                # (h119_caller_explicit) because it could not tell PN100's
                # grant from a caller's. Neither override nor provisional ever
                # incremented — that pair being 0 under real traffic is the
                # signature, and it is the reason this file exists.
                _alarm(out, "CONSUMER_NEVER_APPLIED",
                       "consumer flag is on and it has taken over NOTHING: "
                       f"h119_pn100_override=0 h119_provisional_added=0 over "
                       f"{decisions} decisions "
                       f"(caller_explicit={c.get('caller_explicit')}, "
                       f"tier0_respected={c.get('tier0_respected')}) — the "
                       "route is being computed and thrown away",
                       decisions=decisions,
                       caller_explicit=int(c.get("caller_explicit", 0)),
                       tier0_respected=int(c.get("tier0_respected", 0)))

        # ── FALLBACK_STORM ─────────────────────────────────────────────────
        # Unscoreable requests take the fallback route, which is "deep" by
        # design (fail-safe on accuracy). That makes a storm the most
        # expensive quiet failure available: the router pays the champion's
        # cost for everyone and still reports itself alive and scoring.
        if (decisions >= _ALARM_FALLBACK_MIN_N
                and float(rt.get("fallback_rate", 0.0)) > _ALARM_FALLBACK_RATE):
            reasons = {k: v for k, v in st.items()
                       if k.startswith("unscoreable_")}
            _alarm(out, "FALLBACK_STORM",
                   f"{rt.get('fallback_rate', 0.0):.1%} of {decisions} decisions "
                   f"took the {r.get('fallback_route')} fallback instead of a "
                   f"score (>{_ALARM_FALLBACK_RATE:.0%}) — reasons: "
                   f"{reasons or 'none recorded'}",
                   decisions=decisions,
                   fallback_rate=float(rt.get("fallback_rate", 0.0)),
                   reasons=reasons)

        # ── UNROUTABLE_TRAFFIC ─────────────────────────────────────────────
        # Requests the router never reached a decision for: no request state,
        # no prompt length, or a consumer forced to invent a route via
        # route_for(). Structural rather than statistical — the expected count
        # is zero, so the threshold is only there to tolerate a request racing
        # a shutdown.
        if (decisions >= _ALARM_UNROUTABLE_MIN_N
                and float(rt.get("unroutable_rate", 0.0)) > _ALARM_UNROUTABLE_RATE):
            _alarm(out, "UNROUTABLE_TRAFFIC",
                   f"{t.get('unroutable')} of {decisions} requests reached no "
                   "routing decision at all (route_for misses / missing req "
                   "state / missing prompt length) — the router is attached to "
                   "a request lifecycle it does not fully see",
                   unroutable=int(t.get("unroutable", 0)), decisions=decisions,
                   rate=float(rt.get("unroutable_rate", 0.0)))

        # ── THINK_MARKERS_DIVERGED ─────────────────────────────────────────
        # The env told this module one thing and the holder that actually
        # forces `</think>` uses another. Whichever is wrong, every `thinking`
        # flag and every rtok in the sink is now suspect — and the router
        # cannot tell which rows. Critical, and it fires from the first boot
        # because the divergence is a static property of the configuration.
        if int(st.get("think_marker_divergence", 0)) > 0:
            _alarm(out, "THINK_MARKERS_DIVERGED",
                   "PN119_THINK_START_ID/PN119_THINK_END_ID disagree with the "
                   "holder's reasoning_config token-id sequences — the holder's "
                   "were adopted, but the env is lying and every historical "
                   "label written under it is suspect",
                   markers=st.get("think_marker_divergence"))

        # ── CENSORED_STORM (BUG-139) ───────────────────────────────────────
        # rtok landed within SLACK of the request's own budget: a lower bound
        # recorded as a spend. In shadow this caps what any refit can learn;
        # in enforce it makes lean self-confirming.
        cens_n = int(t.get("finished_thinking", 0))
        if (cens_n >= _ALARM_CENSORED_MIN_N
                and float(rt.get("censored_rate", 0.0)) > _ALARM_CENSORED_RATE):
            _alarm(out, "CENSORED_STORM",
                   f"{t.get('censored')}/{cens_n} thinking finishes "
                   f"({rt.get('censored_rate', 0.0):.1%}) stopped at their own "
                   f"budget (> {_ALARM_CENSORED_RATE:.0%}) — those rtok values "
                   "are lower bounds, and under enforce a truncated lean row "
                   "trains the router to keep routing it lean",
                   censored=int(t.get("censored", 0)), n=cens_n,
                   rate=float(rt.get("censored_rate", 0.0)))

        # ── INDEX_DESYNC ───────────────────────────────────────────────────
        # _state is keyed by BATCH INDEX and indices are recycled. A desync
        # means slot i belonged to a different request between mark and
        # resolve; the identity guard caught it and refused, which is the only
        # reason the wrong request did not get capped. Alarm on the FIRST
        # occurrence: applying one request's route to another's budget is
        # invisible in every output collected, so the guard's hit count is the
        # entire evidence base for whether the assumption holds.
        desync = int(c.get("index_desync", 0))
        if desync > 0:
            _alarm(out, "INDEX_DESYNC",
                   f"{desync} batch slot(s) changed hands between mark and "
                   "resolve — the identity guard refused to rewrite them; "
                   "without it these would have capped the wrong request",
                   count=desync)
        oob = int(c.get("index_out_of_batch", 0))
        resolved = (int(c.get("routed_deep", 0)) + int(c.get("routed_lean", 0))
                    + oob)
        if (resolved >= _ALARM_OUT_OF_BATCH_MIN_N
                and _rate(oob, resolved) > _ALARM_OUT_OF_BATCH_RATE):
            _alarm(out, "INDEX_DESYNC",
                   f"{oob}/{resolved} provisional rows resolved outside the "
                   "current batch — a row that never returns sits at the "
                   "provisional DEEP budget for its whole life",
                   out_of_batch=oob, resolved=resolved,
                   rate=_rate(oob, resolved))
    except Exception as e:  # noqa: BLE001 — a health check must never raise
        out.append({"id": "ALARM_ENGINE_ERROR", "severity": "warn",
                    "detail": f"{type(e).__name__}: {e}"})
    return out


# The live worker-side router instance (set by maybe_create). The enforce
# consumer needs it for two things it cannot get from the holder alone: the
# mode (never act while the router is in shadow) and runner.input_batch, which
# is the ONLY place batch-index -> req_id is available inside the sampler.
ROUTER: "PN119Router | None" = None


def route_for(req_id: str) -> str:
    """Authoritative route for `req_id` — NEVER None.

    Enforce-mode consumers MUST use this rather than `SCORES.get(req_id)`.
    A missing entry means the router never got a prefill decision for this
    request at all; that is a bug, so it is counted and it still returns the
    defined fallback route instead of silently doing nothing.
    """
    r = ROUTES.get(req_id)
    if r is not None:
        return r
    _bump("route_for_miss")
    logger.warning("[PN119] route_for(%s): no decision on record — "
                   "falling back to %s (%s)", req_id, _FALLBACK_ROUTE,
                   stats_line())
    return _FALLBACK_ROUTE


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _truthy(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "on")


class PN119Router:
    @classmethod
    def maybe_create(cls, runner):
        """Called from load_model tail. Returns router or None; never raises."""
        try:
            if not _truthy(_env("GENESIS_ENABLE_PN119_ROUTER")):
                return None
            npz_path = _env("GENESIS_PN119_PROBE")
            if not npz_path or not os.path.isfile(npz_path):
                logger.warning("[PN119] probe npz missing (%r) — router disabled", npz_path)
                return None
            model = runner.get_model() if hasattr(runner, "get_model") else runner.model
            if not hasattr(model, "set_aux_hidden_state_layers"):
                logger.warning("[PN119] model lacks set_aux_hidden_state_layers — disabled")
                return None
            model.set_aux_hidden_state_layers(tuple(LAYERS))
            # Deliberately NOT setting runner.use_aux_hidden_state_outputs:
            # that flag also reroutes the DRAFTER'S target_hidden_states to an
            # aux CONCAT (eagle3 semantics) — under MTP that fed a 15360-dim
            # tensor into a 5120-dim proposer buffer and killed the engine on
            # the first spec-decode step (live crash 2026-07-25 15:18Z).
            # Instead the model returns (hidden, aux) tuples on its own and
            # the PN119 patcher makes BOTH unpack sites tuple-tolerant while
            # every drafter/eagle3 site keeps stock flag-off behavior.
            inst = cls(runner, npz_path)
            global ROUTER
            ROUTER = inst
            # Re-publish now that the instance owns the global: health_snapshot
            # reports router.present as "ROUTER is me", and the write inside
            # __init__ necessarily ran before this assignment.
            # force: the counters have not moved since the write in __init__,
            # and the publish-on-change guard would otherwise skip this one —
            # leaving a health file that says router.present=False.
            inst._health_write(force=True)
            # Belt and braces for the discarded-logs class above: if this
            # logger still cannot emit INFO, attach a stderr handler rather than
            # run blind. A router nobody can observe is how two degenerate boots
            # (0% and 100% deep) survived a whole afternoon unnoticed.
            if not logger.isEnabledFor(logging.INFO):
                _h = logging.StreamHandler(sys.stderr)
                _h.setFormatter(logging.Formatter(
                    "%(asctime)s [h119] %(levelname)s %(message)s"))
                logger.addHandler(_h)
                logger.setLevel(logging.INFO)
                logger.propagate = False
                logger.info("[PN119] attached own stderr handler — the 'vllm' "
                            "logging config did not reach %s", logger.name)
            logger.info(
                "[PN119] router active: mode=%s tdeep=%.3f probe=%s(%s) "
                "feat_dim=%d blocks=%s aux layers=%s sink=%s fallback_route=%s "
                "async_score=%s acc_max=%d think_start=%s think_end=%s "
                "censor_slack=%d boot_id=%s health=%s",
                inst.mode, inst.tdeep, os.path.basename(npz_path),
                inst.probe_sig_short(), FEAT_DIM, ",".join(FEAT_BLOCKS), LAYERS,
                inst.sink_dir or "-", inst.fallback_route,
                "on" if inst._async_want else "off", inst._acc_max,
                inst._think_start_ids, inst._think_end_ids,
                inst._censor_slack, inst.boot_id, inst.health_path or "-",
            )
            return inst
        except Exception as e:  # noqa: BLE001 — never brick model load
            logger.warning("[PN119] init failed: %s — router disabled", e)
            return None

    def __init__(self, runner, npz_path: str):
        global _FALLBACK_ROUTE
        self.runner = runner
        # Boot identity. A sink file, a health file and a log line all have to
        # be attributable to ONE router instance: the 2026-07-25 forensics
        # failed mainly because forty sink files from a dozen boots could only
        # be told apart by their filename timestamps, and a health.json left
        # behind by a dead boot reads exactly like a live one.
        self.boot_id = uuid.uuid4().hex
        self.started = time.time()
        try:
            self.hostname = socket.gethostname()   # = the container id
        except OSError:
            self.hostname = ""
        self.mode_requested = _env("PN119_MODE", "shadow")
        self.mode = _env("PN119_MODE", "shadow").lower() or "shadow"
        if self.mode not in ("shadow", "enforce"):
            # Everything downstream tests `mode != "enforce"`, so a typo like
            # "enforced" or "ENFORCE " silently degrades to shadow: the router
            # scores, logs, sinks, and acts on nothing, which is exactly what a
            # working enforce boot looks like from the outside.
            logger.error("[PN119] PN119_MODE=%r is not shadow|enforce — "
                         "falling back to shadow; NOTHING will be enforced",
                         self.mode)
            self.mode = "shadow"
        try:
            self.tdeep = float(_env("PN119_TDEEP", "0.5") or 0.5)
        except ValueError:
            # A non-numeric threshold used to take the whole router down at
            # load (maybe_create turns the raise into "disabled"). Degrading to
            # the default and alarming keeps the boot serving AND visible.
            logger.error("[PN119] PN119_TDEEP=%r is not a number — using 0.5",
                         _env("PN119_TDEEP"))
            self.tdeep = 0.5
        # Route for requests we cannot score (see module docstring §2).
        self.fallback_requested = _env("PN119_FALLBACK_ROUTE", ROUTE_DEEP)
        fb = _env("PN119_FALLBACK_ROUTE", ROUTE_DEEP).lower() or ROUTE_DEEP
        if fb not in ROUTE_CHOICES:
            logger.warning("[PN119] PN119_FALLBACK_ROUTE=%r invalid — using %s",
                           fb, ROUTE_DEEP)
            fb = ROUTE_DEEP
        self.fallback_route = _FALLBACK_ROUTE = fb
        self._stats_every = max(int(_env("PN119_STATS_EVERY", "200") or 200), 1)
        self._decisions = 0
        # ── step-level early-out (latency hygiene) ─────────────────────────
        # A step with no newly scheduled request and nothing mid-prefill has
        # nothing this module can possibly finalize, so the whole batch scan
        # (a python loop over req_ids plus a `sum()` over the schedule dict)
        # is pure overhead on every decode step — which is the overwhelming
        # majority of steps. The periodic FULL scan is the safety valve: if
        # `scheduled_new_reqs` ever fails to mean what we think it means, a
        # request stuck in the accumulator is still found within
        # PN119_FULLSCAN_EVERY steps rather than never.
        self._step = 0
        self._fullscan_every = max(
            int(_env("PN119_FULLSCAN_EVERY", "256") or 256), 1)
        # ── bounded in-flight state ────────────────────────────────────────
        # Every per-request map here is popped in on_finish. on_finish is
        # driven by scheduler_output.finished_req_ids, so a request the engine
        # aborts without reporting (or one racing a router that attached late)
        # leaks an entry forever. 4x max_num_seqs is enough headroom that the
        # reaper never touches a live request and small enough that a leak is
        # bounded by megabytes rather than by uptime.
        self._acc_max = self._resolve_acc_max()
        # v2 explore knob (BUILD-PACK §v2): fraction of requests flagged for
        # generous caps regardless of score. Deterministic per req_id so the
        # sink row and the enforce-side consumer always agree.
        try:
            self.explore_rate = min(max(float(_env("PN119_EXPLORE", "0") or 0.0), 0.0), 1.0)
        except ValueError:
            self.explore_rate = 0.0
        # v2 hot-reload: refit timer atomically swaps the npz; we re-load on a
        # CONTENT HASH change, throttled to one read per PN119_RELOAD_S.
        # (mtime_ns, size) was the old signature and it is blind to `cp -p`,
        # `rsync -a` and snapshot rollbacks — and the two probes we actually
        # swap between have IDENTICAL sizes, so that blindness is not
        # theoretical. See §v2 loop in the module docstring.
        self._probe_path = npz_path
        self._reload_every = max(float(_env("PN119_RELOAD_S", "60") or 60.0), 1.0)
        self._next_reload_check = time.time() + self._reload_every
        self._probe_sig = self._content_sig(npz_path)
        self._failed_sig = None
        # Health: the fold self-check residual of the weights actually in use.
        self._probe_fold_resid = None
        self._probe_canary = None
        self._probe_loads = 0
        self.pv, self.pb = self._load_probe(npz_path)
        # v2 label plumbing: think-region markers. Resolved against the
        # holder's reasoning_config SEQUENCES — see _resolve_think_markers.
        self._think_start_ids, self._think_end_ids = self._resolve_think_markers()
        # Kept as scalars for the tests and log lines that name a single id;
        # nothing in the scan path reads them.
        self._think_start = self._think_start_ids[0] if self._think_start_ids else None
        self._think_end = self._think_end_ids[0] if self._think_end_ids else None
        # BUG-139 slack: how close to its own grant an rtok has to land before
        # the stop is attributed to the budget rather than to the model. The
        # measured offset on 43 live rows is exactly 5; +8 over the end-marker
        # length absorbs a multi-token `</think>` and the spec-decode lookahead
        # without ever reaching far enough down to swallow a natural stop
        # (the nearest non-grid modal value in the same window is 881 against
        # a 1300 grant).
        self._censor_slack = max(len(self._think_end_ids) + 8, 8)
        # Tail window for the thinking-label scan. MUST be wide enough to reach
        # back past whatever PN102 seeded INSIDE the think region, otherwise the
        # <think> marker falls out of view and the row labels thinking=None —
        # i.e. the refit trains on nothing. The old 8-token window only worked
        # under the v5-family seed ("Step 1:", ~3 tokens); the v3 seed
        # ("Budget: ~13 short steps.\nStep 1:", ~12 tokens) and the v3-echo /
        # v8b seeds are longer, and v3 is the validated prod path.
        # 64 clears every PN102 variant (longest ≈ 20 tokens) with 3x headroom.
        # Widening is safe under the last-marker-wins scan below: a marker from
        # an EARLIER conversation turn can never outrank the current turn's.
        self._tail_window = max(
            int(_env("PN119_TAIL_WINDOW", "64") or 64), 8)
        # In-flight prefill tracker: req_id -> {"seen", "plen", "step"}. With
        # last-only there is nothing to accumulate, so this is bookkeeping
        # only — it is what tells a chunked prefill apart from a request the
        # router has never seen, and it is what the step early-out consults.
        self._acc: dict[str, dict] = {}
        self.scored: dict[str, float] = {}
        # req_id -> prompt length AT SCORE TIME. A streaming/multi-turn client
        # that re-prefills under the SAME req_id used to keep turn 1's route
        # for the rest of the session and then write turn N's labels against
        # turn 1's feature row — a wrong training row, not merely a stale one.
        # A mismatch here drops the score and re-enters the accumulator.
        self._scored_plen: dict[str, int] = {}
        # req_id -> reason, for requests that got the fallback route instead of
        # a score. Keeps the warning + the sink line to one per request.
        self.unscored: dict[str, str] = {}
        # req_id -> reason, for requests deliberately NOT scored because the
        # router has no lever on them (thinking-off). Distinct from `unscored`
        # so a design decision never reads as a failure in the health surface.
        self.skipped: dict[str, str] = {}
        # req_id -> budget H119's consumer actually applied. The only way the
        # finish line can report the EFFECTIVE grant: the consumer rewrites
        # the holder's state entry, never SamplingParams.
        self._h119_applied: dict[str, int] = {}
        # Bounded window of routable scores, for the per-row `pctl` field.
        self._pctl_window = max(int(_env("PN119_PCTL_WINDOW", "512") or 512), 8)
        self._score_win: collections.deque = collections.deque()
        self._score_sorted: list[float] = []
        # ── deferred score readback (PN119_ASYNC_SCORE) ────────────────────
        self._async_want = _truthy(_env("PN119_ASYNC_SCORE", "0"))
        self._async_slots = max(int(_env("PN119_ASYNC_SLOTS", "0") or 0), 0)
        self._async_ready = False
        self._pin_score = None
        self._pin_feat = None
        self._free_slots: list[int] = []
        self._pending: list[dict] = []
        self._warned = 0
        # v2 sink — RAM-buffered, flushed off the request path (see
        # _sink_append).  The sink is pure observability; the routing decision
        # never reads it, so it has no business owning a syscall per request.
        self.sink_dir = _env("PN119_SINK")
        self._sink_feat = self._sink_meta = None
        self._sink_rows = 0
        self._sbuf_feat: list[bytes] = []
        self._sbuf_meta: list[str] = []
        self._sink_lock = threading.Lock()      # guards the two buffers
        self._sink_io_lock = threading.Lock()   # serialises writers (ORDER)
        self._sink_wake = threading.Event()
        self._sink_thread = None
        self._sink_stop = False
        self._sink_pid = os.getpid()
        # Thresholds are counted in SINK EVENTS (= meta lines), because every
        # event writes one: a scored request emits 2 (score + finish) and one
        # feature row, an unscoreable one emits 2 and no row.  So the default
        # 64 events is ~32 requests of exposure, bounded by BUF_SECS anyway.
        try:
            self._buf_rows = int(_env("PN119_SINK_BUF_ROWS", "64") or 64)
            self._buf_secs = max(
                float(_env("PN119_SINK_BUF_SECS", "2") or 2), 0.05)
            self._buf_max = max(
                int(_env("PN119_SINK_BUF_MAX", "512") or 512),
                max(self._buf_rows, 1))
        except ValueError:
            logger.warning("[PN119] bad PN119_SINK_BUF_* — using defaults")
            self._buf_rows, self._buf_secs, self._buf_max = 64, 2.0, 512
        # ── health surface (module docstring §HEALTH SURFACE) ───────────────
        # Defaults into the sink dir, which is already a rw mount everywhere
        # the router runs — so the health file appears at the next boot with
        # no compose change. PN119_HEALTH overrides.
        self.health_path = _env("PN119_HEALTH") or (
            os.path.join(self.sink_dir, "health.json") if self.sink_dir else "")
        self._health_fails = 0
        # Publish-on-change plus a heartbeat. The flusher ticks every
        # buf_secs (2 s), and rewriting a ~3 KB document 43,200 times a day on
        # an idle boot is pure write amplification for no added information.
        # The heartbeat must stay well inside pn119-doctor's staleness floor
        # (30 s) so that "not written recently" keeps meaning "the writer
        # stopped", not "nothing happened".
        try:
            self._health_every = max(
                float(_env("PN119_HEALTH_EVERY_S", "10") or 10.0), 1.0)
        except ValueError:
            self._health_every = 10.0
        self._h_next_beat = 0.0
        self._h_prev_fp = None
        # Derived timestamps. The request path must not pay for these, so the
        # flusher infers them by watching the counters move: accurate to one
        # PN119_SINK_BUF_SECS tick, which is 2 s by default and is two orders
        # finer than anything anyone asks this file.
        self._h_first_scored_ts = None
        self._h_last_scored_ts = None
        self._h_last_decision_ts = None
        self._h_prev_scored = 0
        self._h_prev_decisions = 0
        if self.sink_dir:
            try:
                os.makedirs(self.sink_dir, exist_ok=True)
                tag = time.strftime("%Y%m%d-%H%M%S")
                self._sink_feat = open(
                    os.path.join(self.sink_dir, f"feats-{tag}.bin"), "ab")
                self._sink_meta = open(
                    os.path.join(self.sink_dir, f"meta-{tag}.jsonl"), "a",
                    encoding="utf-8")
                self._sink_header()
                self._sink_start()
                _SINKS.add(self)
                logger.info("[PN119] sink buffered: buf_rows=%d buf_secs=%.2f "
                            "buf_max=%d thread=%s health=%s", self._buf_rows,
                            self._buf_secs, self._buf_max,
                            self._sink_thread is not None,
                            self.health_path or "-")
            except OSError as e:
                logger.warning("[PN119] sink unavailable (%s) — logging only", e)
                self._sink_feat = self._sink_meta = None
        if self.health_path and self._sink_thread is None:
            # Health without a sink (or with the synchronous sink escape).
            # Deliberately reuses the SAME thread rather than adding one: the
            # loop already exists, and a second timer thread in the worker
            # process is a cost with no matching benefit.
            self._sink_start()
            _SINKS.add(self)
        # Publish once immediately, so `health.json` exists from boot rather
        # than from the first flush — "no file" and "file with no traffic" are
        # different diagnoses and the doctor must be able to tell them apart.
        self._health_write()

    # ── probe loading + hot-reload ──────────────────────────────────────────
    def _load_probe(self, path: str):
        """Load npz -> (v, b): ONE folded vector on device, plus a bias scalar.

        The scoring pipeline is a chain of affine maps, so it collapses:

            score = w[:-1] . (Vt @ ((x - mu) / sd)) + w[-1]
                  = ((Vt^T w[:-1]) / sd) . (x - mu)  + w[-1]
                  =  v . x  +  b        with v = (Vt^T w[:-1]) / sd
                                             b =  w[-1] - v . mu

        Folding at LOAD time instead of per request:
          * VRAM collapses to one [FEAT_DIM] fp32 vector — mu, sd and the
            [10, FEAT_DIM] Vt matrix stop being resident at all;
          * six kernels (sub, div, matvec, dot, add, and their temporaries)
            collapse to one dot product;
          * numerically BETTER, not worse: the fold is computed in float64 on
            the host, so the per-request path carries one rounding instead of
            four. A 10xFEAT_DIM GEMV is bandwidth-bound, which is also why fp16
            would buy nothing here and would cost precision.

        Raises on any problem — callers decide whether that is fatal
        (__init__) or a keep-old-weights warning (_maybe_reload).
        """
        import numpy as np

        z = np.load(path, allow_pickle=True)
        dev = self.runner.device
        # float64 host-side: this runs once per load, and it is the only place
        # the algebra can lose precision.
        mu = np.asarray(z["mu"], dtype=np.float64).reshape(-1)
        sd = np.asarray(z["sd"], dtype=np.float64).reshape(-1)
        vt = np.asarray(z["Vt10"], dtype=np.float64)   # [pcs, FEAT_DIM]
        w = np.asarray(z["w"], dtype=np.float64).reshape(-1)  # [pcs+1] incl bias
        if mu.size != FEAT_DIM or sd.size != FEAT_DIM:
            raise ValueError(f"probe dim {mu.size} != {FEAT_DIM}")
        # A degenerate probe is the WORST failure this router has, because it is
        # completely silent: scoring divides by sd, so a zero entry yields inf
        # or NaN, and `NaN >= tdeep` is False — every request routes LEAN, at
        # full accuracy cost, with no exception, no warning and a deep rate of
        # 0% that looks like a calibration question rather than a broken file.
        # Refuse to load instead; maybe_create turns a raise into "router
        # disabled", which is loud and leaves upstream behaviour intact.
        for name, t in (("mu", mu), ("sd", sd), ("Vt10", vt), ("w", w)):
            if not bool(np.isfinite(t).all()):
                raise ValueError(f"probe {name} contains non-finite values")
        sd_min = float(sd.min())
        if sd_min < 1e-4:
            raise ValueError(
                f"probe sd has a near-zero entry ({sd_min:.3e} < 1e-4) — "
                "scores would be inf/NaN and every request would route lean")
        if vt.shape[1] != FEAT_DIM or w.size != vt.shape[0] + 1:
            raise ValueError(f"probe shapes Vt{vt.shape} w{w.shape} inconsistent")
        # BLOCK ORDER. mu/sd/Vt are fit on a specific concatenation order and
        # nothing about a [FEAT_DIM] vector reveals which one. A probe trained
        # on [last, mean] per layer has exactly the same shape as one trained
        # on [last] over twice the layers, and mis-ordering the blocks produces
        # finite, plausible, meaningless scores. When the npz declares its
        # order (`blocks`), it must match ours.
        blocks = z["blocks"] if "blocks" in z.files else None
        if blocks is not None:
            got_blocks = tuple(str(b) for b in np.asarray(blocks).reshape(-1))
            if got_blocks != FEAT_BLOCKS:
                raise ValueError(
                    f"probe block order {got_blocks} != router {FEAT_BLOCKS} — "
                    "the feature vector would be assembled in the wrong order")

        v64 = (vt.T @ w[:-1]) / sd            # [FEAT_DIM]
        b64 = float(w[-1] - float(v64 @ mu))
        if not np.isfinite(v64).all() or not np.isfinite(b64):
            raise ValueError("probe fold produced non-finite values")

        # Prove the algebra on this actual probe rather than trusting the
        # derivation: score the unfolded and folded forms on a fixed pseudo-
        # random vector and require agreement. A silent fold error would shift
        # every score by a constant, which looks exactly like a threshold that
        # needs recalibrating.
        rng = np.random.default_rng(0)
        xt = rng.standard_normal(FEAT_DIM)
        ref = float(w[:-1] @ (vt @ ((xt - mu) / sd)) + w[-1])
        got = float(v64 @ xt + b64)
        if abs(ref - got) > 1e-6 * max(1.0, abs(ref)):
            raise ValueError(
                f"probe fold disagrees with the staged form: {got} vs {ref}")

        # ── CANARY BLOCK ───────────────────────────────────────────────────
        # The fold check above proves the folded form agrees with the STAGED
        # form of the SAME file. It cannot detect that the file itself is the
        # wrong file: both forms of a wrong probe agree perfectly. A canary is
        # the trainer's signed claim about what this probe outputs — a
        # feature vector and the score it must produce — re-scored here, on
        # the folded weights, at every load and every hot-reload.
        # Schema (all optional; a probe without them loads and counts
        # `probe_canary_absent`, which is the state every npz on disk is in
        # today):
        #     canary_score : float  — required to activate the check
        #     canary_x     : [FEAT_DIM] float  — the input, or
        #     canary_seed  : int    — np.random.default_rng(seed)
        #                             .standard_normal(FEAT_DIM) instead
        #     canary_tol   : float  — relative tolerance, default 1e-6
        canary = None
        if "canary_score" in z.files:
            want = float(np.asarray(z["canary_score"]).reshape(-1)[0])
            if "canary_x" in z.files:
                cx = np.asarray(z["canary_x"], dtype=np.float64).reshape(-1)
                src = "canary_x"
            elif "canary_seed" in z.files:
                seed = int(np.asarray(z["canary_seed"]).reshape(-1)[0])
                cx = np.random.default_rng(seed).standard_normal(FEAT_DIM)
                src = f"canary_seed={seed}"
            else:
                raise ValueError(
                    "probe declares canary_score but neither canary_x nor "
                    "canary_seed — the claim cannot be checked")
            if cx.size != FEAT_DIM:
                raise ValueError(
                    f"probe canary_x dim {cx.size} != {FEAT_DIM}")
            tol = (float(np.asarray(z["canary_tol"]).reshape(-1)[0])
                   if "canary_tol" in z.files else 1e-6)
            cgot = float(v64 @ cx + b64)
            if abs(cgot - want) > tol * max(1.0, abs(want)):
                raise ValueError(
                    f"probe CANARY FAILED ({src}): scored {cgot!r}, npz claims "
                    f"{want!r} (tol {tol:g}) — this is not the probe the "
                    "trainer validated")
            canary = {"source": src, "want": want, "got": cgot,
                      "resid": abs(cgot - want), "tol": tol}
        else:
            _bump("probe_canary_absent")

        v = torch.from_numpy(np.ascontiguousarray(v64, dtype=np.float32)).to(dev)
        # Health surface: the residual of the check above, on the weights that
        # are actually about to serve. Published rather than only logged —
        # every INFO this module emitted about its own health was discarded
        # for a whole afternoon, which is the entire reason health.json exists.
        self._probe_fold_resid = abs(ref - got)
        self._probe_canary = canary
        self._probe_loads += 1
        logger.info("[PN119] probe folded: %d B on device (was %d B), "
                    "fold check |d|=%.3e canary=%s", v.numel() * 4,
                    (mu.size + sd.size + vt.size + w.size) * 4, abs(ref - got),
                    "ok" if canary else "absent")
        return v, b64

    @staticmethod
    def _content_sig(path: str):
        """(size, sha256) of the probe file — the reload trigger.

        Deliberately NOT (mtime_ns, size). Every mechanism that would put a
        DIFFERENT probe at this path without moving the mtime is a mechanism
        we use: `cp -p` from the staging dir, `rsync -a` from a backup, a
        btrfs snapshot rollback. And the two probes this deployment actually
        alternates between (tap-trained vs offline-trained) have byte-equal
        sizes, so `size` alone catches nothing either. Reading 1.4 MB once per
        PN119_RELOAD_S is ~0.3 ms of a background check.
        """
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return (os.path.getsize(path), h.hexdigest())
        except OSError:
            return None

    def probe_sig_short(self) -> str:
        sig = self._probe_sig
        return sig[1][:12] if sig else "-"

    def _maybe_reload(self) -> None:
        """Pick up an atomically-swapped probe without a restart. The swap
        writer (pn119_atomic) guarantees the file at this name is always a
        complete npz, so the only failure modes here are transient FS errors
        — on ANY failure we keep the current weights and remember the bad
        signature so we don't retry-spin until the file changes again."""
        now = time.time()
        if now < self._next_reload_check:
            return
        self._next_reload_check = now + self._reload_every
        sig = self._content_sig(self._probe_path)
        if sig is None or sig == self._probe_sig or sig == self._failed_sig:
            return
        try:
            new = self._load_probe(self._probe_path)
        except Exception as e:  # noqa: BLE001 — never break serving on a bad swap
            self._failed_sig = sig
            # Counted, not just logged: a refused swap leaves the router
            # serving weights the refit loop believes it replaced, forever and
            # silently. PROBE_CANARY_FAIL reads this counter.
            _bump("probe_reload_failed")
            logger.warning("[PN119] probe hot-reload FAILED (%s) — keeping current weights", e)
            return
        self.pv, self.pb = new
        self._probe_sig = sig
        self._failed_sig = None
        logger.info("[PN119] probe hot-reloaded from %s (size=%d sha256=%s)",
                    self._probe_path, sig[0], sig[1][:12])

    def _is_explore(self, req_id: str) -> bool:
        if self.explore_rate <= 0.0:
            return False
        h = int.from_bytes(hashlib.sha1(req_id.encode()).digest()[:4], "big")
        return (h / 0x100000000) < self.explore_rate

    # ── observation (called from execute_model postprocess) ────────────────
    def observe(self, scheduler_output, aux_hidden_states) -> None:
        try:
            self._maybe_reload()
            self._observe(scheduler_output, aux_hidden_states)
        except Exception as e:  # noqa: BLE001 — never break a serving step
            self._warned += 1
            if self._warned <= 5:
                logger.warning("[PN119] observe error (%d/5 shown): %s", self._warned, e)

    def _observe(self, scheduler_output, aux) -> None:
        if aux is None or len(aux) != len(LAYERS):
            return
        self._step += 1
        runner = self.runner
        # Resolve any score whose D2H was deferred from an EARLIER step. Under
        # normal operation h119_resolve_routes drains within the same step; this
        # is the backstop for a boot with no holder (no reasoning_config) where
        # update_state never runs at all.
        if self._pending:
            self._drain_pending()
        sched = scheduler_output.num_scheduled_tokens  # dict req_id -> n
        # ── step early-out ────────────────────────────────────────────────
        # Nothing can be finalized on a step that admitted no new request and
        # has nothing mid-prefill. The full scan every _fullscan_every steps
        # is the safety valve for the case where that reasoning is wrong.
        new_reqs = getattr(scheduler_output, "scheduled_new_reqs", _MISSING)
        if new_reqs is _MISSING:
            _bump("no_scheduled_new_reqs_attr")
        elif (not new_reqs and not self._acc
                and (self._step - 1) % self._fullscan_every):
            # `_step - 1` so the FIRST step this router ever sees is always a
            # full scan: a router that attached mid-flight has requests in the
            # batch that were never "new" and are not in the accumulator, and
            # making them wait a whole fullscan period to be noticed is the
            # silent-unrouted-request failure this module exists to not have.
            _bump("step_early_out")
            return
        req_ids = list(runner.input_batch.req_ids)
        start = 0
        total = sum(sched.get(r, 0) for r in req_ids)
        # slice padded aux tensors down to real tokens (upstream convention)
        aux = [h[:total] for h in aux]
        for req_id in req_ids:
            n = sched.get(req_id, 0)
            if n <= 0:
                continue
            end = start + n
            try:
                self._observe_one(runner, req_id, n, aux, start)
            finally:
                # `start` MUST advance for every scheduled request whatever
                # happens above, or every later request in the batch reads
                # another request's hidden states.
                start = end
        self._reap()

    def _observe_one(self, runner, req_id, n, aux, start) -> None:
        state = runner.requests.get(req_id)
        if req_id in self.unscored or req_id in self.skipped:
            return                                   # decided, and not scored
        if req_id in self.scored:
            # RE-PREFILL GUARD. A streaming / multi-turn client that reuses one
            # req_id across turns re-prefills a LONGER prompt. The old code saw
            # "already scored" and skipped forever, so turn 1's route governed
            # the whole session and turn N's finish labels were written against
            # turn 1's feature row — a wrong training row, silently.
            plen_now = getattr(state, "num_prompt_tokens", None) if state else None
            if not plen_now or plen_now == self._scored_plen.get(req_id):
                return
            _bump("rescore_reprefill")
            self.scored.pop(req_id, None)
            self._scored_plen.pop(req_id, None)
            SCORES.pop(req_id, None)
            ROUTES.pop(req_id, None)
            self._acc.pop(req_id, None)
        if state is None:
            _bump("skip_no_req_state")
            return
        prompt_len = getattr(state, "num_prompt_tokens", None)
        if not prompt_len:
            _bump("skip_no_prompt_len")
            return
        # `num_computed_tokens` is the engine's PRE-step progress (set in
        # _update_states, which runs before the forward and before this
        # postprocess hook). On a first prefill step it is exactly the
        # APC-cached prefix length: those positions were NOT forwarded.
        base = int(getattr(state, "num_computed_tokens", None) or 0)
        base = min(max(base, 0), prompt_len)
        acc = self._acc.get(req_id)
        if acc is None:
            if base >= prompt_len:
                # The engine says this prompt is fully computed and we have
                # never seen a chunk of it: the last prompt token was forwarded
                # before this router existed (attach mid-flight) or on a step an
                # observe error swallowed. vLLM guarantees the last prompt token
                # is recomputed on a cache hit, so this is NOT the prefix-cache
                # case — it is the only remaining unscoreable case.
                self._unscoreable(req_id, "prefill_not_observed",
                                  prompt_len, prompt_len)
                return
            acc = self._acc[req_id] = {"seen": base, "plen": prompt_len,
                                       "cached": base, "step": self._step}
            if base > 0:
                # Prefix-cache hit. Scoreable now — the mean-pool that used to
                # make it unscoreable is gone.
                _bump("prefill_partial_cached")
        elif base < acc["seen"]:
            # Preemption-recompute: the engine restarted this prefill from a
            # lower offset. Re-derive rather than trust the old progress.
            _bump("acc_reset_recompute")
            acc["seen"] = base
            acc["cached"] = base
        remaining = prompt_len - base
        if remaining <= 0:
            return
        take = min(n, remaining)
        acc["seen"] = base + take
        acc["step"] = self._step
        if acc["seen"] < prompt_len:
            return                                   # chunked prefill, not done
        # The last prompt token sits at slice offset take-1 — the same row the
        # accumulator version used, which is why the last-only feature vector is
        # a strict SUBSET of the old one and needs no new capture plumbing.
        last_rows = [aux[li][start + take - 1].float()
                     for li in range(len(LAYERS))]
        self._finalize(req_id, state, last_rows, prompt_len, acc.get("cached", 0))

    # ── bounded state: one reaper for every per-request map ────────────────
    def _resolve_acc_max(self) -> int:
        """PN119_ACC_MAX, defaulting to 4x the engine's max_num_seqs."""
        try:
            explicit = int(_env("PN119_ACC_MAX", "") or 0)
            if explicit > 0:
                return explicit
        except ValueError:
            logger.warning("[PN119] PN119_ACC_MAX is not an integer — derived")
        seqs = int(getattr(self.runner, "max_num_reqs", 0) or 0)
        if not seqs:
            cfg = getattr(self.runner, "vllm_config", None)
            sch = getattr(cfg, "scheduler_config", None) if cfg else None
            seqs = int(getattr(sch, "max_num_seqs", 0) or 0)
        return max(4 * (seqs or 256), 64)

    def _reap(self) -> None:
        """Evict the oldest entries from every per-request map over the bound.

        These maps are all popped in on_finish, which is driven by
        `scheduler_output.finished_req_ids`. A request that leaves the engine
        without appearing there — an abort racing a shutdown, or anything the
        router saw before it was fully attached — leaks one entry per map for
        the life of the process. Insertion order is arrival order, so the head
        of each dict is the oldest and the least likely to still be live.
        ONE reaper for all of them: the maps are keyed by the same req_id and
        letting them drift apart is how a route outlives its score.
        """
        lim = self._acc_max
        maps = (("acc", self._acc), ("scored", self.scored),
                ("scored_plen", self._scored_plen),
                ("unscored", self.unscored), ("skipped", self.skipped),
                ("h119_applied", self._h119_applied),
                ("routes", ROUTES), ("scores", SCORES))
        for name, m in maps:
            over = len(m) - lim
            if over <= 0:
                continue
            for key in list(itertools.islice(iter(m), over)):
                m.pop(key, None)
                EXPLORE.discard(key)
            _bump(f"reaped_{name}", over)
        over = len(EXPLORE) - lim
        if over > 0:
            for _ in range(over):
                EXPLORE.pop()
            _bump("reaped_explore", over)

    # ── v2 sink plumbing: no disk on the request path ───────────────────────
    # observe()/on_finish() only append to two RAM lists under a lock; a daemon
    # thread does every write()+flush().  Measured on the live btrfs sink dir
    # (2000 rows, 61440 B feature + meta line each; last-only halved the
    #  feature half of that to 30720 B, so these are upper bounds now):
    #   inline write+flush per request   13.66 us mean /  58 us max
    #   inline batch-of-64               (mean drops, but 1.56 ms max — the
    #                                     64th request eats the whole 3.9 MB)
    #   buffered + background flush       0.58 us mean / 8.3 us max
    # The on-disk BYTES are unchanged: feature rows are the same b"".join of
    # the same per-row .tobytes(), meta lines the same json.dumps + "\n", in
    # the same order — see test_pn119_sink_buffer.py B1.
    def _sink_start(self) -> None:
        """(Re)arm the flusher for the CURRENT process."""
        self._sink_pid = os.getpid()
        self._sink_stop = False
        if self._buf_rows <= 0 and not self.health_path:
            self._sink_thread = None    # PN119_SINK_BUF_ROWS=0 => sync escape
            return
        # With the sync escape AND a health path, the thread still runs: it is
        # the health publisher, and _sink_flush on two empty buffers is two
        # uncontended lock acquisitions. Health is not worth a SECOND thread,
        # but it is worth this one.
        self._sink_thread = threading.Thread(
            target=self._sink_loop, name="pn119-sink", daemon=True)
        self._sink_thread.start()

    def _sink_header(self) -> None:
        """First meta line of every sink file: who wrote it.

        Written DIRECTLY (not through the buffer) so it is on disk before the
        first request, and so a file that ends up with exactly one line is
        unambiguous. Before this, a 0-byte meta-*.jsonl could mean the tap
        never fired, no traffic arrived, or the router never started — twenty
        of the forty files in the 2026-07-25 sink are 0 bytes and none of them
        can be attributed to a boot after the fact. refit_pn119_probe's
        load_sink ignores the line: it keys score rows on "row" and label rows
        on "finish", and skips the whole pair when feats-*.bin is empty.
        """
        try:
            if self._sink_meta is None:
                return
            self._sink_meta.write(json.dumps({
                "pn119_header": 1, "boot_id": self.boot_id, "pid": os.getpid(),
                "hostname": self.hostname, "ts": self.started,
                "mode": self.mode, "tdeep": self.tdeep,
                "probe": os.path.basename(self._probe_path),
                "fallback_route": self.fallback_route,
                "explore_rate": self.explore_rate,
                "consumer_flag": _truthy(_env(_CONSUMER_FLAG, "0")),
                # A feats-*.bin row is FEAT_DIM bf16 values and nothing on
                # disk says which FEAT_DIM. The 30720-dim and 15360-dim eras
                # produce files that differ only in length, so a reader that
                # guesses wrong silently reinterprets every row.
                "feat_dim": FEAT_DIM,
                "blocks": list(FEAT_BLOCKS),
                "probe_sig": self.probe_sig_short(),
                "think_start_ids": self._think_start_ids,
                "think_end_ids": self._think_end_ids,
                "censor_slack": self._censor_slack,
            }) + "\n")
            self._sink_meta.flush()
        except Exception as e:  # noqa: BLE001 — a header is never worth a boot
            logger.warning("[PN119] sink header failed (%s)", e)

    def _sink_loop(self) -> None:
        while not self._sink_stop:
            self._sink_wake.wait(self._buf_secs)
            self._sink_wake.clear()
            if self._sink_meta is not None:
                self._sink_flush()
            elif not self.health_path:
                return                  # sink closed and nothing else to do
            # Health rides this tick: the thread is already awake and the
            # request path stays untouched. See §HEALTH SURFACE.
            self._health_write()

    def _sink_append(self, feat: bytes | None, line: str) -> None:
        """Queue one sink event.  Never raises; never touches the disk unless
        the hard cap is hit, where blocking this one request beats dropping
        training data on the floor."""
        try:
            if self._sink_meta is None:
                return
            if self._sink_pid != os.getpid():
                # Forked: the flusher thread did not come across.  Re-arm here
                # rather than let the buffer grow to the cap every time.
                self._sink_start()
            with self._sink_lock:
                if feat is not None:
                    self._sbuf_feat.append(feat)
                self._sbuf_meta.append(line)
                n = len(self._sbuf_meta)
            if self._buf_rows <= 0 or n >= self._buf_max:
                self._sink_flush()      # sync mode, or hard-cap backpressure
            elif n >= self._buf_rows:
                self._sink_wake.set()
        except Exception as e:  # noqa: BLE001 — any sink failure disables it
            logger.warning("[PN119] sink append failed (%s) — disabling sink", e)
            self._sink_feat = self._sink_meta = None

    def _sink_flush(self) -> None:
        """Drain both buffers to disk.

        The snapshot is taken INSIDE the io lock: meta "row" values are
        positional indices into feats-*.bin, so two concurrent flushers that
        snapshotted in one order and wrote in the other would silently
        mis-align every row after them.  The buffer lock is held only for the
        swap, so an appending request never waits on IO.
        """
        with self._sink_io_lock:
            with self._sink_lock:
                if not self._sbuf_meta and not self._sbuf_feat:
                    return
                feats = b"".join(self._sbuf_feat)
                lines = "".join(self._sbuf_meta)
                self._sbuf_feat.clear()
                self._sbuf_meta.clear()
            f, m = self._sink_feat, self._sink_meta
            if m is None:
                return
            try:
                # Feature rows FIRST: the only torn state this can leave is
                # rows with no meta line, which load_sink cannot even see (it
                # keys on "row").  The reverse leaves dangling indices that
                # only load_sink's ridx bound check catches.
                if feats and f is not None:
                    f.write(feats)
                    f.flush()
                m.write(lines)
                m.flush()
            except Exception as e:  # noqa: BLE001 — any sink failure disables it
                logger.warning("[PN119] sink write failed (%s) — disabling sink", e)
                self._sink_feat = self._sink_meta = None

    def _sink_close(self) -> None:
        """Stop the flusher and drain.  A clean shutdown loses NOTHING."""
        self._sink_stop = True
        self._sink_wake.set()
        t = self._sink_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._sink_thread = None
        self._sink_flush()
        for h in (self._sink_feat, self._sink_meta):
            try:
                if h is not None:
                    h.close()
            except Exception:  # noqa: BLE001
                pass
        self._sink_feat = self._sink_meta = None
        # Final publish: a health file left behind by a CLEAN shutdown says so,
        # so the doctor can distinguish "the container was stopped" from "the
        # worker died mid-step", which look identical through a stale file.
        self._health_write(shutdown=True, force=True)

    # ── health surface: published from the flusher, never from a request ────
    def health_snapshot(self, *, present=None, shutdown=False) -> dict:
        """Everything a reader needs to judge this boot, as a plain dict."""
        probe = {
            "path": self._probe_path,
            "basename": os.path.basename(self._probe_path),
            "loads": self._probe_loads,
            "fold_resid": self._probe_fold_resid,
            "canary": self._probe_canary,
            "feat_dim": FEAT_DIM,
            "blocks": list(FEAT_BLOCKS),
            "readable": os.path.isfile(self._probe_path),
        }
        sig = self._probe_sig
        if sig:
            probe["size"], probe["sha256"] = sig
        cs = dict(_consumer_state)
        cs.pop("warned", None)
        consumer = {
            "flag_env": _truthy(_env(_CONSUMER_FLAG, "0")),
            "checked": bool(_consumer_state.get("checked")),
            "on": bool(_consumer_state.get("on")),
            "deep_budget": _consumer_state.get("deep"),
            "lean_budget": _consumer_state.get("lean"),
            "override_pn100": bool(_consumer_state.get("override_pn100")),
            "warned": int(_consumer_state.get("warned", 0)),
        }
        return make_snapshot(
            stats=STATS, boot_id=self.boot_id, pid=os.getpid(),
            hostname=self.hostname, started=self.started,
            mode=self.mode, mode_requested=self.mode_requested,
            router_present=(ROUTER is self) if present is None else present,
            router_enabled=_truthy(_env("GENESIS_ENABLE_PN119_ROUTER")),
            tdeep=self.tdeep, fallback_route=self.fallback_route,
            fallback_requested=self.fallback_requested,
            explore_rate=self.explore_rate, probe=probe, consumer=consumer,
            first_scored_ts=self._h_first_scored_ts,
            last_scored_ts=self._h_last_scored_ts,
            last_decision_ts=self._h_last_decision_ts,
            sink={
                "dir": self.sink_dir,
                "enabled": self._sink_meta is not None,
                "rows": self._sink_rows,
                "buf_rows": self._buf_rows, "buf_secs": self._buf_secs,
                "buf_max": self._buf_max,
                "pending": len(self._sbuf_meta),
                "thread": self._sink_thread is not None,
            },
            extra={"shutdown": bool(shutdown),
                   "health_path": self.health_path,
                   "warned": self._warned,
                   "inflight": {"acc": len(self._acc),
                                "scored": len(self.scored),
                                "unscored": len(self.unscored),
                                "skipped": len(self.skipped),
                                "routes": len(ROUTES),
                                "pending_async": len(self._pending),
                                "max": self._acc_max},
                   "async_score": {"requested": self._async_want,
                                   "ready": self._async_ready,
                                   "slots_free": len(self._free_slots)},
                   "think_markers": {"start": self._think_start_ids,
                                     "end": self._think_end_ids,
                                     "censor_slack": self._censor_slack}},
        )

    def _health_write(self, shutdown: bool = False, force: bool = False) -> None:
        """Publish health.json ATOMICALLY. Never raises, never blocks a request.

        Called only from the flusher thread (plus once at init and once at
        shutdown), so the request path pays nothing. os.replace on the same
        filesystem is atomic, which is what lets pn119-doctor read the file
        with no lock and never see a torn document.
        """
        if not self.health_path or self._health_fails > 5:
            return
        try:
            now = time.time()
            # Cheap change detector: STATS is monotonic, so the sum plus the
            # key count moves whenever anything at all happened.
            fp = (len(STATS), sum(STATS.values()))
            if not (force or shutdown or fp != self._h_prev_fp
                    or now >= self._h_next_beat):
                return
            self._h_prev_fp = fp
            self._h_next_beat = now + self._health_every
            # Derive the traffic timestamps from counter movement — the
            # alternative is a time.time() per request, which this module
            # does not get to spend.
            scored = int(STATS.get("scored", 0))
            decisions = (scored + int(STATS.get("scored_unknown", 0))
                         + int(STATS.get("unscoreable", 0))
                         + int(STATS.get("skip_thinking_off", 0)))
            if scored > self._h_prev_scored:
                if self._h_first_scored_ts is None:
                    self._h_first_scored_ts = now
                self._h_last_scored_ts = now
                self._h_prev_scored = scored
            if decisions > self._h_prev_decisions:
                self._h_last_decision_ts = now
                self._h_prev_decisions = decisions
            snap = self.health_snapshot(shutdown=shutdown)
            tmp = f"{self.health_path}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snap, f, indent=1, sort_keys=False, default=str)
                f.write("\n")
                f.flush()
            os.replace(tmp, self.health_path)
            self._health_fails = 0
        except Exception as e:  # noqa: BLE001 — health must never cost a request
            self._health_fails += 1
            if self._health_fails <= 3:
                logger.warning("[PN119] health write failed (%d/6): %s",
                               self._health_fails, e)
            if self._health_fails > 5:
                logger.warning("[PN119] health writes disabled after 6 failures "
                               "— the router keeps serving, blind")

    # ── explicit fallback for requests that cannot be scored ────────────────
    def _unscoreable(self, req_id, reason: str, missing: int, prompt_len: int) -> None:
        """The one thing that must never happen is a silent pass-through."""
        route = self.fallback_route
        self.unscored[req_id] = reason
        self._acc.pop(req_id, None)
        _bump("unscoreable")
        _bump(f"unscoreable_{reason}")
        _bump(f"fallback_{route}")
        self._decisions += 1
        explore = self._is_explore(req_id)
        logger.warning(
            "[PN119] req=%s UNSCOREABLE reason=%s missing=%d/%d prompt_tok=%d "
            "-> FALLBACK route=%s mode=%s explore=%s | %s",
            req_id, reason, missing, prompt_len, prompt_len, route, self.mode,
            explore, stats_line())
        if self.mode == "enforce":
            ROUTES[req_id] = route
            # Compatibility shim: consumers still reading SCORES get a value
            # that lands on the intended side of TDEEP. route_for() is the
            # contract; this only keeps a legacy reader from seeing "absent".
            SCORES[req_id] = self.tdeep if route == ROUTE_DEEP else self.tdeep - 1.0
            if explore:
                EXPLORE.add(req_id)
        if self._sink_meta is not None:
            # NO "row" key and NO feature write: there is no valid feature
            # vector, and refit_pn119_probe.load_sink keys score lines on
            # "row", so this line can never become training data.
            self._sink_append(None, json.dumps({
                "req_id": req_id, "unscoreable": True, "reason": reason,
                "missing": missing, "route": route, "prompt_tok": prompt_len,
                "ts": time.time(), "mode": self.mode, "explore": explore,
            }) + "\n")

    # ── the thinking-ON gate (module docstring §THINKING-ON GATE) ──────────
    def _resolve_think_markers(self):
        """Reconcile PN119_THINK_*_ID with the holder's marker SEQUENCES.

        The sidecar was configured with two single token ids; the component
        that actually forces `</think>` reads
        `reasoning_config.reasoning_start_token_ids` / `…end_token_ids`, which
        `initialize_token_ids` fills from `tokenizer.encode()` and which are
        therefore LISTS of arbitrary length. Nothing checked that the two
        agreed, and a disagreement is invisible: the sink would simply label
        every row `thinking=None` (or, worse, `False`), the refit would train
        on the wrong subset, and the health surface would report a happy boot.

        On disagreement the HOLDER wins — it is the thing whose behaviour the
        labels are meant to describe — and the divergence is counted so the
        env that lied is fixed rather than silently overridden forever.
        """
        # Defaults are the thinkingcap-gptq-pro-v2 / Qwen3.6 ids, verified in
        # the served tokenizer.json: <think> = 248068, </think> = 248069, both
        # single tokens. Keeping them as the ENV DEFAULT (rather than deriving
        # solely from the holder) means a boot with no reasoning_config still
        # labels correctly instead of labelling nothing.
        env_start = _env("PN119_THINK_START_ID", "248068")
        env_end = _env("PN119_THINK_END_ID", "248069")
        try:
            start = [int(env_start)]
            end = [int(env_end)]
        except ValueError:
            logger.error("[PN119] PN119_THINK_*_ID not integers (%r/%r) — "
                         "using 248068/248069", env_start, env_end)
            start, end = [248068], [248069]
        held_start = held_end = None
        try:
            cfg = getattr(self.runner, "vllm_config", None)
            rc = getattr(cfg, "reasoning_config", None) if cfg else None
            if rc is not None:
                held_start = list(getattr(rc, "reasoning_start_token_ids", None) or [])
                held_end = list(getattr(rc, "reasoning_end_token_ids", None) or [])
        except Exception as e:  # noqa: BLE001 — never brick load over a lookup
            logger.warning("[PN119] reasoning_config unreadable (%s)", e)
        if not held_start and not held_end:
            # No holder to compare against (no --reasoning-parser, or a runner
            # that does not carry vllm_config). Keep the env ids and say so:
            # "unverified" is a different claim from "verified equal".
            _bump("think_markers_unverified")
            logger.warning("[PN119] reasoning_config has no marker ids — using "
                           "PN119_THINK_*_ID unverified: start=%s end=%s",
                           start, end)
            return start, end
        if start == held_start and end == held_end:
            logger.info("[PN119] think markers agree with reasoning_config: "
                        "start=%s end=%s", held_start, held_end)
            return held_start, held_end
        if not (held_start and held_end):
            # Only ONE side is populated. Adopting that leaves the scan with an
            # empty pattern on the other side, which silently makes every
            # prompt read thinking-off — worse than the divergence itself.
            # Keep the env, alarm, say why.
            _bump("think_marker_divergence")
            logger.error("[PN119] reasoning_config has a PARTIAL marker pair "
                         "(start=%s end=%s) — keeping PN119_THINK_*_ID "
                         "start=%s end=%s; an empty pattern would label every "
                         "prompt thinking-off", held_start, held_end, start, end)
            return start, end
        _bump("think_marker_divergence")
        logger.error(
            "[PN119] THINK MARKER DIVERGENCE — env says start=%s end=%s, the "
            "holder uses start=%s end=%s. ADOPTING THE HOLDER'S: the holder is "
            "what forces </think>, so it defines what a thinking row IS. Every "
            "label written under the old env values is suspect.",
            start, end, held_start, held_end)
        return held_start, held_end

    @staticmethod
    def _last_subseq(seq, pat) -> int:
        """Index of the LAST occurrence of `pat` in `seq`, or -1.

        Same shape as the holder's `_find_last_sequence_index`, deliberately:
        the two must agree about where a think region opens and closes or the
        labels describe a different model than the one being served.
        """
        if not pat or len(pat) > len(seq):
            return -1
        for i in range(len(seq) - len(pat), -1, -1):
            if seq[i:i + len(pat)] == pat:
                return i
        return -1

    def _prompt_thinking(self, req_state):
        """True / False / None — does this PROMPT open a think region?

        Hoisted out of `_label_fields` so `_finalize` can consult it BEFORE
        spending a matvec and a 30,720-byte sink row on a request the router
        has no lever on. Reads only `prompt_token_ids`, which the caller has
        in hand.
        """
        prompt_ids = getattr(req_state, "prompt_token_ids", None)
        if not prompt_ids:
            return None
        # LAST marker wins, not "any </think> anywhere in the window".
        # thinking-off pre-closes the region, so the tail holds BOTH markers
        # and </think> is last; thinking-on with a PN102 seed holds only
        # <think>, several seed tokens back from the end.
        tail = list(prompt_ids[-self._tail_window:])
        last_end = self._last_subseq(tail, self._think_end_ids)
        last_start = self._last_subseq(tail, self._think_start_ids)
        if last_start >= 0 and last_start > last_end:
            return True
        if last_end >= 0:
            return False
        return None

    # ── budget provenance (BUG-139) ───────────────────────────────────────
    def _budget_fields(self, req_id, req_state):
        """(budget_grant, budget_source) — the EFFECTIVE thinking budget.

        `sampling_params.thinking_token_budget` is what the frontend asked
        for. It is NOT what the request ran under when H119's consumer took
        the row over: the consumer rewrites the holder's live state entry and
        never touches SamplingParams, so reading the params alone reports a
        grant the request never had. `_h119_applied` is the consumer's own
        record and wins when present.
        """
        sp = getattr(req_state, "sampling_params", None)
        budget = getattr(sp, "thinking_token_budget", None) if sp is not None else None
        applied = self._h119_applied.get(req_id)
        if applied is not None:
            return int(applied[0]), applied[1]
        if budget is None:
            return None, "none"
        stamp = _h119_overridable_stamp(sp)
        # stamp == 1 means PN100 chose this budget and marked it overridable.
        # stamp == 0 means PN100 deliberately kept out — which it does for a
        # client-pinned numeric, so the budget on the params is the caller's.
        return int(budget), ("pn100" if stamp == 1 else "caller")

    @staticmethod
    def _extra_arg(sp, *names):
        """First present of `names` in SamplingParams.extra_args, else None.

        `caller` / `suite` are not set by anything today; this is the plumbing
        so a bench harness can stamp a run through `vllm_xargs` and have every
        sink row carry it. Wire data: never raise, never coerce.
        """
        try:
            xargs = getattr(sp, "extra_args", None)
            if not xargs:
                return None
            for n in names:
                v = xargs.get(n)
                if v is not None:
                    return v if isinstance(v, (int, float)) else str(v)[:64]
        except (AttributeError, TypeError, ValueError):
            return None
        return None

    def _lane_key(self, route: str, explore: bool) -> str:
        """The treatment lane this row belongs to, as one joinable string.

        A sink row is only comparable to another row that got the same
        TREATMENT, and the treatment is (mode, route, explore, consumer-on) —
        four fields that were previously spread across three files and a boot
        log. Analyses that joined on `mode` alone silently mixed a shadow row
        with an enforce row whose budget had been rewritten.
        """
        return "{}:{}:{}:{}".format(
            self.mode, route, "x" if explore else "-",
            "c" if _consumer_state.get("on") else "-")

    def _pctl(self, score: float) -> float:
        """Percentile of `score` within a bounded window of RECENT routable
        scores. The threshold is absolute, so an operator cannot tell from a
        score alone whether it sat at the edge of the distribution or in the
        bulk; that is exactly the question a retune asks.
        """
        import bisect
        win, srt = self._score_win, self._score_sorted
        pos = bisect.bisect_right(srt, score)
        pctl = pos / len(srt) if srt else 0.5
        bisect.insort(srt, score)
        win.append(score)
        while len(win) > self._pctl_window:
            old = win.popleft()
            i = bisect.bisect_left(srt, old)
            if i < len(srt) and srt[i] == old:
                srt.pop(i)
        return pctl

    # ── deferred score readback (PN119_ASYNC_SCORE) ────────────────────────
    def _async_init(self) -> bool:
        """Allocate the pinned staging buffers once. False = stay synchronous."""
        if self._async_ready:
            return True
        if not self._async_want:
            return False
        try:
            dev = getattr(self.pv, "device", None)
            if dev is None or dev.type != "cuda" or not torch.cuda.is_available():
                self._async_want = False
                logger.info("[PN119] PN119_ASYNC_SCORE ignored — probe is not "
                            "on a CUDA device; scoring stays synchronous")
                return False
            slots = self._async_slots or max(min(self._acc_max, 128), 8)
            self._pin_score = torch.empty(slots, dtype=torch.float32,
                                          pin_memory=True)
            if self._sink_feat is not None:
                # 15360 dims x 2 B x slots — 3.9 MB at 128 slots. Halved by
                # last-only; this is the whole reason deferring the feature
                # copy is affordable at all.
                self._pin_feat = torch.empty(slots, FEAT_DIM,
                                             dtype=torch.uint16, pin_memory=True)
            self._free_slots = list(range(slots))
            self._async_ready = True
            logger.info("[PN119] async score readback ON: %d pinned slots "
                        "(%d B score + %d B feat)", slots, slots * 4,
                        0 if self._pin_feat is None else slots * FEAT_DIM * 2)
            return True
        except Exception as e:  # noqa: BLE001 — never fail a step over an optimisation
            self._async_want = False
            self._async_ready = False
            logger.warning("[PN119] async score init failed (%s) — synchronous", e)
            return False

    def _drain_pending(self) -> None:
        """Resolve every deferred readback whose copy has landed.

        Called from `h119_resolve_routes` (sampler time, SAME engine step as
        the observe() that queued it) and, as a backstop, from the top of the
        next observe(). Both are the runner thread — the flusher must never
        touch CUDA. `event.synchronize()` on a 60 kB copy issued a forward ago
        is expected to be already complete; the call is there for correctness,
        not for waiting.
        """
        if not self._pending:
            return
        pend, self._pending = self._pending, []
        for p in pend:
            try:
                ev = p.get("event")
                if ev is not None:
                    ev.synchronize()
                score = float(self._pin_score[p["slot"]]) + self.pb
                feat = None
                if p["want_feat"] and self._pin_feat is not None:
                    feat = self._pin_feat[p["slot"]].numpy().tobytes()
            except Exception as e:  # noqa: BLE001
                _bump("async_resolve_failed")
                self._warned += 1
                if self._warned <= 5:
                    logger.warning("[PN119] async resolve failed (%s) — request "
                                   "%s left to the fallback route", e, p["req_id"])
                self._free_slots.append(p["slot"])
                continue
            self._free_slots.append(p["slot"])
            self._publish(p["req_id"], score, p["prompt_len"], p["cached"],
                          p["thinking"], p["budget"], p["source"],
                          p["caller"], p["suite"], feat)

    def _finalize(self, req_id, state, last_rows, prompt_len, cached) -> None:
        # ── the thinking-ON gate ──────────────────────────────────────────
        thinking = self._prompt_thinking(state)
        if thinking is False:
            # The prompt pre-closed </think>: this request will spend zero
            # reasoning tokens whatever the probe says. Scoring it would buy a
            # decision with no lever, a 30,720-byte sink row that can never be
            # training data, and a place in every rate denominator. Publish the
            # cheap side and stop.
            self._skip_thinking_off(req_id, state, prompt_len)
            return
        x = torch.cat(last_rows)  # [FEAT_DIM] f32 on device
        sp = getattr(state, "sampling_params", None)
        budget, source = self._budget_fields(req_id, state)
        caller = self._extra_arg(sp, "caller", "x_caller")
        suite = self._extra_arg(sp, "suite", "bench_suite")
        want_feat = self._sink_feat is not None
        if self._async_init() and self._free_slots:
            # DEFERRED PATH. `float(torch.dot(...))` is `.item()` in disguise:
            # a full device sync in the middle of a step, on a stack vLLM keeps
            # sync-free everywhere else. Copy into pinned host memory, record an
            # event, and read it back at the last responsible moment — which is
            # still inside this same engine step (h119_resolve_routes runs at
            # sampler time, after this postprocess), so the route is on record
            # before the sampler that emits the first token. Semantics identical,
            # sync removed.
            try:
                slot = self._free_slots.pop()
                s = torch.dot(self.pv, x)
                self._pin_score[slot].copy_(s, non_blocking=True)
                if want_feat and self._pin_feat is not None:
                    self._pin_feat[slot].copy_(
                        x.to(torch.bfloat16).view(torch.uint16),
                        non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()
                self._pending.append({
                    "req_id": req_id, "slot": slot, "event": ev,
                    "prompt_len": prompt_len, "cached": cached,
                    "thinking": thinking, "budget": budget, "source": source,
                    "caller": caller, "suite": suite,
                    "want_feat": want_feat and self._pin_feat is not None,
                })
                return
            except Exception as e:  # noqa: BLE001 — fall back, never fail
                _bump("async_enqueue_failed")
                if self._warned <= 5:
                    self._warned += 1
                    logger.warning("[PN119] async enqueue failed (%s) — "
                                   "synchronous readback", e)
        _bump("sync_fallback_used")
        # One fused dot against the folded probe (see _load_probe): the
        # centre/scale/PCA/readout chain is pre-collapsed at load, so the whole
        # per-request decision is a single 15,360-term reduction on device.
        score = float(torch.dot(self.pv, x)) + self.pb
        feat = None
        if want_feat:
            try:
                # bf16 has no numpy dtype — reinterpret as uint16 for the raw
                # write (reader: np.fromfile(uint16).view via torch bf16).
                # `x` is already the concatenation of `last_rows` and is
                # contiguous, so reuse it rather than building a second copy.
                feat = (x.to(torch.bfloat16)
                        .view(torch.uint16).cpu().numpy().tobytes())
            except Exception as e:  # noqa: BLE001 — any sink failure disables it
                logger.warning("[PN119] sink encode failed (%s) — disabling sink", e)
                self._sink_feat = self._sink_meta = None
        self._publish(req_id, score, prompt_len, cached, thinking, budget,
                      source, caller, suite, feat)

    def _skip_thinking_off(self, req_id, state, prompt_len) -> None:
        """Thinking-OFF: route lean, no matvec, no feature row, out of the rates."""
        self.skipped[req_id] = "thinking_off"
        self._acc.pop(req_id, None)
        _bump("skip_thinking_off")
        self._decisions += 1
        explore = self._is_explore(req_id)
        if self.mode == "enforce":
            ROUTES[req_id] = ROUTE_LEAN
            # Compatibility shim for legacy SCORES readers: a value on the
            # lean side of tdeep. route_for() is the contract.
            SCORES[req_id] = self.tdeep - 1.0
            if explore:
                EXPLORE.add(req_id)
        if self._sink_meta is not None:
            budget, source = self._budget_fields(req_id, state)
            self._sink_append(None, json.dumps({
                "req_id": req_id, "routable": False, "reason": "thinking_off",
                "route": ROUTE_LEAN, "prompt_tok": prompt_len,
                "ts": time.time(), "mode": self.mode, "explore": explore,
                "budget_grant": budget, "budget_source": source,
                "lane_key": self._lane_key(ROUTE_LEAN, explore),
                "boot_id": self.boot_id, "pid": self._sink_pid,
                "probe_sig": self.probe_sig_short(),
            }) + "\n")

    def _publish(self, req_id, score, prompt_len, cached, thinking, budget,
                 source, caller, suite, feat) -> None:
        """Everything that happens once a score EXISTS — sync or deferred."""
        self.scored[req_id] = score
        self._scored_plen[req_id] = prompt_len
        route = ROUTE_DEEP if score >= self.tdeep else ROUTE_LEAN
        explore = self._is_explore(req_id)
        routable = bool(thinking)
        if routable:
            _bump("scored")
            _bump(f"scored_{route}")
        else:
            # thinking is None: a raw / completion prompt with no marker
            # either way. Still routed — the treatment might matter — but it
            # is NOT evidence about the router's design point, so it stays out
            # of deep_frac and out of every percentile.
            _bump("scored_unknown")
        if cached:
            _bump("scored_prefix_cached")
        self._decisions += 1
        pctl = self._pctl(score) if routable else None
        if logger.isEnabledFor(logging.DEBUG):
            # DEBUG, not INFO: at 61 requests these two lines were noise, at
            # bench rates they are the module's dominant log volume and they
            # say nothing an operator acts on. The periodic rollup below is
            # the line worth reading.
            logger.debug("[PN119] req=%s score=%.4f route=%s routable=%s "
                         "explore=%s prompt_tok=%d cached_prefix=%d mode=%s",
                         req_id, score, route, routable, explore, prompt_len,
                         cached, self.mode)
        if self._decisions % self._stats_every == 0:
            logger.info("[PN119] stats: %s", stats_line())
        if self.mode == "enforce":
            ROUTES[req_id] = route
            SCORES[req_id] = score
            if explore:
                # Consumer contract: explore requests get generous caps
                # regardless of score (keeps v2 labels uncensored).
                EXPLORE.add(req_id)
        if self._sink_meta is not None:
            self._sink_append(feat, json.dumps({
                "req_id": req_id,
                # "row" is present ONLY when a feature row was written —
                # load_sink keys training rows on it.
                **({"row": self._sink_rows} if feat is not None else {}),
                "score": score, "route": route, "prompt_tok": prompt_len,
                "ts": time.time(), "mode": self.mode, "explore": explore,
                "routable": routable if thinking is not None else None,
                "pctl": pctl, "T_used": self.tdeep,
                "probe_sig": self.probe_sig_short(),
                "lane_key": self._lane_key(route, explore),
                "boot_id": self.boot_id, "pid": self._sink_pid,
                "budget_grant": budget, "budget_source": source,
                "caller": caller, "suite": suite,
                "cached_prefix": cached,
            }) + "\n")
            if feat is not None:
                self._sink_rows += 1

    # ── finish sink (called from _update_states removal loop) ──────────────
    def on_finish(self, req_id, req_state) -> None:
        try:
            self._acc.pop(req_id, None)
            score = self.scored.pop(req_id, None)
            self._scored_plen.pop(req_id, None)
            reason = self.unscored.pop(req_id, None)
            skipped = self.skipped.pop(req_id, None)
            applied = self._h119_applied.pop(req_id, None)
            SCORES.pop(req_id, None)
            ROUTES.pop(req_id, None)
            EXPLORE.discard(req_id)
            if reason is not None or skipped is not None:
                # Fallback-routed or deliberately-skipped: no score, but the
                # finish MUST still be visible. No "row" key => load_sink never
                # joins it into a training Row (its features do not exist).
                if self._sink_meta is not None:
                    self._sink_append(None, json.dumps({
                        "req_id": req_id, "finish": True,
                        "unscoreable": reason is not None,
                        "routable": None if reason is not None else False,
                        "reason": reason or skipped, "ts": time.time(),
                        "mode": self.mode, "boot_id": self.boot_id,
                    }) + "\n")
                return
            if score is None or req_state is None:
                return
            out_ids = getattr(req_state, "output_token_ids", None)
            # `num_computed_tokens - num_prompt_tokens` was wrong under MTP:
            # the computed counter moves in accept/reject JUMPS (a rejected
            # draft still advanced it, an accepted batch advances it by more
            # than one), so `generated` drifted from the token count the label
            # is actually about. The output list is the tokens the request
            # kept, which is the definition.
            generated = len(out_ids) if out_ids is not None else 0
            if applied is not None:
                budget, source = int(applied[0]), applied[1]
            else:
                budget, source = self._budget_fields(req_id, req_state)
            thinking, rtok, cap_hit, censored = self._label_fields(
                req_state, generated, budget)
            if thinking:
                _bump("finish_thinking")
                if censored:
                    _bump("finish_censored")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[PN119] finish req=%s score=%.4f generated=%d "
                             "thinking=%s rtok=%s cap_hit=%s censored=%s "
                             "budget=%s(%s)", req_id, score, generated,
                             thinking, rtok, cap_hit, censored, budget, source)
            if self._sink_meta is not None:
                self._sink_append(None, json.dumps({
                    "req_id": req_id, "finish": True, "score": score,
                    "generated": generated, "ts": time.time(),
                    "thinking": thinking, "rtok": rtok, "cap_hit": cap_hit,
                    "censored": censored, "budget_grant": budget,
                    "budget_source": source,
                    "routable": thinking if thinking is not None else None,
                    "explore": self._is_explore(req_id), "mode": self.mode,
                    "boot_id": self.boot_id,
                }) + "\n")
        except Exception as e:  # noqa: BLE001
            self._warned += 1
            if self._warned <= 5:
                logger.warning("[PN119] finish error: %s", e)

    def _label_fields(self, req_state, generated: int, budget=None):
        """v2 label plumbing (BUILD-PACK §v2 items 3+4) + BUG-139 censoring.

        thinking: True  = prompt tail opens a <think> region (spend signal
                          exists — the ONLY rows the refit may learn from);
                  False = template pre-closed </think> (thinking-off);
                  None  = neither marker found (raw/completion prompt) —
                          refit treats unknown as ineligible.
        rtok:     tokens generated BEFORE </think> = true thinking spend
                  (better label than total `generated`, which includes the
                  answer). Falls back to `generated` when </think> never
                  appeared (capped inside the think region).
        cap_hit:  UNCHANGED, for compatibility with the sinks already on disk:
                  </think> never emitted while thinking, or generated >=
                  max_tokens. It is a poor censoring signal — it fired on 4 of
                  79 thinking rows while 43 were truncated — because
                  max_tokens is not the cap that binds. `censored` is.
        censored: BUG-139. The request was STOPPED at its thinking budget:
                  rtok landed within SLACK of the grant, which is the holder
                  forcing `</think>`. rtok is then a LOWER BOUND, not a spend,
                  and any fit that reads it as a spend inherits a ceiling.
                  Requires a known budget — with none there is nothing that
                  could have truncated it here.
        """
        thinking = self._prompt_thinking(req_state)
        rtok = None
        cap_hit = False
        censored = False
        out_ids = getattr(req_state, "output_token_ids", None)
        if thinking and out_ids is not None:
            seq = list(out_ids)
            # FIRST occurrence, not last: the spend ends at the first
            # `</think>` the model emits. A later one belongs to quoted text
            # in the answer and would inflate rtok by the whole answer.
            idx = -1
            pat = self._think_end_ids
            for i in range(0, len(seq) - len(pat) + 1) if pat else ():
                if seq[i:i + len(pat)] == pat:
                    idx = i
                    break
            if idx >= 0:
                rtok = idx
            else:
                rtok = generated      # never closed the think region
                # An EMPTY output never closed anything either, but that is a
                # request that produced nothing — an abort, a zero max_tokens,
                # a client disconnect. Calling it a cap hit put a y=1 censoring
                # label on a row with no evidence in it at all.
                cap_hit = bool(seq)
        elif thinking:
            rtok = generated
        sp = getattr(req_state, "sampling_params", None)
        max_tokens = getattr(sp, "max_tokens", None) if sp is not None else None
        if max_tokens and generated >= max_tokens:
            cap_hit = True
        if thinking and budget and rtok is not None:
            censored = rtok >= int(budget) - self._censor_slack
        return thinking, rtok, cap_hit, censored


# ═══════════════════════════════════════════════════════════════════════════
# H119 ENFORCE ROUTE CONSUMER — the BUDGET-CAP half
# ═══════════════════════════════════════════════════════════════════════════
# Added 2026-07-25, default OFF behind GENESIS_ENABLE_H119_ROUTE_BUDGET=1.
#
# WHY HERE AND NOWHERE ELSE. The module docstring records (and
# fixes/test_h119_route_consumer_timing.py asserts from source) that the
# frontend PN100 site cannot host this: wrong time, no req_id yet, and — the
# fatal one — the wrong PROCESS, since AsyncLLM always builds an out-of-process
# EngineCore client, so ROUTES would read {} forever. This consumer runs inside
# ``ThinkingBudgetStateHolder`` in the ENGINE/worker process, which is the same
# process that imports this module and mutates ROUTES, and it keys off
# ``runner.input_batch.req_ids`` — the same req_id namespace observe() writes.
#
# THE TIMING, per engine step (verified against gpu_model_runner.py on both
# pinned images):
#   1. _update_states -> input_batch.refresh_metadata()
#        -> holder.sync_batch(batch_update)      <- h119_on_batch_add() here
#        -> _make_sampling_metadata()            <- reads has_tracked_requests()
#   2. execute_model: model forward -> aux unpack -> router.observe(...)
#        -> ROUTES[req_id] is written HERE, at the end of the prefill step
#   3. sample_tokens -> _sample -> Sampler.forward
#        -> holder.update_state(...)             <- h119_resolve_routes() here
#        -> holder.apply_to_logits(...)
# So a request's route is on record BEFORE the sampler that emits its FIRST
# token runs. think_count is still 0 at that point: lowering the cap binds in
# full, no thinking token has escaped it.
#
# WHY A PROVISIONAL ENTRY IS CREATED AT STEP 1. `_make_sampling_metadata()`
# only populates `output_token_ids` when `holder.has_tracked_requests()` is
# true, and that is evaluated at step 1 — before the route exists. A state
# entry conjured at step 3 would therefore be handed an EMPTY output-token list
# for that step and be skipped by update_state()'s own `seq_idx >= len(...)`
# guard. So a request the caller left unbudgeted gets its entry at step 1 with
# the DEEP budget (fail-safe: the champion ceiling), and step 3 rewrites the
# number in place once the route is known. Provisional-until-resolved, never
# unbudgeted-then-tracked.
#
# WHAT THIS DOES *NOT* DO — state it plainly, do not imply parity with the
# original deep/lean result. The PN102 v5-vs-v3 BANNER is rendered into the
# PROMPT, pre-prefill. The route is derived FROM that prefill. Selecting the
# banner from the route is therefore circular in a single pass and is NOT
# attempted here. This ships the BUDGET half of the deep/lean split only; the
# 25-to-v5 / 75-to-v3 banner half needs a separate cheap probe request, or
# nothing. Expect a fraction of the measured saving, not all of it.
#
# COMPOSITION WITH PN100. PN100 sets request.thinking_token_budget on the
# frontend, so those requests arrive with an EXPLICIT budget and this consumer
# never touches them (counted as h119_caller_explicit). The two do not fight:
# H119 only fills the gap where nothing else expressed an opinion.

_CONSUMER_FLAG = "GENESIS_ENABLE_H119_ROUTE_BUDGET"

# Marker key inside a holder state entry: True while the entry carries the
# provisional deep budget and is still waiting for its route.
H119_PROVISIONAL = "_h119_provisional"

# Deep = the champion path's own ceiling (GENESIS_PN100_BUDGET_CEIL 10240 on
# the continuous k260 path). It is a CEILING, not a target: champion prod
# traffic measures ~1575 reasoning tokens/req, so on deep-routed requests this
# binds only on runaways (the uncapped-native arm measured 2.9-3.4K rtok/req).
_DEEP_DEFAULT = 10240

# Lean = 1600.
#
# It was 800, from ~/shared/REPORT-prodmatrix-goal80-20260724.md §1: the v3fam
# arm measured 778 reasoning tokens/req with 24/24 consolidate parse, rounded
# onto PN100's 100-token grid. That reasoning was wrong in a specific way, and
# the 2026-07-25 lens review quantified it from three directions:
#   * 778 is the MEAN of a BANNER-shaped arm, not a cap that arm respected. As a
#     hard cap it does not sit above the distribution, it bisects it.
#   * Live thinking traffic: p10 794, p50 1295, p90 3095. A cap of 800 truncates
#     61-87% of it (two independent counts, on prod and on GPQA).
#   * Truncation is the expensive direction. Over-granting is nearly free
#     because 12 of 31 live rows stop naturally below their grant, while an
#     under-grant forces </think> mid-thought and costs the answer.
# 1600 truncates 4/31 and covers ~90.8% of measured need. It is an interim
# number: the durable form is a conditional quantile Q_t(need | score), which
# needs the regression head, not this constant.
_LEAN_DEFAULT = 1600

# PN100 stamps this into SamplingParams.extra_args (via the request's
# vllm_xargs) on every budget IT chose, and sets it to 0 on the two it must keep
# absolute authority over: tier-0/thinking-off, and a client-pinned numeric.
# Without it the consumer cannot tell PN100's budget from a real caller's, and
# since PN100 runs with AUTO_DEFAULT=1 it budgets ~every request — which made
# the 2026-07-25 consumer boot a measured no-op (GPQA-30 identical to control on
# every compared row) even with all seven patch sites correctly installed.
# The protocol types vllm_xargs as str|int|float, so this is 1/0, never a bool.
H119_OVERRIDABLE = "h119_overridable"

# Kill switch for the PN100 override alone; the rest of the consumer is
# unaffected. Default ON — a consumer that defers to PN100 does nothing at all.
_OVERRIDE_PN100_FLAG = "H119_OVERRIDE_PN100"

# ROUTE GRACE. A provisional row used to be committed to route_for()'s fallback
# the instant it had produced ANY output token. That is a zero-token deadline
# for a decision made in the same step, and it left no room at all for the
# deferred score readback (PN119_ASYNC_SCORE) or for a chunked prefill that
# spills one token past its last chunk. Waiting a few tokens is arithmetically
# free: the budgets in play are 800/1600 lean and 10240 deep, and
# `_h119_apply_budget` preserves the countdown, so applying the lean cap at
# token 4 instead of token 0 caps the request at exactly the same place. 8 is
# ~1% of the smallest cap under discussion.
_ROUTE_GRACE_DEFAULT = 8

# Identity witnesses recorded when we mark an entry, checked before we rewrite
# it. `_state` is keyed by BATCH INDEX, and indices are reused as requests come
# and go — so between marking and resolving, index i can belong to a different
# request entirely. Rewriting then applies one request's route to another's
# budget: silent, and it looks exactly like "the router capped the wrong row".
H119_PLEN = "_h119_plen"
H119_OUT = "_h119_out"

# Set on an entry we took over from PN100, holding the budget PN100 had chosen.
# If the route never lands, this is what the row falls back to — PN100's own
# grant is a far better fail-safe than route_for()'s generic default, because it
# is what the request would have got if H119 had never touched it.
H119_PRIOR_BUDGET = "_h119_prior_budget"

_consumer_state: dict = {"checked": False, "on": False, "deep": _DEEP_DEFAULT,
                         "lean": _LEAN_DEFAULT, "warned": 0,
                         "override_pn100": True,
                         "grace": _ROUTE_GRACE_DEFAULT}


def _consumer_int(name: str, default: int) -> int:
    try:
        return int(_env(name, "") or default)
    except ValueError:
        logger.warning("[H119] %s is not an integer — using %d", name, default)
        return default


def reset_consumer_cache() -> None:
    """Re-read the H119 consumer env on the next call (tests only)."""
    _consumer_state.update({"checked": False, "on": False,
                            "deep": _DEEP_DEFAULT, "lean": _LEAN_DEFAULT,
                            "warned": 0, "override_pn100": True,
                            "grace": _ROUTE_GRACE_DEFAULT})


def _consumer_active() -> bool:
    """True only when the operator asked for it AND the router can back it.

    Three independent conditions, all required:
      * GENESIS_ENABLE_H119_ROUTE_BUDGET=1 (default OFF -> next boot unchanged);
      * a live router instance (ROUTER) — without one nothing ever writes
        ROUTES, so every request would resolve to the fallback and the whole
        thing would be the exact 100%-miss failure this design exists to avoid;
      * that router in ENFORCE mode — shadow means "act on nothing", and this
        is an act.
    """
    st = _consumer_state
    if not st["checked"]:
        st["checked"] = True
        st["on"] = _truthy(_env(_CONSUMER_FLAG, "0"))
        st["deep"] = _consumer_int("H119_DEEP_BUDGET", _DEEP_DEFAULT)
        st["lean"] = _consumer_int("H119_LEAN_BUDGET", _LEAN_DEFAULT)
        st["override_pn100"] = _truthy(_env(_OVERRIDE_PN100_FLAG, "1"))
        st["grace"] = max(_consumer_int("H119_ROUTE_GRACE_TOKENS",
                                        _ROUTE_GRACE_DEFAULT), 0)
        if st["on"] and st["deep"] <= 0:
            # A non-positive deep budget would mean "install no entry", and
            # without an entry at step 1 the LEAN route can never be applied
            # either (see the has_tracked_requests() note above). Refuse rather
            # than silently route nothing.
            logger.warning("[H119] H119_DEEP_BUDGET=%d is not positive — the "
                           "route consumer needs a provisional cap to attach "
                           "to; consumer DISABLED", st["deep"])
            st["on"] = False
        if st["on"]:
            logger.info("[H119] route->budget consumer ON: deep=%d lean=%d "
                        "grace=%d (BUDGET CAP ONLY — the PN102 banner half of "
                        "the deep/lean split is not reachable from here)",
                        st["deep"], st["lean"], st["grace"])
    if not st["on"]:
        return False
    router = ROUTER
    if router is None:
        _bump("h119_no_router")
        return False
    if getattr(router, "mode", "shadow") != "enforce":
        _bump("h119_router_not_enforce")
        return False
    return True


def _consumer_warn(msg: str, exc: Exception) -> None:
    _consumer_state["warned"] += 1
    if _consumer_state["warned"] <= 5:
        logger.warning("[H119] %s (%d/5 shown): %s",
                       msg, _consumer_state["warned"], exc)


def _h119_overridable_stamp(params) -> int | None:
    """PN100's ownership stamp off SamplingParams.extra_args, or None.

    None means "nobody stamped this" — an un-stamped budget is a real caller's
    and is never touched. Read defensively: extra_args is operator-supplied data
    that arrives from the wire, so a bad value must degrade to None (defer),
    never raise inside sync_batch.
    """
    try:
        xargs = getattr(params, "extra_args", None)
        if not xargs:
            return None
        raw = xargs.get(H119_OVERRIDABLE)
        if raw is None:
            return None
        return 1 if int(raw) else 0
    except (AttributeError, TypeError, ValueError):
        return None


def h119_on_batch_add(holder, index, params, prompt_tok_ids,
                      output_tok_ids) -> bool:
    """Called from ThinkingBudgetStateHolder.sync_batch for EVERY added row.

    Returns True iff this installed a provisional state entry for `index`, in
    which case the caller must NOT pop the row. Returns False for every other
    case — including a caller-supplied budget, which always wins (requirement:
    an explicit thinking_token_budget outranks the router) — so stock
    behaviour is reached by simply doing what the unpatched code did.
    """
    try:
        budget = getattr(params, "thinking_token_budget", None)
        stamp = _h119_overridable_stamp(params)
        if budget is not None:
            # sync_batch's own branch has already built the entry by the time we
            # are called, on BOTH F variants — so taking it over is a matter of
            # marking it, not rebuilding it.
            if not _consumer_active():
                return False
            if stamp != 1 or not _consumer_state["override_pn100"]:
                # A real caller's budget (or the override switched off): an
                # explicit client budget outranks the router, unconditionally.
                _bump("h119_caller_explicit")
                return False
            entry = holder._state.get(index)
            if entry is None:
                # Shouldn't happen on either variant, but never assume the
                # upstream branch above us ran: decline rather than corrupt.
                _bump("h119_pn100_entry_missing")
                return False
            entry[H119_PRIOR_BUDGET] = budget
            entry[H119_PLEN] = len(prompt_tok_ids or ())
            entry[H119_OUT] = output_tok_ids
            entry[H119_PROVISIONAL] = True
            _bump("h119_pn100_override")
            # False = "caller keeps its own entry", which is right here: we
            # amended the entry in place rather than installing one.
            return False
        if not _consumer_active():
            return False
        if stamp == 0:
            # PN100 tier-0 / thinking-off. It deliberately left the budget unset
            # AND told us to keep out; installing a provisional deep entry here
            # would start tracking a request that never enters <think>.
            _bump("h119_tier0_respected")
            return False
        deep = _consumer_state["deep"]
        entry = holder._init_state_entry(prompt_tok_ids, deep)
        entry["output_tok_ids"] = output_tok_ids
        entry["spec_token_ids"] = []
        entry[H119_PLEN] = len(prompt_tok_ids or ())
        entry[H119_OUT] = output_tok_ids
        entry[H119_PROVISIONAL] = True
        holder._state[index] = entry
        _bump("h119_provisional_added")
        return True
    except Exception as e:  # noqa: BLE001 — never fail a request over routing
        _consumer_warn("on_batch_add failed — request left unbudgeted", e)
        return False


def _h119_apply_budget(state: dict, budget: int) -> None:
    """Rewrite a live state entry's thinking budget, preserving its countdown.

    ``check_count_down`` is initialised to ``budget - think_count`` (think_count
    is non-zero when the PROMPT already sits inside <think>, which is exactly
    what the Qwen3.6 thinking-on template produces). Shifting it by the same
    delta as the budget keeps that invariant without assuming anything about
    how far the request has already got.
    """
    old = state.get("thinking_token_budget", budget)
    delta = budget - old
    state["thinking_token_budget"] = budget
    state["check_count_down"] = state.get("check_count_down", old) + delta
    # Re-evaluate the "already exhausted inside the prompt" flag the same way
    # _init_state_entry does; only meaningful when the prompt opened <think>.
    if state.get("continue_thinking", False):
        state["in_end"] = (budget - state.get("think_count", 0)) <= 0


def _h119_index_matches(index: int, req_id: str, state: dict) -> bool:
    """True iff batch slot `index` still holds the request we marked.

    Two witnesses, cheap and independent:
      * prompt length — a scalar the runner keeps per request;
      * the OUTPUT TOKEN LIST'S IDENTITY (`is`, not `==`). The holder entry and
        the runner's request state share one list object, so identity is a
        proof of provenance that no amount of equal content can fake.
    Missing witnesses (an entry from before this code, or a runner that does not
    expose `requests`) return True: this guard exists to catch a recycled slot,
    not to disable the consumer on an unfamiliar runner.
    """
    try:
        plen = state.get(H119_PLEN)
        out = state.get(H119_OUT)
        if plen is None and out is None:
            return True
        reqs = getattr(ROUTER.runner, "requests", None)
        if reqs is None:
            return True
        rs = reqs.get(req_id)
        if rs is None:
            return False
        if plen is not None:
            actual = getattr(rs, "num_prompt_tokens", None)
            if actual is not None and actual != plen:
                return False
        if out is not None:
            rs_out = getattr(rs, "output_token_ids", None)
            if rs_out is not None and rs_out is not out:
                return False
        return True
    except Exception:  # noqa: BLE001 — a guard must never fail the step
        return True


def _record_applied(req_id: str, budget, source: str) -> None:
    """Note the EFFECTIVE budget for `req_id`, if there is a router to note it on.

    Deliberately defensive: this runs inside the sampler, and the only thing
    worse than a sink row with a missing budget is an exception raised out of
    update_state. A test harness (or a partially-constructed router) that does
    not carry the map simply gets no record.
    """
    try:
        m = getattr(ROUTER, "_h119_applied", None)
        if isinstance(m, dict):
            m[req_id] = (int(budget), source)
    except (TypeError, ValueError):
        pass


def h119_resolve_routes(holder) -> None:
    """Called at the top of ThinkingBudgetStateHolder.update_state.

    Converts every provisional entry whose route is now on record into a real
    routed budget. Entries whose route has not landed yet AND which have not
    produced a token are left alone — they are still in (chunked) prefill and
    the deep budget they carry is the fail-safe. An entry that has started
    generating with no route on record is a genuine miss: it is committed to
    route_for()'s defined fallback, counted, and never revisited.

    ALSO the resolution point for the deferred score readback. This runs at
    sampler time, LATER IN THE SAME ENGINE STEP than the observe() that queued
    it (step order: sync_batch -> forward+observe -> update_state), so draining
    here keeps the original guarantee — a route on record before the sampler
    that emits the first token — while removing the mid-step device sync. The
    drain deliberately happens BEFORE `_consumer_active()`: in shadow mode, or
    with the consumer flag off, there is still a score waiting in pinned memory
    and a sink row that has to be written.
    """
    try:
        router = ROUTER
        if getattr(router, "_pending", None):
            router._drain_pending()
        if not _consumer_active():
            return
        state_map = getattr(holder, "_state", None)
        if not state_map:
            return
        pending = [(i, s) for i, s in state_map.items()
                   if s.get(H119_PROVISIONAL)]
        if not pending:
            return
        req_ids = ROUTER.runner.input_batch.req_ids
        n = len(req_ids)
        # Floor of 1: "no output token yet" has ALWAYS meant "still prefilling,
        # wait", and H119_ROUTE_GRACE_TOKENS=0 must not turn that into "commit
        # to the fallback before the request has generated anything".
        grace = max(int(_consumer_state.get("grace", _ROUTE_GRACE_DEFAULT)), 1)
        for index, state in pending:
            if index >= n:
                # Batch index outside the current batch: the row is mid-move.
                # Leave it provisional; it resolves on a later step. Counted,
                # because a row that NEVER comes back sits at the provisional
                # deep budget forever and this is the only trace of it.
                _bump("h119_index_out_of_batch")
                continue
            req_id = req_ids[index]
            if not req_id:
                continue
            if not _h119_index_matches(index, req_id, state):
                # The batch slot was recycled to a different request between the
                # add and now. Refuse: applying this route would cap the wrong
                # request, which is invisible in every output we collect.
                _bump("h119_index_desync")
                state[H119_PROVISIONAL] = False
                continue
            route = ROUTES.get(req_id)
            if route is None:
                if len(state.get("output_tok_ids") or ()) < grace:
                    # Still prefilling (chunked prefill spans steps), or inside
                    # the grace window. Expected, not a miss — do NOT burn
                    # route_for()'s fallback here.
                    continue
                prior = state.get(H119_PRIOR_BUDGET)
                if prior is not None:
                    # We took this row over from PN100 and the route never
                    # landed. PN100's own grant is the honest fail-safe: it is
                    # exactly what the request would have run with had H119
                    # never touched it, so an unrouted row costs nothing.
                    _h119_apply_budget(state, prior)
                    state[H119_PROVISIONAL] = False
                    # The number is PN100's, not H119's — record the
                    # provenance with it so the sink does not credit a grant
                    # to a decision that never happened.
                    _record_applied(req_id, prior, "pn100")
                    _bump("h119_route_missing_kept_pn100")
                    continue
                # Generation started with no decision on record: this request
                # will never be routed. Commit to the defined fallback once.
                route = route_for(req_id)
                _bump("h119_route_missing")
            budget = (_consumer_state["lean"] if route == ROUTE_LEAN
                      else _consumer_state["deep"])
            _h119_apply_budget(state, budget)
            state[H119_PROVISIONAL] = False
            # BUG-139: the EFFECTIVE grant, recorded where the finish line can
            # find it. The consumer rewrites the holder's state entry and never
            # SamplingParams, so without this the sink reports the budget the
            # frontend asked for and the censoring test is run against a number
            # the request never ran under.
            _record_applied(req_id, budget, "h119")
            _bump("h119_routed_lean" if route == ROUTE_LEAN
                  else "h119_routed_deep")
    except Exception as e:  # noqa: BLE001 — never fail a sampling step
        _consumer_warn("resolve_routes failed — provisional budgets stand", e)
