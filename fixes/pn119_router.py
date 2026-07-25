"""PN119 lens-router sidecar (installed as vllm/_genesis_pn119.py at boot).

Per-request deep/mass thinking router on the serving model's OWN prefill
hidden states — no extra model, no extra VRAM. Basis: needfit lens probe
(nested-LOO AUC 0.930 lens-only; PN119-BUILD-PACK.md is the spec).

Capture: vLLM's native EAGLE3 aux-hidden-state mechanism
(model.set_aux_hidden_state_layers((42, 47, 51)) — cudagraph/compile-safe,
no python hooks). Feature vector per request (order = lens_pilot.py /
train_pn119_probe.py ground truth, [6, 5120] row-major):
  L42-last, L42-mean, L47-last, L47-mean, L51-last, L51-mean
Score: xs = (x-mu)/sd; p = xs @ Vt10.T; score = concat(p,[1]) @ w.

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

PARTIAL PREFILLS / PREFIX-CACHE HITS (fixed 2026-07-25)
-------------------------------------------------------
The tap only sees the tokens the engine actually FORWARDS this pass. On an
APC prefix-cache hit the first `num_computed_tokens` prompt positions are
served from the KV cache and their layer-42/47/51 residual states are never
materialised — so the mean-pooled half of the feature vector cannot be
formed from this pass alone. The accumulator used to start at 0 regardless,
never reach `prompt_len`, and the request was SILENTLY never scored: under
enforce it fell through to default handling with no score, no log, no
counter. Harmless only while prefix caching never hit (BUG-131); the moment
`prefix_match_unit: 258` (compose/single/cache-override-apc.yaml, 5884a5f9)
is enabled, exactly the shared-system-prompt agent traffic goes unrouted.

Three behaviours now, in order of preference:
  1. EXACT RECONSTRUCTION (PN119_PREFIX_MEMO=1, default OFF): cumulative
     pooled sums are memoised at PN119_MEMO_UNIT boundaries keyed by the
     hash of the token prefix, so a later cache-hit request whose cached
     length lands on a stored checkpoint gets a mathematically identical
     feature vector. Its validity assumption is *the same one APC itself
     makes* — identical token prefix => identical hidden states (b3 proved
     this prefill bit-deterministic for identical token ids). Default OFF
     because its cost (one D2H snapshot per checkpoint) and its hit rate
     (needs PN119_MEMO_UNIT == the live prefix_match_unit) are unmeasured
     until a boot; the memo NEVER produces a score from partial data — a
     miss falls through to (2).
  2. EXPLICIT FALLBACK: a request whose prompt was not fully observed is
     UNSCOREABLE — a partial-feature score is not a degraded score, it is a
     different quantity (mu/sd/PCA were fit on whole-prompt pooling), so it
     is never computed. Instead the router publishes a defined default route
     (PN119_FALLBACK_ROUTE, default "deep" = fail-safe on accuracy), logs it
     at WARNING and counts it. No feature row reaches the v2 sink.
  3. Consumers call `route_for(req_id)`, which can never return None. The
     old `SCORES.get(req_id) -> None -> whatever the default is` shape is
     what made the failure silent; it stays populated for compatibility.

v2 self-training sink (PN119-BUILD-PACK §v2): every finalized prefill
appends (bf16 features row, meta line w/ score+mode+explore) to PN119_SINK;
request finish appends a label line (generated, thinking flag, true rtok =
tokens before </think>, cap_hit) keyed by req_id. Shadow traffic is
uncensored → doubles as the v2 training bootstrap.
The sink is BUFFERED IN RAM and drained by a daemon thread, so the request
path holds no disk I/O at all (PN119_SINK_BUF_ROWS / _BUF_SECS / _BUF_MAX;
0 rows = legacy synchronous mode). Clean shutdown always drains via atexit;
a hard kill loses at most one buffer. On-disk bytes are unchanged.

v2 loop (fixes/refit_pn119_probe.py + pn119-refit.timer): refits the probe
from the sink on CPU and ATOMICALLY swaps the npz (pn119_atomic.py); this
router hot-reloads it on (mtime,size) change — PN119_RELOAD_S throttle,
no restart. PN119_EXPLORE=<frac> flags a deterministic ~frac of requests
for generous caps in enforce mode so labels stay uncensored (EXPLORE set).

Never raises into serving: every entry point is fully guarded.
"""
from __future__ import annotations

import atexit
import collections
import hashlib
import json
import logging
import os
import threading
import time
import weakref

import torch

logger = logging.getLogger("genesis.pn119")

LAYERS = (42, 47, 51)
D_MODEL = 5120
FEAT_DIM = len(LAYERS) * 2 * D_MODEL  # 30720

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


def _rindex(seq, value):
    """Index of the LAST occurrence of `value` in `seq`, or None."""
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] == value:
            return i
    return None


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
            logger.info(
                "[PN119] router active: mode=%s tdeep=%.3f probe=%s aux layers=%s "
                "sink=%s fallback_route=%s prefix_memo=%s(unit=%d max=%d)",
                inst.mode, inst.tdeep, os.path.basename(npz_path), LAYERS,
                inst.sink_dir or "-", inst.fallback_route,
                "on" if inst.memo_on else "off", inst.memo_unit, inst.memo_max,
            )
            return inst
        except Exception as e:  # noqa: BLE001 — never brick model load
            logger.warning("[PN119] init failed: %s — router disabled", e)
            return None

    def __init__(self, runner, npz_path: str):
        global _FALLBACK_ROUTE
        self.runner = runner
        self.mode = _env("PN119_MODE", "shadow").lower() or "shadow"
        self.tdeep = float(_env("PN119_TDEEP", "0.5") or 0.5)
        # Route for requests we cannot score (see module docstring §2).
        fb = _env("PN119_FALLBACK_ROUTE", ROUTE_DEEP).lower() or ROUTE_DEEP
        if fb not in ROUTE_CHOICES:
            logger.warning("[PN119] PN119_FALLBACK_ROUTE=%r invalid — using %s",
                           fb, ROUTE_DEEP)
            fb = ROUTE_DEEP
        self.fallback_route = _FALLBACK_ROUTE = fb
        # Exact-reconstruction memo for cached prefixes (module docstring §1).
        # OFF by default: correctness is not at stake either way (a miss falls
        # back), but the D2H snapshot cost and the hit rate are unmeasured
        # until a boot, and OFF keeps the full-recompute path byte-identical.
        self.memo_on = _truthy(_env("PN119_PREFIX_MEMO", "0"))
        # MUST equal the engine's APC prefix_match_unit for checkpoints to line
        # up with cache-hit lengths (cache-override-apc.yaml ships 258).
        self.memo_unit = max(int(_env("PN119_MEMO_UNIT", "258") or 258), 1)
        self.memo_max = max(int(_env("PN119_MEMO_MAX", "256") or 256), 1)
        # key -> [len(LAYERS), D_MODEL] float32 CPU cumulative sums, LRU.
        self._memo: "collections.OrderedDict[tuple, torch.Tensor]" = (
            collections.OrderedDict())
        self._stats_every = max(int(_env("PN119_STATS_EVERY", "200") or 200), 1)
        self._decisions = 0
        # v2 explore knob (BUILD-PACK §v2): fraction of requests flagged for
        # generous caps regardless of score. Deterministic per req_id so the
        # sink row and the enforce-side consumer always agree.
        try:
            self.explore_rate = min(max(float(_env("PN119_EXPLORE", "0") or 0.0), 0.0), 1.0)
        except ValueError:
            self.explore_rate = 0.0
        # v2 hot-reload: refit timer atomically swaps the npz; we re-load on
        # (mtime, size) change, throttled to one stat() per PN119_RELOAD_S.
        self._probe_path = npz_path
        self._reload_every = max(float(_env("PN119_RELOAD_S", "60") or 60.0), 1.0)
        self._next_reload_check = time.time() + self._reload_every
        self._probe_sig = self._stat_sig(npz_path)
        self._failed_sig = None
        self.mu, self.sd, self.vt, self.w = self._load_probe(npz_path)
        # v2 label plumbing: think-region markers of the SERVED model's
        # tokenizer (defaults = thinkingcap-gptq-pro-v2 / Qwen3.6 template:
        # thinking-ON prompt ends "<think>\n", thinking-OFF ends
        # "<think>\n\n</think>\n\n"; </think> in OUTPUT marks end of spend).
        self._think_start = int(_env("PN119_THINK_START_ID", "248068") or 248068)
        self._think_end = int(_env("PN119_THINK_END_ID", "248069") or 248069)
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
        # per-request prefill accumulators: req_id -> state dict
        self._acc: dict[str, dict] = {}
        self.scored: dict[str, float] = {}
        # req_id -> reason, for requests that got the fallback route instead of
        # a score. Keeps the warning + the sink line to one per request.
        self.unscored: dict[str, str] = {}
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
        if self.sink_dir:
            try:
                os.makedirs(self.sink_dir, exist_ok=True)
                tag = time.strftime("%Y%m%d-%H%M%S")
                self._sink_feat = open(
                    os.path.join(self.sink_dir, f"feats-{tag}.bin"), "ab")
                self._sink_meta = open(
                    os.path.join(self.sink_dir, f"meta-{tag}.jsonl"), "a",
                    encoding="utf-8")
                self._sink_start()
                _SINKS.add(self)
                logger.info("[PN119] sink buffered: buf_rows=%d buf_secs=%.2f "
                            "buf_max=%d thread=%s", self._buf_rows,
                            self._buf_secs, self._buf_max,
                            self._sink_thread is not None)
            except OSError as e:
                logger.warning("[PN119] sink unavailable (%s) — logging only", e)
                self._sink_feat = self._sink_meta = None

    # ── probe loading + hot-reload ──────────────────────────────────────────
    def _load_probe(self, path: str):
        """Load npz -> (mu, sd, vt, w) tensors on the runner device.
        Raises on any problem — callers decide whether that is fatal
        (__init__) or a keep-old-weights warning (_maybe_reload)."""
        import numpy as np

        z = np.load(path, allow_pickle=True)
        dev = self.runner.device
        mu = torch.from_numpy(np.asarray(z["mu"])).float().to(dev)
        sd = torch.from_numpy(np.asarray(z["sd"])).float().to(dev)
        vt = torch.from_numpy(np.asarray(z["Vt10"])).float().to(dev)  # [10, FEAT_DIM]
        w = torch.from_numpy(np.asarray(z["w"])).float().to(dev)      # [pcs+1] incl bias
        if mu.numel() != FEAT_DIM or sd.numel() != FEAT_DIM:
            raise ValueError(f"probe dim {mu.numel()} != {FEAT_DIM}")
        if vt.shape[1] != FEAT_DIM or w.numel() != vt.shape[0] + 1:
            raise ValueError(f"probe shapes Vt{tuple(vt.shape)} w{tuple(w.shape)} inconsistent")
        return mu, sd, vt, w

    @staticmethod
    def _stat_sig(path: str):
        try:
            st = os.stat(path)
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

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
        sig = self._stat_sig(self._probe_path)
        if sig is None or sig == self._probe_sig or sig == self._failed_sig:
            return
        try:
            new = self._load_probe(self._probe_path)
        except Exception as e:  # noqa: BLE001 — never break serving on a bad swap
            self._failed_sig = sig
            logger.warning("[PN119] probe hot-reload FAILED (%s) — keeping current weights", e)
            return
        self.mu, self.sd, self.vt, self.w = new
        self._probe_sig = sig
        self._failed_sig = None
        logger.info("[PN119] probe hot-reloaded from %s (mtime_ns=%d size=%d)",
                    self._probe_path, sig[0], sig[1])

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
        runner = self.runner
        sched = scheduler_output.num_scheduled_tokens  # dict req_id -> n
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
            state = runner.requests.get(req_id)
            # Already decided (scored or explicitly fallen back) => nothing to
            # do on this step. Both maps are popped in on_finish.
            if req_id in self.scored or req_id in self.unscored:
                start = end
                continue
            if state is None:
                _bump("skip_no_req_state")
                start = end
                continue
            prompt_len = getattr(state, "num_prompt_tokens", None)
            if not prompt_len:
                _bump("skip_no_prompt_len")
                start = end
                continue
            acc = self._acc.get(req_id)
            # Preemption-recompute guard: a request preempted mid-prefill
            # restarts from a lower num_computed_tokens; a stale accumulator
            # would double-count. Reset when the engine is behind us — the
            # rebuilt accumulator re-derives its own starting offset below.
            engine_computed = getattr(state, "num_computed_tokens", None)
            if (acc is not None and engine_computed is not None
                    and engine_computed < acc["seen"]):
                _bump("acc_reset_recompute")
                acc = None
            if acc is None:
                # `num_computed_tokens` is the engine's PRE-step progress (set
                # in _update_states, which runs before the forward and before
                # this postprocess hook). For a first-prefill step it is
                # exactly the APC-cached prefix length: those positions were
                # NOT forwarded, so their aux rows do not exist this pass.
                base = int(engine_computed or 0)
                base = min(max(base, 0), prompt_len)
                if base >= prompt_len:
                    # We never saw any of this prompt's prefill (router
                    # attached late, or an earlier observe error dropped the
                    # accumulator). Nothing to reconstruct from.
                    self._unscoreable(req_id, "prefill_not_observed",
                                      prompt_len, prompt_len)
                    start = end
                    continue
                acc = self._acc[req_id] = {
                    "seen": base,
                    "sum": torch.zeros(len(LAYERS), D_MODEL,
                                       dtype=torch.float32, device=aux[0].device),
                    "last": None,
                    "cached": base,
                    "missing": base,
                }
                if base > 0:
                    _bump("prefill_partial_cached")
                    prefix = self._memo_get(state, base, aux[0].device)
                    if prefix is not None:
                        acc["sum"] += prefix
                        acc["missing"] = 0
                        _bump("memo_hit")
                    else:
                        _bump("memo_miss")
            remaining = prompt_len - acc["seen"]
            if remaining <= 0:
                start = end
                continue
            take = min(n, remaining)
            self._accumulate(state, acc, aux, start, take)
            if acc["seen"] >= prompt_len:
                if acc["missing"]:
                    # The mean-pool half of the feature vector would be pooled
                    # over a different token set than mu/sd/Vt were fit on.
                    # That is not a degraded score, it is a different quantity
                    # — refuse to compute one.
                    self._unscoreable(req_id, "partial_prefill",
                                      acc["missing"], prompt_len)
                else:
                    last_rows = [aux[li][start + take - 1].float()
                                 for li in range(len(LAYERS))]
                    self._finalize(req_id, acc, last_rows, prompt_len)
            start = end

    def _accumulate(self, state, acc, aux, start: int, take: int) -> None:
        """Add aux rows [start, start+take) to the running pooled sum.

        With the memo OFF this is a single fused `sum(0)` over the whole slice
        — byte-identical to the pre-2026-07-25 accumulator. With the memo ON
        the slice is split at PN119_MEMO_UNIT boundaries so a complete-from-
        zero cumulative sum can be snapshotted at each one.
        """
        if not self.memo_on:
            for li in range(len(LAYERS)):
                acc["sum"][li] += aux[li][start:start + take].float().sum(0)
            acc["seen"] += take
            return
        pos = acc["seen"]
        off = 0
        unit = self.memo_unit
        while off < take:
            nxt = ((pos // unit) + 1) * unit
            step = min(take - off, nxt - pos)
            for li in range(len(LAYERS)):
                acc["sum"][li] += (
                    aux[li][start + off:start + off + step].float().sum(0))
            pos += step
            off += step
            if pos % unit == 0 and not acc["missing"]:
                self._memo_put(state, pos, acc["sum"])
        acc["seen"] = pos

    # ── cached-prefix feature memo (module docstring §1) ────────────────────
    def _memo_key(self, state, n: int):
        ids = getattr(state, "prompt_token_ids", None)
        if not ids or len(ids) < n:
            return None
        h = hashlib.sha1()
        h.update(b"pn119.1")
        h.update(repr(list(ids[:n])).encode())
        return (n, h.digest())

    def _memo_put(self, state, n: int, cum: torch.Tensor) -> None:
        key = self._memo_key(state, n)
        if key is None:
            return
        if key in self._memo:
            self._memo.move_to_end(key)
            return
        self._memo[key] = cum.detach().to("cpu", copy=True)
        _bump("memo_store")
        while len(self._memo) > self.memo_max:
            self._memo.popitem(last=False)
            _bump("memo_evict")

    def _memo_get(self, state, n: int, device):
        key = self._memo_key(state, n)
        if key is None:
            return None
        cum = self._memo.get(key)
        if cum is None:
            return None
        self._memo.move_to_end(key)
        return cum.to(device)

    # ── v2 sink plumbing: no disk on the request path ───────────────────────
    # observe()/on_finish() only append to two RAM lists under a lock; a daemon
    # thread does every write()+flush().  Measured on the live btrfs sink dir
    # (2000 rows, 61440 B feature + meta line each):
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
        if self._buf_rows <= 0:
            self._sink_thread = None    # PN119_SINK_BUF_ROWS=0 => sync escape
            return
        self._sink_thread = threading.Thread(
            target=self._sink_loop, name="pn119-sink", daemon=True)
        self._sink_thread.start()

    def _sink_loop(self) -> None:
        while not self._sink_stop:
            self._sink_wake.wait(self._buf_secs)
            self._sink_wake.clear()
            if self._sink_meta is None:
                return                  # sink disabled or closed
            self._sink_flush()

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

    def _finalize(self, req_id, acc, last_rows, prompt_len) -> None:
        mean_rows = [acc["sum"][li] / float(prompt_len) for li in range(len(LAYERS))]
        # feature order: per layer [last, mean] (lens_pilot raw_rows order)
        rows = []
        for li in range(len(LAYERS)):
            rows.append(last_rows[li])
            rows.append(mean_rows[li])
        x = torch.cat(rows)  # [FEAT_DIM] f32 on device
        xs = (x - self.mu) / self.sd
        p = self.vt @ xs  # [10]
        score = float(torch.dot(self.w[:-1], p) + self.w[-1])
        self.scored[req_id] = score
        cached = acc.get("cached", 0)
        self._acc.pop(req_id, None)
        route = ROUTE_DEEP if score >= self.tdeep else ROUTE_LEAN
        explore = self._is_explore(req_id)
        _bump("scored")
        _bump(f"scored_{route}")
        if cached:
            _bump("scored_via_memo")
        self._decisions += 1
        logger.info("[PN119] req=%s score=%.4f route=%s explore=%s prompt_tok=%d "
                    "cached_prefix=%d mode=%s",
                    req_id, score, route, explore, prompt_len, cached, self.mode)
        if self._decisions % self._stats_every == 0:
            logger.info("[PN119] stats: %s", stats_line())
        if self.mode == "enforce":
            ROUTES[req_id] = route
            SCORES[req_id] = score
            if explore:
                # Consumer contract: explore requests get generous caps
                # regardless of score (keeps v2 labels uncensored).
                EXPLORE.add(req_id)
        if self._sink_feat is not None:
            try:
                # bf16 has no numpy dtype — reinterpret as uint16 for the raw
                # write (reader: np.fromfile(uint16).view via torch bf16).
                # The D2H stays HERE, on the runner thread, deliberately: this
                # runs in execute_model's postprocess, which has already
                # synchronised for the sampled tokens, so the marginal cost is
                # the 61 kB PCIe copy.  Deferring it to the flusher would mean
                # issuing a CUDA copy from a non-runner thread while the runner
                # may be replaying a cudagraph — a real hazard traded for a
                # copy that is not the bottleneck.  (Cost if we ever do defer
                # it: 30720 dims x 2 B x buf = 3.75 MB VRAM at the default 64,
                # 30 MB at the 512 hard cap.)
                feat = (torch.stack(rows).to(torch.bfloat16)
                        .view(torch.uint16).cpu().numpy().tobytes())
            except Exception as e:  # noqa: BLE001 — any sink failure disables it
                logger.warning("[PN119] sink encode failed (%s) — disabling sink", e)
                self._sink_feat = self._sink_meta = None
            else:
                self._sink_append(feat, json.dumps({
                    "req_id": req_id, "row": self._sink_rows, "score": score,
                    "route": route, "prompt_tok": prompt_len, "ts": time.time(),
                    "mode": self.mode, "explore": explore,
                }) + "\n")
                self._sink_rows += 1

    # ── finish sink (called from _update_states removal loop) ──────────────
    def on_finish(self, req_id, req_state) -> None:
        try:
            self._acc.pop(req_id, None)
            score = self.scored.pop(req_id, None)
            reason = self.unscored.pop(req_id, None)
            SCORES.pop(req_id, None)
            ROUTES.pop(req_id, None)
            EXPLORE.discard(req_id)
            if reason is not None:
                # Fallback-routed request: no score, but the finish MUST still
                # be visible. No "row" key => load_sink never joins it into a
                # training Row (its features do not exist).
                if self._sink_meta is not None:
                    self._sink_append(None, json.dumps({
                        "req_id": req_id, "finish": True, "unscoreable": True,
                        "reason": reason, "ts": time.time(), "mode": self.mode,
                    }) + "\n")
                return
            if score is None or req_state is None:
                return
            prompt_len = getattr(req_state, "num_prompt_tokens", 0) or 0
            computed = getattr(req_state, "num_computed_tokens", 0) or 0
            generated = max(computed - prompt_len, 0)
            thinking, rtok, cap_hit = self._label_fields(req_state, generated)
            logger.info("[PN119] finish req=%s score=%.4f generated=%d thinking=%s rtok=%s cap_hit=%s",
                        req_id, score, generated, thinking, rtok, cap_hit)
            if self._sink_meta is not None:
                self._sink_append(None, json.dumps({
                    "req_id": req_id, "finish": True, "score": score,
                    "generated": generated, "ts": time.time(),
                    "thinking": thinking, "rtok": rtok, "cap_hit": cap_hit,
                    "explore": self._is_explore(req_id), "mode": self.mode,
                }) + "\n")
        except Exception as e:  # noqa: BLE001
            self._warned += 1
            if self._warned <= 5:
                logger.warning("[PN119] finish error: %s", e)

    def _label_fields(self, req_state, generated: int):
        """v2 label plumbing (BUILD-PACK §v2 items 3+4).

        thinking: True  = prompt tail opens a <think> region (spend signal
                          exists — the ONLY rows the refit may learn from);
                  False = template pre-closed </think> (thinking-off);
                  None  = neither marker found (raw/completion prompt) —
                          refit treats unknown as ineligible.
        rtok:     tokens generated BEFORE </think> = true thinking spend
                  (better label than total `generated`, which includes the
                  answer). Falls back to `generated` when </think> never
                  appeared (capped inside the think region).
        cap_hit:  spend was truncated (</think> never emitted while
                  thinking, or generated >= max_tokens). Censoring guard:
                  refit must treat cap_hit as positive-label evidence, and
                  under enforce a lean-routed cap_hit is the ONE lean row
                  that may be learned from (y=1).
        """
        thinking = None
        rtok = None
        cap_hit = False
        prompt_ids = getattr(req_state, "prompt_token_ids", None)
        if prompt_ids:
            # LAST marker wins, not "any </think> anywhere in the window".
            # thinking-off pre-closes the region, so the tail holds BOTH
            # markers and </think> is last; thinking-on with a PN102 seed
            # holds only <think>, several seed tokens back from the end.
            tail = list(prompt_ids[-self._tail_window:])
            last_end = _rindex(tail, self._think_end)
            last_start = _rindex(tail, self._think_start)
            if last_start is not None and (
                    last_end is None or last_start > last_end):
                thinking = True
            elif last_end is not None:
                thinking = False
        out_ids = getattr(req_state, "output_token_ids", None)
        if thinking and out_ids is not None:
            try:
                rtok = list(out_ids).index(self._think_end)
            except ValueError:
                rtok = generated  # never closed the think region => capped
                cap_hit = True
        elif thinking:
            rtok = generated
        sp = getattr(req_state, "sampling_params", None)
        max_tokens = getattr(sp, "max_tokens", None) if sp is not None else None
        if max_tokens and generated >= max_tokens:
            cap_hit = True
        return thinking, rtok, cap_hit


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

# Lean = 800. Provenance and its ONE honest caveat:
#   ~/shared/REPORT-prodmatrix-goal80-20260724.md §1 measures the v3fam arm at
#   778 reasoning tokens/req with 24/24 (100%) consolidate parse — the cheapest
#   arm that stayed reliable. 800 is 778 rounded onto PN100's 100-token grid.
#   CAVEAT (do not lose this): 778 is a MEAN of a BANNER-shaped arm, not a cap
#   that arm respected, and this consumer cannot reproduce that banner (see
#   above). As a hard cap, 800 will force </think> on the upper part of the
#   spend distribution rather than sit above it. That is the same mechanism
#   PN100's own tiers already use in prod, but it is a cap, not a nudge — which
#   is exactly why it is tunable and why the flag ships OFF.
_LEAN_DEFAULT = 800

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

# Set on an entry we took over from PN100, holding the budget PN100 had chosen.
# If the route never lands, this is what the row falls back to — PN100's own
# grant is a far better fail-safe than route_for()'s generic default, because it
# is what the request would have got if H119 had never touched it.
H119_PRIOR_BUDGET = "_h119_prior_budget"

_consumer_state: dict = {"checked": False, "on": False, "deep": _DEEP_DEFAULT,
                         "lean": _LEAN_DEFAULT, "warned": 0,
                         "override_pn100": True}


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
                            "warned": 0, "override_pn100": True})


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
                        "(BUDGET CAP ONLY — the PN102 banner half of the "
                        "deep/lean split is not reachable from here)",
                        st["deep"], st["lean"])
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


def h119_resolve_routes(holder) -> None:
    """Called at the top of ThinkingBudgetStateHolder.update_state.

    Converts every provisional entry whose route is now on record into a real
    routed budget. Entries whose route has not landed yet AND which have not
    produced a token are left alone — they are still in (chunked) prefill and
    the deep budget they carry is the fail-safe. An entry that has started
    generating with no route on record is a genuine miss: it is committed to
    route_for()'s defined fallback, counted, and never revisited.
    """
    try:
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
        for index, state in pending:
            if index >= n:
                # Batch index outside the current batch: the row is mid-move.
                # Leave it provisional; it resolves on a later step.
                continue
            req_id = req_ids[index]
            if not req_id:
                continue
            route = ROUTES.get(req_id)
            if route is None:
                if not state.get("output_tok_ids"):
                    # Still prefilling (chunked prefill spans steps). Expected,
                    # not a miss — do NOT burn route_for()'s fallback here.
                    continue
                prior = state.get(H119_PRIOR_BUDGET)
                if prior is not None:
                    # We took this row over from PN100 and the route never
                    # landed. PN100's own grant is the honest fail-safe: it is
                    # exactly what the request would have run with had H119
                    # never touched it, so an unrouted row costs nothing.
                    _h119_apply_budget(state, prior)
                    state[H119_PROVISIONAL] = False
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
            _bump("h119_routed_lean" if route == ROUTE_LEAN
                  else "h119_routed_deep")
    except Exception as e:  # noqa: BLE001 — never fail a sampling step
        _consumer_warn("resolve_routes failed — provisional budgets stand", e)
