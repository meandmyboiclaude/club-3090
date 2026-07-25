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

WHERE IT CAN LIVE. Worker-side, same process, same req_id namespace:
vllm/v1/sample/thinking_budget_state.py keeps a MUTABLE per-request
"thinking_token_budget" in _state[batch_index], re-read by update_state() on
every decode step, and gpu_input_batch.py carries req_id_to_index. A request's
first prefill and its ROUTES write happen in the same engine step (observe()
runs in execute_model's postprocess, after the forward); the first sampled
token comes on the NEXT step, so a route-driven budget rewrite still binds —
no thinking token has been produced yet.
LIMIT: that site reaches the budget CAP only. The deep/lean treatments also
differ by PN102 banner (v5-class vs v3-class), which is rendered into the
PROMPT before prefill. The route is derived FROM that prefill, so banner
selection from the route is circular and cannot be done in one pass — it needs
a separate cheap prefill-only probe request, or nothing.

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

v2 loop (fixes/refit_pn119_probe.py + pn119-refit.timer): refits the probe
from the sink on CPU and ATOMICALLY swaps the npz (pn119_atomic.py); this
router hot-reloads it on (mtime,size) change — PN119_RELOAD_S throttle,
no restart. PN119_EXPLORE=<frac> flags a deterministic ~frac of requests
for generous caps in enforce mode so labels stay uncensored (EXPLORE set).

Never raises into serving: every entry point is fully guarded.
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import time

import torch

logger = logging.getLogger("genesis.pn119")

LAYERS = (42, 47, 51)
D_MODEL = 5120
FEAT_DIM = len(LAYERS) * 2 * D_MODEL  # 30720

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
        # v2 sink
        self.sink_dir = _env("PN119_SINK")
        self._sink_feat = self._sink_meta = None
        self._sink_rows = 0
        if self.sink_dir:
            try:
                os.makedirs(self.sink_dir, exist_ok=True)
                tag = time.strftime("%Y%m%d-%H%M%S")
                self._sink_feat = open(
                    os.path.join(self.sink_dir, f"feats-{tag}.bin"), "ab")
                self._sink_meta = open(
                    os.path.join(self.sink_dir, f"meta-{tag}.jsonl"), "a",
                    encoding="utf-8")
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
            try:
                # NO "row" key and NO feature write: there is no valid feature
                # vector, and refit_pn119_probe.load_sink keys score lines on
                # "row", so this line can never become training data.
                self._sink_meta.write(json.dumps({
                    "req_id": req_id, "unscoreable": True, "reason": reason,
                    "missing": missing, "route": route, "prompt_tok": prompt_len,
                    "ts": time.time(), "mode": self.mode, "explore": explore,
                }) + "\n")
                self._sink_meta.flush()
            except Exception as e:  # noqa: BLE001 — any sink failure disables it
                logger.warning("[PN119] sink write failed (%s) — disabling sink", e)
                self._sink_feat = self._sink_meta = None

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
                self._sink_feat.write(
                    torch.stack(rows).to(torch.bfloat16).view(torch.uint16)
                    .cpu().numpy().tobytes())
                self._sink_feat.flush()
                self._sink_meta.write(json.dumps({
                    "req_id": req_id, "row": self._sink_rows, "score": score,
                    "route": route, "prompt_tok": prompt_len, "ts": time.time(),
                    "mode": self.mode, "explore": explore,
                }) + "\n")
                self._sink_meta.flush()
                self._sink_rows += 1
            except Exception as e:  # noqa: BLE001 — any sink failure disables it
                logger.warning("[PN119] sink write failed (%s) — disabling sink", e)
                self._sink_feat = self._sink_meta = None

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
                    self._sink_meta.write(json.dumps({
                        "req_id": req_id, "finish": True, "unscoreable": True,
                        "reason": reason, "ts": time.time(), "mode": self.mode,
                    }) + "\n")
                    self._sink_meta.flush()
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
                self._sink_meta.write(json.dumps({
                    "req_id": req_id, "finish": True, "score": score,
                    "generated": generated, "ts": time.time(),
                    "thinking": thinking, "rtok": rtok, "cap_hit": cap_hit,
                    "explore": self._is_explore(req_id), "mode": self.mode,
                }) + "\n")
                self._sink_meta.flush()
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
