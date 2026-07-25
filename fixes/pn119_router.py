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
enforce = additionally publish the score to the SCORES registry for the
PN100/PN102 holder side (the route-action consumer wiring is a follow-up —
v1 enforce publishes, never mutates requests itself).

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

# Enforce-mode consumers (PN100/PN102 holder side) read this: req_id -> score.
SCORES: dict[str, float] = {}
# PN119_EXPLORE (BUILD-PACK §v2 censoring guard): req_ids selected for
# exploration. Enforce-mode consumers MUST give these generous caps
# regardless of score — that is what keeps the self-training labels honest.
EXPLORE: set[str] = set()


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
            logger.info(
                "[PN119] router active: mode=%s tdeep=%.3f probe=%s aux layers=%s sink=%s",
                inst.mode, inst.tdeep, os.path.basename(npz_path), LAYERS,
                inst.sink_dir or "-",
            )
            return inst
        except Exception as e:  # noqa: BLE001 — never brick model load
            logger.warning("[PN119] init failed: %s — router disabled", e)
            return None

    def __init__(self, runner, npz_path: str):
        self.runner = runner
        self.mode = _env("PN119_MODE", "shadow").lower() or "shadow"
        self.tdeep = float(_env("PN119_TDEEP", "0.5") or 0.5)
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
        # per-request prefill accumulators: req_id -> state dict
        self._acc: dict[str, dict] = {}
        self.scored: dict[str, float] = {}
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
            if state is None or req_id in self.scored:
                start = end
                continue
            prompt_len = getattr(state, "num_prompt_tokens", None)
            if not prompt_len:
                start = end
                continue
            acc = self._acc.get(req_id)
            # Preemption-recompute guard: a request preempted mid-prefill
            # restarts from num_computed_tokens=0; a stale accumulator would
            # double-count. Reset when the engine's progress is behind ours.
            engine_computed = getattr(state, "num_computed_tokens", None)
            if (acc is not None and engine_computed is not None
                    and engine_computed < acc["seen"]):
                acc = None
            if acc is None:
                acc = self._acc[req_id] = {
                    "seen": 0,
                    "sum": torch.zeros(len(LAYERS), D_MODEL,
                                       dtype=torch.float32, device=aux[0].device),
                    "last": None,
                }
            remaining = prompt_len - acc["seen"]
            if remaining <= 0:
                start = end
                continue
            take = min(n, remaining)
            for li in range(len(LAYERS)):
                chunk = aux[li][start:start + take].float()  # [take, d]
                acc["sum"][li] += chunk.sum(0)
            acc["seen"] += take
            if acc["seen"] >= prompt_len:
                last_rows = [aux[li][start + take - 1].float() for li in range(len(LAYERS))]
                self._finalize(req_id, acc, last_rows, prompt_len)
            start = end

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
        self._acc.pop(req_id, None)
        route = "deep" if score >= self.tdeep else "lean"
        explore = self._is_explore(req_id)
        logger.info("[PN119] req=%s score=%.4f route=%s explore=%s prompt_tok=%d mode=%s",
                    req_id, score, route, explore, prompt_len, self.mode)
        if self.mode == "enforce":
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
            SCORES.pop(req_id, None)
            EXPLORE.discard(req_id)
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
            tail = list(prompt_ids[-8:])
            if self._think_end in tail:
                thinking = False
            elif self._think_start in tail:
                thinking = True
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
