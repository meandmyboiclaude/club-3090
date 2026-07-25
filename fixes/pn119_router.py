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
appends (bf16 features row, meta line w/ score) to PN119_SINK; request
finish appends a generated-token line keyed by req_id. Shadow traffic is
uncensored → doubles as the v2 training bootstrap.

Never raises into serving: every entry point is fully guarded.
"""
from __future__ import annotations

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
            runner.use_aux_hidden_state_outputs = True
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
        import numpy as np

        self.runner = runner
        self.mode = _env("PN119_MODE", "shadow").lower() or "shadow"
        self.tdeep = float(_env("PN119_TDEEP", "0.5") or 0.5)
        z = np.load(npz_path, allow_pickle=True)
        dev = runner.device
        self.mu = torch.from_numpy(z["mu"]).float().to(dev)
        self.sd = torch.from_numpy(z["sd"]).float().to(dev)
        self.vt = torch.from_numpy(z["Vt10"]).float().to(dev)  # [10, FEAT_DIM]
        self.w = torch.from_numpy(z["w"]).float().to(dev)      # [11] incl bias
        if self.mu.numel() != FEAT_DIM:
            raise ValueError(f"probe dim {self.mu.numel()} != {FEAT_DIM}")
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

    # ── observation (called from execute_model postprocess) ────────────────
    def observe(self, scheduler_output, aux_hidden_states) -> None:
        try:
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
        logger.info("[PN119] req=%s score=%.4f route=%s prompt_tok=%d mode=%s",
                    req_id, score, route, prompt_len, self.mode)
        if self.mode == "enforce":
            SCORES[req_id] = score
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
            if score is None or req_state is None:
                return
            prompt_len = getattr(req_state, "num_prompt_tokens", 0) or 0
            computed = getattr(req_state, "num_computed_tokens", 0) or 0
            generated = max(computed - prompt_len, 0)
            logger.info("[PN119] finish req=%s score=%.4f generated=%d", req_id, score, generated)
            if self._sink_meta is not None:
                self._sink_meta.write(json.dumps({
                    "req_id": req_id, "finish": True, "score": score,
                    "generated": generated, "ts": time.time(),
                }) + "\n")
                self._sink_meta.flush()
        except Exception as e:  # noqa: BLE001
            self._warned += 1
            if self._warned <= 5:
                logger.warning("[PN119] finish error: %s", e)
