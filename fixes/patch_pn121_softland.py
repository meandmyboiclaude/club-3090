#!/usr/bin/env python3
"""PN121 soft-landing grafts (2026-07-26) — apply AFTER patch_pn114_forced_span.py.

PN121 lands the thinking-budget close on a sentence boundary instead of
guillotining mid-sentence with a bare </think>. The STATE MACHINE lives in
vllm/_genesis/plateau/pn121_softland.py (driven from PN114's observe seat,
which the forced-span graft G already installs). This file adds the two
things the state machine cannot reach from there:

  N) SOFT NUDGE at the raw-logits seat in _apply_forcing_to_logits: while a
     row is in the soft phase (think >= budget - PN121_SOFT_RESERVE), add a
     small additive bump to the newline ids and the </think> ids — Mueller's
     >95%-of-budget nudge, quoted in the P7 research doc §2. Nudge only: no
     mask, no forcing, no budget change, so it cannot pre-empt the cap the
     way the KILLED GENESIS_PN112_WRAPUP_AT_CAP arm did.

     CUDA-graph safety: the id/bump tensors are built ONCE and cached on the
     holder (self._pn121_ids_t / _pn121_bump_t) — sized from the boot ids
     file, never per step. Per step this is one fused index-add per active
     row, no allocation. Same seat and same shape as the P-pen graft (H) it
     sits next to; that seat is the eager sampler, outside any captured
     forward, and applied AFTER penalties like the stock forcing.

  X) GRAMMAR ROW STAMP in the gpu_model_runner sampling seat: record which
     batch rows have a constrained-decoding grammar active this step onto
     each holder state, so PN121 can SUPPRESS injection entirely for them.
     Upstream #44676 (budget forcing injects </think> mid-JSON) is UNMERGED;
     the research doc measures reasoning overflow at ~30% WITHOUT structured
     output, so the structured path is precisely where we must not inject.
     (The <tool_call> half of the guard needs no graft — PN121 scans the
     live think slice for the opener itself.)

Dark by default: with GENESIS_ENABLE_PN121_SOFTLAND unset, N sees no row in
the soft phase and X's stamp is never read. Neither graft changes any
existing default.

Idempotent by marker; anchor drift = FATAL exit 1 (loud bad boot); stale-pyc
drop — house style (patch_pn108/pn112/pr44812/pn114 lineage).
"""
import pathlib
import sys

LOG = "[patch_pn121_softland]"
BASE = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
HOLDER = BASE / "v1/sample/thinking_budget_state.py"
RUNNER = BASE / "v1/worker/gpu_model_runner.py"

# ── N: soft-phase logit nudge ───────────────────────────────────────────────
MARK_N = "# PN121 N:"
ANCH_N = (
    "        # Build the active index / forced-token lists entirely on CPU so we\n"
)
REPL_N = '''        # PN121 N: soft-phase boundary nudge — runtime-gated, dark unless
        # GENESIS_ENABLE_PN121_SOFTLAND=1 puts a row in the soft phase.
        # Upweights the newline ids and </think> for rows within
        # PN121_SOFT_RESERVE of their cap so the close can LAND on a
        # sentence boundary. Additive bump only (no -inf mask): the model
        # keeps every other option, so this cannot shorten a deep item the
        # way the killed WRAPUP_AT_CAP forcing did.
        try:
            import os as _pn121_os
            if (
                _pn121_os.environ.get(
                    'GENESIS_ENABLE_PN121_SOFTLAND', ''
                ).strip().lower() in ('1', 'true', 'yes', 'on')
                and self._state
            ):
                from vllm._genesis.plateau import pn121_softland as _pn121
                _pn121_rows = _pn121.nudge_rows(self)
                if _pn121_rows:
                    _pn121_t = getattr(self, '_pn121_ids_t', None)
                    if _pn121_t is None:
                        # built ONCE, sized from the boot ids file; never a
                        # per-step allocation (CUDA-graph / capture safety).
                        import json as _pn121_json
                        import torch as _pn121_torch
                        try:
                            with open(
                                '/tmp/genesis_pn114_ids.json'
                            ) as _pn121_f:
                                _pn121_ids = _pn121_json.load(_pn121_f)
                        except Exception:
                            _pn121_ids = {}
                        _pn121_c = _pn121.cfg()
                        _pn121_nl = list(
                            _pn121_ids.get('nl_end', [])
                            if _pn121_os.environ.get(
                                'PN121_NUDGE_ALL_NL', ''
                            ).strip().lower() in ('1', 'true', 'yes', 'on')
                            else _pn121_ids.get('newline', [])
                        )
                        _pn121_end = list(self.think_end_token_ids)
                        _pn121_flat = _pn121_nl + _pn121_end
                        _pn121_bumps = (
                            [float(_pn121_c['nudge_nl'])] * len(_pn121_nl)
                            + [float(_pn121_c['nudge_end'])] * len(_pn121_end)
                        )
                        if _pn121_flat:
                            _pn121_t = _pn121_torch.tensor(
                                _pn121_flat, dtype=_pn121_torch.long,
                                device=logits.device,
                            )
                            self._pn121_bump_t = _pn121_torch.tensor(
                                _pn121_bumps, dtype=logits.dtype,
                                device=logits.device,
                            )
                        else:
                            _pn121_t = False
                            self._pn121_bump_t = None
                        self._pn121_ids_t = _pn121_t
                    if _pn121_t is not False and len(_pn121_t) > 0:
                        for _pn121_si in _pn121_rows:
                            _pn121_row = self.cu_num_tokens.get(_pn121_si)
                            if (_pn121_row is None
                                    or _pn121_row >= logits.shape[0]):
                                continue
                            _pn121_e = min(
                                self.cu_num_tokens.get(
                                    _pn121_si + 1, logits.shape[0]
                                ),
                                logits.shape[0],
                            )
                            logits[_pn121_row:_pn121_e, _pn121_t] += (
                                self._pn121_bump_t
                            )
        except Exception:
            import logging as _pn121_log
            _pn121_log.getLogger('vllm.genesis.pn121').warning(
                'PN121 nudge raised', exc_info=True
            )
        # Build the active index / forced-token lists entirely on CPU so we
'''

# ── X: structured-output row stamp ──────────────────────────────────────────
MARK_X = "# PN121 X:"
ANCH_X = """        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )
"""
REPL_X = """        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )
        # PN121 X: stamp which batch rows have a constrained-decoding grammar
        # active this step onto the thinking-budget holder states, so PN121
        # can suppress its injection there (upstream #44676 is unmerged, and
        # a forced span would fight the grammar bitmask for the same rows).
        # Cleared every step; absent stamp == no grammar == no suppression.
        try:
            _pn121_h = self.input_batch.thinking_budget_state_holder
            if _pn121_h is not None and _pn121_h._state:
                _pn121_rows = set()
                if grammar_output is not None:
                    for _pn121_rid in (
                        grammar_output.structured_output_request_ids
                    ):
                        _pn121_i = self.input_batch.req_id_to_index.get(
                            _pn121_rid
                        )
                        if _pn121_i is not None:
                            _pn121_rows.add(_pn121_i)
                for _pn121_st in _pn121_h._state.values():
                    _pn121_st['_pn121_grammar_rows'] = _pn121_rows
        except Exception:
            import logging as _pn121_xlog
            _pn121_xlog.getLogger('vllm.genesis.pn121').warning(
                'PN121 grammar stamp raised', exc_info=True
            )
"""

GRAFTS = [
    (HOLDER, MARK_N, ANCH_N, REPL_N, "N soft-phase logit nudge"),
    (RUNNER, MARK_X, ANCH_X, REPL_X, "X structured-output row stamp"),
]


def _apply(target: pathlib.Path, marker: str, anchor: str, repl: str,
           what: str) -> int:
    if not target.exists():
        print(f"{LOG} FATAL: target missing: {target}", flush=True)
        return 1
    src = target.read_text(encoding="utf-8")
    if marker in src:
        print(f"{LOG} SKIP (already applied): {what}", flush=True)
        return 0
    count = src.count(anchor)
    if count != 1:
        print(f"{LOG} FATAL: anchor occurs {count}x (need exactly 1) for "
              f"{what} in {target.name}", flush=True)
        return 1
    assert marker in repl, f"marker missing from replacement for {what}"
    target.write_text(src.replace(anchor, repl, 1), encoding="utf-8")
    print(f"{LOG} applied: {what}", flush=True)
    return 0


def _drop_stale_pyc() -> None:
    """Same-second pyc race (2026-07-22): boot scripts import vllm before the
    text patches rewrite these files; a rewrite landing within the same mtime
    second leaves a stale pyc that survives timestamp validation."""
    for target in (HOLDER, RUNNER):
        cache = target.parent / "__pycache__"
        if not cache.is_dir():
            continue
        for pyc in cache.glob(target.stem + ".*.pyc"):
            try:
                pyc.unlink()
                print(f"{LOG} dropped stale pyc {pyc.name}", flush=True)
            except OSError as exc:
                print(f"{LOG} WARN: could not drop {pyc.name}: {exc}",
                      flush=True)


def main() -> int:
    rc = 0
    for target, marker, anchor, repl, what in GRAFTS:
        rc |= _apply(target, marker, anchor, repl, what)
    if rc == 0:
        _drop_stale_pyc()
    return rc


if __name__ == "__main__":
    sys.exit(main())
