#!/usr/bin/env python3
"""PN112 settled-stop grafts (2026-07-22) — apply AFTER patch_pn108_plateau_cap.py.

Two targets in vllm/v1/sample/thinking_budget_state.py:
  A) confidence tap in _apply_forcing_to_logits: logits ARE in scope there;
     computes per-seq C = mean(logsumexp - top20) (big C = peaked = confident)
     using the cu_num_tokens row map the method just built. No float() copies
     (bf16 ops), <=16 rows, one small .tolist() sync — eager sampler phase,
     outside cudagraph. Gated at runtime on GENESIS_ENABLE_PN112_SETTLED_STOP.
  B) pn112.observe_state call after PN108's observe block (anchor depends on
     the PN108 rewrite — ordering enforced by the compose boot sequence).

Fail-open: every inserted region is try/except-wrapped; a raise disables
nothing but that step's sample. Idempotent by marker. Anchor-drift = FATAL
(exit 1) so a bad boot is loud, mirroring patch_pn108_plateau_cap.py.
"""
import pathlib
import sys

LOG = "[patch_pn112_conf_tap]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/"
    "thinking_budget_state.py"
)

# ── Graft A: confidence tap ──────────────────────────────────────────────────
MARKER_A = "# PN112 tap:"
ANCHOR_A = (
    "        # Build the active index / forced-token lists entirely on CPU so we\n"
)
REPLACEMENT_A = (
    "        # PN112 tap: per-seq sampling confidence C for the settled-stop\n"
    "        # detector (_genesis/plateau/pn112.py). Uses the row map built\n"
    "        # above; bf16 throughout; <=max_num_seqs*(spec+1) rows. Runtime-\n"
    "        # gated so the tap costs nothing when PN112 is off.\n"
    "        try:\n"
    "            import os as _pn112_os\n"
    "            if (\n"
    "                _pn112_os.environ.get(\n"
    "                    'GENESIS_ENABLE_PN112_SETTLED_STOP', ''\n"
    "                ).strip().lower() in ('1', 'true', 'yes', 'on')\n"
    "                and self._state\n"
    "            ):\n"
    "                import torch as _pn112_t\n"
    "                with _pn112_t.no_grad():\n"
    "                    _pn112_lse = _pn112_t.logsumexp(logits, dim=-1)\n"
    "                    _pn112_top = _pn112_t.topk(\n"
    "                        logits, min(20, logits.shape[-1]), dim=-1\n"
    "                    ).values\n"
    "                    _pn112_c = (\n"
    "                        _pn112_lse.unsqueeze(-1) - _pn112_top\n"
    "                    ).mean(dim=-1).tolist()\n"
    "                _pn112_conf = {}\n"
    "                for _pn112_si in self._state:\n"
    "                    _pn112_row = self.cu_num_tokens.get(_pn112_si)\n"
    "                    if _pn112_row is None or _pn112_row >= len(_pn112_c):\n"
    "                        continue\n"
    "                    _pn112_end = min(\n"
    "                        self.cu_num_tokens.get(\n"
    "                            _pn112_si + 1, len(_pn112_c)\n"
    "                        ),\n"
    "                        len(_pn112_c),\n"
    "                    )\n"
    "                    if _pn112_end <= _pn112_row:\n"
    "                        _pn112_end = _pn112_row + 1\n"
    "                    _pn112_rows = _pn112_c[_pn112_row:_pn112_end]\n"
    "                    _pn112_conf[_pn112_si] = sum(_pn112_rows) / len(_pn112_rows)\n"
    "                self._genesis_pn112_conf = _pn112_conf\n"
    "        except Exception:\n"
    "            import logging as _pn112_alog\n"
    "            _pn112_alog.getLogger(\n"
    "                'genesis.plateau.pn112'\n"
    "            ).warning('PN112 conf tap raised', exc_info=True)\n"
    "        # Build the active index / forced-token lists entirely on CPU so we\n"
)

# ── Graft B: observe call after PN108's block ───────────────────────────────
MARKER_B = "# PN112 observe:"
ANCHOR_B = (
    "                ).debug('PN108 observe raised; ignored', exc_info=True)\n"
    "            self._update_think_state(state)\n"
)
REPLACEMENT_B = (
    "                ).debug('PN108 observe raised; ignored', exc_info=True)\n"
    "            # PN112 observe: settled-stop — cut the post-answer tail once\n"
    "            # sampling confidence says the answer is settled. Conf comes\n"
    "            # from the apply_to_logits tap (same step). Fail-open.\n"
    "            try:\n"
    "                from vllm._genesis.plateau import pn112 as _pn112\n"
    "                _pn112.observe_state(\n"
    "                    state, len(self.think_start_token_ids), seq_idx,\n"
    "                    conf=getattr(\n"
    "                        self, '_genesis_pn112_conf', {}\n"
    "                    ).get(seq_idx),\n"
    "                    req_id=getattr(\n"
    "                        self, '_genesis_req_id_by_index', {}\n"
    "                    ).get(seq_idx),\n"
    "                )\n"
    "            except Exception:\n"
    "                import logging as _pn112_elog\n"
    "                _pn112_elog.getLogger(\n"
    "                    'genesis.plateau.pn112'\n"
    "                ).warning('PN112 observe raised', exc_info=True)\n"
    "            self._update_think_state(state)\n"
)


def _apply(marker: str, anchor: str, replacement: str, what: str) -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: target missing: {TARGET}", flush=True)
        return 1
    src = TARGET.read_text(encoding="utf-8")
    if marker in src:
        print(f"{LOG} already applied ({what}) — skipping", flush=True)
        return 0
    count = src.count(anchor)
    if count != 1:
        print(
            f"{LOG} FATAL: anchor occurs {count}x (need exactly 1) for {what} "
            f"— run AFTER patch_pn108_plateau_cap.py / upstream drifted",
            flush=True,
        )
        return 1
    TARGET.write_text(src.replace(anchor, replacement, 1), encoding="utf-8")
    print(f"{LOG} applied — {what}", flush=True)
    return 0


def _drop_stale_pyc() -> None:
    """Same-second pyc race (found 2026-07-22): an earlier boot-script vllm
    import compiles the STOCK holder's pyc; the text patches then rewrite the
    .py within the same mtime second, so timestamp validation keeps the stale
    pyc and the engine runs bytecode WITHOUT any of the file grafts (pn108's
    included — its runtime behavior came from the genesis apply_all hook).
    Deleting the cache forces a recompile at engine import."""
    cache = TARGET.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob(TARGET.stem + ".*.pyc"):
        try:
            pyc.unlink()
            print(f"{LOG} dropped stale bytecode: {pyc.name}", flush=True)
        except OSError as exc:
            print(f"{LOG} WARN: could not drop {pyc.name}: {exc}", flush=True)


def main() -> int:
    rc = _apply(MARKER_A, ANCHOR_A, REPLACEMENT_A, "confidence tap in forcing pass")
    if rc:
        return rc
    rc = _apply(MARKER_B, ANCHOR_B, REPLACEMENT_B, "pn112 observe after pn108")
    if rc:
        return rc
    _drop_stale_pyc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
