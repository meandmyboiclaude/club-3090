#!/usr/bin/env python3
"""PN122 graft (2026-07-26) — suppress the forced </think> on grammar rows.

Leak class L6, second half. PR44812 (fixes/patch_pr44812_tool_guard.py) only
teaches the holder that a literal <tool_call> ends reasoning. That covers the
Qwen3 tool-call surface and NOTHING else: an OpenAI-style structured request
(`guided_json` / `response_format` / `tool_choice` driving an xgrammar mask)
never emits <tool_call>, so `implicit_think_end_token_ids` never matches and a
budget expiry still force-closes </think> straight into the JSON.

Ordering is what makes it reachable, not a race:

    gpu_model_runner.py  apply_grammar_bitmask(...)   # masks off-grammar ids
    gpu_model_runner.py  self._sample(logits, ...)
      sampler.py           holder.apply_to_logits(...)
        thinking_budget_state.py  logits.index_put_(..., 1e9)   # </think>

The holder writes +1e9 AFTER the mask, so the forced token wins over the
grammar and an out-of-grammar </think> lands mid-arguments.

Fix: `apply_grammar_bitmask` already computes the exact LOGIT ROWS of every
structured request (`struct_out_req_batch_indices`, plus each row's spec-decode
span) — the same coordinate space as the holder's `mask_idx`. Publish those
rows, and have `_apply_forcing_to_logits` skip them. The budget then simply
does not force inside a constrained region; it resumes forcing as soon as the
grammar is no longer active for that request.

BOOT ORDER IS LOAD-BEARING: this must run AFTER patch_pr44812_tool_guard.py
and BEFORE patch_pn114_forced_span.py. PN114's graft F2 rewrites the very line
graft D anchors on (`active_indices_cpu.append(mask_idx)` gains an indent and a
`# PN114 F2` tail), so PN122-after-PN114 exits 1 on anchor drift — verified
against the prod pin (image sha256:e4f8554…) by applying the real chain both
ways: correct order rc=0, reversed rc=1.

Targets (in-container):
  A) vllm/v1/structured_output/utils.py — module-level row registry
  B) vllm/v1/structured_output/utils.py — publish rows (RUNTIME-GATED on
                                          GENESIS_ENABLE_PN122_STRUCTURED_FORCE_GUARD)
  C) vllm/v1/worker/gpu_model_runner.py  — clear the registry every step, so a
                                           step with no grammar can never
                                           inherit the previous step's rows
  D) vllm/v1/sample/thinking_budget_state.py — skip registered rows

Dark by default: with the env flag unset (B) never publishes, so the registry
stays empty and (D) changes nothing. Flipping the flag ON is a BEHAVIOURAL
change (structured requests stop being budget-clamped mid-object) and wants a
bench arm before it ships — see AUDIT-leak-paths-20260726.md §L6.

The registry is deliberately a plain module global, not holder state: the
holder is constructed once per engine while the rows change every step, and
(C) guarantees it is cleared before each forward's sampling.

Idempotent by marker; anchor drift = FATAL exit 1, mirroring
patch_pr44812_tool_guard.py.
"""
import pathlib
import sys

LOG = "[patch_pn122_structured_force_guard]"
BASE = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")

SOUTIL = BASE / "v1/structured_output/utils.py"
RUNNER = BASE / "v1/worker/gpu_model_runner.py"
HOLDER = BASE / "v1/sample/thinking_budget_state.py"

# ── A: module-level registry ────────────────────────────────────────────────
MARK_A = "# PN122 graft: row registry"
ANCH_A = "def apply_grammar_bitmask(\n"
REPL_A = (
    "# PN122 graft: row registry — logits rows under an active grammar this\n"
    "# step. Written by apply_grammar_bitmask, cleared by the model runner\n"
    "# before every forward, read by the thinking-budget holder.\n"
    "PN122_STRUCTURED_ROWS: set[int] = set()\n"
    "\n"
    "\n"
    "def apply_grammar_bitmask(\n"
)

# ── B: publish the rows (runtime gate lives HERE) ───────────────────────────
MARK_B = "# PN122 graft: publish rows"
ANCH_B = (
    "            struct_out_req_batch_indices[req_id] = logit_index\n"
    "\n"
    "    out_indices = []\n"
)
REPL_B = (
    "            struct_out_req_batch_indices[req_id] = logit_index\n"
    "\n"
    "    # PN122 graft: publish rows — every logits row this step that is under\n"
    "    # a grammar, INCLUDING the request's speculative span (the holder can\n"
    "    # force onto any offset inside it). Empty unless the flag is set.\n"
    "    try:\n"
    "        import os as _pn122_os\n"
    "        if _pn122_os.environ.get(\n"
    "            'GENESIS_ENABLE_PN122_STRUCTURED_FORCE_GUARD', ''\n"
    "        ).strip().lower() in ('1', 'true', 'yes', 'on'):\n"
    "            for _pn122_req, _pn122_row in struct_out_req_batch_indices.items():\n"
    "                _pn122_span = len(spec_tokens.get(_pn122_req, ()))\n"
    "                PN122_STRUCTURED_ROWS.update(\n"
    "                    range(_pn122_row, _pn122_row + _pn122_span + 1)\n"
    "                )\n"
    "    except Exception:\n"
    "        import logging as _pn122_log\n"
    "        _pn122_log.getLogger('vllm.pn122').warning(\n"
    "            'PN122 row publish raised', exc_info=True\n"
    "        )\n"
    "\n"
    "    out_indices = []\n"
)

# ── C: clear the registry every step ────────────────────────────────────────
MARK_C = "# PN122 graft: clear rows"
ANCH_C = (
    "        # Apply structured output bitmasks if present.\n"
    "        if grammar_output is not None:\n"
    "            apply_grammar_bitmask(\n"
)
REPL_C = (
    "        # PN122 graft: clear rows — must happen on EVERY step, including\n"
    "        # steps with no grammar, or a later unconstrained batch inherits\n"
    "        # the previous step's rows and silently loses budget enforcement.\n"
    "        try:\n"
    "            from vllm.v1.structured_output import utils as _pn122_sou\n"
    "            _pn122_sou.PN122_STRUCTURED_ROWS.clear()\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
    "        # Apply structured output bitmasks if present.\n"
    "        if grammar_output is not None:\n"
    "            apply_grammar_bitmask(\n"
)

# ── D: holder skips registered rows ─────────────────────────────────────────
MARK_D = "# PN122 graft: skip grammar row"
ANCH_D = (
    "                            if (\n"
    "                                mask_idx < self._mask_capacity\n"
    "                                and mask_idx < logits.shape[0]\n"
    "                            ):\n"
    "                                active_indices_cpu.append(mask_idx)\n"
)
REPL_D = (
    "                            if (\n"
    "                                mask_idx < self._mask_capacity\n"
    "                                and mask_idx < logits.shape[0]\n"
    "                                # PN122 graft: skip grammar row — forcing\n"
    "                                # </think> here would overwrite the\n"
    "                                # xgrammar mask applied moments earlier\n"
    "                                # and emit an off-grammar token into the\n"
    "                                # middle of tool-call arguments.\n"
    "                                and not self._pn122_row_masked(mask_idx)\n"
    "                            ):\n"
    "                                active_indices_cpu.append(mask_idx)\n"
)

# ── D2: the helper D calls (fail-open) ──────────────────────────────────────
MARK_D2 = "# PN122 graft: row helper"
ANCH_D2 = "    def _apply_forcing_to_logits(\n"
REPL_D2 = (
    "    @staticmethod\n"
    "    def _pn122_row_masked(mask_idx: int) -> bool:\n"
    "        # PN122 graft: row helper — True when this logits row is under an\n"
    "        # active grammar this step. Fail-OPEN (return False) on any error:\n"
    "        # losing the guard is a rare corrupted tool call, losing the force\n"
    "        # is an unbounded think block on every request.\n"
    "        try:\n"
    "            from vllm.v1.structured_output import utils as _pn122_sou\n"
    "            return mask_idx in _pn122_sou.PN122_STRUCTURED_ROWS\n"
    "        except Exception:\n"
    "            return False\n"
    "\n"
    "    def _apply_forcing_to_logits(\n"
)

GRAFTS = [
    (SOUTIL, MARK_A, ANCH_A, REPL_A, "A row registry"),
    (SOUTIL, MARK_B, ANCH_B, REPL_B, "B publish rows"),
    (RUNNER, MARK_C, ANCH_C, REPL_C, "C clear rows per step"),
    (HOLDER, MARK_D2, ANCH_D2, REPL_D2, "D2 holder row helper"),
    (HOLDER, MARK_D, ANCH_D, REPL_D, "D holder skip grammar row"),
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
        print(
            f"{LOG} FATAL: anchor occurs {count}x (need exactly 1) for {what} "
            f"in {target.name}",
            flush=True,
        )
        return 1
    assert marker in repl, f"marker missing from replacement for {what}"
    target.write_text(src.replace(anchor, repl, 1), encoding="utf-8")
    print(f"{LOG} applied: {what}", flush=True)
    return 0


def _drop_stale_pyc() -> None:
    """Same-second pyc race (2026-07-22): boot scripts import vllm before the
    text patches rewrite these files."""
    for target in (SOUTIL, RUNNER, HOLDER):
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
