#!/usr/bin/env python3
"""PR44812 graft (2026-07-23) — tool-call implicit reasoning-end guard.

Upstream vllm-project/vllm#44812 (open, merge-conflicted, stale since 07-11):
when Qwen3 emits <tool_call> INSIDE a think block, the thinking-budget holder
keeps counting "reasoning" tokens and can force-close </think> into the tool
call (~75% repro at budget~=256 upstream). The fix teaches the holder that
parser-declared strings (Qwen3: <tool_call>, thinking-mode only) implicitly
terminate reasoning.

Grafted with the KNOWN BUG FIXED (flagged in-PR, unfixed upstream): the
implicit-terminator search must scan the SAME scan_offset slice as the
explicit </think> search. Upstream scans the full output_tok_ids, so a stale
pre-thinking <tool_call> from an earlier segment permanently satisfies
end_thinking and locks the holder out of thinking-mode detection.

Targets (in-container, applied at boot AFTER patch_pn112_conf_tap.py):
  A) vllm/parser/qwen3.py           — implicit_reasoning_end_strs property
  B) vllm/config/reasoning.py       — collect+tokenize ids (RUNTIME-GATED on
                                      GENESIS_ENABLE_PR44812_TOOL_GUARD)
  C) holder __init__                — implicit_think_end_token_ids attr
  D) holder helper                  — _find_last_any_sequence_index
  E) holder _update_think_state     — implicit-end max() fold, slice-fixed

Dark by default: with the env flag unset, (B) never populates the id list,
so (C)-(E) see an empty list and change nothing. Delete this file if #44812
ever merges upstream (rebase-milestone manifest step 0).

Idempotent by marker; anchor drift = FATAL exit 1 (loud bad boot), mirroring
patch_pn108_plateau_cap.py / patch_pn112_conf_tap.py.
"""
import pathlib
import sys

LOG = "[patch_pr44812_tool_guard]"
BASE = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")

QWEN3 = BASE / "parser/qwen3.py"
RCONF = BASE / "config/reasoning.py"
HOLDER = BASE / "v1/sample/thinking_budget_state.py"

# ── A: qwen3 parser property ────────────────────────────────────────────────
MARK_A = "# PR44812 graft: implicit strs"
ANCH_A = (
    "        return super().extract_reasoning(model_output, request)\n"
    "\n"
    "    def is_reasoning_end(self, input_ids: list[int]) -> bool:\n"
)
REPL_A = (
    "        return super().extract_reasoning(model_output, request)\n"
    "\n"
    "    @property\n"
    "    def implicit_reasoning_end_strs(self) -> list[str]:\n"
    "        # PR44812 graft: implicit strs that terminate reasoning without\n"
    "        # an emitted </think> (thinking mode only).\n"
    "        if not self.thinking_enabled:\n"
    "            return []\n"
    "        return [TOOL_CALL_START]\n"
    "\n"
    "    def is_reasoning_end(self, input_ids: list[int]) -> bool:\n"
)

# ── B: reasoning config collection (runtime gate lives HERE) ────────────────
MARK_B = "# PR44812 graft: collect implicit"
ANCH_B = (
    "            end_token = reasoning_parser.reasoning_end_str\n"
    "            if end_token and not reasoning_end_str:\n"
    "                reasoning_end_str = end_token\n"
)
REPL_B = (
    "            end_token = reasoning_parser.reasoning_end_str\n"
    "            if end_token and not reasoning_end_str:\n"
    "                reasoning_end_str = end_token\n"
    "\n"
    "            # PR44812 graft: collect implicit reasoning terminators from\n"
    "            # the parser (Qwen3 -> <tool_call>). Runtime-gated: with the\n"
    "            # flag unset the attr stays absent and the holder no-ops.\n"
    "            try:\n"
    "                import os as _pr44812_os\n"
    "                if _pr44812_os.environ.get(\n"
    "                    'GENESIS_ENABLE_PR44812_TOOL_GUARD', ''\n"
    "                ).strip().lower() in ('1', 'true', 'yes', 'on'):\n"
    "                    _pr44812_ids = [\n"
    "                        _t\n"
    "                        for _t in (\n"
    "                            tokenizer.encode(\n"
    "                                _s, add_special_tokens=False\n"
    "                            )\n"
    "                            for _s in getattr(\n"
    "                                reasoning_parser,\n"
    "                                'implicit_reasoning_end_strs',\n"
    "                                [],\n"
    "                            )\n"
    "                        )\n"
    "                        if _t\n"
    "                    ]\n"
    "                    try:\n"
    "                        self.implicit_reasoning_end_token_ids = _pr44812_ids\n"
    "                    except Exception:\n"
    "                        object.__setattr__(\n"
    "                            self,\n"
    "                            'implicit_reasoning_end_token_ids',\n"
    "                            _pr44812_ids,\n"
    "                        )\n"
    "                    import logging as _pr44812_ilog\n"
    "                    _pr44812_ilog.getLogger('vllm.pr44812').info(\n"
    "                        'PR44812: implicit reasoning-end ids %s',\n"
    "                        _pr44812_ids,\n"
    "                    )\n"
    "            except Exception:\n"
    "                import logging as _pr44812_log\n"
    "                _pr44812_log.getLogger('vllm.pr44812').warning(\n"
    "                    'PR44812 implicit-end collection raised', exc_info=True\n"
    "                )\n"
)

# ── C: holder __init__ attr ─────────────────────────────────────────────────
MARK_C = "# PR44812 graft: holder attr"
ANCH_C = (
    "            self.think_end_token_ids = re if re else []\n"
    "\n"
    "        self.device = device\n"
)
REPL_C = (
    "            self.think_end_token_ids = re if re else []\n"
    "\n"
    "        # PR44812 graft: holder attr — implicit terminators; empty unless\n"
    "        # GENESIS_ENABLE_PR44812_TOOL_GUARD collected them at init.\n"
    "        self.implicit_think_end_token_ids: list[list[int]] = (\n"
    "            getattr(\n"
    "                reasoning_config, 'implicit_reasoning_end_token_ids', []\n"
    "            )\n"
    "            or []\n"
    "        )\n"
    "\n"
    "        self.device = device\n"
)

# ── D: holder helper classmethod ────────────────────────────────────────────
MARK_D = "# PR44812 graft: any-sequence helper"
ANCH_D = "    def _init_state_entry(\n"
REPL_D = (
    "    @classmethod\n"
    "    def _find_last_any_sequence_index(\n"
    "        cls, target_list: list[int], token_id_sequences: list[list[int]]\n"
    "    ) -> int:\n"
    "        # PR44812 graft: any-sequence helper — latest match across all\n"
    "        # implicit terminator sequences, -1 when none/empty.\n"
    "        return max(\n"
    "            (\n"
    "                cls._find_last_sequence_index(target_list, token_ids)\n"
    "                for token_ids in token_id_sequences\n"
    "            ),\n"
    "            default=-1,\n"
    "        )\n"
    "\n"
    "    def _init_state_entry(\n"
)

# ── E: holder _update_think_state fold (slice-fixed) ────────────────────────
MARK_E = "# PR44812 graft (slice fix)"
ANCH_E = (
    "        if state[\"end_thinking\"] == -1:\n"
    "            scan_offset = state.get(\"scan_offset\", 0)\n"
    "            output_slice = state.get(\"output_tok_ids\", [])[scan_offset:]\n"
    "            end_thinking = self._find_last_sequence_index(\n"
    "                output_slice, self.think_end_token_ids\n"
    "            )\n"
    "            if end_thinking >= 0:\n"
    "                end_thinking += scan_offset\n"
    "            state[\"end_thinking\"] = end_thinking\n"
)
REPL_E = (
    "        if state[\"end_thinking\"] == -1:\n"
    "            scan_offset = state.get(\"scan_offset\", 0)\n"
    "            output_slice = state.get(\"output_tok_ids\", [])[scan_offset:]\n"
    "            end_thinking = self._find_last_sequence_index(\n"
    "                output_slice, self.think_end_token_ids\n"
    "            )\n"
    "            if end_thinking >= 0:\n"
    "                end_thinking += scan_offset\n"
    "            # PR44812 graft (slice fix): implicit terminators scan the\n"
    "            # SAME post-offset slice; scanning full output_tok_ids would\n"
    "            # let a stale pre-thinking <tool_call> lock out detection.\n"
    "            if self.implicit_think_end_token_ids:\n"
    "                implicit_end_thinking = self._find_last_any_sequence_index(\n"
    "                    output_slice, self.implicit_think_end_token_ids\n"
    "                )\n"
    "                if implicit_end_thinking >= 0:\n"
    "                    implicit_end_thinking += scan_offset\n"
    "                    end_thinking = max(end_thinking, implicit_end_thinking)\n"
    "            state[\"end_thinking\"] = end_thinking\n"
)

GRAFTS = [
    (QWEN3, MARK_A, ANCH_A, REPL_A, "A qwen3 implicit strs"),
    (RCONF, MARK_B, ANCH_B, REPL_B, "B reasoning-config collect"),
    (HOLDER, MARK_C, ANCH_C, REPL_C, "C holder attr"),
    (HOLDER, MARK_D, ANCH_D, REPL_D, "D holder helper"),
    (HOLDER, MARK_E, ANCH_E, REPL_E, "E update_think_state fold"),
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
    text patches rewrite these files; a rewrite landing within the same mtime
    second leaves a stale pyc that survives timestamp validation."""
    for target in (QWEN3, RCONF, HOLDER):
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
