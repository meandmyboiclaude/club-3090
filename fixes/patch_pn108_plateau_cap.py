#!/usr/bin/env python3
"""PN108 — call-site for the plateau-triggered dynamic thinking cap.

Inserts one observe call into ThinkingBudgetStateHolder.update_state, right
before self._update_think_state(state). The detector itself lives in the
mounted Genesis tree (vllm/_genesis/plateau/pn108.py): rolling new-trigram
novelty over the active think stream; on sustained collapse it lowers the
request's own thinking_token_budget to think_count(+grace), and the holder's
existing spec-aware forcing closes the segment — byte-identical to a static
cap hit, the proven MTP-safe path.

Why here: custom logits processors are hard-rejected under spec decode on
this pin (STR_SPEC_DEC_REJECTS_LOGITSPROCS), and prod runs MTP n=3 — grafting
into the holder is the only spec-compatible seat. The anchor call site runs
in sample_tokens' eager phase (outside cudagraph capture), after the spec
suffix strip, so the detector sees clean accepted tokens.

Fail-open at runtime (observe_state swallows nothing itself but is wrapped
here in try/except); fail-LOUD at boot if the anchor drifts (house style).
Gate: GENESIS_ENABLE_PN108_PLATEAU_CAP (patch applies always, hook no-ops
while the gate is unset — ship-dark).

Bench context: v6a/v6b instruction-side candidates KILLED −9/−10 (2026-07-19
window); PN108 is the observation-side successor. Calibration defaults in
the module docstring; see ~/shared/pn108/.
"""
import pathlib
import sys

LOG = "[pn108-plateau-cap]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/"
    "thinking_budget_state.py"
)
MARKER = "# PN108:"

ANCHOR = "            self._update_think_state(state)\n"
REPLACEMENT = (
    "            # PN108: plateau-triggered dynamic cap (house — see\n"
    "            # _genesis/plateau/pn108.py). Observation-side: watches the\n"
    "            # think stream's trigram novelty; on sustained collapse it\n"
    "            # lowers this request's thinking_token_budget to what is\n"
    "            # already spent, and _update_think_state below forces the\n"
    "            # close through the normal spec-aware machinery. Inert\n"
    "            # unless GENESIS_ENABLE_PN108_PLATEAU_CAP=1. Fail-open.\n"
    "            try:\n"
    "                from vllm._genesis.plateau import pn108 as _pn108\n"
    "                _pn108.observe_state(\n"
    "                    state, len(self.think_start_token_ids), seq_idx\n"
    "                )\n"
    "            except Exception:\n"
    "                import logging as _pn108_logging\n"
    "                _pn108_logging.getLogger(\n"
    "                    'genesis.plateau.pn108'\n"
    "                ).debug('PN108 observe raised; ignored', exc_info=True)\n"
    "            self._update_think_state(state)\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: target missing: {TARGET}", flush=True)
        return 1
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"{LOG} already applied — skipping", flush=True)
        return 0
    count = src.count(ANCHOR)
    if count != 1:
        print(
            f"{LOG} FATAL: anchor occurs {count}x (need exactly 1) — "
            "upstream drifted; re-anchor before boot",
            flush=True,
        )
        return 1
    TARGET.write_text(src.replace(ANCHOR, REPLACEMENT, 1), encoding="utf-8")
    print(f"{LOG} applied — observe hook inserted before _update_think_state",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
