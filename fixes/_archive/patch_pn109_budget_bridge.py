#!/usr/bin/env python3
"""PN109 — bridge request.thinking_token_budget into SamplingParams.

RETIRED 2026-07-20 (BUG-107h): the 107e diagnosis this patch fixed was a
MISDIAGNOSIS — this pin's chat_completion/protocol.py natively carries
thinking_token_budget (field :244 -> to_sampling_params :774); the "zero
reads" grep hit the legacy flat openai/protocol.py path, which does not
exist on this pin (grep over a missing file reads as zero matches).
The holder has tracked requests since PN100 shipped; this patch's
client-wins guard finds the param already set and skips — that is why
"PN109: bridged" never logged. Kept in _archive because it becomes
load-bearing again ONLY if a future pin drops the protocol field
(detection: re-run the 107e greps against chat_completion/protocol.py).

BUG-107e (2026-07-20, second live window): SamplingParams HAS a native
thinking_token_budget field and the engine-side ThinkingBudgetStateHolder
enforces it — but the OpenAI protocol layer never carries the value:
protocol.py and serving.py contain ZERO reads of thinking_token_budget, so
PN100's `request.thinking_token_budget = N` (auto_budget.py:340/376) dies at
the API boundary. Consequence, proven live: the holder tracked ZERO requests
across two windows (9 real think blocks, 0 observed) — N67, the +1e9 forcing
and the PN108 graft have been unreachable on the auto path since PN100
shipped; every observed "budget respected" behaviour was ThinkingCap's
TRAINED self-capping plus the PN102 banner, not engine enforcement.

This patch inserts the one missing hop: right after serving.py builds
sampling_params, copy the request attribute over (only when a positive int
and not already client-set). Gate check downstream (input_processor.py:102)
passes on this deployment: --reasoning-parser qwen3 derives <think>/</think>
via initialize_token_ids -> reasoning_config._enabled=True.

BEHAVIORAL when enabled: tier budgets become ENGINE-ENFORCED hard caps for
the first time (10240 on tiers 1-3) — bench-gated, ships env-dark:
GENESIS_ENABLE_PN109_BUDGET_BRIDGE=1 to engage. Budget 0 (tier-0 off) is
deliberately NOT bridged — enable_thinking=false already handles it and a
0-budget holder entry force-closes at step 0, which is redundant risk.
Fail-open at runtime; fail-LOUD at boot on anchor drift (house style).
"""
import pathlib
import sys

LOG = "[pn109-budget-bridge]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/"
    "chat_completion/serving.py"
)
MARKER = "# PN109:"

ANCHOR = (
    "                sampling_params = request.to_sampling_params(\n"
    "                    max_tokens,\n"
    "                    self.default_sampling_params,\n"
    "                )\n"
)
REPLACEMENT = ANCHOR + (
    "                # PN109: bridge request.thinking_token_budget ->\n"
    "                # SamplingParams (BUG-107e — the protocol layer never\n"
    "                # carries it, so the engine budget holder tracked zero\n"
    "                # requests). Positive ints only; client-set params win;\n"
    "                # inert unless GENESIS_ENABLE_PN109_BUDGET_BRIDGE=1.\n"
    "                try:\n"
    "                    import os as _pn109_os\n"
    "                    if _pn109_os.environ.get(\n"
    "                        'GENESIS_ENABLE_PN109_BUDGET_BRIDGE', ''\n"
    "                    ).strip().lower() in ('1', 'true', 'yes', 'on'):\n"
    "                        _pn109_b = getattr(\n"
    "                            request, 'thinking_token_budget', None)\n"
    "                        if (isinstance(_pn109_b, int)\n"
    "                                and _pn109_b > 0\n"
    "                                and sampling_params.thinking_token_budget\n"
    "                                is None):\n"
    "                            sampling_params.thinking_token_budget = _pn109_b\n"
    "                            # vllm's logger prints INFO in-server;\n"
    "                            # plain root logger may not (BUG-107g)\n"
    "                            try:\n"
    "                                from vllm.logger import (\n"
    "                                    init_logger as _pn109_il)\n"
    "                                _pn109_log = _pn109_il(\n"
    "                                    'vllm.genesis.pn109')\n"
    "                            except Exception:\n"
    "                                import logging as _pn109_lg\n"
    "                                _pn109_log = _pn109_lg.getLogger(\n"
    "                                    'genesis.pn109')\n"
    "                            _pn109_log.info(\n"
    "                                'PN109: bridged thinking_token_budget=%d',\n"
    "                                _pn109_b)\n"
    "                except Exception:\n"
    "                    import logging as _pn109_lg2\n"
    "                    _pn109_lg2.getLogger('genesis.pn109').debug(\n"
    "                        'PN109 bridge raised; ignored', exc_info=True)\n"
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
    print(f"{LOG} applied — budget bridge inserted after to_sampling_params",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
