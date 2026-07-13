#!/usr/bin/env python3
"""PN89 — vllm#46324 (align spec-decode cudagraph capture sizes for PIECEWISE):
documented NO-OP stub — redundant with Genesis P66.

Upstream #46324 (+7/-1, vllm/config/compilation.py) widens the condition
guarding adjust_cudagraph_sizes_for_spec_decode() in
resolve_cudagraph_mode_and_sizes() from
    cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
to
    cudagraph_mode != CUDAGraphMode.NONE
so PIECEWISE decode graphs also get capture sizes rounded to multiples of
uniform_decode_query_len. Without it, a partial-acceptance spec-decode step
dispatches to a graph keyed on a non-multiple size → slot_mapping/positions
read at offsets from the wrong graph → cudaErrorIllegalAddress (issue #28207).

Why NO-OP here: Genesis P66 (opt-in via GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER,
set =1 in every compose file in this repo) already filters
cudagraph_capture_sizes at config time (config/vllm.py::_set_cudagraph_sizes)
to sizes divisible by uniform_decode_query_len, MODE-INDEPENDENTLY — it runs
before and regardless of the FULL-vs-PIECEWISE resolution. Boot log proof:
"[Genesis P66] Filtered cudagraph_capture_sizes ... kept [4, 8, 16, 24, 32, 40]".
On a P66-filtered list, upstream's adjust (round_up to multiple + dedup) is
the identity, so #46324 adds nothing; applying it anyway would only invite
double-filtering drift. Per operator decision 2026-07-13: documented no-op.

What this stub still enforces (FAIL-LOUD safety):
  1. self-retire if upstream already carries the #46324 widening;
  2. verify the un-widened anchor is still recognizable (drift detection);
  3. verify P66 actually covers this deployment (P66 marker present in
     config/vllm.py after genesis apply_all, or the P66 env flag set).
     If NEITHER upstream nor P66 protects capture-size divisibility,
     exit 1 — spec-decode + PIECEWISE would be exposed to #28207.
"""
import os
import pathlib
import sys

LOG = "[pn89-piecewise-spec-capture-sizes]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py"
)
GENESIS_P66_TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py"
)
P66_ENV = "GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER"

ANCHOR_UNFIXED = (
    "        if (\n"
    "            not use_v2_model_runner\n"
    "            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL\n"
    "            and uniform_decode_query_len > 1\n"
    "        ):\n"
    "            self.adjust_cudagraph_sizes_for_spec_decode(\n"
)
ANCHOR_FIXED = (
    "            not use_v2_model_runner\n"
    "            and cudagraph_mode != CUDAGraphMode.NONE\n"
    "            and uniform_decode_query_len > 1\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    text = TARGET.read_text()

    if ANCHOR_FIXED in text:
        print(f"{LOG} upstream drift: vllm#46324 widening already present "
              f"— self-retire (no-op)")
        return 0

    if ANCHOR_UNFIXED not in text:
        print(f"{LOG} FATAL: neither the #46324-fixed form nor the un-widened "
              f"anchor found in resolve_cudagraph_mode_and_sizes — upstream "
              f"refactor; re-derive the P66-vs-#46324 redundancy analysis "
              f"before boot", file=sys.stderr)
        return 1

    p66_marker = False
    if GENESIS_P66_TARGET.exists():
        p66_marker = "[Genesis P66]" in GENESIS_P66_TARGET.read_text()
    p66_env = os.environ.get(P66_ENV, "") == "1"

    if p66_marker or p66_env:
        via = "marker in config/vllm.py" if p66_marker else f"{P66_ENV}=1"
        print(f"{LOG} no-op by design: Genesis P66 covers capture-size "
              f"divisibility for ALL cudagraph modes ({via}); vllm#46324 "
              f"is redundant — not double-filtering")
        return 0

    print(f"{LOG} FATAL: vllm#46324 absent upstream AND Genesis P66 inactive "
          f"(no [Genesis P66] marker in {GENESIS_P66_TARGET}, {P66_ENV} unset) "
          f"— spec-decode + PIECEWISE cudagraphs exposed to non-multiple "
          f"capture sizes (vllm issue #28207, cudaErrorIllegalAddress). "
          f"Set {P66_ENV}=1 (compose default) or backport #46324 for real.",
          file=sys.stderr)
    return 1


sys.exit(main())
