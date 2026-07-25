#!/usr/bin/env python3
"""PN91g — clamp spec-decode recurrent-state index for zero
num_accepted_tokens in the GDN/FLA fused kernels.

Backport of vllm#48475 to nightly-9e57de71 (dev1060). ('g' suffix avoids a
name clash with Genesis P91.)

In fused_recurrent_gated_delta_rule_fwd_kernel and
fused_sigmoid_gating_delta_rule_update_kernel (IS_SPEC_DECODING path), the
initial-state slot is selected as

    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1

A zero entry in num_accepted_tokens (stale or padded batch row) yields
i_t = -1, which indexes OUTSIDE this request's row of ssm_state_indices —
i.e. reads the previous request's last slot (or OOB for row 0): a real
crash/corruption class on our MTP K=3 + GDN hybrid path. Upstream fix
clamps with tl.maximum(..., 0), matching the clamp already used by the
align kernel in vllm/v1/worker/mamba_utils.py. Zero-accepted entries then
read slot 0, exactly like an entry of 1 (verified by upstream's
test_spec_decoding_zero_accepted_tokens_reads_first_slot).

Anchors verified byte-exact in-image (both files, line ~106, unique)
2026-07-13. Retire when upstream contains the tl.maximum clamp in both
files — this patcher then self-retires per file.
"""
import pathlib
import sys

LOG = "[pn91g-gdn-spec-state-index-clamp]"
VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")


def _fla(name: str) -> pathlib.Path:
    # [2026-07-25] vllm#48500 moved fla ops to third_party/ — resolve
    # whichever home this image has (old path = dev1060cherry/prod).
    new = VLLM / "third_party/flash_linear_attention/ops" / name
    return new if new.exists() else VLLM / "model_executor/layers/fla/ops" / name


TARGETS = (
    _fla("fused_recurrent.py"),
    _fla("fused_sigmoid_gating.py"),
)
MARKER = "# PN91g:"

OLD = "                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1\n"
NEW = (
    "                # PN91g: vllm#48475 backport — clamp so a zero (stale or\n"
    "                # padded batch row) entry in num_accepted_tokens cannot\n"
    "                # index outside this request's row of ssm_state_indices;\n"
    "                # matches the clamp in the align kernel in\n"
    "                # vllm/v1/worker/mamba_utils.py.\n"
    "                i_t = tl.maximum(tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1, 0)\n"
)

UPSTREAM_FIX = "tl.maximum(tl.load(num_accepted_tokens + i_n)"


def main() -> int:
    rc = 0
    for target in TARGETS:
        if not target.exists():
            print(f"{LOG} FATAL: {target} not present", file=sys.stderr)
            return 1
        text = target.read_text()
        if MARKER in text:
            print(f"{LOG} {target.name}: already applied (idempotent)")
            continue
        if UPSTREAM_FIX in text:
            print(
                f"{LOG} {target.name}: upstream drift — clamp already present, "
                f"self-retire (no-op)"
            )
            continue
        if OLD not in text:
            print(
                f"{LOG} FATAL: anchor-not-found in {target.name} — upstream "
                f"refactor; re-derive before boot (zero num_accepted_tokens "
                f"reads a foreign recurrent-state slot under MTP+GDN)",
                file=sys.stderr,
            )
            rc = 1
            continue
        if text.count(OLD) != 1:
            print(
                f"{LOG} FATAL: ambiguous anchor in {target.name} "
                f"({text.count(OLD)} matches)",
                file=sys.stderr,
            )
            rc = 1
            continue
        target.write_text(text.replace(OLD, NEW, 1))
        print(
            f"{LOG} {target.name}: applied — spec-decode initial-state index "
            f"clamped to >= 0 (vllm#48475 backport)"
        )
    return rc


sys.exit(main())
