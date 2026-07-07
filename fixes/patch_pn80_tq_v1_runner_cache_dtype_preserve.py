#!/usr/bin/env python3
"""PN80 — preserve TQ KV-cache dtype in the V1 runner reshape path.

Backport of vllm#47609 (fa4321de3, merged 2026-07-05) to the V1
gpu_model_runner. Upstream's fix only covers the Model Runner V2 worker
(vllm/v1/worker/gpu/attn_utils.py); the V1 runner has the identical bug:

The --kv-cache-dtype-skip-layers feature (#46xxx) computes
    layer_cache_dtype_str = "auto" if kv_cache_spec.kv_quant_mode == NONE
                            else cache_dtype
in _reshape_kv_cache_tensors. TQFullAttentionSpec encodes its quantization
in the spec subclass (kv_quant_mode stays NONE), so TQ layers get
cache_dtype_str='auto' -> TurboQuantConfig.from_cache_dtype('auto') raises
ValueError at boot ("Unknown TurboQuant cache dtype: 'auto'").

Observed: nightly-69715823 (dev799) boot crash in
_init_minimal_kv_cache_for_profiling on TQ3 KV. Retire when upstream applies
the #47609 exemption to the V1 runner path.
"""
import pathlib
import sys

LOG = "[pn80-tq-v1-runner-cache-dtype]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
)
MARKER = "# PN80:"

IMPORT_OLD = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    AttentionSpec,\n"
)
IMPORT_NEW = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    AttentionSpec,\n"
    "    TQFullAttentionSpec,  # PN80: vllm#47609 backport (V1 runner path)\n"
)

OLD = (
    "                    layer_cache_dtype_str = (\n"
    '                        "auto"\n'
    "                        if kv_cache_spec.kv_quant_mode == KVQuantMode.NONE\n"
    "                        else self.cache_config.cache_dtype\n"
    "                    )\n"
)
NEW = (
    "                    # PN80: vllm#47609 backport — TQ specs keep kv_quant_mode==NONE\n"
    "                    # but MUST receive the real cache_dtype (TQ backend rejects 'auto').\n"
    "                    layer_cache_dtype_str = (\n"
    '                        "auto"\n'
    "                        if kv_cache_spec.kv_quant_mode == KVQuantMode.NONE\n"
    "                        and not isinstance(kv_cache_spec, TQFullAttentionSpec)\n"
    "                        else self.cache_config.cache_dtype\n"
    "                    )\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    text = TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} already applied (idempotent)")
        return 0
    if "def _reshape_kv_cache_tensors" in text:
        body = text.split("def _reshape_kv_cache_tensors", 1)[1][:6000]
        if "TQFullAttentionSpec" in body:
            print(f"{LOG} upstream drift: exemption already present — self-retire (no-op)")
            return 0
    for name, anchor in (("import", IMPORT_OLD), ("dtype-select", OLD)):
        if anchor not in text:
            print(f"{LOG} FATAL: anchor-not-found ({name}) — upstream refactor; "
                  f"re-derive before boot (TQ3 boot WILL crash without this fix)",
                  file=sys.stderr)
            return 1
        if text.count(anchor) != 1:
            print(f"{LOG} FATAL: ambiguous anchor ({name})", file=sys.stderr)
            return 1
    text = text.replace(IMPORT_OLD, IMPORT_NEW, 1).replace(OLD, NEW, 1)
    TARGET.write_text(text)
    print(f"{LOG} applied: TQFullAttentionSpec exempted from 'auto' cache-dtype "
          f"in _reshape_kv_cache_tensors (vllm#47609 V1-runner backport)")
    return 0


sys.exit(main())
