#!/usr/bin/env python3
"""PN80 — preserve TQ KV-cache dtype in the V1 runner reshape path.

Backport of vllm#47609 (fa4321de3, merged 2026-07-05) to the V1
gpu_model_runner. Upstream's fix only covers the Model Runner V2 worker
(vllm/v1/worker/gpu/attn_utils.py); the V1 runner has the identical bug:

The --kv-cache-dtype-skip-layers feature (#46xxx) computes
    layer_cache_dtype_str = "auto" if kv_cache_spec.kv_quant_mode == NONE
                            else <cache dtype>
in _reshape_kv_cache_tensors. TQFullAttentionSpec encodes its quantization
in the spec subclass (kv_quant_mode stays NONE), so TQ layers get
cache_dtype_str='auto' -> TurboQuantConfig.from_cache_dtype('auto') raises
ValueError at boot ("Unknown TurboQuant cache dtype: 'auto'").

Observed: nightly-69715823 (dev799) boot crash in
_init_minimal_kv_cache_for_profiling on TQ3 KV. Retire when upstream applies
the #47609 exemption to the V1 runner path.

[2026-07-13 rebase for nightly-9e57de71 (dev1060)] 51878e5b (KV layout
refactor) + follow-ups rewrote the dtype-select to add a
getattr(spec, "cache_dtype_str", ...) fallback — but TQFullAttentionSpec
still has kv_quant_mode==NONE and no cache_dtype_str attr (verified in
image), so the 'auto' branch still fires and TQ3 boot still crashes.
Patcher now carries BOTH anchor forms (dev799 + dev1060) so rollback to the
validated dev799 image keeps working.
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

# (OLD, NEW) anchor candidates, newest first. First OLD found in the file wins.
DTYPE_SELECT_FORMS = (
    # nightly-9e57de71 (dev1060, post-51878e5b) form
    (
        "                    layer_cache_dtype_str = (\n"
        '                        "auto"\n'
        "                        if kv_cache_spec.kv_quant_mode == KVQuantMode.NONE\n"
        "                        else getattr(\n"
        "                            kv_cache_spec,\n"
        '                            "cache_dtype_str",\n'
        "                            None,\n"
        "                        )\n"
        "                        or self.cache_config.cache_dtype\n"
        "                    )\n",
        "                    # PN80: vllm#47609 backport — TQ specs keep kv_quant_mode==NONE\n"
        "                    # but MUST receive the real cache_dtype (TQ backend rejects 'auto').\n"
        "                    layer_cache_dtype_str = (\n"
        '                        "auto"\n'
        "                        if kv_cache_spec.kv_quant_mode == KVQuantMode.NONE\n"
        "                        and not isinstance(kv_cache_spec, TQFullAttentionSpec)\n"
        "                        else getattr(\n"
        "                            kv_cache_spec,\n"
        '                            "cache_dtype_str",\n'
        "                            None,\n"
        "                        )\n"
        "                        or self.cache_config.cache_dtype\n"
        "                    )\n",
    ),
    # nightly-69715823 (dev799) form
    (
        "                    layer_cache_dtype_str = (\n"
        '                        "auto"\n'
        "                        if kv_cache_spec.kv_quant_mode == KVQuantMode.NONE\n"
        "                        else self.cache_config.cache_dtype\n"
        "                    )\n",
        "                    # PN80: vllm#47609 backport — TQ specs keep kv_quant_mode==NONE\n"
        "                    # but MUST receive the real cache_dtype (TQ backend rejects 'auto').\n"
        "                    layer_cache_dtype_str = (\n"
        '                        "auto"\n'
        "                        if kv_cache_spec.kv_quant_mode == KVQuantMode.NONE\n"
        "                        and not isinstance(kv_cache_spec, TQFullAttentionSpec)\n"
        "                        else self.cache_config.cache_dtype\n"
        "                    )\n",
    ),
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

    if IMPORT_OLD not in text:
        print(f"{LOG} FATAL: anchor-not-found (import) — upstream refactor; "
              f"re-derive before boot (TQ3 boot WILL crash without this fix)",
              file=sys.stderr)
        return 1
    if text.count(IMPORT_OLD) != 1:
        print(f"{LOG} FATAL: ambiguous anchor (import)", file=sys.stderr)
        return 1

    for old, new in DTYPE_SELECT_FORMS:
        if old in text:
            if text.count(old) != 1:
                print(f"{LOG} FATAL: ambiguous anchor (dtype-select)", file=sys.stderr)
                return 1
            text = text.replace(IMPORT_OLD, IMPORT_NEW, 1).replace(old, new, 1)
            TARGET.write_text(text)
            print(f"{LOG} applied: TQFullAttentionSpec exempted from 'auto' cache-dtype "
                  f"in _reshape_kv_cache_tensors (vllm#47609 V1-runner backport)")
            return 0

    print(f"{LOG} FATAL: anchor-not-found (dtype-select) — upstream refactor; "
          f"re-derive before boot (TQ3 boot WILL crash without this fix)",
          file=sys.stderr)
    return 1


sys.exit(main())
