"""
GPU memory bypass patch for kv_cache_utils.py

Allows vLLM to proceed past tight VRAM conditions during boot instead of
raising ValueError. When available_memory is insufficient:
- Forces available_memory to 256 MB floor (enough for minimal KV cache)
- Logs warnings instead of crashing
"""

import logging
import os

log = logging.getLogger("patch_kv_cache_memory_bypass")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
TARGET = f"{VLLM}/v1/core/kv_cache_utils.py"

SENTINEL = "# [GPU-PV-BYPASS]"

OLD_AVAILABLE_LE_ZERO = '''    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine. "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )'''

NEW_AVAILABLE_LE_ZERO = '''    if available_memory <= 0:  # [GPU-PV-BYPASS]
        import warnings
        available_memory = 256 * 1024 * 1024  # 256 MB floor
        warnings.warn(
            "GPU-PV bypass: available_memory <= 0, forcing to 256 MB. "
            "KV cache will be minimal. This is expected on tight-VRAM configs."
        )'''

OLD_NEEDED_GT_AVAIL = '''    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        raise ValueError(
            f"To serve at least one request with the models's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )'''

NEW_NEEDED_GT_AVAIL = '''    if needed_memory > available_memory:  # [GPU-PV-BYPASS]
        import warnings
        warnings.warn(
            f"GPU-PV bypass: needed {needed_memory/(1024**3):.2f} GiB > "
            f"available {available_memory/(1024**3):.2f} GiB for KV cache. "
            f"Proceeding anyway — may rely on GreenBoost memory spill."
        )
        return  # skip the raise, allow boot to continue'''


def main():
    if not os.path.isfile(TARGET):
        log.warning(f"[kv_cache_bypass] target not found: {TARGET}")
        return

    text = open(TARGET).read()

    if SENTINEL in text:
        log.info("[kv_cache_bypass] already applied, skipping")
        return

    if OLD_AVAILABLE_LE_ZERO not in text:
        log.warning("[kv_cache_bypass] cannot find available_memory <= 0 block — vLLM version mismatch?")
        return

    text = text.replace(OLD_AVAILABLE_LE_ZERO, NEW_AVAILABLE_LE_ZERO)

    if OLD_NEEDED_GT_AVAIL in text:
        text = text.replace(OLD_NEEDED_GT_AVAIL, NEW_NEEDED_GT_AVAIL)
        log.info("[kv_cache_bypass] patched needed > available block")
    else:
        log.warning("[kv_cache_bypass] could not find needed > available block (may have changed upstream)")

    open(TARGET, "w").write(text)
    log.info("[kv_cache_bypass] applied GPU-PV memory bypass patch")


if __name__ == "__main__":
    main()
else:
    main()
