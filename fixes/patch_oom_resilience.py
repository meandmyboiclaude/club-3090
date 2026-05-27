"""OOM Resilience — wraps the EngineCore busy loop to survive OOM crashes.

When a CUDA OOM kills the model execution step, this patch catches it at
the outermost level and restarts the busy loop instead of killing the
entire EngineCore process.
"""
import logging
from pathlib import Path

log = logging.getLogger("patch_oom_resilience")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

MARKER = "# PATCH: oom_resilience_v5"
CORE_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py"
)


def apply():
    if not CORE_TARGET.exists():
        log.warning("[oom_resilience] core.py not found")
        return

    text = CORE_TARGET.read_text()

    # Remove any previous version markers/patches
    if "PATCH: oom_resilience" in text and MARKER not in text:
        # Strip old patches line by line
        lines = text.split("\n")
        cleaned = []
        skip_block = False
        for line in lines:
            if "PATCH: oom_resilience" in line and MARKER not in line:
                skip_block = True
                continue
            if skip_block:
                if line.strip().startswith(("import gc", "_oom_", "logger.error",
                    "torch.cuda.empty_cache", "if _oom", "raise", "while True:",
                    "try:", "except torch", "break", "_oom_loop")):
                    continue
                skip_block = False
            cleaned.append(line)
        text = "\n".join(cleaned)

    if MARKER in text:
        log.info("[oom_resilience] already applied (v5)")
        return

    # Ensure torch import
    if "import torch\n" not in text and "import torch " not in text:
        lines = text.split("\n")
        last_import = 0
        for i, line in enumerate(lines):
            if line.startswith(("import ", "from ")):
                last_import = i
        lines.insert(last_import + 1, "import torch  # oom_resilience")
        text = "\n".join(lines)

    # Wrap engine_core.run_busy_loop() with OOM retry
    old = "            engine_core.run_busy_loop()"
    if old in text:
        new = f"""            {MARKER}
            _oom_crashes = 0
            while True:
                try:
                    engine_core.run_busy_loop()
                    break
                except torch.cuda.OutOfMemoryError:
                    import gc
                    _oom_crashes += 1
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.error("[oom_resilience] OOM in busy loop (crash #%d), restarting", _oom_crashes)
                    if _oom_crashes >= 20:
                        logger.error("[oom_resilience] 20 OOM crashes, giving up")
                        raise"""
        text = text.replace(old, new)
        log.info("[oom_resilience] core.py busy loop wrapped (v5)")
    else:
        log.warning("[oom_resilience] run_busy_loop() not found")

    CORE_TARGET.write_text(text)
    log.info("[oom_resilience] v5 applied")


apply()
