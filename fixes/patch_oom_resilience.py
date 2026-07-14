"""OOM Resilience — survive CUDA OOM in the engine step without killing EngineCore.

v7 (2026-07-14): fixes two defects in v6:

  1. HANGING CLIENTS (BUG-2). v6 called ``self.abort_requests(ids)``, which
     only runs ``scheduler.finish_requests(...)`` and DISCARDS its return —
     the aborted requests' clients were never sent finish outputs, so their
     HTTP streams hung until timeout. v7 mirrors the engine's own shutdown
     path: ``aborted = scheduler.finish_requests(ids, FINISHED_ABORTED)`` then
     ``self._send_abort_outputs(aborted)``, so every aborted client gets a
     proper abort response immediately.

  2. UNBOUNDED RECOVERY ON A POISONED CONTEXT (BUG-3). The broadened
     message-guard (``aten::empty`` + ``api call failed``) also matches an
     "illegal memory access" / "device-side assert" — an UNRECOVERABLE CUDA
     context — so v6 could abort-loop forever instead of dying. v7 (a)
     re-raises immediately on illegal-memory-access / device-side-assert text,
     and (b) caps consecutive recoveries at 8 (streak resets on any successful
     step): past the cap it re-raises so the supervisor does a clean restart.

v6 history (still true in v7): wraps the actual OOM site — ``self.step_fn()``
inside ``_process_engine_step`` — because an OOM from an AOTI/inductor-compiled
graph surfaces as a plain ``RuntimeError`` (``torch_call_dispatcher
("aten::empty", ...) API call failed``), not ``torch.cuda.OutOfMemoryError``,
and the outer-loop restart of v5 never freed the culprit request.

On a recoverable OOM this handler ABORTS ALL currently-running requests (not
just the culprit — the scheduler gives no attribution at this level), notifies
their clients, empties the allocator cache, and returns so the loop continues.
A clean cudaMalloc OOM leaves the CUDA context intact, so recovery is sound;
illegal-memory-access / device-side-assert are explicitly re-raised (see #2
above) rather than relying on the match text alone to exclude them.
"""
import logging
from pathlib import Path

log = logging.getLogger("patch_oom_resilience")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

MARKER = "# PATCH: oom_resilience_v7"
CORE_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py"
)


def apply():
    if not CORE_TARGET.exists():
        log.warning("[oom_resilience] core.py not found")
        return

    text = CORE_TARGET.read_text()

    if MARKER in text:
        log.info("[oom_resilience] already applied (v7)")
        return

    # Ensure torch import (idempotent).
    if "import torch\n" not in text and "import torch " not in text:
        lines = text.split("\n")
        last_import = 0
        for i, line in enumerate(lines):
            if line.startswith(("import ", "from ")):
                last_import = i
        lines.insert(last_import + 1, "import torch  # oom_resilience")
        text = "\n".join(lines)

    # Wrap the model-execution step. This is the exact line that OOMs.
    old = "        outputs, model_executed = self.step_fn()"
    if old not in text:
        log.warning(
            "[oom_resilience] step_fn() call site not found — patch NOT applied "
            "(vLLM internals drifted; review _process_engine_step)"
        )
        return

    new = (
        "        " + MARKER + "\n"
        "        try:\n"
        "            outputs, model_executed = self.step_fn()\n"
        "            self._oom_recov_streak = 0\n"
        "        except Exception as _oom_e:\n"
        "            _oom_s = str(_oom_e).lower()\n"
        "            if \"illegal memory access\" in _oom_s or \"device-side assert\" in _oom_s:\n"
        "                raise  # poisoned CUDA context — unrecoverable, die for clean restart\n"
        "            _oom_hit = isinstance(_oom_e, torch.cuda.OutOfMemoryError) or (\n"
        "                (\"out of memory\" in _oom_s)\n"
        "                or (\"aten::empty\" in _oom_s and \"api call failed\" in _oom_s)\n"
        "            )\n"
        "            if not _oom_hit:\n"
        "                raise\n"
        "            self._oom_recov_streak = getattr(self, \"_oom_recov_streak\", 0) + 1\n"
        "            if self._oom_recov_streak > 8:\n"
        "                logger.error(\n"
        "                    \"[oom_resilience] %d consecutive OOM recoveries without a \"\n"
        "                    \"successful step — giving up so the supervisor restarts us.\",\n"
        "                    self._oom_recov_streak,\n"
        "                )\n"
        "                raise\n"
        "            import gc as _gc\n"
        "            try:\n"
        "                _run_ids = [r.request_id for r in list(self.scheduler.running)]\n"
        "            except Exception:\n"
        "                _run_ids = []\n"
        "            logger.error(\n"
        "                \"[oom_resilience] CUDA OOM during engine step (%s). Aborting \"\n"
        "                \"%d running request(s) and continuing; engine stays up. \"\n"
        "                \"streak=%d ids=%s\",\n"
        "                type(_oom_e).__name__, len(_run_ids),\n"
        "                self._oom_recov_streak, _run_ids,\n"
        "            )\n"
        "            try:\n"
        "                if _run_ids:\n"
        "                    _ab = self.scheduler.finish_requests(\n"
        "                        _run_ids, RequestStatus.FINISHED_ABORTED\n"
        "                    )\n"
        "                    try:\n"
        "                        self._send_abort_outputs(_ab)\n"
        "                    except Exception as _nt_e:\n"
        "                        logger.error(\n"
        "                            \"[oom_resilience] client abort-notify failed: %s\", _nt_e\n"
        "                        )\n"
        "            except Exception as _ab_e:\n"
        "                logger.error(\"[oom_resilience] abort failed: %s\", _ab_e)\n"
        "            _gc.collect()\n"
        "            try:\n"
        "                torch.cuda.empty_cache()\n"
        "            except Exception:\n"
        "                pass\n"
        "            return False"
    )
    text = text.replace(old, new, 1)
    CORE_TARGET.write_text(text)
    log.info(
        "[oom_resilience] v7 applied — step_fn() OOM-guarded, all-running abort "
        "+ client notify, IMA re-raise, recovery-streak cap"
    )


apply()
