# SPDX-License-Identifier: Apache-2.0
"""Give the ``genesis.*`` / ``sndr.*`` logger trees a sink inside `vllm serve`.

vLLM's ``DEFAULT_LOGGING_CONFIG`` (vllm/logger.py) configures exactly one
logger — ``"vllm"`` — with one handler and ``propagate: False``.  It never
touches the root logger.  Every module in this tree that does
``logging.getLogger("genesis.x")`` therefore ends up, inside the API server
and inside the EngineCore / worker processes, with:

  * no handler anywhere on its ancestor chain (root is bare), and
  * an effective level of WARNING (root's default).

So its INFO calls are dropped and its WARNING calls go to
``logging.lastResort`` at best.  Measured on the 2026-07-25 boot: the H119
lens router made 61 ``logger.info`` calls while scoring 61 requests and
produced zero lines in a 1,613-line container log.  Any patch here whose
"I degraded / I bailed / I hit the fallback" announcement is a runtime
WARNING is announcing it into the void.

The apply-time process is *not* affected: ``patches/apply_all.py`` installs
its own root handler via ``basicConfig`` before it runs anything, which is
where the ``[INFO:genesis.apply_all]`` boot lines come from.  That handler
is a load-bearing interface — ``scripts/report.sh`` greps its exact format
— so this bridge deliberately stays out of that process (see
``_is_genesis_own_main``) and ``apply_all`` calls :func:`uninstall` before
configuring, as a second line of defence.

Why a handler bridge instead of renaming the loggers to ``vllm.genesis.*``:
many of these modules log in BOTH roles.  ``sndr/dispatcher`` alone emits
397 ``[INFO:genesis.dispatcher]`` lines at apply time and is also imported
into the worker; renaming its logger would silently rewrite the boot log
that report.sh and the operator docs read.  Adding a sink where one is
missing changes nothing that already works.

Nothing in here may raise into serving: every entry point swallows.
"""
from __future__ import annotations

import logging
import os
import sys

# Logger trees that live outside vLLM's "vllm" namespace and would otherwise
# have no handler at all in a serving process.
_ROOTS = ("genesis", "sndr")

# Saved (logger, handlers, level, propagate) so uninstall() can restore the
# exact pre-bridge state. Empty == not installed.
_SAVED: list[tuple[logging.Logger, list[logging.Handler], int, bool]] = []

_HERE = os.path.dirname(os.path.abspath(__file__))


def _is_genesis_own_main() -> bool:
    """True when this interpreter's __main__ is a Genesis/sndr entry point.

    Those all configure logging themselves (``basicConfig``) and own their
    output format.  Detection has to work from the *package* ``__init__``,
    which CPython imports before the target module's body runs:

      * ``python3 -m vllm._genesis.patches.apply_all`` — argv[0] is literally
        ``"-m"`` at that moment (verified on CPython 3.12, the container's
        interpreter); it only becomes the module path once the body starts.
      * ``python3 .../_genesis/<anything>.py`` — argv[0] is inside this tree.

    A real server is ``vllm serve`` (argv[0] == ``.../bin/vllm``), and
    multiprocessing restores that argv in spawned EngineCore/worker children,
    so both sides of the fence are stable.
    """
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 == "-m":
        return True
    if not argv0:
        return False
    try:
        return os.path.abspath(argv0).startswith(_HERE + os.sep)
    except Exception:
        return False


def _target_level(vllm_logger: logging.Logger) -> int:
    """Level for the bridged trees: vLLM's, unless GENESIS_LOG_LEVEL says otherwise.

    Inheriting means ``VLLM_LOGGING_LEVEL=DEBUG`` turns these on too, and an
    operator who needs to quieten a chatty patch without a rebuild has
    ``GENESIS_LOG_LEVEL=WARNING``.
    """
    raw = os.environ.get("GENESIS_LOG_LEVEL", "").strip().upper()
    if raw:
        level = logging.getLevelName(raw)
        if isinstance(level, int):
            return level
        if raw.isdigit():
            return int(raw)
    return vllm_logger.getEffectiveLevel()


def install() -> None:
    """Point ``genesis.*`` / ``sndr.*`` at vLLM's handler. Idempotent, never raises."""
    try:
        if _SAVED or _is_genesis_own_main():
            return

        vllm_logger = logging.getLogger("vllm")
        handlers = list(vllm_logger.handlers)
        if not handlers:
            # VLLM_CONFIGURE_LOGGING=0, or vllm.logger not imported yet. The
            # operator opted out of vLLM's logging; don't invent a sink.
            return

        level = _target_level(vllm_logger)
        bridged = []
        for name in _ROOTS:
            logger = logging.getLogger(name)
            if logger.handlers:
                # Someone already gave this tree a sink — leave it alone
                # rather than emit every record twice.
                continue
            _SAVED.append(
                (logger, list(logger.handlers), logger.level, logger.propagate)
            )
            for handler in handlers:
                logger.addHandler(handler)
            logger.setLevel(level)
            # vLLM's own logger stops propagation at "vllm"; do the same here
            # so a root handler added later can't double every line.
            logger.propagate = False
            bridged.append(name)

        if bridged:
            # Boot check. One line per process (API server + EngineCore +
            # each worker): `podman logs <c> | grep 'genesis log bridge'`.
            logging.getLogger("genesis.log_bridge").info(
                "[Genesis] genesis log bridge installed for %s at level %s "
                "(pid=%d) — these trees are outside vLLM's 'vllm' logger and "
                "had no handler; override with GENESIS_LOG_LEVEL",
                "/".join(f"{n}.*" for n in bridged),
                logging.getLevelName(level),
                os.getpid(),
            )
    except Exception:  # pragma: no cover - logging must never break serving
        pass


def uninstall() -> None:
    """Restore the pre-bridge state of every tree install() touched."""
    try:
        while _SAVED:
            logger, handlers, level, propagate = _SAVED.pop()
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
            for handler in handlers:
                logger.addHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate
    except Exception:  # pragma: no cover
        pass


def is_installed() -> bool:
    """True if the bridge is currently active in this process."""
    return bool(_SAVED)


__all__ = ["install", "uninstall", "is_installed"]
