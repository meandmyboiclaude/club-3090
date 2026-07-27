"""v0.8.0 Pull-Emit-Derived `[E]` — STEP E3: the derived boot stage.

CONTRACT-4 (the brief's locked E3 spec, boot half). This module owns ONLY
the derived-compose BOOT; it does NOT wire into `run_pull()` (E4), does NOT
download (E2 owns that), does NOT emit the §6 capture artifacts (that is
`capture.py`, also E3 but a separate module), does NOT do a real on-rig boot
(E5), does NOT write docs (E5).

Boot discipline (the proven on-rig `[D]`/Pull-Gate lesson — reused here)
-----------------------------------------------------------------------
A derived compose boots ONLY with:

  * `docker compose --project-directory <dir> -f <compose> up`  (the
    `--project-directory` discipline — relative volume/path resolution is
    project-dir-relative);
  * the model weights bind-mounted from the HF_HOME host dir
    `<hf_home>/club3090/pulls/<slug-sanitized>/` to the container path
    `/models/club3090/pulls/<slug-sanitized>` — which is EXACTLY the
    `host:container:ro` volume E1's `generate_from_profile()` already emits
    in `compose_text`. It is **NOT** `MODEL_DIR=/mnt/models` (that is the
    curated Path-A convention; a DERIVED model's weights live under HF_HOME,
    per CONTRACT-2 / CONTRACT-3).

`runner` is INJECTABLE. The default runner shells the real `docker compose`;
E3 tests pass a fixture runner so there is NO real Docker / GPU in CI (the
real boot is E5, on-rig). Whatever happens, the boot is ALWAYS torn down on
exit (mirrors the Pull-Gate on-rig harness "always teardown" rule — no
orphaned container/project state, ever).

Public API (stable for E4)
--------------------------

    from scripts.lib.profiles import booter

    with booter.booted_derived(einput, compose_text, *, runner=None) as bt:
        #   bt: BootResult — the server is ALIVE for the ENTIRE `with` body
        #     .ok       -> bool
        #     .seconds  -> float        (wall time of the up->ready probe)
        #     .failure  -> None | "<container-died-reason>"
        #     .endpoint -> str | None   (the OpenAI-compatible base URL when
        #                                ok; None on failure)
        if bt.ok:
            smoke(bt.endpoint)          # server is UP here
            capture(...)                # emitted BEFORE teardown
    # <- teardown (runner.down + rmtree) happens HERE, on context-manager
    #    exit, AFTER the with-body completes (success OR exception); ALWAYS.

`booted_derived` is a `@contextlib.contextmanager`: it brings the server up,
`yield`s a live `BootResult`-shaped handle while the server is UP (so the
caller can smoke + capture AGAINST A LIVE SERVER), and ALWAYS tears the
compose down in its `finally` on `__exit__` — preserving the no-orphan
guarantee (the Pull-Gate on-rig harness lesson) but at the CORRECT scope.

WHY a context manager (the on-rig E5-caught defect this module's history
records): the prior `boot_derived(...) -> BootResult` did teardown in a
`finally` BEFORE the `return` reached the caller — Python runs `finally`
before delivering `return`, so the server was torn down BEFORE the
orchestrator could smoke it -> `ConnectionRefused`, every time, by
construction. Moving teardown to the CM's `__exit__` keeps the always-
teardown guarantee while keeping the server alive for boot -> smoke ->
capture. This is also the reusable lifecycle primitive that the future
`--keep`/serve path, the §7 soak-continuous path, and `[F]` extended
capture become additive consumers of (NOT control-flow rewrites).

`BootResult` IS the §6 capture-point-3 payload source (`capture.py` maps it
to the `{point:"boot", ok, seconds, failure}` artifact — note `endpoint` is
runtime-only plumbing for the smoke prober and is NOT part of the redacted
artifact).
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from .downloader import sanitize_slug

# The container-side port the derived-vllm template binds (E1 emits
# `... :8000` and `vllm serve --port 8000` inside the container). The host
# port is the compose `${PORT:-<default_port>}`; for the boot readiness probe
# E3 talks to the host-published port carried on einput.runtime.
_CONTAINER_PORT = 8000
_DEFAULT_READY_TIMEOUT_S = 600  # generous; a fixture runner returns instantly


# F3/G6-A-ii: a bounded tail of the CONTAINER log captured on boot failure.
# `docker compose logs --no-color --tail=N`. Bounded by BOTH a line tail
# (N) AND a byte cap so a runaway log can never blow the artifact up. The
# in-container vLLM OOM traceback AND the gpu_worker.py measured-peak line
# both land here (NOT in compose-up stderr — that was C2).
_FAILURE_LOG_TAIL_LINES = 400
_FAILURE_LOG_BYTE_CAP = 64 * 1024  # 64 KiB hard ceiling (post-tail)


@dataclass
class BootResult:
    ok: bool
    seconds: float = 0.0
    # None on success; a short "<container-died-reason>" string on failure
    # (E3 does NOT classify into §6.1 classes — that is [F]'s job; this is
    # just the raw reason the runner surfaced).
    failure: Optional[str] = None
    # The OpenAI-compatible base URL the smoke prober talks to (ok only).
    endpoint: Optional[str] = None
    # F3/G6-A-ii (ADDITIVE; default None — success-path unchanged): on a
    # boot FAILURE the CM captures a bounded, `_redact_text`-scrubbed
    # `docker compose logs --no-color` excerpt of the project BEFORE
    # teardown (the container still exists then). `capture.py:emit_capture`
    # reads this (or its explicit `failure_log_excerpt=` param) into
    # `pt3.failure_log_excerpt` + parses `pt3.actual` from it. Stays None on
    # success and whenever the runner can't surface a log (honest degrade).
    failure_log_excerpt: Optional[str] = None


# ---------------------------------------------------------------------------
# Runner abstraction (injectable; default = real `docker compose`).
#
# A runner must provide:
#   .up(project_dir, compose_path)   -> None
#       bring the derived compose up (the proven `docker compose
#       --project-directory <project_dir> -f <compose_path> up -d` shape).
#       Raise BootError(reason) if the container dies / never becomes ready.
#   .wait_ready(endpoint)            -> None
#       block until the server answers (a cheap GET on the OpenAI base) or
#       raise BootError(reason) on container death / timeout.
#   .down(project_dir, compose_path) -> None
#       tear the compose down (ALWAYS called from booted_derived's
#       __exit__/finally — i.e. AFTER the `with` body, not before the
#       handle reaches the caller).
#   .capture_failure_log(project_dir, compose_path) -> str | None
#       F3/G6-A-ii (ADDITIVE, optional): on a boot FAILURE, return a
#       bounded `docker compose logs --no-color --tail=N` excerpt of the
#       project (the CONTAINER log — where the in-container vLLM OOM
#       traceback + the gpu_worker.py measured-peak line land). Called
#       BEFORE `.down()` (the container still exists then). A runner
#       WITHOUT this method (legacy fixture) -> the CM degrades to None
#       (no excerpt; honest, never raises). `--no-color` is MANDATORY
#       (ANSI escapes corrupt the Tier-1 regex — Kimi-r2 L1).
#
# E3 tests inject a fixture runner (no Docker). The real runner is below.
# ---------------------------------------------------------------------------
class BootError(RuntimeError):
    """The runner raises this with a short container-died reason; the stage
    maps it to BootResult.failure and ALWAYS tears down."""


class DockerComposeRunner:
    """Real runner. `docker compose --project-directory <dir>` discipline.

    NOT exercised in E3 CI (a fixture runner is injected — no Docker/GPU in
    unit tests). The real boot is E5, on-rig. This codifies the EXACT
    proven shape so E5 has nothing to invent."""

    ready_timeout_s = _DEFAULT_READY_TIMEOUT_S

    def _compose(self, project_dir: str, compose_path: str, *args: str) -> None:
        cmd = [
            "docker", "compose",
            "--project-directory", project_dir,
            "-f", compose_path,
            *args,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            reason = tail[-1] if tail else f"exit {proc.returncode}"
            raise BootError(f"docker compose {args[0]} failed: {reason}")

    def up(self, project_dir: str, compose_path: str) -> None:
        self._compose(project_dir, compose_path, "up", "-d")

    def wait_ready(self, endpoint: str) -> None:
        import urllib.error
        import urllib.request

        deadline = time.monotonic() + self.ready_timeout_s
        url = f"{endpoint}/models"
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(3)
        raise BootError("server did not become ready before timeout")

    def capture_failure_log(
        self, project_dir: str, compose_path: str
    ) -> Optional[str]:
        """F3/G6-A-ii: bounded redacted CONTAINER-log excerpt on boot fail.

        `docker compose --project-directory <dir> -f <compose> logs
        --no-color --tail=N` — the project's container log, where the
        in-container vLLM `torch.cuda.OutOfMemoryError` traceback AND the
        `gpu_worker.py` measured-peak line both land (C2: NOT in compose-up
        stderr). Bounded by `--tail=N` AND a post-fetch byte cap. Scrubbed
        with `capture.py`'s SAME `_redact_text` (reuse — NOT a weaker
        re-impl). NEVER raises (best-effort, like `down()`): any failure to
        obtain a log -> `None` (honest degrade; Tier-1 then can't route as
        a kv-calc bug, never a confidently-wrong filing). `--no-color` is
        MANDATORY (ANSI escapes corrupt the Tier-1 regex)."""
        try:
            cmd = [
                "docker", "compose",
                "--project-directory", project_dir,
                "-f", compose_path,
                "logs", "--no-color",
                f"--tail={_FAILURE_LOG_TAIL_LINES}",
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )
            raw = (proc.stdout or "") + (
                ("\n" + proc.stderr) if proc.stderr else ""
            )
            if not raw.strip():
                return None
            # Byte-cap AFTER the tail (keep the TAIL — the OOM traceback +
            # gpu_worker peak land at the END of a failing boot).
            if len(raw) > _FAILURE_LOG_BYTE_CAP:
                raw = raw[-_FAILURE_LOG_BYTE_CAP:]
            # REUSE capture.py's _redact_text (NOT a weaker scrub).
            from scripts.lib.profiles.capture import _redact_text
            return _redact_text(raw)
        except Exception:  # pragma: no cover - never raise out of capture
            return None

    def down(self, project_dir: str, compose_path: str) -> None:
        # Best-effort teardown; never raise out of teardown.
        try:
            self._compose(
                project_dir, compose_path, "down", "-v", "--remove-orphans"
            )
        except BootError:
            pass


# ---------------------------------------------------------------------------
# THE boot stage — as a live-server lifecycle CONTEXT MANAGER.
#
# Teardown is in the CM's `finally` (runs on `__exit__`, i.e. AFTER the
# `with` body completes — success OR exception), NOT before the handle
# reaches the caller. This keeps the always-teardown / no-orphan guarantee
# at the CORRECT scope: the server stays alive for the ENTIRE with-body
# (boot -> smoke -> capture), then is ALWAYS torn down. Teardown NEVER
# raises (so an exception in the with-body still propagates, with teardown
# having run).
# ---------------------------------------------------------------------------
def _safe_capture_failure_log(
    runner: Any, project_dir: str, compose_path: str
) -> Optional[str]:
    """F3/G6-A-ii: call `runner.capture_failure_log` if the runner provides
    it (the real `DockerComposeRunner` does; a legacy fixture runner may
    not). NEVER raises and is called ONLY in the boot-FAILURE branches of
    `booted_derived`, BEFORE the `finally` teardown — so the container the
    log comes from still exists. Returns the bounded redacted excerpt, or
    `None` (no method / empty / any error — honest degrade; downstream
    Tier-1 then can't route a confidently-wrong kv-calc bug)."""
    fn = getattr(runner, "capture_failure_log", None)
    if not callable(fn):
        return None
    try:
        out = fn(project_dir, compose_path)
        return out if (isinstance(out, str) and out.strip()) else None
    except Exception:  # pragma: no cover - capture must never raise
        return None


@contextlib.contextmanager
def booted_derived(
    einput, compose_text: str, *, runner: Optional[Any] = None
) -> Iterator[BootResult]:
    """Bring up the E1-emitted derived `compose_text` using the proven
    `docker compose --project-directory` discipline against the HF_HOME
    host->container `:ro` mount E1 already wrote into `compose_text` (NOT
    `MODEL_DIR`), `yield` a live `BootResult`-shaped handle WHILE THE SERVER
    IS UP, then ALWAYS tear the compose down on context-manager exit.

    Contract:
      * The compose is written to a fresh project directory; `runner.up` /
        `runner.wait_ready` are invoked. On success a `BootResult(ok=True,
        endpoint=...)` is yielded — **the server is alive for the entire
        `with` body** (so the orchestrator can smoke + capture against a
        LIVE server, not a torn-down one).
      * On an EXPECTED boot failure (`BootError`) a
        `BootResult(ok=False, failure=..., endpoint=None)` is yielded (the
        CM does NOT raise for an expected boot failure — the orchestrator
        must still emit pt3 + capture). Only a truly defensive/unexpected
        error surfaces (and even then teardown still runs).
      * Teardown (`runner.down` + `shutil.rmtree(project_dir)`) happens in
        the `finally` — i.e. on `__exit__`, AFTER the `with` body completes
        (success OR exception). This PRESERVES the no-orphan guarantee (the
        Pull-Gate on-rig harness lesson) at the CORRECT scope. Teardown
        NEVER raises.

    `runner` is injectable: default = real `docker compose`
    (`DockerComposeRunner`); E3 tests pass a fixture runner so there is NO
    real Docker / GPU in CI (the real boot is E5, on-rig). The
    `endpoint`/`host_port` derivation, the `--project-directory` discipline,
    the HF_HOME absolute-mount reliance, and the `BootResult` field set are
    byte-unchanged vs the prior implementation — ONLY the teardown *scope*
    moved from "before the call returns" to "on context-manager exit, after
    the with-body".
    """
    if runner is None:
        runner = DockerComposeRunner()

    slug = einput.slug
    san = sanitize_slug(slug)
    rt = einput.runtime or {}
    host_port = int(rt.get("default_port") or rt.get("port") or _CONTAINER_PORT)
    endpoint = f"http://127.0.0.1:{host_port}/v1"

    # Fresh, isolated project directory — the `--project-directory`
    # discipline resolves the compose's relative paths against THIS dir;
    # the derived compose's volume is an ABSOLUTE host HF_HOME path
    # (`<hf_home>/club3090/pulls/<san>:/models/...:ro`, emitted by E1), so
    # the bind mount is the HF_HOME weights dir — never MODEL_DIR.
    project_dir = tempfile.mkdtemp(prefix=f"club3090-derived-{san}-")
    compose_path = str(Path(project_dir) / "docker-compose.yml")
    Path(compose_path).write_text(compose_text, encoding="utf-8")

    started = time.monotonic()
    try:
        try:
            runner.up(project_dir, compose_path)
            runner.wait_ready(endpoint)
        except BootError as exc:
            # EXPECTED boot failure — yield an ok=False handle (do NOT raise
            # out of the CM; the orchestrator must still emit pt3 + capture).
            # F3/G6-A-ii: capture the bounded redacted CONTAINER log NOW —
            # the container still exists (teardown is the `finally`, AFTER
            # this branch). Best-effort: None when the runner can't / has no
            # `capture_failure_log` (legacy fixture) — honest degrade.
            yield BootResult(
                ok=False,
                seconds=round(time.monotonic() - started, 3),
                failure=str(exc) or "container died (no reason surfaced)",
                endpoint=None,
                failure_log_excerpt=_safe_capture_failure_log(
                    runner, project_dir, compose_path
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive
            # Truly unexpected (runner contract breach) — still yield an
            # ok=False handle so the orchestrator captures it; teardown in
            # the finally still runs.
            yield BootResult(
                ok=False,
                seconds=round(time.monotonic() - started, 3),
                failure=f"unexpected boot error: {exc!r}",
                endpoint=None,
                failure_log_excerpt=_safe_capture_failure_log(
                    runner, project_dir, compose_path
                ),
            )
        else:
            # Server is UP. Yield the live handle for the ENTIRE with-body
            # (boot -> smoke -> capture). Teardown is deferred to the
            # finally below (on __exit__, AFTER the body).
            yield BootResult(
                ok=True,
                seconds=round(time.monotonic() - started, 3),
                failure=None,
                endpoint=endpoint,
            )
    finally:
        # ALWAYS teardown — on __exit__, AFTER the with-body completes
        # (success OR exception). Mirrors the Pull-Gate on-rig harness
        # always-teardown rule; never leaves an orphaned container/project;
        # never lets teardown raise (so a with-body exception still
        # propagates with teardown having run).
        try:
            runner.down(project_dir, compose_path)
        except Exception:  # pragma: no cover - teardown must never raise
            pass
        shutil.rmtree(project_dir, ignore_errors=True)
