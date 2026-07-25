#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the vLLM capability ledger.

Stdlib only — this box's python3 has neither PyYAML, sklearn nor scipy, and the
tooling has to keep working inside a bare container.

Design note (the load-bearing lesson):
    "Applied" in a boot log does NOT mean effective.  Every verification in this
    package is an EFFECT check — a marker string physically present in the
    installed file, a symbol present in the wheel, or a file present on disk.
    Nothing here ever reads a log line to decide a capability is live.

Author: house tooling, 2026-07-25
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from typing import Iterable

REPO = os.environ.get("CLUB3090", "/home/user/club-3090")
VLLM_DIR = os.path.join(REPO, "models/qwen3.6-27b/vllm")
GENESIS = os.path.join(VLLM_DIR, "patches/genesis/vllm/_genesis")
SNDR = os.path.join(GENESIS, "sndr")
FIXES = os.path.join(REPO, "fixes")
LEDGER_DIR = os.path.join(REPO, "ops/vllm-capability-ledger")

# Where the installed vLLM lives inside our images.
CONTAINER_VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"

# Host paths -> container paths for the read-only lane mounts.  Presence of a
# lane SOURCE file in the container is therefore equivalent to presence in the
# host tree; the interesting container-only signal is the marker that a text
# patch writes INTO the installed vllm at boot.
MOUNTS = {
    GENESIS: CONTAINER_VLLM + "/_genesis",
    SNDR: "/usr/local/lib/python3.12/dist-packages/sndr",
    FIXES: "/fixes",
}


# ---------------------------------------------------------------- process ---

def run(cmd: list[str], cwd: str | None = None, check: bool = False,
        timeout: int = 120) -> tuple[int, str, str]:
    """Run a command, never raise on non-zero unless check=True."""
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, errors="replace")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return 127, "", f"{type(e).__name__}: {e}"
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} -> {p.returncode}\n{p.stderr}")
    return p.returncode, p.stdout, p.stderr


def git(repo: str, *args: str, timeout: int = 300) -> str:
    rc, out, err = run(["git", "-C", repo, *args], timeout=timeout)
    if rc != 0:
        return ""
    return out


# ------------------------------------------------------------- inspectors ---

class Inspector:
    """Reads files out of a target: the host tree, a live container, or an image.

    The three back-ends share one API so the verifier is written once:

        read(path) -> str | None
        exists(path) -> bool
        grep_count(path, needle) -> int

    Paths are CONTAINER-ABSOLUTE for the container/image back-ends and
    HOST-ABSOLUTE for the tree back-end; `Capability.resolve_path()` picks.
    """

    kind = "abstract"
    label = "abstract"

    def read(self, path: str) -> str | None:  # pragma: no cover - interface
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        return self.read(path) is not None

    def contains(self, path: str, needle: str) -> bool:
        body = self.read(path)
        return body is not None and needle in body

    def count(self, path: str, needle: str) -> int:
        body = self.read(path)
        return 0 if body is None else body.count(needle)


class TreeInspector(Inspector):
    """Reads the host working tree (or any directory root)."""

    kind = "tree"

    def __init__(self, root: str = "/"):
        self.root = root.rstrip("/")
        self.label = f"tree:{root}"
        self._cache: dict[str, str | None] = {}

    def read(self, path: str) -> str | None:
        if path in self._cache:
            return self._cache[path]
        full = path if self.root in ("", "/") else self.root + path
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                body = fh.read()
        except OSError:
            body = None
        self._cache[path] = body
        return body


class _PodmanInspector(Inspector):
    """Common cat-based reader; one subprocess per distinct file, cached.

    A batch reader (`prefetch`) pulls many files in a single container call so a
    500-capability sweep does not spawn 500 containers.
    """

    def __init__(self, sudo: bool = True):
        self.sudo = sudo
        self._cache: dict[str, str | None] = {}

    def _pre(self) -> list[str]:
        return ["sudo", "podman"] if self.sudo else ["podman"]

    def _batch_cmd(self, script: str) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def prefetch(self, paths: Iterable[str]) -> None:
        """Read many files in ONE container invocation.

        Emits a NUL-delimited `path\\0len\\0bytes` stream so filenames and file
        bodies containing newlines survive the round-trip.
        """
        want = [p for p in dict.fromkeys(paths) if p not in self._cache]
        if not want:
            return
        # Chunk so we never blow the ARG_MAX of the container shell.
        for i in range(0, len(want), 200):
            chunk = want[i:i + 200]
            listing = " ".join(shlex.quote(p) for p in chunk)
            # base64 the whole framed stream.  A raw byte stream does NOT
            # survive the trip: some targets are binaries (libgreenboost*.so),
            # subprocess text mode replaces undecodable bytes with U+FFFD, and
            # re-encoding then desyncs every length prefix — which crashed the
            # first version mid-sweep on a .so.
            script = (
                "import sys,base64\n"
                f"paths = {chunk!r}\n"
                "buf = bytearray()\n"
                "for p in paths:\n"
                "    try:\n"
                "        b = open(p,'rb').read()\n"
                "    except OSError:\n"
                "        b = None\n"
                "    buf += p.encode()+b'\\0'\n"
                "    buf += (b'-1' if b is None else str(len(b)).encode())+b'\\0'\n"
                "    if b is not None:\n"
                "        buf += b\n"
                "sys.stdout.write(base64.b64encode(bytes(buf)).decode())\n"
            )
            del listing
            rc, out, err = self._run_python(script)
            if rc != 0:
                for p in chunk:
                    self._cache.setdefault(p, None)
                continue
            self._decode_stream(out, chunk)

    def _decode_stream(self, blob: str, chunk: list[str]) -> None:
        import base64
        try:
            data = base64.b64decode(blob.strip(), validate=False)
        except Exception:
            for p in chunk:
                self._cache.setdefault(p, None)
            return
        pos = 0
        seen = set()
        while pos < len(data):
            nul = data.find(b"\0", pos)
            if nul < 0:
                break
            path = data[pos:nul].decode("utf-8", "replace")
            pos = nul + 1
            nul = data.find(b"\0", pos)
            if nul < 0:
                break
            length = int(data[pos:nul] or b"-1")
            pos = nul + 1
            if length < 0:
                self._cache[path] = None
            else:
                self._cache[path] = data[pos:pos + length].decode("utf-8", "replace")
                pos += length
            seen.add(path)
        for p in chunk:
            if p not in seen:
                self._cache.setdefault(p, None)

    def _run_python(self, script: str) -> tuple[int, str, str]:  # pragma: no cover
        raise NotImplementedError

    def read(self, path: str) -> str | None:
        if path not in self._cache:
            self.prefetch([path])
        return self._cache.get(path)


class ContainerInspector(_PodmanInspector):
    """Reads files out of a RUNNING container via `podman exec`.

    Safe: exec'ing `python3 -c` that only reads files touches no GPU and does
    not import torch or vllm.  It never restarts or signals the container.
    """

    kind = "container"

    def __init__(self, name: str, sudo: bool = True):
        super().__init__(sudo)
        self.name = name
        self.label = f"container:{name}"

    def _run_python(self, script: str) -> tuple[int, str, str]:
        return run(self._pre() + ["exec", "-i", self.name, "python3", "-c", script],
                   timeout=180)


class ImageInspector(_PodmanInspector):
    """Reads files out of an IMAGE without starting the real entrypoint.

    `podman run --rm --entrypoint python3 <image> -c <script>` is a short-lived
    CPU-only container: no GPU device is requested, no server is started.
    NOTE: an image holds the PRISTINE installed vllm — boot-time text patches
    have not run.  So an image sweep verifies WHEEL-level capabilities
    (cherry-picks compiled into the wheel, vendored lane sources baked in) and
    is expected to report boot-patch markers as absent.
    """

    kind = "image"

    def __init__(self, ref: str, sudo: bool = True):
        super().__init__(sudo)
        self.ref = ref
        self.label = f"image:{ref}"

    def _run_python(self, script: str) -> tuple[int, str, str]:
        return run(self._pre() + ["run", "--rm", "--network=none",
                                  "--entrypoint", "python3", self.ref, "-c", script],
                   timeout=300)


# ------------------------------------------------------------------- misc ---

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


def container_env(name: str, sudo: bool = True) -> dict[str, str]:
    """Env of a running container (catches 'module present but flag off')."""
    pre = ["sudo", "podman"] if sudo else ["podman"]
    rc, out, _ = run(pre + ["inspect", name, "--format", "{{json .Config.Env}}"])
    if rc != 0:
        return {}
    try:
        pairs = json.loads(out)
    except Exception:
        return {}
    env = {}
    for kv in pairs or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
    return env


def image_of(container: str, sudo: bool = True) -> str:
    pre = ["sudo", "podman"] if sudo else ["podman"]
    rc, out, _ = run(pre + ["inspect", container, "--format", "{{.ImageName}}"])
    return out.strip() if rc == 0 else ""


def eprint(*a) -> None:
    print(*a, file=sys.stderr)
