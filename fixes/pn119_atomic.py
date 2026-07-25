#!/usr/bin/env python3
"""PN119 v2 — atomic npz artifact swap (PN119-BUILD-PACK.md §v2 step 2).

Contract: a reader (the PN119 router's hot-reload, or anything np.load-ing
the probe) must NEVER observe a half-written file. Guarantees:

  1. The payload is serialized to a TEMP FILE created in the target's own
     directory (same filesystem — os.replace across filesystems is not
     atomic; it would fall back to copy+unlink).
  2. The temp file is flushed and fsync'd BEFORE the rename, so the bytes
     are durable when the new name appears.
  3. os.replace() atomically points the target name at the complete file.
     A reader holding the old file open keeps a consistent old view; a
     reader opening after the replace sees the complete new file. There is
     no instant at which the target name resolves to partial content.
  4. The directory is fsync'd after the rename so the swap itself survives
     a crash (otherwise the rename can be lost and the OLD file remains —
     which is still a consistent state, never a torn one).
  5. Any failure mid-write unlinks the temp file and re-raises; the target
     is untouched.

Container-visibility note (why the probe is mounted as a DIRECTORY in the
compose): a single-file bind mount pins the container to the original
inode — os.replace on the host creates a NEW inode, which the running
container would never see (it keeps the old, fully-consistent file until
next boot). Mounting the parent directory makes the rename visible inside
the container, which is what enables hot-reload without a restart.
"""
from __future__ import annotations

import os
import tempfile


def atomic_write_bytes(target: str, payload: bytes) -> None:
    """Atomically replace `target` with `payload` (temp + fsync + rename)."""
    target = os.path.abspath(target)
    d = os.path.dirname(target)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".pn119-swap-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        _fsync_dir(d)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_npz(target: str, arrays: dict) -> None:
    """np.savez(**arrays) into `target` atomically (see module docstring)."""
    import io

    import numpy as np

    buf = io.BytesIO()
    np.savez(buf, **arrays)
    atomic_write_bytes(target, buf.getvalue())


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


if __name__ == "__main__":
    # CLI used by the kill-test: loop forever alternately writing two known
    # payload versions to argv[1]; the test SIGKILLs this at random moments
    # and asserts the target always loads as exactly one full version.
    import sys

    import numpy as np

    target = sys.argv[1]
    v = 0
    while True:
        v += 1
        val = float(v % 2)
        atomic_write_npz(target, {
            "mu": np.full(30720, val, dtype=np.float32),
            "sd": np.full(30720, 1.0, dtype=np.float32),
            "Vt10": np.full((10, 30720), val, dtype=np.float32),
            "w": np.full(11, val, dtype=np.float32),
            "version": np.array([v]),
        })
