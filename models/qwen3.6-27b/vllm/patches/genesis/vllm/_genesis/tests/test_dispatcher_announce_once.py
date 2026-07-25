# SPDX-License-Identifier: Apache-2.0
"""Regression: the dispatcher announces each decision ONCE per boot.

Spec-only patches are gated twice — `_apply_spec_module` records the gate
result (sndr/apply/orchestrator.py) and the patch module's own `apply()` calls
`log_decision` again with the identical pair. `_DECISIONS` was deduped by
patch_id, but the `log.info` above it was not, so every such patch emitted two
byte-identical `[Genesis Dispatcher] APPLY <id>` lines. 27 ids doubled in the
2026-07-14 cold boot (PN384, PN79_V2_MD5_CHUNK, PN79_V2_MD5_CHUNK_DELTA_H, …),
which inflated every announcement-derived patch count — the same blind spot
that let the phantom-patch class hide.

A repeat that carries a genuinely DIFFERENT decision still logs: the legacy
and spec phases can evaluate one id twice with different reasons, and losing
the second line would hide a real state change.
"""
from __future__ import annotations

import logging
import pathlib
import sys

import pytest

# …/genesis/vllm/_genesis — `sndr` is registered as a top-level package there.
_GEN = pathlib.Path(__file__).resolve().parent.parent
if str(_GEN) not in sys.path:
    sys.path.insert(0, str(_GEN))

from sndr.dispatcher import decision as dec  # noqa: E402

ANNOUNCE = "[Genesis Dispatcher]"


@pytest.fixture(autouse=True)
def _clean_decisions():
    saved = list(dec._DECISIONS)
    dec._DECISIONS.clear()
    yield
    dec._DECISIONS.clear()
    dec._DECISIONS.extend(saved)


def _announcements(caplog, patch_id: str) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if ANNOUNCE in r.getMessage() and patch_id in r.getMessage()
    ]


def test_identical_repeat_announces_once(caplog):
    """The double-emit itself: same id, same decision, same reason."""
    with caplog.at_level(logging.INFO):
        dec.log_decision("PXTEST1", True, "opt-in env (config: neutral)")
        dec.log_decision("PXTEST1", True, "opt-in env (config: neutral)")

    assert len(_announcements(caplog, "PXTEST1")) == 1
    # and the matrix still holds exactly one row
    rows = [d for d in dec._DECISIONS if d["patch_id"] == "PXTEST1"]
    assert len(rows) == 1
    assert rows[0]["applied"] is True


def test_changed_reason_still_announces(caplog):
    """A re-evaluation with new information must not be swallowed."""
    with caplog.at_level(logging.INFO):
        dec.log_decision("PXTEST2", False, "opt-in only")
        dec.log_decision("PXTEST2", False, "LIFECYCLE: retired")

    assert len(_announcements(caplog, "PXTEST2")) == 2
    rows = [d for d in dec._DECISIONS if d["patch_id"] == "PXTEST2"]
    assert len(rows) == 1
    assert rows[0]["reason"] == "LIFECYCLE: retired"


def test_flipped_verdict_still_announces(caplog):
    """SKIP → APPLY (or back) is a real state change; keep both lines."""
    with caplog.at_level(logging.INFO):
        dec.log_decision("PXTEST3", False, "same reason text")
        dec.log_decision("PXTEST3", True, "same reason text")

    lines = _announcements(caplog, "PXTEST3")
    assert len(lines) == 2
    assert "SKIP" in lines[0] and "APPLY" in lines[1]
    rows = [d for d in dec._DECISIONS if d["patch_id"] == "PXTEST3"]
    assert len(rows) == 1
    assert rows[0]["applied"] is True


def test_distinct_patches_each_announce(caplog):
    """The guard is per patch_id — it must not suppress other patches."""
    with caplog.at_level(logging.INFO):
        dec.log_decision("PXTEST4", True, "opt-in env")
        dec.log_decision("PXTEST5", True, "opt-in env")
        dec.log_decision("PXTEST4", True, "opt-in env")

    assert len(_announcements(caplog, "PXTEST4")) == 1
    assert len(_announcements(caplog, "PXTEST5")) == 1
    assert len(dec._DECISIONS) == 2


def test_first_seen_order_is_preserved(caplog):
    """Dedup replaces in place; the matrix keeps first-seen ordering."""
    with caplog.at_level(logging.INFO):
        dec.log_decision("PXTEST6", True, "a")
        dec.log_decision("PXTEST7", True, "b")
        dec.log_decision("PXTEST6", False, "c")

    assert [d["patch_id"] for d in dec._DECISIONS] == ["PXTEST6", "PXTEST7"]
    assert dec._DECISIONS[0]["applied"] is False
    assert dec._DECISIONS[0]["reason"] == "c"
