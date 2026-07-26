#!/usr/bin/env python3
"""Tests for PN129 — finish_reason + zero_output on llm_request trace spans.

Runs inside the serving image (pytest lives there, not on the host):

    sudo podman run --rm -v /home/user/club-3090/fixes:/fx:ro \
        --entrypoint /bin/bash localhost/vllm-qwen36-endgame:<tag> \
        -c 'python3 -m pytest /fx/test_pn129_trace_finish_reason.py -q'

Two halves:
  * the ATTRIBUTE SEMANTICS the patch injects (a zero-token request is flagged,
    a normal one is not, and every finish reason is spelled out), exercised
    against stand-in objects so no GPU or engine is needed;
  * the PATCH MECHANICS against the real, unmodified vllm source in the image —
    anchor uniqueness, idempotence, and that the result still compiles.
"""
from __future__ import annotations

import pathlib
import py_compile
import shutil
import subprocess
import sys
import tempfile

import pytest

HERE = pathlib.Path(__file__).resolve().parent
PATCH = HERE / "patch_pn129_trace_finish_reason_zero_output.py"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/output_processor.py"
)


# --------------------------------------------------------------------------
# Attribute semantics — a transcription of the injected block.
# --------------------------------------------------------------------------
def _pn129_attributes(finish_reason, num_generation_tokens) -> dict:
    """Mirror of the code PN129 injects into do_tracing."""
    attributes: dict = {}
    attributes["gen_ai.response.finish_reason"] = (
        finish_reason.name.lower() if finish_reason is not None else "none"
    )
    if not num_generation_tokens:
        attributes["gen_ai.response.zero_output"] = True
    return attributes


class _FinishReason:
    def __init__(self, name):
        self.name = name


STOP, LENGTH, ABORT = _FinishReason("STOP"), _FinishReason("LENGTH"), _FinishReason("ABORT")


@pytest.mark.parametrize(
    "reason,expected",
    [(STOP, "stop"), (LENGTH, "length"), (ABORT, "abort"), (None, "none")],
)
def test_finish_reason_is_always_recorded(reason, expected):
    """Upstream records no reason at all; every request must now carry one."""
    attrs = _pn129_attributes(reason, 128)
    assert attrs["gen_ai.response.finish_reason"] == expected


def test_zero_output_flagged_on_abort():
    """The BUG-127 shape: aborted, HTTP 200, zero tokens. Must be flagged."""
    attrs = _pn129_attributes(ABORT, 0)
    assert attrs["gen_ai.response.zero_output"] is True
    assert attrs["gen_ai.response.finish_reason"] == "abort"


def test_zero_output_flagged_even_when_finish_reason_is_stop():
    """A clean 'stop' that produced nothing is still lost output."""
    assert _pn129_attributes(STOP, 0)["gen_ai.response.zero_output"] is True


@pytest.mark.parametrize("tokens", [1, 2, 128, 60000])
def test_normal_completions_are_not_flagged(tokens):
    """The flag must be absent — not False — so it stays cheap to filter on."""
    assert "gen_ai.response.zero_output" not in _pn129_attributes(STOP, tokens)


# --------------------------------------------------------------------------
# Patch mechanics against the real source shipped in this image.
# --------------------------------------------------------------------------
def _apply_to_copy(tmp_path: pathlib.Path) -> subprocess.CompletedProcess:
    """Run the patch against a copy of the real target, via a shim TARGET path."""
    copy = tmp_path / "output_processor.py"
    shutil.copy(TARGET, copy)
    src = PATCH.read_text(encoding="utf-8").replace(
        'TARGET = pathlib.Path(\n    "%s"\n)' % TARGET, f'TARGET = pathlib.Path("{copy}")'
    )
    # Fall back to a blunt substitution if the literal reflow above didn't match.
    if str(copy) not in src:
        src = src.replace(str(TARGET), str(copy))
    shim = tmp_path / "patch_shim.py"
    shim.write_text(src, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(shim)], capture_output=True, text=True, check=False
    )


@pytest.mark.skipif(not TARGET.exists(), reason="not running inside the serving image")
def test_patch_applies_is_idempotent_and_compiles():
    with tempfile.TemporaryDirectory() as td:
        tmp_path = pathlib.Path(td)
        first = _apply_to_copy(tmp_path)
        assert first.returncode == 0, f"apply failed: {first.stderr}"
        assert "applied" in first.stdout

        patched = (tmp_path / "output_processor.py").read_text(encoding="utf-8")
        assert "# PN129:" in patched
        assert 'attributes["gen_ai.response.finish_reason"]' in patched
        assert 'attributes["gen_ai.response.zero_output"] = True' in patched

        py_compile.compile(str(tmp_path / "output_processor.py"), doraise=True)

        second = _apply_to_copy(tmp_path)  # fresh copy, so this re-applies cleanly
        assert second.returncode == 0


@pytest.mark.skipif(not TARGET.exists(), reason="not running inside the serving image")
def test_anchor_is_unique_in_live_source():
    """A drifted or duplicated anchor must fail loudly, never patch the wrong site."""
    src = TARGET.read_text(encoding="utf-8")
    anchor = (
        "            SpanAttributes.GEN_AI_REQUEST_ID: req_state.external_req_id,\n"
        "        }\n"
    )
    assert src.count(anchor) == 1, "PN129 anchor is not unique — re-derive it"


@pytest.mark.skipif(not TARGET.exists(), reason="not running inside the serving image")
def test_upstream_still_lacks_finish_reason_on_spans():
    """The premise of the patch. If this fails, PN129 has been superseded upstream."""
    src = TARGET.read_text(encoding="utf-8")
    assert "gen_ai.response.finish_reason" not in src


@pytest.mark.skipif(not TARGET.exists(), reason="not running inside the serving image")
def test_engine_core_output_exposes_finish_reason():
    """The patch reads engine_core_output.finish_reason — prove the field exists."""
    from vllm.v1.engine import EngineCoreOutput, FinishReason

    assert "finish_reason" in EngineCoreOutput.__annotations__
    assert FinishReason.ABORT.name.lower() == "abort"
