#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""H119 router-enable flag — canonical-first resolution, both call sites.

WHY THIS FILE EXISTS
--------------------
The lens router was renamed PN119 -> H119 on 2026-07-25 (the id "PN119" already
belonged to lane-2's TurboQuant k8v4 GQA kernel — PATCH-ID-COLLISION-AUDIT-
20260725.md, and it ships live as the unrelated `GENESIS_ENABLE_PN119=1`).
The sidecar kept reading the OLD flag name at two sites, and a ':-' chain in the
compose entrypoint papered over it:

    export GENESIS_ENABLE_PN119_ROUTER="$${GENESIS_ENABLE_PN119_ROUTER:-\
                                          $${GENESIS_ENABLE_H119_LENS_ROUTER:-}}"

That shim is why the router runs at all today: on 2026-07-26 the container's
CONFIG env carried only `GENESIS_ENABLE_H119_LENS_ROUTER=1`, and the legacy name
existed *solely* in the engine process's environ because the entrypoint minted
it. Deleting the shim without moving the resolution into the sidecar would have
turned live routing OFF silently — the flag is the router's master switch
(`maybe_create` returns None and nothing else ever complains).

WHAT IS PINNED
--------------
T1  only-new-set        — canonical alone drives on and off.
T2  only-old-set        — BACK-COMPAT: legacy alone still drives on and off.
T3  both-set agreeing   — no interaction, either value.
T4  both-set DISAGREE   — the CANONICAL name wins, in both directions.
T5  neither set         — off.
T6  empty == unset      — an empty canonical falls THROUGH to the legacy name
                          (a `${FOO:-}` chain yields "" on docker-compose).
T7  live shape          — today's container env, and the same env with the shim
                          removed, both resolve ON.
T8  call site 1 (gate)  — `PN119Router.maybe_create` really consults the helper:
                          flag-off short-circuits BEFORE the probe check.
T9  call site 2 (health)— `health_snapshot` reports through the same helper.
T10 no stragglers       — no raw read of either flag name is left anywhere in
                          the module outside the helper. This is the guard that
                          stops a future edit from silently un-fixing one site.

PRECEDENCE, AND WHY (T4)
------------------------
Canonical-first, FIRST-SET-WINS: if `GENESIS_ENABLE_H119_LENS_ROUTER` is set to
a non-empty value it decides, even against a contradicting legacy alias.
  * an OR over both names would make the canonical name unable to turn the
    router OFF — a stale `GENESIS_ENABLE_PN119_ROUTER=1` export (which is
    exactly what the old shim used to leave lying around) would pin it on, and
    the canonical `=0` kill-switch an operator reaches for first would be inert;
  * an AND would break T2 and T1's only-new-set case;
  * legacy-first would keep the deprecated name authoritative forever, which is
    the opposite of the rename.
The rule matches the shim's own semantics with the priority inverted onto the
canonical name, so the shim becomes redundant rather than merely tolerated.

No GPU, no vllm, no torch, no container: the module is loaded BY PATH with a
stub `torch` in sys.modules (its only heavy import; every module-level
statement in it is a plain constant).

Run (pytest, anywhere):

    python3 -m pytest -q --noconftest fixes/test_h119flag_precedence.py

Run (no pytest installed — the bare aibox host has none):

    python3 fixes/test_h119flag_precedence.py
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import logging
import os
import sys
import tempfile
import types
from pathlib import Path

FIXES = Path(__file__).resolve().parent
ROUTER_PY = FIXES / "pn119_router.py"

CANON = "GENESIS_ENABLE_H119_LENS_ROUTER"
LEGACY = "GENESIS_ENABLE_PN119_ROUTER"
# The lane-2 TurboQuant kernel flag. Sits in the same environment, is NOT this
# router, and must never be mistaken for the master switch by a prefix match.
FOREIGN = "GENESIS_ENABLE_PN119"


def _stub_torch() -> None:
    """Install a no-op `torch` so the sidecar imports on a CPU-only host."""
    if "torch" in sys.modules:
        return
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.float32 = "float32"
    torch.no_grad = contextlib.nullcontext
    sys.modules["torch"] = torch


def _load():
    """Import fixes/pn119_router.py by path. No package context, no vllm."""
    assert ROUTER_PY.is_file(), f"router sidecar missing at {ROUTER_PY}"
    _stub_torch()
    spec = importlib.util.spec_from_file_location("_h119flag_router", ROUTER_PY)
    mod = importlib.util.module_from_spec(spec)
    # In sys.modules BEFORE exec: @dataclass resolves its owning module out of
    # sys.modules on py3.12+ and raises AttributeError if it is absent.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load()
SRC = ROUTER_PY.read_text(encoding="utf-8")
TREE = ast.parse(SRC)


@contextlib.contextmanager
def env(**pairs):
    """Set/clear exactly these vars; restore afterwards.

    A value of None means UNSET. Both flag names plus the foreign lane-2 flag
    are cleared first so an inherited shell export cannot make a case pass.
    """
    keys = set(pairs) | {CANON, LEGACY, FOREIGN}
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        for k, v in pairs.items():
            if v is not None:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def enabled(**pairs) -> bool:
    with env(**pairs):
        return R._router_enabled()


# ══════════════════════════════════════════════════════════ T1  only-new-set
def test_only_canonical_set():
    for on in ("1", "true", "TRUE", "yes", "on"):
        assert enabled(**{CANON: on}) is True, f"canonical {on!r} should enable"
    for off in ("0", "false", "no", "off", "banana"):
        assert enabled(**{CANON: off}) is False, f"canonical {off!r} should disable"


# ══════════════════════════════════════════════════════════ T2  only-old-set
def test_only_legacy_set_still_works():
    """An operator whose compose sets ONLY the old name keeps old behaviour."""
    for on in ("1", "true", "yes", "on"):
        assert enabled(**{LEGACY: on}) is True, f"legacy {on!r} must still enable"
    for off in ("0", "false", "no", "off"):
        assert enabled(**{LEGACY: off}) is False, f"legacy {off!r} must disable"


# ═════════════════════════════════════════════════════ T3  both, in agreement
def test_both_set_agreeing():
    assert enabled(**{CANON: "1", LEGACY: "1"}) is True
    assert enabled(**{CANON: "0", LEGACY: "0"}) is False


# ══════════════════════════════════════════════════ T4  both, DISAGREEING
def test_both_set_disagreeing_canonical_wins():
    """INTENDED PRECEDENCE: the canonical H119 name wins when it is set.

    Direction A is the one that matters operationally: a stale legacy export
    (the old shim minted one on every boot) must NOT override an explicit
    canonical kill-switch. Direction B proves the rule is precedence, not an OR.
    """
    # A: canonical OFF beats legacy ON.
    assert enabled(**{CANON: "0", LEGACY: "1"}) is False, (
        "canonical kill-switch must beat a stale legacy alias — an OR here "
        "would make the router impossible to turn off by its own flag name"
    )
    # B: canonical ON beats legacy OFF.
    assert enabled(**{CANON: "1", LEGACY: "0"}) is True, (
        "canonical must beat a stale legacy OFF — an AND here would strand "
        "anyone who set only the new name"
    )


# ═════════════════════════════════════════════════════════ T5  neither set
def test_neither_set():
    assert enabled() is False


# ═══════════════════════════════════════════════════════ T6  empty == unset
def test_empty_canonical_falls_through():
    """`${FOO:-}` yields "" on docker-compose; that must not read as "off"."""
    assert enabled(**{CANON: "", LEGACY: "1"}) is True
    assert enabled(**{CANON: "   ", LEGACY: "1"}) is True
    assert enabled(**{CANON: "", LEGACY: "0"}) is False
    assert enabled(**{CANON: "", LEGACY: None}) is False


def test_foreign_pn119_flag_is_not_the_router():
    """GENESIS_ENABLE_PN119 is lane-2's TurboQuant kernel, not this router."""
    assert enabled(**{FOREIGN: "1"}) is False


# ══════════════════════════════════════════════════════════ T7  live shapes
def test_live_container_shape_and_shim_removal():
    """Both today's env and the post-shim-deletion env must resolve ON.

    Measured 2026-07-26 on vllm-tcbench-8021: the container CONFIG carries
    GENESIS_ENABLE_H119_LENS_ROUTER=1 and NO legacy name; the engine process
    environ additionally carries GENESIS_ENABLE_PN119_ROUTER=1, minted by the
    entrypoint shim. If either shape came out False, deleting the shim would
    silently disable live routing.
    """
    # engine process today (shim present, both names, agreeing)
    assert enabled(**{CANON: "1", LEGACY: "1", FOREIGN: "1"}) is True
    # container config today / after the shim is deleted (canonical only)
    assert enabled(**{CANON: "1", FOREIGN: "1"}) is True


# ═══════════════════════════════════════════ T8  call site 1 — the real gate
def _capture(fn):
    """Run fn() capturing records from the sidecar's logger."""
    recs: list[logging.LogRecord] = []

    class Sink(logging.Handler):
        def emit(self, record):
            recs.append(record)

    h = Sink()
    lg = R.logger
    lg.addHandler(h)
    prev_level, prev_prop = lg.level, lg.propagate
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    try:
        return fn(), recs
    finally:
        lg.removeHandler(h)
        lg.setLevel(prev_level)
        lg.propagate = prev_prop


def test_maybe_create_gate_uses_the_helper():
    """maybe_create must short-circuit on the flag BEFORE touching the probe.

    Distinguishing evidence: with a runner that cannot host the tap, a
    flag-ON call gets far enough to log a diagnostic; a flag-OFF call returns
    None silently. Same inputs, different reason — that is the gate firing,
    not an incidental None from the probe/model checks.
    """
    runner = object()  # no set_aux_hidden_state_layers, no get_model
    with tempfile.NamedTemporaryFile(suffix=".npz") as probe:
        base = {"GENESIS_PN119_PROBE": probe.name}

        # flag OFF via the canonical name, legacy contradicting -> still OFF
        with env(**{CANON: "0", LEGACY: "1", **base}):
            out, recs = _capture(lambda: R.PN119Router.maybe_create(runner))
        assert out is None
        assert not recs, (
            "flag-off must short-circuit before the probe/model checks; got "
            f"{[r.getMessage() for r in recs]}"
        )

        # flag ON via the LEGACY name alone -> gate passes, later check speaks
        with env(**{LEGACY: "1", **base}):
            out, recs = _capture(lambda: R.PN119Router.maybe_create(runner))
        assert out is None  # runner is a stub; the tap cannot attach
        assert recs, "legacy-only flag must still pass the gate (back-compat)"

        # flag ON via the CANONICAL name alone -> same, post-shim shape
        with env(**{CANON: "1", **base}):
            out, recs = _capture(lambda: R.PN119Router.maybe_create(runner))
        assert out is None
        assert recs, "canonical-only flag must pass the gate (shim deleted)"


# ═══════════════════════════════════════ T9/T10  static: both sites, no strays
def _func(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in {ROUTER_PY.name}")


def _calls(node: ast.AST) -> set[str]:
    return {
        c.func.id
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    }


def test_both_call_sites_resolve_through_the_helper():
    for site in ("maybe_create", "health_snapshot"):
        fn = _func(site)
        assert "_router_enabled" in _calls(fn), (
            f"{site}() no longer calls _router_enabled() — the H119 rename is "
            "un-fixed at this site"
        )
        strays = [
            s.value for s in ast.walk(fn)
            if isinstance(s, ast.Constant) and s.value in (CANON, LEGACY)
        ]
        assert not strays, f"{site}() reads a flag name directly: {strays}"


def test_no_raw_flag_reads_outside_the_helper():
    """The only textual uses of either name are the two module constants."""
    hits = [
        node.lineno for node in ast.walk(TREE)
        if isinstance(node, ast.Constant) and node.value in (CANON, LEGACY)
    ]
    const_lines = {
        n.lineno for n in TREE.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in ("ENABLE_FLAG",
                                                     "ENABLE_FLAG_LEGACY")
                for t in n.targets)
    }
    assert len(const_lines) == 2, f"expected 2 flag constants, got {const_lines}"
    stray = sorted(set(hits) - const_lines)
    assert not stray, (
        f"raw flag-name string(s) outside the ENABLE_FLAG constants at "
        f"line(s) {stray} — route every read through _router_enabled()"
    )


def test_helper_reads_canonical_before_legacy():
    """Guards the ORDER, which no value-level test can see once both agree."""
    fn = _func("_router_enabled")
    # ast.walk is breadth-first, so sort back into SOURCE order before reading
    # the sequence — otherwise this test would pin nothing.
    names = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Name) and n.id in ("ENABLE_FLAG", "ENABLE_FLAG_LEGACY")
    ]
    order = [n.id for n in sorted(names, key=lambda n: (n.lineno, n.col_offset))]
    assert order[:2] == ["ENABLE_FLAG", "ENABLE_FLAG_LEGACY"], (
        f"_router_enabled must read the canonical name first; saw {order}"
    )
    assert R.ENABLE_FLAG == CANON and R.ENABLE_FLAG_LEGACY == LEGACY


# ═══════════════════════════════════════════════════════════════════════════
# Bare-python fallback: the aibox host has no pytest, and a gate that cannot be
# run on the box it guards is not a gate.
# ═══════════════════════════════════════════════════════════════════════════

def _main() -> int:
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"ok   {name}")
    print("FAILED" if failed else "PASSED", f"({failed} failure(s))")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
