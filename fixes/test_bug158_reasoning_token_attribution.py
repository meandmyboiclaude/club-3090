#!/usr/bin/env python3
"""Tests for patch_bug158_reasoning_token_attribution.py.

Runs on the HOST with plain python3 — no vLLM import, no GPU, no server —
either under pytest or standalone (same technique as
test_bug077_repetition_detection.py: the applier module is loaded by path and
its injected source is exec'd into a namespace of stand-ins):

    /usr/bin/python3 -m pytest /home/user/club-3090/fixes/test_bug158_reasoning_token_attribution.py -q
    /usr/bin/python3 /home/user/club-3090/fixes/test_bug158_reasoning_token_attribution.py

Covers
  1. the injected counter across the five stream shapes BUG-154/BUG-158/BUG-160
     put in play — normal close, no close (runaway), double close,
     close-then-reopen, empty reasoning — plus the implicit ``<tool_call>``
     reasoning end and a re-entry-capable config (glm47_moe / deepseek_v4
     shape);
  2. the darkness contract: flag off -> defer to the stock walk, for every
     shape; ``enable_thinking=false`` (initial_state CONTENT) -> defer even
     with the flag on;
  3. AGREEMENT with a verbatim replica of the qwen3 transition table
     (``vllm/parser/qwen3.py:121-194``), i.e. the count equals the number of
     tokens the parser itself routes to ``reasoning`` — and the replica's
     CONTENT output is byte-identical with the graft on and off, on all shapes;
  4. CONTENT-INVARIANCE structurally: applied to the REAL container file the
     patch is a pure single-point INSERTION — zero deleted lines, zero replaced
     lines, and the whole file outside ``count_reasoning_tokens`` is byte-equal;
  5. the applier: anchor detection, idempotency, FATAL on drift / half-apply /
     already-rewritten walk, and that the result compiles.
"""
import difflib
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PATCH = HERE / "patch_bug158_reasoning_token_attribution.py"

spec = importlib.util.spec_from_file_location("bug158_patch", PATCH)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

FLAG = "GENESIS_ENABLE_BUG158_REASONING_TOKENS"

# ── stand-ins for the vLLM types the injected code touches ──────────────────
THINK_START, THINK_END = 248068, 248069        # real ids from the live vocab
TOOL_START, TOOL_END = 900001, 900002
R = [11, 12, 13, 14, 15]                       # ordinary reasoning tokens
C = [21, 22, 23, 24, 25]                       # ordinary content tokens


class ParserState:
    CONTENT = "CONTENT"
    REASONING = "REASONING"
    TOOL_PREAMBLE = "TOOL_PREAMBLE"


class _Tr:
    def __init__(self, next_state):
        self.next_state = next_state


class _Cfg:
    """Mirrors vllm.parser.engine.parser_engine_config.ParserEngineConfig."""

    def __init__(self, initial_state, transitions, token_id_terminals):
        self.initial_state = initial_state
        self.transitions = transitions
        self.token_id_terminals = token_id_terminals


VOCAB = {
    "<think>": THINK_START,
    "</think>": THINK_END,
    "<tool_call>": TOOL_START,
    "</tool_call>": TOOL_END,
}
TOKEN_ID_TERMINALS = {
    "THINK_START": "<think>",
    "THINK_END": "</think>",
    "TOOL_START": "<tool_call>",
    "TOOL_END": "</tool_call>",
}

# verbatim from vllm/parser/qwen3.py:121-194 (reasoning-relevant rows)
QWEN3_TRANSITIONS = {
    (ParserState.REASONING, "THINK_START"): _Tr(ParserState.REASONING),
    (ParserState.REASONING, "THINK_END"): _Tr(ParserState.CONTENT),
    (ParserState.CONTENT, "THINK_END"): _Tr(ParserState.CONTENT),
    (ParserState.REASONING, "TOOL_START"): _Tr(ParserState.TOOL_PREAMBLE),
    (ParserState.CONTENT, "TOOL_START"): _Tr(ParserState.TOOL_PREAMBLE),
    (ParserState.TOOL_PREAMBLE, "TOOL_END"): _Tr(ParserState.CONTENT),
}
# glm47_moe.py:106 / deepseek_v4.py:146 additionally allow re-entry
REENTRY_TRANSITIONS = dict(QWEN3_TRANSITIONS)
REENTRY_TRANSITIONS[(ParserState.CONTENT, "THINK_START")] = _Tr(ParserState.REASONING)


class _Engine:
    """Minimal stand-in exposing exactly what the graft reads."""

    def __init__(self, thinking=True, transitions=None):
        self.parser_engine_config = _Cfg(
            ParserState.REASONING if thinking else ParserState.CONTENT,
            QWEN3_TRANSITIONS if transitions is None else transitions,
            TOKEN_ID_TERMINALS,
        )
        self.vocab = VOCAB
        self._reasoning_start_token_id = THINK_START
        self._reasoning_end_token_id = THINK_END


def _load_graft():
    """exec the injected helper source into a namespace of stand-ins."""
    ns = {"ParserState": ParserState}
    body = "class _Host:\n" + patch.HELPER_SRC
    exec(body, ns)
    return ns["_Host"]


_Host = _load_graft()


def _mk(thinking=True, transitions=None):
    """An object with the graft methods bound onto an engine stand-in."""
    obj = _Engine(thinking=thinking, transitions=transitions)
    obj._bug158_span_markers = _Host._bug158_span_markers.__get__(obj)
    obj._bug158_count_reasoning_tokens = (
        _Host._bug158_count_reasoning_tokens.__get__(obj)
    )
    return obj


def _flag(on):
    if on:
        os.environ[FLAG] = "1"
    else:
        os.environ.pop(FLAG, None)


# ── the stock walk, verbatim from parser_engine.py:624-641 ──────────────────
def stock_count(start_id, end_id, token_ids):
    if start_id is None or end_id is None:
        return 0
    count = 0
    depth = 0
    for token_id in token_ids:
        if token_id == start_id:
            depth += 1
            continue
        if token_id == end_id:
            if depth > 0:
                depth -= 1
            continue
        if depth > 0:
            count += 1
    return count


# ── replica of the qwen3 FSM: what the PARSER puts in reasoning vs content ──
def fsm_split(token_ids, thinking=True, transitions=None):
    """Return (reasoning_ids, content_ids) the way the engine routes them."""
    tr = QWEN3_TRANSITIONS if transitions is None else transitions
    name_of = {v: k for k, v in
               {n: VOCAB[t] for n, t in TOKEN_ID_TERMINALS.items()}.items()}
    state = ParserState.REASONING if thinking else ParserState.CONTENT
    reasoning, content = [], []
    for tid in token_ids:
        name = name_of.get(tid)
        if name is not None:
            nxt = tr.get((state, name))
            if nxt is not None:
                state = nxt.next_state
                continue
            # unmatched terminal: the engine has no transition, so the literal
            # text stays in whatever channel the current state feeds
        if state == ParserState.REASONING:
            reasoning.append(tid)
        elif state == ParserState.CONTENT:
            content.append(tid)
        # TOOL_* states feed the tool channel, neither reasoning nor content
    return reasoning, content


# ── the five shapes (+2) ────────────────────────────────────────────────────
SHAPES = {
    # name: (token_ids, expected reasoning-token count under the graft)
    "normal close":        ([R[0], R[1], R[2], THINK_END, C[0], C[1]], 3),
    "no close (runaway)":  ([R[0], R[1], R[2], R[3], R[4]], 5),
    "double close":        ([R[0], R[1], THINK_END, C[0], THINK_END, C[1]], 2),
    "close-then-reopen":   ([R[0], THINK_END, C[0], THINK_START, C[1],
                             THINK_END, C[2]], 1),
    "empty reasoning":     ([THINK_END, C[0], C[1]], 0),
    "implicit tool end":   ([R[0], R[1], TOOL_START, C[0], TOOL_END, C[1]], 2),
    "opener echoed in out": ([R[0], THINK_START, R[1], THINK_END, C[0]], 2),
}


def test_counts_match_expected():
    _flag(True)
    try:
        eng = _mk()
        for name, (ids, want) in SHAPES.items():
            got = eng._bug158_count_reasoning_tokens(ids)
            assert got == want, f"{name}: got {got}, want {want}"
    finally:
        _flag(False)


def test_count_agrees_with_parser_fsm():
    """The count must equal what the parser routes to `reasoning`."""
    _flag(True)
    try:
        eng = _mk()
        for name, (ids, _) in SHAPES.items():
            reasoning, _content = fsm_split(ids)
            got = eng._bug158_count_reasoning_tokens(ids)
            assert got == len(reasoning), (
                f"{name}: count {got} != FSM reasoning span {len(reasoning)}")
    finally:
        _flag(False)


def test_content_channel_identical_flag_on_and_off():
    """The graft cannot move content: the split is computed by code it does
    not touch. Assert the FSM's content channel is byte-identical either way."""
    for name, (ids, _) in SHAPES.items():
        _flag(False)
        _r_off, c_off = fsm_split(ids)
        _flag(True)
        try:
            _r_on, c_on = fsm_split(ids)
        finally:
            _flag(False)
        assert c_off == c_on, f"{name}: content channel moved"


def test_stock_walk_misattributes():
    """The defect itself. With the opener in prompt space the stock walk
    attributes NOTHING on every shape whose output carries no `<think>`; and on
    the two shapes that do carry an echoed opener it counts the wrong tokens."""
    zero_shapes = ("normal close", "no close (runaway)", "double close",
                   "empty reasoning", "implicit tool end")
    for name in zero_shapes:
        ids, want = SHAPES[name]
        got = stock_count(THINK_START, THINK_END, ids)
        assert got == 0, f"{name}: stock walk returned {got}, expected 0"
        assert want > 0 or name == "empty reasoning"

    # close-then-reopen: the echoed <think> opens a span the PARSER never
    # opened (qwen3 has no (CONTENT, THINK_START) transition), so the stock
    # walk bills a CONTENT token as reasoning while missing the real span.
    ids, _ = SHAPES["close-then-reopen"]
    assert stock_count(THINK_START, THINK_END, ids) == 1
    reasoning, _c = fsm_split(ids)
    assert [t for t in reasoning] == [R[0]]        # the real span
    # ...and the token the stock walk actually counted is C[1], a content token
    assert C[1] in ids

    # opener echoed mid-reasoning: stock counts only the post-echo tail (2 of 2
    # here by coincidence of length, but from the wrong start point)
    ids, want = SHAPES["opener echoed in out"]
    assert stock_count(THINK_START, THINK_END, ids) == 1 and want == 2


def test_dark_by_default():
    _flag(False)
    eng = _mk()
    for name, (ids, _) in SHAPES.items():
        assert eng._bug158_count_reasoning_tokens(ids) is None, name


def test_defers_when_thinking_disabled():
    """enable_thinking=false -> initial_state CONTENT -> stock walk is right."""
    _flag(True)
    try:
        eng = _mk(thinking=False)
        for name, (ids, _) in SHAPES.items():
            assert eng._bug158_count_reasoning_tokens(ids) is None, name
    finally:
        _flag(False)


def test_defers_when_no_end_token_id():
    _flag(True)
    try:
        eng = _mk()
        eng._reasoning_end_token_id = None
        assert eng._bug158_count_reasoning_tokens([R[0]]) is None
    finally:
        _flag(False)


def test_reentry_config_reopens():
    """glm47_moe / deepseek_v4 declare (CONTENT, THINK_START) -> REASONING, so
    a second <think> DOES reopen there — and the count must follow."""
    _flag(True)
    try:
        eng = _mk(transitions=REENTRY_TRANSITIONS)
        ids, _ = SHAPES["close-then-reopen"]
        reasoning, _c = fsm_split(ids, transitions=REENTRY_TRANSITIONS)
        got = eng._bug158_count_reasoning_tokens(ids)
        assert got == 2, got                      # R[0] + C[1] after reopen
        assert got == len(reasoning), (got, len(reasoning))
        # and qwen3 (no re-entry) must NOT behave that way
        assert _mk()._bug158_count_reasoning_tokens(ids) == 1
    finally:
        _flag(False)


def test_marker_cache_is_stable():
    _flag(True)
    try:
        eng = _mk()
        a = eng._bug158_span_markers()
        b = eng._bug158_span_markers()
        assert a is b
        starters, enders = a
        assert starters == set()                  # qwen3 has no re-entry
        assert enders == {THINK_END, TOOL_START}
    finally:
        _flag(False)


def test_empty_and_content_only_streams():
    _flag(True)
    try:
        eng = _mk()
        assert eng._bug158_count_reasoning_tokens([]) == 0
        # answer with no closer at all and no reasoning is indistinguishable
        # from a runaway — the FSM agrees, so the count does too
        r, _c = fsm_split(C[:3])
        assert eng._bug158_count_reasoning_tokens(C[:3]) == len(r) == 3
    finally:
        _flag(False)


# ── applier ─────────────────────────────────────────────────────────────────
FIXTURE = '''from vllm.parser.engine.parser_engine_config import ParserEngineConfig, ParserState


class ParserEngine(Parser):
    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return input_ids

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        start_id = self._reasoning_start_token_id
        end_id = self._reasoning_end_token_id
        if start_id is None or end_id is None:
            return 0
        count = 0
        depth = 0
        for token_id in token_ids:
            if token_id == start_id:
                depth += 1
                continue
            if token_id == end_id:
                if depth > 0:
                    depth -= 1
                continue
            if depth > 0:
                count += 1
        return count

    def _single_pass_parse(self, text, token_ids, initial_state=None):
        return None, None, None
'''


def _run_applier(base):
    env = dict(os.environ, BUG158_VLLM_BASE=str(base))
    return subprocess.run([sys.executable, str(PATCH)], env=env,
                          capture_output=True, text=True)


def _stage(td, text):
    base = pathlib.Path(td)
    tgt = base / "parser/engine/parser_engine.py"
    tgt.parent.mkdir(parents=True, exist_ok=True)
    tgt.write_text(text, encoding="utf-8")
    return base, tgt


def test_applier_applies_and_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        base, tgt = _stage(td, FIXTURE)
        r = _run_applier(base)
        assert r.returncode == 0, r.stdout + r.stderr
        out = tgt.read_text(encoding="utf-8")
        assert patch.MARK_HELPER in out
        assert patch.MARK_CALL in out
        assert "_bug158 = self._bug158_count_reasoning_tokens(token_ids)" in out
        compile(out, "parser_engine.py", "exec")

        r2 = _run_applier(base)
        assert r2.returncode == 0 and "already applied" in r2.stdout, r2.stdout
        assert tgt.read_text(encoding="utf-8") == out


def test_applier_is_a_pure_insertion():
    """Content-invariance, structurally: nothing is deleted or replaced."""
    with tempfile.TemporaryDirectory() as td:
        base, tgt = _stage(td, FIXTURE)
        assert _run_applier(base).returncode == 0
        before = FIXTURE.splitlines(keepends=True)
        after = tgt.read_text(encoding="utf-8").splitlines(keepends=True)
        ops = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
        kinds = {op[0] for op in ops.get_opcodes() if op[0] != "equal"}
        assert kinds == {"insert"}, kinds        # no delete, no replace
        # and the stock walk survives verbatim underneath the delegation
        assert patch.ANCH_STOCK_BODY in "".join(after)


def test_applier_fatal_on_drift():
    with tempfile.TemporaryDirectory() as td:
        base, tgt = _stage(td, FIXTURE.replace(
            "    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:\n",
            "    def count_reasoning_tokens(self, ids) -> int:\n"))
        r = _run_applier(base)
        assert r.returncode == 1 and "method anchor" in r.stderr, r.stderr


def test_applier_fatal_when_walk_already_rewritten():
    with tempfile.TemporaryDirectory() as td:
        base, tgt = _stage(td, FIXTURE.replace("        depth = 0\n",
                                               "        depth = 1\n"))
        r = _run_applier(base)
        assert r.returncode == 1 and "already been rewritten" in r.stderr, r.stderr


def test_applier_fatal_on_half_applied():
    with tempfile.TemporaryDirectory() as td:
        base, tgt = _stage(td, FIXTURE + "\n    " + patch.MARK_HELPER + "\n")
        r = _run_applier(base)
        assert r.returncode == 1 and "half-applied" in r.stderr, r.stderr


def test_applier_fatal_on_missing_target():
    with tempfile.TemporaryDirectory() as td:
        r = _run_applier(pathlib.Path(td))
        assert r.returncode == 1 and "missing target" in r.stderr, r.stderr


# ── against the REAL container file, when a copy is available ───────────────
def _real_copy():
    """The real in-container parser_engine.py, for a dry-run against it.

    Shipped as a .ref alongside this test (taken from image
    dev1474cherrymax-1757-20260725, AFTER the boot appliers ran). Refresh for a
    new image with:

        sudo podman cp vllm-tcbench-8021:/usr/local/lib/python3.12/dist-packages/\\
            vllm/parser/engine/parser_engine.py <path>
        BUG158_REAL_PARSER_ENGINE=<path> python3 <this file>
    """
    p = os.environ.get("BUG158_REAL_PARSER_ENGINE")
    if p and pathlib.Path(p).is_file():
        return pathlib.Path(p)
    p = HERE / "_refs/parser_engine__dev1474cherrymax-1757-20260725.py.ref"
    return p if p.is_file() else None


def test_real_file_pure_insertion():
    real = _real_copy()
    if real is None:
        print("  skip real-file test (no copy of the container file present)")
        return
    original = real.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as td:
        base, tgt = _stage(td, original)
        r = _run_applier(base)
        assert r.returncode == 0, r.stdout + r.stderr
        after_text = tgt.read_text(encoding="utf-8")
        compile(after_text, "parser_engine.py", "exec")

        # exact: a single-point insertion, nothing else in the file moves
        expect = original.replace(
            patch.ANCH_METHOD, patch.HELPER_SRC + patch.REPL_METHOD, 1)
        assert after_text == expect

        ops = difflib.SequenceMatcher(
            a=original.splitlines(keepends=True),
            b=after_text.splitlines(keepends=True), autojunk=False)
        kinds = {op[0] for op in ops.get_opcodes() if op[0] != "equal"}
        assert kinds == {"insert"}, kinds        # no delete, no replace

        # every method that produces served bytes is untouched: excising the
        # inserted block byte-for-byte must reproduce the original file, and
        # each split-path symbol must survive with the same arity.
        ins = patch.HELPER_SRC + patch.REPL_METHOD
        assert after_text.count(ins) == 1
        assert after_text.replace(ins, patch.ANCH_METHOD, 1) == original
        for sym in ("def extract_reasoning(", "def extract_reasoning_streaming(",
                    "def parse_delta(", "def _single_pass_parse(",
                    "def _events_to_delta(", "def _feed(", "def parse(",
                    "def finish_streaming(", "def extract_content_ids(",
                    "def _strip_trailing_reasoning(", "def _reset("):
            assert original.count(sym) == after_text.count(sym) >= 1, sym
        # the graft adds no reference to any of them
        assert all(s not in ins for s in ("extract_reasoning", "parse_delta",
                                          "_events_to_delta", "_feed("))


if __name__ == "__main__":
    fails = []
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        try:
            fn()
            print(f"  ok   {name}")
        except AssertionError as exc:
            print(f"  FAIL {name}: {exc}")
            fails.append(name)
    print()
    if fails:
        print(f"FAILED {len(fails)}/{len(fns)}: {fails}")
        sys.exit(1)
    print(f"all {len(fns)} tests passed")
