#!/usr/bin/env python3
"""BUG-139 (P1) — a budget-truncated row must be identifiable as one.

Run IN-CONTAINER (pytest 9.1.1 is in the image, not on the host), against a
THROWAWAY container — never inside the serving one:

    sudo podman run --rm \
      -v /home/user/club-3090:/repo:ro \
      -v /home/user/shared/needfit/pn119-sink:/sink:ro \
      -e PN119_SINK_DIR=/sink --entrypoint /bin/bash \
      localhost/vllm-qwen36-endgame:<pin> \
      -c 'cd /tmp && python -m pytest /repo/fixes/test_bug139_censoring.py -v --noconftest -p no:cacheprovider'

WHAT THIS PINS
--------------
BUG-139 is that `cap_hit` cannot see thinking-budget truncation, so a forced
`</think>` lands in the training corpus as a natural stop and "lean" becomes an
absorbing state. Its FIRST fix added `censored = rtok >= budget - SLACK` with
SLACK = len(think_end_ids) + 8 = 9. That was measured against one truncation
mode and there are two:

    gap = budget_grant - rtok, over all 1046 thinking finishes in the live
    sink that carry a budget_grant (2026-07-26):
        gap  5 : 94 rows        gap 13 : 223 rows
        gap 6..12 : 1 row       gap 14..30 : 7       gap >30 : 721

9 sits exactly BETWEEN the two spikes, so it resolved the 94 and wrote the 223
into the corpus as natural stops. Class of failure: a constant fitted to one
sample, with nothing asserting it still separates the populations it names.

So the tests here are in three layers, weakest last:
  A. GROUND TRUTH. The REAL upstream `ThinkingBudgetStateHolder` is driven to
     an actual budget overrun and `_entry_is_forcing` must see it — and must
     NOT see anything on a row that closed `</think>` by itself. No constant is
     involved, which is the point: this is the detector that cannot rot.
  B. LATCHING + LABEL PLUMBING. The flag lives one step and the finish is
     thousands of steps later.
  C. THE ARITHMETIC FALLBACK, pinned at BOTH measured modes, plus the refit's
     trust rules — the part a future re-tune can silently narrow again.
  D. REPLAY over the live sink when it is mounted: the gap-13 population must
     stop resolving to y=0.
"""
from __future__ import annotations

import collections
import glob
import importlib.util
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TBS_IN_IMAGE = ("/usr/local/lib/python3.12/dist-packages/vllm/v1/sample"
                "/thinking_budget_state.py")

THINK_START = [900]
THINK_END = [901]


# ── loading the units under test ──────────────────────────────────────────
class MoveDirectionality:
    UNIDIRECTIONAL = 0
    SWAP = 1


class BatchUpdate:
    def __init__(self, added=(), removed=(), moved=()):
        self.added = list(added)
        self.removed = list(removed)
        self.moved = list(moved)

    def __bool__(self):
        return bool(self.added or self.removed or self.moved)


def _install_vllm_stubs() -> None:
    """Enough of vllm for thinking_budget_state.py to exec standalone.

    Mirrors fixes/test_h119_budget_consumer.py — same stub set, same reason:
    importing real vllm to test a 400-line state machine costs a model load.
    """
    import torch  # noqa: F401,PLC0415 — the real thing; the module imports it

    for name in ("vllm", "vllm.platforms", "vllm.utils", "vllm.utils.torch_utils",
                 "vllm.v1", "vllm.v1.sample", "vllm.v1.sample.logits_processor",
                 "vllm.v1.sample.logits_processor.interface", "vllm.config",
                 "vllm.config.reasoning"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vllm.platforms"].current_platform = types.SimpleNamespace(
        is_rocm=lambda: False, is_cuda=lambda: True)
    sys.modules["vllm.utils.torch_utils"].async_tensor_h2d = lambda *a, **k: None
    iface = sys.modules["vllm.v1.sample.logits_processor.interface"]
    iface.BatchUpdate = BatchUpdate
    iface.MoveDirectionality = MoveDirectionality
    sys.modules["vllm.config.reasoning"].ReasoningConfig = object
    sys.modules["vllm"].platforms = sys.modules["vllm.platforms"]


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def side():
    """The router sidecar, loaded under the name the boot shims look for."""
    _install_vllm_stubs()
    return _load("vllm._genesis_pn119", os.path.join(HERE, "pn119_router.py"))


@pytest.fixture(scope="module")
def refit():
    return _load("refit_pn119_probe_under_test",
                 os.path.join(HERE, "refit_pn119_probe.py"))


@pytest.fixture(scope="module")
def tbs():
    """The REAL upstream holder, exec'd from wherever it can be found.

    PN119_TBS_SRC lets a host run point at an extracted copy; in-container the
    image path is used directly. No stand-in: a hand-written fake holder would
    assert my model of `in_end`, which is the exact thing under test.
    """
    src_path = os.environ.get("PN119_TBS_SRC") or TBS_IN_IMAGE
    if not os.path.isfile(src_path):
        pytest.skip(f"upstream thinking_budget_state.py not found at {src_path}")
    _install_vllm_stubs()
    mod = types.ModuleType("tbs_real")
    with open(src_path, encoding="utf-8") as fh:
        exec(compile(fh.read(), src_path, "exec"), mod.__dict__)  # noqa: S102
    return mod


class _Params:
    def __init__(self, budget=None):
        self.thinking_token_budget = budget
        self.max_tokens = 4096


class _FakeRouter:
    def __init__(self, req_ids):
        self.mode = "shadow"
        self._h119_applied = {}
        self._h119_forced = {}
        self._pending = []
        self.runner = types.SimpleNamespace(
            input_batch=types.SimpleNamespace(req_ids=list(req_ids)))


def _holder(tbs, budget_slots):
    import torch  # noqa: PLC0415
    rc = types.SimpleNamespace(reasoning_start_token_ids=THINK_START,
                               reasoning_end_token_ids=THINK_END)
    h = tbs.ThinkingBudgetStateHolder(rc, 8, 0, torch.device("cpu"), False)
    prompt = [1, 2, 3, THINK_START[0]]          # thinking-on template shape
    h.sync_batch(BatchUpdate(added=[(i, _Params(b), prompt, [])
                                    for i, b in enumerate(budget_slots)]))
    return h


# ══ A. GROUND TRUTH — the real holder, no constants ═══════════════════════
def test_real_holder_overrun_is_seen_as_forcing(tbs, side):
    """A row that blows its thinking budget must read as forcing.

    Nothing here mentions 5, 13 or any slack: the holder is driven past its
    own budget and asked what it did.
    """
    budget = 12
    h = _holder(tbs, [budget])
    out: list[int] = []
    seen = False
    for _step in range(budget + 8):
        out.append(7)                            # a thinking token
        h.update_state([list(out)], [[]])
        if side._entry_is_forcing(h._state[0]):
            seen = True
            break
    assert seen, (f"the holder never registered as forcing after "
                  f"{len(out)} tokens against a budget of {budget}; "
                  f"state={h._state[0]}")
    assert h._state[0]["in_end"] or h._state[0]["force_index"]


def test_real_holder_natural_close_is_not_forcing(tbs, side):
    """A row that emits `</think>` on its own, well inside budget, is NOT
    censored — the detector must not simply return True."""
    h = _holder(tbs, [4096])
    out = [7, 7, 7, THINK_END[0]]
    h.update_state([list(out)], [[]])
    assert not side._entry_is_forcing(h._state[0]), h._state[0]


def test_relaxed_sentinel_row_is_never_forcing(tbs, side):
    """Budget-less (`-1` sentinel) rows cannot be truncated by a budget."""
    state = {"thinking_token_budget": -1, "in_end": True, "force_index": [0]}
    assert side._entry_is_forcing(state) is False
    assert side._entry_is_forcing({"in_end": True}) is False   # no budget key


def test_force_index_alone_counts(side):
    """An observation arriving a step late (in_end already cleared) still lands."""
    assert side._entry_is_forcing(
        {"thinking_token_budget": 100, "in_end": False, "force_index": [0]})


def test_pn114_armed_span_is_not_a_budget_stop(side):
    """PN114 `_arm()` drives the SAME machinery for arbitrary spans.

    `_genesis/plateau/pn114.py` sets in_end=True, force_index=[0], force_seq=
    <span> and parks thinking_token_budget at 10_000_000 while a plateau probe
    / wrap-up / PN117 rescue runs. Those rows were never truncated; labelling
    them censored would delete real measurements from the corpus. `force_seq`
    is the discriminator — PN114 is the only writer and it clears it on
    disarm, and the holder's own forcing reads
    `state.get("force_seq") or self.think_end_token_ids`.
    """
    armed = {"thinking_token_budget": 10_000_000, "in_think": False,
             "in_end": True, "end_count": 0, "bonus_token_forced": False,
             "force_index": [0], "force_seq": [1, 2, 3], "force_seq_base": 40}
    assert side._entry_is_forcing(armed) is False
    # ...and once PN114 disarms (force_seq -> None) a real budget stop on the
    # same request is seen again.
    disarmed = dict(armed, force_seq=None, thinking_token_budget=1300)
    assert side._entry_is_forcing(disarmed) is True


def test_pn114_arm_shape_is_still_what_this_excludes(side):
    """Pin the exclusion to PN114's ACTUAL arm, read out of its source.

    If `_arm` stops setting `force_seq`, or starts setting some other key, the
    exclusion above goes silently inert and every PN114 row becomes a false
    censoring. This is the assertion that notices.
    """
    pn114 = os.path.join(
        HERE, "..", "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis"
        "/plateau/pn114.py")
    if not os.path.isfile(pn114):
        pytest.skip("pn114.py not in this tree")
    src = open(pn114, encoding="utf-8").read()
    arm = src.split("def _arm(", 1)[1].split("\ndef ", 1)[0]
    for needed in ('state["force_seq"] = list(seq)',
                   'state["in_end"] = True',
                   'state["force_index"] = [0]'):
        assert needed in arm, f"PN114 _arm no longer does: {needed}"


# ══ B. LATCHING AND LABEL PLUMBING ════════════════════════════════════════
def test_observer_latches_across_steps(tbs, side):
    """`in_end` is live for ONE step; the finish line is thousands later."""
    side.ROUTER = _FakeRouter(["r0"])
    side.STATS.clear()
    budget = 12
    h = _holder(tbs, [budget])
    out: list[int] = []
    for _step in range(budget + 8):
        out.append(7)
        h.update_state([list(out)], [[]])
        side.h119_observe_forcing(h)
    assert side.ROUTER._h119_forced.get("r0") is True
    assert side.STATS.get("censor_forced_observed") == 1, "latched more than once"

    # The flag going down again must not un-latch it.
    h._state[0]["in_end"] = False
    h._state[0]["force_index"] = []
    side.h119_observe_forcing(h)
    assert side.ROUTER._h119_forced.get("r0") is True


def test_observer_ignores_out_of_batch_and_missing_req_ids(tbs, side):
    """A state index past the end of req_ids must not raise or mislabel."""
    side.ROUTER = _FakeRouter([])                # empty batch
    side.STATS.clear()
    before = side._consumer_state["warned"]
    h = _holder(tbs, [12])
    h._state[0]["in_end"] = True
    side.h119_observe_forcing(h)                 # index 0 >= len(req_ids) == 0
    assert side.ROUTER._h119_forced == {}
    assert side._consumer_state["warned"] == before, (
        "an out-of-batch index is an expected condition, not an error")


def test_observer_never_raises_without_a_router(side):
    side.ROUTER = None
    side.h119_observe_forcing(object())          # must be a no-op, not a crash


# ══ C. THE ARITHMETIC FALLBACK AND ITS TWO MODES ══════════════════════════
def _router(side, slack=None):
    r = object.__new__(side.PN119Router)
    r._think_start_ids = list(THINK_START)
    r._think_end_ids = list(THINK_END)
    r._tail_window = 64
    r._censor_slack = (slack if slack is not None
                       else len(r._think_end_ids) + 12)
    r._h119_applied = {}
    r._h119_forced = {}
    return r


def _req(rtok, generated=None, max_tokens=4096, closed=True):
    sp = _Params()
    sp.max_tokens = max_tokens
    out = [5] * rtok + ([THINK_END[0]] + [9] * 40 if closed else [])
    return types.SimpleNamespace(
        prompt_token_ids=[7] * 40 + [THINK_START[0]] + [11] * 12,
        output_token_ids=out, sampling_params=sp), len(out)


def test_shipped_slack_covers_both_measured_modes(side):
    """The regression witness. 1300-grant rows at BOTH live offsets."""
    r = _router(side)
    assert r._censor_slack >= 13, (
        "SLACK must cover the gap-13 truncation mode; 9 (len+8) provably "
        "missed 223 of 317 truncated rows in the live sink")
    for gap in (5, 13):
        req, gen = _req(1300 - gap)
        thinking, rtok, cap_hit, censored, src = r._label_fields(
            req, generated=gen, budget=1300)
        assert thinking is True
        assert rtok == 1300 - gap
        assert censored is True, f"gap={gap} not detected (slack={r._censor_slack})"
        assert src == "slack"
        assert cap_hit is False, "max_tokens is not what bound this row"


def test_old_slack_9_is_the_documented_failure(side):
    """Witness that 9 really did miss the gap-13 mode — the reason for 13."""
    r = _router(side, slack=9)
    req, gen = _req(1300 - 13)
    _t, _rtok, _c, censored, src = r._label_fields(req, generated=gen, budget=1300)
    assert censored is False and src == "none"


def test_natural_stop_well_inside_the_grant_is_not_censored(side):
    r = _router(side)
    req, gen = _req(300)
    _t, rtok, _c, censored, src = r._label_fields(req, generated=gen, budget=1300)
    assert rtok == 300 and censored is False and src == "none"


def test_forced_observation_outranks_the_arithmetic(side):
    """A row nowhere near its grant is still censored if the holder forced it —
    that is what makes the label independent of the constant."""
    r = _router(side)
    req, gen = _req(300)
    _t, _rtok, _c, censored, src = r._label_fields(
        req, generated=gen, budget=1300, forced=True)
    assert censored is True and src == "forced"


def test_forced_is_reported_even_when_the_arithmetic_agrees(side):
    r = _router(side)
    req, gen = _req(1295)
    _t, _rtok, _c, censored, src = r._label_fields(
        req, generated=gen, budget=1300, forced=True)
    assert censored is True and src == "forced", (
        "censor_src must show whether the constant is still doing any work")


def test_no_budget_means_nothing_could_have_truncated_it(side):
    r = _router(side)
    req, gen = _req(1295)
    _t, _rtok, _c, censored, src = r._label_fields(req, generated=gen, budget=None)
    assert censored is False and src == "none"


def test_thinking_off_row_is_never_censored(side):
    r = _router(side)
    sp = _Params()
    req = types.SimpleNamespace(
        prompt_token_ids=[7] * 40 + [THINK_START[0], THINK_END[0]],
        output_token_ids=[5] * 1295, sampling_params=sp)
    thinking, _rtok, _c, censored, src = r._label_fields(
        req, generated=1295, budget=1300, forced=True)
    assert thinking is False and censored is False and src == "none"


def test_censor_slack_env_override(side, monkeypatch):
    monkeypatch.setenv("PN119_CENSOR_SLACK", "40")
    assert side._int_env("PN119_CENSOR_SLACK", 13) == 40
    monkeypatch.setenv("PN119_CENSOR_SLACK", "not-an-int")
    assert side._int_env("PN119_CENSOR_SLACK", 13) == 13, "must fail soft"


# ══ C2. THE REFIT SIDE ════════════════════════════════════════════════════
def _row(**kw):
    base = {"rtok": 1287, "generated": 1330, "cap_hit": False, "censored": None,
            "censor_forced": None, "budget_grant": 1300, "censor_schema": 1,
            "slack": 9}
    base.update(kw)
    return base


def test_refit_slack_covers_both_modes(refit):
    assert refit.CENSOR_SLACK >= 13
    for gap in (5, 13):
        cens, lb, prov = refit.censoring_of(_row(rtok=1300 - gap, censored=None))
        assert cens is True, f"gap={gap} prov={prov}"
        assert lb == 1300 - refit.CENSOR_SLACK


def test_refit_forced_flag_wins(refit):
    cens, _lb, prov = refit.censoring_of(_row(rtok=300, censor_forced=True))
    assert cens is True and prov == "forced"


def test_refit_distrusts_a_schema1_negative(refit):
    """The heart of the corpus fix: a schema-1 window's `censored: false` was
    produced by the 9-token test and must be RE-DERIVED, not believed."""
    cens, lb, prov = refit.censoring_of(
        _row(rtok=1287, censored=False, censor_schema=1, slack=9))
    assert cens is True and prov == "budget", (
        "a gap-13 row flagged uncensored by the old detector must not be "
        "imported as a natural stop")
    assert lb == 1300 - refit.CENSOR_SLACK


def test_refit_trusts_a_schema2_negative(refit):
    cens, _lb, prov = refit.censoring_of(
        _row(rtok=800, censored=False,
             censor_schema=refit.TRUSTED_CENSOR_SCHEMA, slack=13))
    assert cens is False and prov == "sink"


def test_refit_schema2_negative_from_a_narrower_window_is_rederived(refit):
    """Schema alone is not enough — the window's own slack must also be at
    least as wide as ours, or a future narrowing re-opens the hole."""
    cens, _lb, prov = refit.censoring_of(
        _row(rtok=1287, censored=False,
             censor_schema=refit.TRUSTED_CENSOR_SCHEMA, slack=5))
    assert cens is True and prov == "budget"


def test_refit_explicit_positive_still_wins(refit):
    cens, _lb, prov = refit.censoring_of(
        _row(rtok=100, censored=True, censor_schema=2, slack=13))
    assert cens is True and prov == "sink"


def test_label_rows_never_narrows_to_the_window_slack(refit):
    """`label_rows` used to let the window's header slack win outright, which
    re-imported the 9-token blind spot per window."""
    rows = [_row(rtok=1287, censored=None, slack=9)]
    y, w, buckets, provs = refit.label_rows(rows, deep_thresh=4000)
    assert buckets[0] == "interval_unresolved", f"got {buckets[0]} / {provs[0]}"
    assert w[0] == 0.0


def test_interval_label_right_censors_below_theta(refit):
    """The absorbing state: a truncated lean row must contribute NOTHING, not
    a y=0 'lean was right'."""
    y, w, bucket, _p = refit.interval_label(
        _row(rtok=1287, censored=False, censor_schema=1, slack=9),
        deep_thresh=4000)
    assert y is None and w == 0.0 and bucket == "interval_unresolved"


def test_interval_label_resolves_a_censored_row_above_theta(refit):
    y, w, bucket, _p = refit.interval_label(
        _row(rtok=1287, censored=True, budget_grant=1300), deep_thresh=800)
    assert y == 1.0 and w == 1.0 and bucket == "censored_pos"


def test_uncensored_row_still_labels_normally(refit):
    y, w, bucket, _p = refit.interval_label(
        _row(rtok=200, generated=400, censored=False, censor_schema=2,
             slack=13, budget_grant=4096), deep_thresh=800)
    assert y == 0.0 and w == 1.0 and bucket == "resolved_neg"


# ══ D. REPLAY OVER THE LIVE SINK ══════════════════════════════════════════
def _live_rows():
    sink = os.environ.get("PN119_SINK_DIR")
    if not sink or not os.path.isdir(sink):
        return None
    out = []
    for f in sorted(glob.glob(os.path.join(sink, "meta-*.jsonl"))):
        for ln in open(f, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if (d.get("finish") and d.get("thinking")
                    and d.get("budget_grant") and d.get("rtok") is not None):
                d.setdefault("slack", 9)
                d.setdefault("censor_schema", 1)
                out.append(d)
    return out or None


def test_live_sink_gap13_population_stops_resolving_negative(refit):
    rows = _live_rows()
    if rows is None:
        pytest.skip("PN119_SINK_DIR not mounted")
    gaps = collections.Counter(
        int(d["budget_grant"]) - int(d["rtok"]) for d in rows)
    assert gaps[5] and gaps[13], (
        f"the two truncation modes this fix is about are not in this sink: "
        f"{sorted(gaps.items())[:6]}")
    # Nothing between the modes to be swallowed by widening 9 -> 13.
    between = sum(v for k, v in gaps.items() if 6 <= k <= 12)
    assert between <= 2, f"{between} rows sit between the modes: {gaps}"

    gap13 = [d for d in rows if int(d["budget_grant"]) - int(d["rtok"]) == 13]
    # As the sink recorded them: the old detector called every one uncensored.
    assert all(d.get("censored") is False for d in gap13)
    y, w, buckets, provs = refit.label_rows(gap13, deep_thresh=4000)
    assert not any(b == "resolved_neg" for b in buckets), (
        f"{sum(1 for b in buckets if b == 'resolved_neg')} of {len(gap13)} "
        f"truncated rows still train as y=0")
    assert all(p in ("forced", "budget") for p in provs), collections.Counter(provs)


def test_live_sink_natural_stops_are_still_usable(refit):
    """The widening must not eat the corpus: rows far from their grant keep
    their exact labels."""
    rows = _live_rows()
    if rows is None:
        pytest.skip("PN119_SINK_DIR not mounted")
    far = [d for d in rows if int(d["budget_grant"]) - int(d["rtok"]) > 30]
    assert len(far) > 100, len(far)
    _y, w, buckets, provs = refit.label_rows(far, deep_thresh=4000)
    assert all(p in ("uncensored", "budget", "cap_hit") for p in set(provs)), \
        collections.Counter(provs)
    resolved = sum(1 for b in buckets if b.startswith("resolved"))
    assert resolved / len(far) > 0.75, (
        f"only {resolved}/{len(far)} natural stops still resolve — the slack "
        f"widening is eating uncensored rows")
