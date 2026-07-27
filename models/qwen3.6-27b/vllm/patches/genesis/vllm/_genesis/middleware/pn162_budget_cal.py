# SPDX-License-Identifier: Apache-2.0
"""PN162 — closed-loop thinking-budget calibrator, CONSUMER half (2026-07-27).

House-original. Dark by default; identity behaviour unless
`GENESIS_ENABLE_PN162_BUDGET_CAL=1` AND a well-formed ledger exists.

WHAT IT IS
----------
PN100 sizes every request as `round100(steps x GENESIS_PN100_TOK_PER_STEP)`.
That grid is open loop: the classifier's step estimate is never compared
against what the request actually SPENT, so a bucket that is systematically
under- or over-granted stays that way forever. PN162 closes the loop with a
per-steps-bucket multiplier `k` learned from the engine's own recorded budget
outcomes:

    grant' = round100(steps x TOK_PER_STEP x k[bucket(steps)])

The feedback signal is NOT answer correctness and costs no extra LLM call. It
is the BUG-139 schema-2 outcome the router already writes to the PN119 sink for
every scored generation:

    censor_forced / censored   -> the cap was BINDING   -> k up   (x1.15)
    grant - rtok > 40% of grant-> the cap was LOOSE     -> k down (x0.97)
    otherwise                  -> the cap was RIGHT     -> no-op

Causal grounding (banked 2026-07-26): force-closed items gain +9.6pt when given
more; natural stops lose nothing. So bumping a bound bucket is the side of the
trade that pays, and decaying a slack bucket is free.

The learning half lives HOST-side in `fixes/pn162_ledger_update.py` (symlinked
as `~/shared/needfit/pn162-ledger-update.py`). This module only READS.

CONTRACT
--------
* Never blocks, never throws. Any error — missing file, bad JSON, bad types,
  a permissions change, a truncated write — degrades to k=1.0, which is exactly
  today's behaviour. There is no failure mode that changes a grant in a way the
  operator did not ask for.
* The ledger read is mtime-cached: at most one `os.stat` per
  `GENESIS_PN162_RESTAT_S` (default 2s) and a re-parse only when
  (mtime_ns, size) moves. The updater writes tmp+rename, so a reader never
  sees a partial file and every real write moves the mtime.

  (Deliberately NOT the sha256 content-signature the PN119 probe reload uses.
  That one exists because `cp -p` / `rsync -a` / a btrfs rollback can install a
  different probe without moving the mtime. Nothing does that to this ledger —
  one writer, one atomic rename, always a fresh mtime — and it is re-read every
  2s rather than every 60, so hashing it would be pure cost.)
* HARD GUARD: refuses to multiply when `GENESIS_PN100_STEP_BUDGET_MAP` is set.
  That map is an absolute per-bucket budget fitted from banked data; folding a
  learned multiplier into it would compound two calibrations of the same thing.
  PN100's map branch returns before the k path anyway — this guard makes the
  refusal explicit, logged, and testable.

FLAGS
-----
  GENESIS_ENABLE_PN162_BUDGET_CAL  0/1  master flag (default 0)
  GENESIS_ENABLE_PN162_EXACT       0/1  per-prompt override leg (default 0,
                                        USER RULING PENDING — see below)
  GENESIS_PN162_LEDGER             path (default /pn162/pn162-ledger.json)
  GENESIS_PN162_RESTAT_S           float re-stat throttle (default 2.0)
  GENESIS_PN162_K_MIN              float clamp floor   (default 0.7)
  GENESIS_PN162_K_MAX              float clamp ceiling (default 3.0)
  GENESIS_PN162_EXACT_MULT         float bump applied to a remembered bound
                                        grant (default 1.25)

THE "EXACT" LEG IS NOT RULED ON, AND NOTHING POPULATES IT
---------------------------------------------------------
`exact` maps a prompt hash to that prompt's last budget outcome, so a repeat of
a request that was force-closed can be granted more immediately instead of
waiting for its whole bucket to drift. That is the user's "run 10 of the same
100" story taken to its literal limit — and it is exactly a per-item bench
history, which the bench-protocol rule forbids for bench traffic. USER has not
ruled. The flag therefore ships separate and default-OFF.

Independently of the ruling, the leg is INERT today: the PN119 sink carries no
prompt hash (fields verified 2026-07-27 — req_id, row, score, prompt_tok,
budget_grant, budget_source, rtok, censored, censor_forced, censor_src,
cap_hit, generated, route, lane_key, ...), so the host updater cannot key an
entry the consumer would recognise, and it writes `"exact": {}` always. The
read path below is implemented and tested against a hand-written ledger so that
closing the gap is one line in the router; see PN162-BUDGET-CAL.md "EXACT".

To make that one line possible, and only when the EXACT flag is on, PN100
stamps `pn162_phash` into `vllm_xargs` (same seat as `h119_overridable`), which
reaches the worker inside the SamplingParams the router already reads via
`_extra_arg`. Stamping is free and dark; the router does not read it yet.

THE BLIND SPOT k CANNOT SEE — ANNOUNCEMENT BIAS
------------------------------------------------
The bound/slack loop is a closed loop on ENFORCEMENT. On the LEAN lane it is
structurally blind to the other failure mode, because that lane ANNOUNCES the
step estimate into the prompt:

    answer_rescue.py `_contract_v3_sized` (the default banner, v3):
        planner_steps = ctk.pop("pn100_steps")
        ... "about {steps} short reasoning steps ..."
        log: "PN102: contract set (v3 sized, steps=%d budget=%d)"

    v4 / v5 / v6a / v6b / v7 all `ctk.pop("pn100_steps")` with the comment
    "planner estimate deliberately unused" — they announce NO per-item N.
    v8 announces only under GENESIS_PN102_BANNER_V8 (dark).

    Lane assignment (BUG-157 autosplit): route "deep" -> pn102_auto_v5 ->
    no announcement; route "lean" -> the env chain -> v3 -> ANNOUNCED N.

Announced N is a hard behavioural anchor (band-limited: holds at N<=5,
collapses above ~12). So an UNDER-announced N self-fulfils: the model paces to
N, finishes clean, nothing binds, and the row lands with modest slack. k reads
that as "the cap was right" — or worse, decays it. Bucket-k fixes BINDING; it
cannot see under-announcement on lean traffic at all.

Two instruments address it, both dark / observational:
  * the ORACLE (telemetry, host side) — deep-lane rows are unanchored, so
    within a lens-score band their realized steps are ground truth for
    comparable difficulty. `deep_realized - lean_announced` per band is an
    estimate of announcement bias. Written to the ledger; steers nothing.
  * EXPLORATION (GENESIS_ENABLE_PN162_EXPLORE, eps default 0.0, USER RULING
    PENDING) — on eps of lean requests announce N+delta (delta in {1,2}, never
    negative) and tag the row so the updater can compare mechanical quality
    proxies between arms. This is the only COUNTERFACTUAL for "N was too low
    despite nothing binding".

k NEVER multiplies the announced N. `budget_multiplier` feeds the token grant
only; `_stash_steps` announces the planner's own N (or N+delta on an explore
row). Announcing a k-corrected N would be a prompt change with its own
quality risk and is a separate, USER-reviewed step.
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from typing import Any

try:  # vllm's logger prints INFO in-server; plain root logger may not
    from vllm.logger import init_logger

    log = init_logger("vllm.genesis.pn162")
except Exception:  # pragma: no cover
    log = logging.getLogger("genesis.middleware.pn162_budget_cal")

FLAG = "GENESIS_ENABLE_PN162_BUDGET_CAL"
EXACT_FLAG = "GENESIS_ENABLE_PN162_EXACT"
EXPLORE_FLAG = "GENESIS_ENABLE_PN162_EXPLORE"
LEDGER_ENV = "GENESIS_PN162_LEDGER"
DEFAULT_LEDGER = "/pn162/pn162-ledger.json"

#: vllm_xargs key carrying the exploration arm. The router already surfaces
#: extra_args onto every sink score line via `_extra_arg(sp, "caller",
#: "x_caller")` -> the `caller` column (pn119_router.py:2440). We take the
#: SECOND name deliberately: a bench harness stamping `caller` still wins, and
#: PN162's label is then simply absent (the updater counts fewer explore rows)
#: rather than overwriting somebody else's provenance.
ARM_XARG = "x_caller"
ARM_PREFIX = "pn162:"

#: steps >= this fold into one bucket. Above ~16 steps the grant is already
#: near GENESIS_PN100_BUDGET_CEIL, so per-value resolution buys nothing and the
#: per-bucket sample count is what limits the fit.
MAX_BUCKET = 16

K_MIN_DEFAULT = 0.7
K_MAX_DEFAULT = 3.0

_STATS: dict[str, int] = {
    "applied": 0,        # a k != 1.0 actually changed a grant
    "identity": 0,       # enabled, ledger read, but k == 1.0
    "no_ledger": 0,      # enabled, no usable ledger
    "map_refused": 0,    # STEP_BUDGET_MAP set -> refused
    "exact_hits": 0,
    "reloads": 0,
    "read_errors": 0,
    "explore_control": 0,
    "explore_arm": 0,
}
_RNG = random.Random()

# (path, mtime_ns, size) -> parsed ledger. One entry; the path never changes
# within a boot in practice, but a change simply forces a re-read.
_CACHE: dict[str, Any] = {
    "path": None,
    "sig": None,
    "data": None,
    "next_stat": 0.0,
}
_LOCK = threading.Lock()
_ANNOUNCED = {"v": False, "map": False}


# ── env helpers (local copies: this module must not import auto_budget) ─────
def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "")
    if val == "":
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def is_enabled() -> bool:
    return _env_bool(FLAG, False)


def exact_enabled() -> bool:
    return _env_bool(EXACT_FLAG, False)


def explore_enabled() -> bool:
    return _env_bool(EXPLORE_FLAG, False)


def ledger_path() -> str:
    return os.environ.get(LEDGER_ENV, "") or DEFAULT_LEDGER


def get_stats() -> dict[str, int]:
    return dict(_STATS)


def bucket_of(steps: Any) -> int:
    """steps -> steps bucket (int 1..MAX_BUCKET). Never raises."""
    try:
        s = int(steps)
    except (TypeError, ValueError):
        return 1
    return max(1, min(MAX_BUCKET, s))


# ── THE KEY SCHEMA — one function, both sides ───────────────────────────────
# The ledger is a shape-keyed lookup table and is meant to GROW: steps today,
# steps x prompt-length band (and later a task-class band) tomorrow. Growing it
# must be a config change, not a rewrite, so key derivation is isolated here
# and mirrored EXACTLY by `pn162_ledger_update.bucket_keys` (the two are
# cross-checked in test_pn162_ledger_update.py).
#
# NEVER prompt-identity. Only shape features — this is a generalising table,
# not a per-item memory (that is the separate, unruled `exact` leg).
#
# Consumer-visible features are limited by WHERE this runs: the PN100 hook is
# in the API-server process, before prefill. steps and prompt LENGTH are
# available; the H119 lens score is NOT (it is a post-prefill, worker-side
# quantity), so a score-band key can only ever be an updater-side analysis
# axis, never a consumer key. Said plainly so nobody wires it later.
#
# The chain is MOST SPECIFIC FIRST and always ends at the marginal steps
# bucket. That is the whole occupancy story: the updater simply does not write
# a composite cell that has fewer than --min-cell observations, and this walk
# then lands on the marginal bucket by construction.
KEY_SCHEMA_STEPS = "steps"
KEY_SCHEMA_STEPS_PTOK = "steps_ptok"
KEY_SCHEMAS = (KEY_SCHEMA_STEPS, KEY_SCHEMA_STEPS_PTOK)

#: prompt-token band edges used when key_schema == "steps_ptok". Overridable
#: per-ledger via `"key_bands": {"ptok": [...]}`.
DEFAULT_PTOK_BANDS = (256, 1024, 4096)

#: chars -> tokens, for the consumer's estimate of prompt_tok. The updater
#: reads the sink's exact `prompt_tok`, so the two agree only approximately —
#: which is exactly why a composite schema needs a screened boot before it is
#: switched on. Tunable: GENESIS_PN162_CHARS_PER_TOK.
DEFAULT_CHARS_PER_TOK = 3.6


def band_index(value: Any, edges) -> int:
    """0..len(edges); -1 when the value is unknown."""
    if value is None:
        return -1
    try:
        v = float(value)
    except (TypeError, ValueError):
        return -1
    for i, e in enumerate(edges):
        if v < e:
            return i
    return len(edges)


def bucket_keys(steps: Any, ptok: Any = None,
                schema: str = KEY_SCHEMA_STEPS, bands=None) -> list:
    """Lookup chain for one request: most specific first, marginal last."""
    b = bucket_of(steps)
    keys = []
    if schema == KEY_SCHEMA_STEPS_PTOK:
        idx = band_index(ptok, bands or DEFAULT_PTOK_BANDS)
        if idx >= 0:
            keys.append(f"{b}|p{idx}")
    keys.append(str(b))
    return keys


def _step_budget_map_set() -> bool:
    return bool(os.environ.get("GENESIS_PN100_STEP_BUDGET_MAP", "").strip())


# ── ledger read (mtime-cached, never raises) ────────────────────────────────
def _read_ledger(force: bool = False) -> dict | None:
    """Parsed ledger dict, or None. Cheap: one stat per RESTAT_S at most."""
    path = ledger_path()
    now = time.monotonic()
    with _LOCK:
        if (not force and _CACHE["path"] == path
                and now < _CACHE["next_stat"]):
            return _CACHE["data"]
        _CACHE["next_stat"] = now + max(
            0.0, _env_float("GENESIS_PN162_RESTAT_S", 2.0))
        try:
            st = os.stat(path)
            sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            _CACHE["path"], _CACHE["sig"], _CACHE["data"] = path, None, None
            return None
        if _CACHE["path"] == path and sig == _CACHE["sig"]:
            return _CACHE["data"]
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            data = _normalise(raw)
        except Exception as exc:  # noqa: BLE001 — a bad ledger is never fatal
            _STATS["read_errors"] += 1
            _CACHE["path"], _CACHE["sig"], _CACHE["data"] = path, sig, None
            log.warning("PN162: ledger %s unreadable (%s) — identity budgets",
                        path, exc)
            return None
        _STATS["reloads"] += 1
        _CACHE["path"], _CACHE["sig"], _CACHE["data"] = path, sig, data
        log.info("PN162: ledger loaded from %s (schema=%s buckets=%d "
                 "exact=%d updated=%s)", path, data.get("schema"),
                 len(data.get("bucket") or {}), len(data.get("exact") or {}),
                 data.get("updated_iso") or data.get("updated_ts"))
        return data


def _normalise(raw: Any) -> dict:
    """Coerce a raw ledger into {bucket: {int: float}, exact: {str: dict}}.

    Anything malformed is DROPPED, not raised on: a ledger with three good
    buckets and one garbage one calibrates the three.
    """
    if not isinstance(raw, dict):
        raise ValueError("ledger root is not an object")
    bucket: dict[str, float] = {}
    for key, val in (raw.get("bucket") or {}).items():
        if not isinstance(key, str):
            continue
        try:
            k = float(val)
        except (TypeError, ValueError):
            continue
        if not (k == k) or k in (float("inf"), float("-inf")):
            continue
        # The marginal part of any key must be a legal steps bucket; composite
        # keys extend it with "|<band>" suffixes. Anything else is dropped.
        head = key.split("|", 1)[0]
        try:
            b = int(head)
        except (TypeError, ValueError):
            continue
        if b < 1 or b > MAX_BUCKET:
            continue
        bucket[key] = k
    exact: dict[str, dict] = {}
    for key, val in (raw.get("exact") or {}).items():
        if isinstance(key, str) and isinstance(val, dict):
            exact[key] = val
    schema = raw.get("key_schema") or KEY_SCHEMA_STEPS
    if schema not in KEY_SCHEMAS:
        log.warning("PN162: unknown key_schema %r — falling back to %r",
                    schema, KEY_SCHEMA_STEPS)
        schema = KEY_SCHEMA_STEPS
    bands = (raw.get("key_bands") or {}).get("ptok")
    try:
        bands = tuple(float(x) for x in bands) if bands else DEFAULT_PTOK_BANDS
    except (TypeError, ValueError):
        bands = DEFAULT_PTOK_BANDS
    return {
        "schema": raw.get("schema"),
        "key_schema": schema,
        "key_bands": bands,
        "updated_ts": raw.get("updated_ts"),
        "updated_iso": raw.get("updated_iso"),
        "bucket": bucket,
        "exact": exact,
    }


# ── the two public legs ─────────────────────────────────────────────────────
def budget_multiplier(steps: Any, ptok: Any = None) -> float:
    """k for this request shape. ALWAYS returns a float; 1.0 == identity.

    `ptok` is an ESTIMATE of the prompt's token count, used only when the
    ledger declares a composite `key_schema`. Under the default "steps" schema
    it is ignored entirely.
    """
    try:
        if not is_enabled():
            return 1.0
        if _step_budget_map_set():
            _STATS["map_refused"] += 1
            if not _ANNOUNCED["map"]:
                _ANNOUNCED["map"] = True
                log.warning(
                    "PN162: GENESIS_PN100_STEP_BUDGET_MAP is set — REFUSING to "
                    "multiply. The map is an absolute fitted budget; stacking a "
                    "learned k on it compounds two calibrations of the same "
                    "quantity. Unset one of them.")
            return 1.0
        data = _read_ledger()
        if not data or not data["bucket"]:
            _STATS["no_ledger"] += 1
            return 1.0
        k = None
        for key in bucket_keys(steps, ptok, data["key_schema"],
                               data["key_bands"]):
            if key in data["bucket"]:
                k = data["bucket"][key]
                break
        if k is None:
            _STATS["identity"] += 1
            return 1.0
        k_min = _env_float("GENESIS_PN162_K_MIN", K_MIN_DEFAULT)
        k_max = _env_float("GENESIS_PN162_K_MAX", K_MAX_DEFAULT)
        k = max(k_min, min(k_max, float(k)))
        if k == 1.0:
            _STATS["identity"] += 1
            return 1.0
        _STATS["applied"] += 1
        if not _ANNOUNCED["v"]:
            _ANNOUNCED["v"] = True
            log.info("PN162: budget calibrator ACTIVE (ledger=%s, %d buckets, "
                     "k range %.3f..%.3f)", ledger_path(),
                     len(data["bucket"]), min(data["bucket"].values()),
                     max(data["bucket"].values()))
        return k
    except Exception:  # noqa: BLE001 — identity is the only safe failure
        _STATS["read_errors"] += 1
        log.debug("PN162: multiplier lookup failed — identity", exc_info=True)
        return 1.0


def exact_floor(prompt_hash: Any, grant: int) -> int:
    """Raise `grant` if this exact prompt was force-closed last time.

    Returns `grant` unchanged unless BOTH flags are on, the hash is in the
    ledger's `exact` map, and its `last_outcome` is "bound". The floor is
    `round100(last_grant * GENESIS_PN162_EXACT_MULT)` — never a reduction, so
    a stale entry can only cost tokens, never truncate an answer.

    Nothing populates `exact` today (the sink carries no prompt hash); this is
    the read half, kept live and tested so wiring it is a one-line router
    change. See the module docstring.
    """
    try:
        if not (is_enabled() and exact_enabled()):
            return grant
        if not isinstance(prompt_hash, str) or not prompt_hash:
            return grant
        data = _read_ledger()
        if not data:
            return grant
        ent = data["exact"].get(prompt_hash)
        if not isinstance(ent, dict):
            return grant
        if str(ent.get("last_outcome", "")) != "bound":
            return grant
        last = int(ent.get("last_grant") or 0)
        if last <= 0:
            return grant
        mult = _env_float("GENESIS_PN162_EXACT_MULT", 1.25)
        floor = int(round(last * mult / 100.0)) * 100
        if floor <= grant:
            return grant
        _STATS["exact_hits"] += 1
        log.info("PN162: exact bump %s -> %d (last_grant=%d bound, x%.2f)",
                 prompt_hash[:12], floor, last, mult)
        return floor
    except Exception:  # noqa: BLE001
        _STATS["read_errors"] += 1
        log.debug("PN162: exact lookup failed — identity", exc_info=True)
        return grant


# ── exploration arm (dark; USER RULING PENDING) ─────────────────────────────
def explore_arm(steps: Any) -> tuple[Any, Any, str | None]:
    """(steps_to_announce, steps_to_size, arm_label). Identity when disarmed.

    On `PN162_EXPLORE_EPS` of calls the announced N is bumped by a delta drawn
    from `PN162_EXPLORE_DELTAS` (default "1,2"); never negative — a downward
    arm would truncate live answers to buy information, which is not a trade
    we make on serving traffic.

    `PN162_EXPLORE_BUDGET_FOLLOWS` (default 1) also sizes the grant from
    N+delta. With it ON the arm is "this item was sized AND announced bigger",
    so a win means "N was too low" — it does NOT isolate the anchor from the
    budget. Set it to 0 for the anchor-only arm (announce N+delta, keep the
    N-sized grant); that arm is cleaner but risks binding, so it is not the
    default.

    Both arms are LABELLED, control included, so the updater compares
    like-for-like inside one window instead of against untagged traffic.
    """
    try:
        if steps is None or not explore_enabled() or not is_enabled():
            return steps, steps, None
        eps = _env_float("PN162_EXPLORE_EPS", 0.0)
        if eps <= 0.0 or _RNG.random() >= min(1.0, eps):
            _STATS["explore_control"] += 1
            return steps, steps, ARM_PREFIX + "c"
        raw = os.environ.get("PN162_EXPLORE_DELTAS", "") or "1,2"
        deltas = [int(x) for x in raw.replace(" ", "").split(",") if x]
        deltas = [d for d in deltas if d > 0] or [1]
        d = _RNG.choice(deltas)
        n_ann = int(steps) + d
        follows = _env_bool("PN162_EXPLORE_BUDGET_FOLLOWS", True)
        _STATS["explore_arm"] += 1
        return n_ann, (n_ann if follows else steps), f"{ARM_PREFIX}e{d}"
    except Exception:  # noqa: BLE001
        return steps, steps, None


def stamp_xargs(request: Any, arm: str | None = None,
                prompt_hash: str | None = None) -> None:
    """Ride the exploration arm / prompt hash out on `vllm_xargs`.

    Same seat and same fail-open contract as PN100's `_stamp_h119`: the stamp
    reaches the worker inside the SamplingParams object the PN119 router reads,
    and a stamp is never worth a failed request.
    """
    if arm is None and prompt_hash is None:
        return
    try:
        xargs = dict(getattr(request, "vllm_xargs", None) or {})
        if arm is not None:
            xargs[ARM_XARG] = str(arm)[:64]
        if prompt_hash is not None:
            xargs["pn162_phash"] = str(prompt_hash)[:64]
        request.vllm_xargs = xargs
    except Exception:  # noqa: BLE001
        log.debug("PN162: could not stamp vllm_xargs", exc_info=True)


def reset_cache() -> None:
    """Drop the mtime cache. Tests only; the server never needs it."""
    with _LOCK:
        _CACHE["path"] = None
        _CACHE["sig"] = None
        _CACHE["data"] = None
        _CACHE["next_stat"] = 0.0
    _ANNOUNCED["v"] = False
    _ANNOUNCED["map"] = False
