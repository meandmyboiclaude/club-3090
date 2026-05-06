"""v0.8.0 Loop `[F]` — STEP F5: the §6.3 canonical-tuple-hash dedup.

CONTRACT-4 (the locked v0.8.0 Loop design brief, "CONTRACT 4";
source-of-truth design §6.3 verbatim; see docs/LOOP.md for the
shipped contributor reference). F1 (`loop_input.py`) produced the validated
`FInput`; F2/F3 (`classifier.py`) produced the §6.1 `ClassificationResult`.
This module CONSUMES both as a library and runs the §6.3 dedup-or-file
submit path: canonical 7-tuple -> sha256[:12] hash label -> query the
issue tracker -> +1 an existing issue (after a body-tuple collision-safe
verify) OR open a new one — degrading to a local spool when `gh` is
missing/unauthenticated/network-fails (CONTRACT-1: `[F]` is NEVER on the
gate path; F5 NEVER raises and NEVER blocks `[E]`).

This module is purely additive — it touches NO shipped code (two NEW files
only: `dedup.py` + `test-dedup.sh`). It imports F1 + F2/F3 as a library;
it modifies none of them.

------------------------------------------------------------------------
CONTRACT-4 — what this module binds (folds Codex-r1 H3 + H4)
------------------------------------------------------------------------

§6.3 dedup key (design verbatim, line 135):
    (model_id, quant_label, arch_family, kv_calc_version,
     engine_version, failure_class, topology_class)

* **The EFFECTIVE key (closes Codex-r1 H3 — all 7 dims hashed):** F1's
  `FInput.dedup_tuple()` builds this 7-tuple normalized, but it was built
  BEFORE classification so its `failure_class` slot is the manifest's
  `None`. F5 substitutes the **classifier's** `failure_class` into that
  one position and hashes with F1's EXACT convention:
  `sha256("\\x1f".join(str(p) for p in tuple))[:12]`
  (`loop_input.py:227-228` — `dedup_hash`; same `\\x1f`+sha256 join `[E]`
  uses for `submission_fingerprint`, truncated to 12 hex). NO dimension is
  droppable from the match — `failure_class` being IN the hashed tuple is
  the §6.1 mislabel safeguard (a misclassification yields a different hash
  so it can NEVER silently merge with a real OOM).

* **Bounded, collision-safe label scheme (closes Codex-r1 H4):**
  - the dedup primitive: `loop:dedup-<hash>` (12 hex chars — bounded).
  - coarse human-filter labels ONLY where the value space is intrinsically
    bounded: `class:<failure_class>` (the 6-value §6.1 enum) and
    `arch:<arch_family>` (a short identifier, slug-sanitized + length-
    capped though it is intrinsically short).
  - **NO raw `model:`/`engine:`/`kvcalc:`/`topo:` labels** — those values
    are unbounded/awkward (arbitrary HF slugs, moving image pins); they
    live in the issue BODY, NEVER as labels.

* **Issue body** carries the FULL canonical 7-tuple (real, un-hashed
  values + the model slug + a stable schema marker) in a fenced ```json
  block — human + machine readable; unbounded values are safe in the body.

* **Submit/dedup op:** compute the effective hash -> query
  `gh issue list --label loop:dedup-<hash> --state all --json
  number,body` -> if a candidate exists, **parse its body JSON tuple and
  verify it equals the full effective 7-tuple BEFORE +1** (the sha12
  collision safeguard — NEVER +1 on a hash-label match alone; a body-tuple
  MISMATCH is treated as no-match => open a new issue). Match => +1 a
  structured machine-parseable dedup comment. No match => open a new issue
  titled deterministically with labels `loop:dedup-<hash>` +
  `class:<failure_class>` + `arch:<arch_family>` and the body 7-tuple.

* **Filing policy (§6.1 acceptance verbatim):** NEVER auto-file when
  `classification.should_file` is False — `benign-cold-start` is
  SUPPRESSED (not filed, not spooled-as-issue); `unknown` goes to the
  maintainer review-queue spool dir (`.pull-captures/_review-queue/`),
  NOT the issue tracker. F5 NEVER files a "kv-calc bug" — that is
  F3-Tier-1's `route_as_kv_calc_bug` signal (a calibration bug, not a
  failure issue); F5 dedups failure ISSUES only (boundary stated here so
  a future reader does not wire calibration-bug filing into F5).

* **Resilience / never block `[E]` (CONTRACT-1):** every `gh` invocation
  goes through an injectable `gh_runner` callable (default = real
  subprocess). If `gh` is missing/unauthenticated/network-fails the op
  degrades to a local spool `.pull-captures/_dedup-queue/<hash>.json`
  (the would-be issue payload the maintainer can replay). F5 NEVER raises
  and NEVER blocks. The test injects a mock `gh_runner` so it has ZERO
  network/`gh` dependency.

PURE-PYTHON, stdlib only (json / re / pathlib / subprocess / dataclasses /
enum). House style mirrors `loop_input.py` / `classifier.py` /
`trust_pipeline.py` (`class …(str, Enum)`, frozen dataclass result,
structured return — never raise for an expected outcome; the redaction
discipline: bodies are ALREADY `[E]`-redacted — F5 does NOT re-redact).
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

from scripts.lib.profiles.classifier import ClassificationResult, FailureClass
# v0.8.2 CONTRACT-1.1 — F5 parameter annotations are retyped `FInput` ->
# `BaseCaptureBundle` (a schema==1 `FInput` AND a schema==2 `FInputGate` both
# satisfy the consumer surface by structural subtyping; pure static retype,
# schema==1 byte-identical incl. dedup-hash).
#
# RED-LINE (V1, FENCED): `effective_dedup_hash` below
# calls the *concrete-class UNBOUND* `FInput.dedup_hash(_EffProxy())` to
# reuse F1's exact serialization primitive. The protocol retype of the
# *parameter annotations* does NOT break this (the concrete `FInput` class
# still carries `dedup_hash`); the `FInput` import is RETAINED for exactly
# that one unbound call. It MUST NOT be "tidied" to
# `BaseCaptureBundle.dedup_hash(...)` (a Protocol has no usable unbound
# body) NOR to `finput.dedup_hash()` (would hash the unsubstituted tuple
# with failure_class=None — silently corrupting every dedup hash AND
# defeating the §6.1 mislabel safeguard).
from scripts.lib.profiles.loop_input import BaseCaptureBundle, FInput

# The §6.1 enum value-set (bounded — `class:<value>` labels are safe).
_CLASS_ENUM_VALUES = tuple(c.value for c in FailureClass)

# Bounded label namespace.
_DEDUP_LABEL_PREFIX = "loop:dedup-"
_CLASS_LABEL_PREFIX = "class:"
_ARCH_LABEL_PREFIX = "arch:"

# `arch_family` is intrinsically short (`config.json["architectures"][0]`,
# e.g. `Qwen3MoeForCausalLM`), but slug-sanitize + length-cap defensively
# so an `arch:` label can never be unbounded/awkward (closes Codex-r1 H4
# for the one human-filter label whose value is not a fixed enum).
_ARCH_LABEL_VALUE_CAP = 48

# Stable schema marker stamped into the issue-body JSON block so a future
# parser can recognise + version the canonical-tuple payload.
_BODY_SCHEMA = "club3090-loop-dedup-v1"

# The fenced json block fence + a stable marker line so the +1 / re-open
# parser can recover the canonical tuple from a heterogeneous issue body.
_BODY_FENCE_OPEN = "```json"
_BODY_FENCE_CLOSE = "```"
_BODY_TUPLE_HEADING = "<!-- club3090-loop-dedup canonical-tuple -->"

# A stable, machine-parseable "+1" marker on the dedup comment so the
# corpus can be counted without re-parsing free-form text.
_PLUSONE_MARKER = "<!-- club3090-loop-dedup +1 -->"

# Spool dirs (under the repo's .pull-captures/, the same root `[E]` writes
# — F5 NEVER touches the issue tracker for these).
_DEDUP_QUEUE_DIRNAME = "_dedup-queue"
_REVIEW_QUEUE_DIRNAME = "_review-queue"


# ---------------------------------------------------------------------------
# gh_runner — the injectable subprocess seam (CONTRACT-1 resilience).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GhResult:
    """The outcome of ONE `gh` invocation.

    `ok`       — True iff `gh` ran AND returned exit 0.
    `returncode` — the process exit code (or a synthetic non-zero when
                   `gh` is missing / raised before exec).
    `stdout`   — captured stdout (decoded; "" on failure).
    `stderr`   — captured stderr (decoded; the failure reason).
    """

    ok: bool
    returncode: int
    stdout: str
    stderr: str


# A `gh_runner` takes the argv AFTER the `gh` binary (e.g.
# `["issue", "list", "--label", "loop:dedup-...", ...]`) and returns a
# `GhResult`. The default = a real, hardened subprocess; the test injects
# a mock so it has ZERO network / `gh` dependency.
GhRunner = Callable[[Sequence[str]], GhResult]


def _real_gh_runner(argv: Sequence[str]) -> GhResult:
    """Default `gh_runner`: a hardened real subprocess.

    NEVER raises — a missing `gh` binary (`FileNotFoundError`), a non-zero
    exit, a timeout, or any OS error all degrade to a non-`ok` `GhResult`
    (the caller then spools locally). CONTRACT-1: F5 must never raise /
    block the gate path.
    """
    try:
        proc = subprocess.run(
            ["gh", *argv],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return GhResult(
            ok=proc.returncode == 0,
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )
    except FileNotFoundError as exc:  # gh not installed at all.
        return GhResult(False, 127, "", f"gh not found: {exc}")
    except subprocess.TimeoutExpired as exc:  # network hang.
        return GhResult(False, 124, "", f"gh timed out: {exc}")
    except Exception as exc:  # pragma: no cover - defensive last resort.
        return GhResult(False, 1, "", f"gh invocation failed: {exc!r}")


# ---------------------------------------------------------------------------
# The EFFECTIVE §6.3 dedup key (CONTRACT-4 — folds the classifier's class
# into F1's pre-classification tuple, hashed by F1's EXACT convention).
# ---------------------------------------------------------------------------
def _failure_class_value(classification: ClassificationResult) -> str:
    """The classifier's §6.1 `failure_class` as its stable string value
    (the enum's `.value`, e.g. `"genuine-oom"`). `FailureClass` is a
    `(str, Enum)` so this is exactly the string F1 would have hashed had
    `[E]` ever classified — keeping the hash convention byte-identical.
    """
    fc = classification.failure_class
    # FailureClass is (str, Enum); .value is the canonical string. Be
    # defensive for any caller that passes the raw string.
    return fc.value if isinstance(fc, FailureClass) else str(fc)


def effective_dedup_tuple(
    finput: BaseCaptureBundle,
    classification: ClassificationResult,
) -> tuple:
    """The §6.3 7-tuple with the CLASSIFIER's `failure_class` substituted
    into the failure_class slot (CONTRACT-4).

    F1's `FInput.dedup_tuple()` (`loop_input.py:193-216`) builds:
        (model_id, quant_label, arch_family, kv_calc_version,
         engine_version, failure_class, topology_class)
    normalized (quant lowercased; arch verbatim; engine≡engine_pin;
    model_id≡model) — but its `failure_class` element is the manifest's
    authoritative `None` (`[E]` NEVER classifies). F5 reuses that exact
    normalized tuple and replaces ONLY position 5 (the 6th element,
    0-indexed `[5]`) with the classifier's class value. Every OTHER
    element is taken verbatim from F1 so the normalization stays SINGLE-
    SOURCED in F1 (F5 never re-normalizes — Codex-r1 H3: no dimension is
    droppable, and none is re-derived divergently here).
    """
    base = list(finput.dedup_tuple())  # F1's normalized 7-tuple (fc=None).
    # §6.3 order: index 5 is `failure_class` (model_id, quant_label,
    # arch_family, kv_calc_version, engine_version, FAILURE_CLASS,
    # topology_class). Substitute the classifier's class.
    base[5] = _failure_class_value(classification)
    return tuple(base)


def effective_dedup_hash(
    finput: BaseCaptureBundle,
    classification: ClassificationResult,
) -> str:
    """`sha256("\\x1f".join(str(p) for p in effective_tuple))[:12]`.

    Byte-exactly F1's `dedup_hash` convention (`loop_input.py:227-228`):
    `\\x1f`-join + sha256 hexdigest truncated to 12 hex — the SAME
    convention `[E]` uses for `submission_fingerprint` (`capture.py`
    `\\x1f`+sha256). F5 does NOT re-implement the primitive: it builds the
    effective tuple (classifier's `failure_class` substituted) and feeds
    it through F1's own hashing. We construct a throwaway `FInput`-shaped
    shim only to reuse F1's `dedup_hash()` method verbatim — guaranteeing
    the hash matches F1's serialization to the byte (the brief's "reuse
    F1's hashing primitive if F1 exposes one" — F1 exposes `dedup_hash`).
    """
    eff = effective_dedup_tuple(finput, classification)

    # Reuse F1's EXACT hashing primitive: F1's `FInput.dedup_hash()` does
    # `sha256("\x1f".join(str(p) for p in self.dedup_tuple()))[:12]`
    # (`loop_input.py:218-228`). We rebind `dedup_tuple` on a lightweight
    # proxy so F1's own method does the join+sha256+[:12] — zero
    # convention drift (single-sourced in loop_input.py).
    class _EffProxy:
        def dedup_tuple(self_proxy) -> tuple:  # noqa: N805
            return eff

    # F1's dedup_hash is a plain method over self.dedup_tuple(); calling it
    # unbound with the proxy yields F1's byte-exact serialization.
    return FInput.dedup_hash(_EffProxy())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Bounded label scheme (CONTRACT-4 — closes Codex-r1 H4).
# ---------------------------------------------------------------------------
def _slug_sanitize(value: str, cap: int) -> str:
    """Lowercase, non-alphanumeric -> `-`, collapse repeats, strip edge
    `-`, length-cap. Used ONLY for the `arch:` label value (intrinsically
    short; this is defensive bounding, not a load-bearing normalization —
    the canonical arch lives un-sanitized in the body tuple).
    """
    s = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    return s[:cap].strip("-")


def dedup_label(dedup_hash: str) -> str:
    """The bounded dedup primitive label: `loop:dedup-<12hex>`."""
    return f"{_DEDUP_LABEL_PREFIX}{dedup_hash}"


def label_set(
    dedup_hash: str,
    classification: ClassificationResult,
    finput: BaseCaptureBundle,
) -> list[str]:
    """The EXACT bounded label set for an issue (CONTRACT-4 — closes H4).

    Exactly three labels, all bounded:
      * `loop:dedup-<12hex>`  — the dedup primitive (12 hex, bounded).
      * `class:<failure_class>` — one of the 6 §6.1 enum values (bounded).
      * `arch:<sanitized>`    — a short identifier, slug-sanitized +
                                length-capped.
    NO raw `model:`/`engine:`/`kvcalc:`/`topo:` label is EVER produced —
    those unbounded values live ONLY in the body 7-tuple.
    """
    return [
        dedup_label(dedup_hash),
        f"{_CLASS_LABEL_PREFIX}{_failure_class_value(classification)}",
        f"{_ARCH_LABEL_PREFIX}"
        f"{_slug_sanitize(finput.arch_family, _ARCH_LABEL_VALUE_CAP)}",
    ]


# ---------------------------------------------------------------------------
# Issue body — the FULL canonical 7-tuple as a fenced json block.
# ---------------------------------------------------------------------------
# §6.3 verbatim field order — the body carries the REAL (un-hashed) values.
_TUPLE_FIELDS = (
    "model_id",
    "quant_label",
    "arch_family",
    "kv_calc_version",
    "engine_version",
    "failure_class",
    "topology_class",
)


def _tuple_payload(
    finput: BaseCaptureBundle,
    classification: ClassificationResult,
) -> dict:
    """The machine-readable body payload: the schema marker + the FULL
    effective 7-tuple as a named dict (real un-hashed values; unbounded
    values are safe in the body) + the model slug + the dedup hash.
    """
    eff = effective_dedup_tuple(finput, classification)
    return {
        "schema": _BODY_SCHEMA,
        "dedup_hash": effective_dedup_hash(finput, classification),
        "model_slug": finput.manifest.get("model"),
        "tuple": {name: eff[i] for i, name in enumerate(_TUPLE_FIELDS)},
        # The ORDERED tuple too, so a reader can compare positionally
        # without trusting dict ordering.
        "tuple_ordered": list(eff),
    }


def build_issue_body(
    finput: BaseCaptureBundle,
    classification: ClassificationResult,
) -> str:
    """The new-issue body: a human header + the canonical 7-tuple in a
    fenced ```json block (CONTRACT-4). The text is ALREADY `[E]`-redacted
    upstream (the manifest fields F1 surfaces went through
    `capture._redact_text`); F5 does NOT re-redact (CONTRACT-3 stage-1 /
    the brief's redaction discipline — do not double-scrub).
    """
    eff = effective_dedup_tuple(finput, classification)
    h = effective_dedup_hash(finput, classification)
    payload = json.dumps(_tuple_payload(finput, classification), indent=2,
                         sort_keys=True)
    es = classification.error_substring or ""
    return (
        f"## club-3090 Loop `[F]` — deduped failure issue\n\n"
        f"Auto-filed by the v0.8.0 Loop submit path (§6.3 dedup). "
        f"Dedup hash `{h}` (the bounded `loop:dedup-{h}` label is the "
        f"collision-safe match primitive; the canonical 7-tuple below is "
        f"the body-tuple collision safeguard — a `+1` is added ONLY when "
        f"a candidate's body tuple equals this one).\n\n"
        f"**§6.1 class:** `{_failure_class_value(classification)}` "
        f"(tier `{classification.tier.value}`)\n\n"
        f"**Effective §6.3 dedup tuple** "
        f"(model_id, quant_label, arch_family, kv_calc_version, "
        f"engine_version, failure_class, topology_class):\n\n"
        f"{_BODY_TUPLE_HEADING}\n"
        f"{_BODY_FENCE_OPEN}\n{payload}\n{_BODY_FENCE_CLOSE}\n\n"
        f"**Redacted error substring** (already `[E]`-scrubbed; not "
        f"re-redacted):\n\n"
        f"```\n{es}\n```\n"
    )


def build_plusone_comment(
    finput: BaseCaptureBundle,
    classification: ClassificationResult,
) -> str:
    """The structured, machine-parseable `+1` comment added on a verified
    body-tuple match (CONTRACT-4). Carries the stable `_PLUSONE_MARKER` +
    the run's `utc_ts` / `club3090_commit` so the dedup corpus can be
    counted without parsing free-form prose.
    """
    m = finput.manifest
    payload = json.dumps(
        {
            "schema": _BODY_SCHEMA,
            "kind": "dedup-plusone",
            "dedup_hash": effective_dedup_hash(finput, classification),
            "utc_ts": m.get("utc_ts"),
            "club3090_commit": m.get("club3090_commit"),
            "failure_class": _failure_class_value(classification),
        },
        sort_keys=True,
    )
    return (
        f"{_PLUSONE_MARKER}\n"
        f"+1 — another occurrence observed by the v0.8.0 Loop "
        f"submit path (same effective §6.3 dedup tuple; body-tuple "
        f"verified before this +1).\n\n"
        f"{_BODY_FENCE_OPEN}\n{payload}\n{_BODY_FENCE_CLOSE}\n"
    )


def build_issue_title(
    finput: BaseCaptureBundle,
    classification: ClassificationResult,
) -> str:
    """A DETERMINISTIC issue title (CONTRACT-4 — "open a new issue titled
    deterministically"). Stable for a given (class, model slug, hash) so
    the same failure always titles identically.
    """
    h = effective_dedup_hash(finput, classification)
    cls = _failure_class_value(classification)
    slug = finput.manifest.get("model") or "<unknown-model>"
    return f"[loop][{cls}] {slug} (dedup {h})"


# ---------------------------------------------------------------------------
# Body-tuple parse + collision-safe verify (CONTRACT-4 — the (c) case).
# ---------------------------------------------------------------------------
def parse_body_tuple(body: str) -> Optional[list]:
    """Recover the canonical ORDERED 7-tuple from a candidate issue body.

    Locates the fenced ```json block carrying our `_BODY_SCHEMA` marker
    and returns its `tuple_ordered` as a list (positional, not dict-order-
    dependent). Returns None when the body has no recognisable / parseable
    canonical block (=> treated by the caller as NO match => open new; the
    honest-degrade default — never a confidently-wrong +1).
    """
    if not body:
        return None
    # Scan every fenced json block; accept the FIRST that parses to our
    # schema. (A heterogeneous issue body may carry other code fences.)
    idx = 0
    while True:
        start = body.find(_BODY_FENCE_OPEN, idx)
        if start == -1:
            return None
        body_start = start + len(_BODY_FENCE_OPEN)
        end = body.find(_BODY_FENCE_CLOSE, body_start)
        if end == -1:
            return None
        chunk = body[body_start:end].strip()
        idx = end + len(_BODY_FENCE_CLOSE)
        try:
            obj = json.loads(chunk)
        except (ValueError, TypeError):
            continue
        if (
            isinstance(obj, dict)
            and obj.get("schema") == _BODY_SCHEMA
            and isinstance(obj.get("tuple_ordered"), list)
        ):
            return list(obj["tuple_ordered"])
    # unreachable


def body_tuple_matches(
    body: str,
    finput: BaseCaptureBundle,
    classification: ClassificationResult,
) -> bool:
    """The sha12 COLLISION SAFEGUARD (CONTRACT-4 (c)): a candidate found
    via the `loop:dedup-<hash>` label is a real duplicate ONLY when its
    parsed body tuple EQUALS the full effective 7-tuple. A 12-hex sha
    truncation can (astronomically rarely, or via a crafted body) collide;
    NEVER +1 on a hash-label match alone. A parse failure or ANY element
    mismatch => not a match => the caller opens a NEW issue (honest
    degrade — never silently merge two distinct failures).
    """
    parsed = parse_body_tuple(body)
    if parsed is None:
        return False
    eff = list(effective_dedup_tuple(finput, classification))
    if len(parsed) != len(eff):
        return False
    # Element-wise; coerce to str so a json round-trip (which may have
    # turned an int ctx etc. — though this 7-tuple is all strings/None —
    # into a number) compares stably, same `str()` discipline F1 uses
    # when it hashes (`loop_input.py:227`).
    return all(str(a) == str(b) for a, b in zip(parsed, eff))


# ---------------------------------------------------------------------------
# Result + the submit/dedup operation.
# ---------------------------------------------------------------------------
class DedupAction(str, Enum):
    """What the submit path DID (returned, never raised). House style:
    `class …(str, Enum)` (mirrors `classifier.FailureClass` /
    `trust_pipeline.TrustStage`).
    """

    OPENED = "opened"            # new issue opened.
    PLUSONE = "plusone"          # +1'd an existing, body-verified issue.
    SUPPRESSED = "suppressed"    # benign-cold-start: NOT filed (§6.1).
    REVIEW_QUEUED = "review-queued"  # unknown -> review-queue spool.
    SPOOLED = "spooled"          # gh unavailable -> _dedup-queue spool.


@dataclass(frozen=True)
class DedupResult:
    """The §6.3 submit-path verdict.

    `action`        — what F5 did (`DedupAction`).
    `dedup_hash`    — the effective 12-hex hash (None for suppressed/
                      review-queued where no failure-issue hash applies —
                      still computed + surfaced for traceability).
    `labels`        — the EXACT bounded label set (empty when not filed).
    `issue_number`  — the matched/created issue number when known
                      (mock/real gh dependent), else None.
    `spool_path`    — the on-disk spool file written, when degraded.
    `reason`        — a stable machine token explaining the action.
    `notes`         — human-readable trace of the decision path.
    """

    action: DedupAction
    dedup_hash: Optional[str]
    labels: tuple = ()
    issue_number: Optional[int] = None
    spool_path: Optional[Path] = None
    reason: str = ""
    notes: tuple = field(default_factory=tuple)


def _spool(spool_dir: Path, name: str, payload: dict) -> Optional[Path]:
    """Write a spool JSON the maintainer can replay. NEVER raises — a
    spool-write failure is itself non-fatal (CONTRACT-1: F5 must never
    block / raise). Returns the path on success, None on failure.
    """
    try:
        spool_dir.mkdir(parents=True, exist_ok=True)
        path = spool_dir / name
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path
    except Exception:  # pragma: no cover - defensive (disk full etc.).
        return None


def _would_be_payload(
    finput: BaseCaptureBundle,
    classification: ClassificationResult,
    dedup_hash: str,
    labels: list[str],
) -> dict:
    """The would-be issue payload spooled when `gh` is unavailable — the
    maintainer can replay it 1:1 into the tracker.
    """
    return {
        "schema": _BODY_SCHEMA,
        "kind": "would-be-issue",
        "dedup_hash": dedup_hash,
        "title": build_issue_title(finput, classification),
        "labels": labels,
        "body": build_issue_body(finput, classification),
        "canonical": _tuple_payload(finput, classification),
    }


def bootstrap_labels(
    *,
    repo: Optional[str] = None,
    gh_runner: GhRunner = _real_gh_runner,
) -> dict:
    """Idempotently create the bounded `class:*` label set (CONTRACT-4).

    `gh label create --force` for the 6 §6.1 `class:<value>` labels (force
    => idempotent: create-or-update, never errors on a pre-existing
    label). `loop:dedup-*` and `arch:*` are created PER-ISSUE on demand
    (their value space is per-failure, not a fixed bootstrap set). ALL
    `gh` calls are `gh_runner`-guarded — a missing/failed `gh` is reported
    (per-label `ok`), NEVER raised (CONTRACT-1).

    Returns `{class_value: bool_ok}`. The test injects a mock `gh_runner`
    (zero network / `gh`).
    """
    out: dict = {}
    for value in _CLASS_ENUM_VALUES:
        argv = [
            "label", "create", f"{_CLASS_LABEL_PREFIX}{value}",
            "--force",
            "--description",
            f"club-3090 Loop [F] §6.1 failure class: {value}",
        ]
        if repo:
            argv += ["-R", repo]
        res = gh_runner(argv)
        out[value] = bool(getattr(res, "ok", False))
    return out


def submit(
    finput: BaseCaptureBundle,
    classification: ClassificationResult,
    *,
    repo_root: Path,
    repo: Optional[str] = None,
    gh_runner: GhRunner = _real_gh_runner,
) -> DedupResult:
    """The §6.3 dedup-or-file submit path (CONTRACT-4). NEVER raises,
    NEVER blocks (CONTRACT-1: `[F]` is off the gate path).

    Decision order:
      1. **Filing policy first (§6.1 acceptance verbatim).** F5 only ever
         dedups *failure ISSUES*; it NEVER files a kv-calc bug
         (`classification.route_as_kv_calc_bug` is F3-Tier-1's calibration
         signal — a different pipeline; F5 deliberately does not touch
         it). When `classification.should_file` is False:
           * `benign-cold-start` -> SUPPRESSED (not filed, not spooled-
             as-issue) — `DedupAction.SUPPRESSED`.
           * `unknown` -> the maintainer review-queue spool
             (`.pull-captures/_review-queue/`), NOT the tracker —
             `DedupAction.REVIEW_QUEUED`.
      2. Compute the effective hash + bounded label set.
      3. Query `gh issue list --label loop:dedup-<hash> --state all
         --json number,body`. `gh` unavailable/failed -> spool the
         would-be payload to `.pull-captures/_dedup-queue/<hash>.json`
         (`DedupAction.SPOOLED`) — replayable, never blocking.
      4. A candidate exists -> parse its body tuple; **verify it equals
         the full effective 7-tuple BEFORE +1** (the sha12 collision
         safeguard). Match -> `gh issue comment` a structured `+1`
         (`DedupAction.PLUSONE`). No candidate, OR a body-tuple MISMATCH
         (simulated/real sha12 collision) -> `gh issue create` a new
         issue with the bounded labels + body 7-tuple
         (`DedupAction.OPENED`). A `gh` failure on the comment/create
         step also degrades to the spool (never raises).
    """
    notes: list[str] = []
    pull_captures = Path(repo_root) / ".pull-captures"

    # The effective hash is always computable (pure function of F1's
    # normalized tuple + the classifier class) — surface it even on the
    # not-filed paths for traceability.
    dedup_h = effective_dedup_hash(finput, classification)

    # ---- 1. Filing policy (§6.1 acceptance) — BEFORE any gh I/O --------
    # Boundary (stated per the brief): F5 dedups FAILURE ISSUES only. It
    # NEVER files a kv-calc bug — `route_as_kv_calc_bug` is F3-Tier-1's
    # calibration signal handled by a DIFFERENT pipeline. F5 does not read
    # it for a filing decision (only `should_file`); this comment is the
    # explicit boundary so a future reader does not wire calibration-bug
    # filing into the dedup submit path.
    if not classification.should_file:
        if classification.failure_class is FailureClass.UNKNOWN:
            payload = {
                "schema": _BODY_SCHEMA,
                "kind": "review-queue",
                "dedup_hash": dedup_h,
                "reason": "classifier=unknown -> maintainer review queue "
                          "(NEVER the issue tracker, NEVER a kv-calc bug)",
                "model_slug": finput.manifest.get("model"),
                "utc_ts": finput.manifest.get("utc_ts"),
                "canonical": _tuple_payload(finput, classification),
                "error_substring": classification.error_substring,
            }
            sp = _spool(
                pull_captures / _REVIEW_QUEUE_DIRNAME,
                f"{dedup_h}.json",
                payload,
            )
            notes.append(
                "filing policy: classifier=unknown -> review-queue spool "
                "(.pull-captures/_review-queue/); NOT the issue tracker"
            )
            return DedupResult(
                action=DedupAction.REVIEW_QUEUED,
                dedup_hash=dedup_h,
                spool_path=sp,
                reason="unknown-review-queued",
                notes=tuple(notes),
            )
        # benign-cold-start (or any other should_file=False class): §6.1
        # acceptance verbatim — SUPPRESSED: not filed, not spooled-as-
        # issue.
        notes.append(
            "filing policy: should_file=False "
            f"(class={_failure_class_value(classification)}) -> SUPPRESSED "
            "(§6.1 acceptance: not filed, not spooled-as-issue)"
        )
        return DedupResult(
            action=DedupAction.SUPPRESSED,
            dedup_hash=dedup_h,
            reason="benign-suppressed",
            notes=tuple(notes),
        )

    # ---- 2. effective hash + bounded label set ------------------------
    labels = label_set(dedup_h, classification, finput)
    notes.append(
        f"effective §6.3 dedup hash={dedup_h}; bounded labels={labels} "
        f"(NO raw model/engine/kvcalc/topo label — those live in the body)"
    )

    def _spool_degrade(reason: str) -> DedupResult:
        sp = _spool(
            pull_captures / _DEDUP_QUEUE_DIRNAME,
            f"{dedup_h}.json",
            _would_be_payload(finput, classification, dedup_h, labels),
        )
        notes.append(
            f"gh unavailable/failed ({reason}) -> spooled would-be issue "
            f"to .pull-captures/{_DEDUP_QUEUE_DIRNAME}/{dedup_h}.json "
            f"(replayable; F5 NEVER blocks [E] — CONTRACT-1)"
        )
        return DedupResult(
            action=DedupAction.SPOOLED,
            dedup_hash=dedup_h,
            labels=tuple(labels),
            spool_path=sp,
            reason=f"gh-degraded:{reason}",
            notes=tuple(notes),
        )

    # ---- 3. query the tracker by the bounded dedup-hash label ---------
    list_argv = [
        "issue", "list",
        "--label", dedup_label(dedup_h),
        "--state", "all",
        "--json", "number,body",
    ]
    if repo:
        list_argv += ["-R", repo]
    list_res = gh_runner(list_argv)
    if not getattr(list_res, "ok", False):
        return _spool_degrade(f"issue-list rc={getattr(list_res,'returncode',None)}")

    # Parse the gh JSON. A malformed payload is treated as "no candidate"
    # (honest degrade -> open new; never a confidently-wrong +1).
    candidates: list = []
    try:
        parsed = json.loads(list_res.stdout or "[]")
        if isinstance(parsed, list):
            candidates = parsed
    except (ValueError, TypeError):
        candidates = []

    # ---- 4. body-tuple collision-safe verify --------------------------
    verified_match = None
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        if body_tuple_matches(cand.get("body") or "", finput,
                              classification):
            verified_match = cand
            break

    if verified_match is not None:
        # A real duplicate (label hit AND body 7-tuple verified) -> +1.
        num = verified_match.get("number")
        comment_argv = ["issue", "comment", str(num),
                        "--body", build_plusone_comment(finput,
                                                        classification)]
        if repo:
            comment_argv += ["-R", repo]
        c_res = gh_runner(comment_argv)
        if not getattr(c_res, "ok", False):
            return _spool_degrade(
                f"issue-comment rc={getattr(c_res,'returncode',None)}"
            )
        notes.append(
            f"candidate #{num} found via {dedup_label(dedup_h)} AND its "
            f"body 7-tuple VERIFIED equal -> +1 structured dedup comment "
            f"(collision-safe: NEVER +1 on a hash-label match alone)"
        )
        return DedupResult(
            action=DedupAction.PLUSONE,
            dedup_hash=dedup_h,
            labels=tuple(labels),
            issue_number=(int(num) if isinstance(num, int)
                          or (isinstance(num, str) and str(num).isdigit())
                          else None),
            reason="plusone-body-verified",
            notes=tuple(notes),
        )

    # No candidate, OR every candidate's body tuple MISMATCHED (a
    # simulated/real sha12 collision: same `loop:dedup-` label, different
    # body 7-tuple) -> open a NEW issue (the collision safeguard: a
    # body-tuple mismatch is treated as no-match -> never silently merge
    # two distinct failures).
    if candidates:
        notes.append(
            f"{len(candidates)} candidate(s) carried {dedup_label(dedup_h)} "
            f"but NONE had a matching body 7-tuple (sha12 collision "
            f"safeguard) -> open NEW issue, do NOT +1"
        )
    else:
        notes.append(
            f"no existing issue for {dedup_label(dedup_h)} -> open new"
        )

    create_argv = ["issue", "create",
                   "--title", build_issue_title(finput, classification),
                   "--body", build_issue_body(finput, classification)]
    for lab in labels:
        create_argv += ["--label", lab]
    if repo:
        create_argv += ["-R", repo]
    cr_res = gh_runner(create_argv)
    if not getattr(cr_res, "ok", False):
        return _spool_degrade(
            f"issue-create rc={getattr(cr_res,'returncode',None)}"
        )

    # gh prints the new issue URL on stdout; recover the trailing number
    # best-effort (None when unparseable — non-fatal, never raises).
    new_num: Optional[int] = None
    m = re.search(r"/issues/(\d+)\s*$", (cr_res.stdout or "").strip())
    if m:
        new_num = int(m.group(1))
    notes.append(
        f"opened new issue (labels={labels}; body carries the full "
        f"canonical 7-tuple json block)"
    )
    return DedupResult(
        action=DedupAction.OPENED,
        dedup_hash=dedup_h,
        labels=tuple(labels),
        issue_number=new_num,
        reason="opened-new",
        notes=tuple(notes),
    )
