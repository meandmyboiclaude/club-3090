"""v0.8.0 Loop `[F]` — STEP F1: the `FInput` capture-bundle reader.

CONTRACT-1 (the locked v0.8.0 Loop design brief, "CONTRACT 1"; see
docs/LOOP.md for the shipped contributor reference). `[E]`
(`capture.py`) is the **producer**; this module is
the **consumer**: it parses ONE on-disk capture directory into a validated
`FInput` object that the later `[F]` STEPs (F2 classifier / F4 trust pipeline
/ F5 dedup) consume. F1 is a STRICT boundary validator — any schema/shape
violation raises a clear typed `CaptureBundleError` (never a silent partial
parse; downstream STEPs trust this object).

This module owns ONLY:
  * `FInput`               — the parsed-bundle dataclass (CONTRACT-1 shape);
  * `read_capture_bundle()`— parse + hard-validate one capture dir;
  * the CONTRACT-1 key-normalization helpers + the §6.2 consensus 9-tuple
    and §6.3 dedup 7-tuple builders (pure functions over the manifest —
    `[F]` owns the canonical construction; `[E]` does NOT normalize).

It is PURE-PYTHON, stdlib only (json / pathlib / dataclasses / hashlib),
matching the no-external-deps style of its `scripts/lib/profiles/` siblings
(`capture.py`, `deriver.py`). It touches NO shipped code — F1 is purely
additive (two NEW files only).

Binding rules enforced in code + comment (CONTRACT-1):
  * F1 MUST NOT treat `manifest["outcome"]` as the §6.1 class enum — it is
    `[E]`'s interim honest 3-state (`failed > partial > ok`,
    `capture.py:524-534`). There is deliberately NO accessor here that
    claims `outcome` is the §6.1 class; `[F]` derives `failure_class` itself
    in a later STEP (F2/F3). The raw value is surfaced (`outcome` property)
    but never re-interpreted as a class.
  * `failure_class` is authoritatively `null` in EVERY `[E]` manifest
    (`capture.py:569`) — F1 just SURFACES that null; it never expects
    `[E]` to have pre-classified.
  * Key normalization is `[F]`'s job: `quant_label` is the raw
    `weight_format` case-as-emitted (`capture.py:389-391`) and is
    LOWERCASED for keying here. `arch_family` is used VERBATIM — it is
    `config.json["architectures"][0]` (`deriver.py:679-680`), already an
    exact identifier, NOT re-normalized.
  * `model_id ≡ manifest["model"]` and `engine_version ≡
    manifest["engine_pin"]` are shipped aliases (`capture.py:553/568`,
    `497-500`) — exposed here as the canonical accessors.
  * `topology_class` (coarse `NxVRAMMiB`) and `topology_summary_canonical`
    (full sorted `(name,vram)` list) are deliberately two resolutions:
    keys use `topology_class`; `submission_fingerprint` uses
    `topology_summary_canonical`. Not interchangeable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

# The schema version `[E]` stamps (`capture.py:42` `SCHEMA = 1`). F1 hard-
# asserts the bundle is this exact schema — a strict boundary refuses an
# unrecognised schema rather than mis-parsing a future shape.
EXPECTED_SCHEMA = 1
# v0.8.2 CONTRACT-1.1 — the gate-only (capture-on-hard-block) bundle schema
# (`capture.py` `GATE_SCHEMA = 2`). The shipped schema==1
# `read_capture_bundle()` path stays byte-identical; a SEPARATE
# `read_gate_bundle()` (schema==2) returns `FInputGate`.
GATE_SCHEMA = 2


# ---------------------------------------------------------------------------
# v0.8.2 CONTRACT-1.1 — `BaseCaptureBundle` (the maintainer-decided Protocol).
#
# F2 (`classifier.py`) + F5 (`dedup.py`) are retyped `FInput` ->
# `BaseCaptureBundle` so a schema==1 `FInput` AND a schema==2 `FInputGate`
# both satisfy the consumer surface by STRUCTURAL subtyping. `FInput`
# satisfies it BY CONSTRUCTION (verified: there is no
# `isinstance(finput, FInput)` anywhere in classifier.py/dedup.py — only
# annotations; the retype is a pure static change, zero behaviour change,
# schema==1 byte-identical incl. dedup-hash). pt2-5 are `Optional[dict]`
# so `FInputGate` (None there) AND `FInput` (dict there) both conform; the
# classifier's existing `or {}` guards already tolerate None.
# ---------------------------------------------------------------------------
@runtime_checkable
class BaseCaptureBundle(Protocol):
    manifest: dict
    pt1_gate: dict
    pt2_download: Optional[dict]
    pt3_boot: Optional[dict]
    pt4_smoke: Optional[dict]
    pt5_override: Optional[dict]
    is_gate_only: bool

    # + properties already on FInput (F2 also reads these):
    #   arch_family, model_id, engine_version, quant_label,
    #   failure_class, outcome; + dedup_tuple(), dedup_hash().
    @property
    def arch_family(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    @property
    def engine_version(self) -> str: ...

    @property
    def quant_label(self) -> str: ...

    @property
    def failure_class(self): ...

    @property
    def outcome(self) -> str: ...

    def dedup_tuple(self) -> tuple: ...

    def dedup_hash(self) -> str: ...


class CaptureBundleError(Exception):
    """Raised on ANY capture-bundle schema/shape violation.

    F1 is a strict boundary validator: a malformed / wrong-schema / shape-
    violating bundle MUST raise this (never a silent partial `FInput`) so
    the downstream `[F]` STEPs can trust every field on a returned object.
    """


# Required per-point capture artifacts and the `point` value each must
# carry (`capture.py:426-430`, `442-489`). pt5 is intentionally NOT here —
# it is conditional (override-accepted only, `pull.py:1082`).
_REQUIRED_POINTS = {
    "pt1-gate.json": "gate",
    "pt2-download.json": "download",
    "pt3-boot.json": "boot",
    "pt4-smoke.json": "smoke",
}
_PT5_FILENAME = "pt5-override-capture.json"
_MANIFEST_FILENAME = "manifest.json"


@dataclass
class FInput:
    """Parsed from a SINGLE capture dir (CONTRACT-1).

    `manifest`     — manifest.json (schema==1 hard-asserted). All
                     consensus/dedup inputs are FIRST-CLASS keys here.
    `pt1_gate` ... `pt4_smoke` — the 4 §6 capture-point artifacts (required).
    `pt5_override` — pt5-override-capture.json as a dict, or None when the
                     file is absent (present iff override-accepted path).
    `raw_bundle_path` — the OPTIONAL `report.sh --redact` human-triage
                     attachment. **NOT produced by `[E]`** (`capture.py`
                     writes no such file — CONTRACT-1 / Kimi-r1 L1); it is
                     an externally-supplied maintainer attachment only.
                     F1 never synthesizes it: it defaults to None and is
                     only ever the explicit value the caller passed.
    """

    manifest: dict
    pt1_gate: dict
    pt2_download: dict
    pt3_boot: dict
    pt4_smoke: dict
    pt5_override: Optional[dict] = None
    # NOT produced by [E] — optional externally-supplied attachment only.
    raw_bundle_path: Optional[Path] = None
    # v0.8.2 — `BaseCaptureBundle` protocol member. ADDITIVE: a schema==1
    # full bundle is NEVER gate-only. Defaults False and is NOT set by
    # `read_capture_bundle()` so the shipped schema==1 parse + the
    # consensus/dedup tuples + dedup_hash are byte-identical (a defaulted
    # dataclass field changes no existing serialization). `FInput` thus
    # satisfies `BaseCaptureBundle` by construction.
    is_gate_only: bool = False

    # ---- CONTRACT-1 canonical accessors -------------------------------
    # These are the SHIPPED aliases (`capture.py:553/568`, `497-500`);
    # exposed as the canonical names so downstream STEPs do not re-derive.
    @property
    def model_id(self) -> str:
        """`model_id ≡ manifest["model"]` (shipped alias)."""
        return self.manifest["model"]

    @property
    def engine_version(self) -> str:
        """`engine_version ≡ manifest["engine_pin"]` (shipped alias)."""
        return self.manifest["engine_pin"]

    @property
    def quant_label(self) -> str:
        """Normalized (LOWERCASED) `quant_label` for keying.

        Raw `manifest["quant_label"]` is `weight_format` case-as-emitted
        (`capture.py:389-391`); CONTRACT-1 binds `[F]` to lowercase it for
        every key it builds. Use this — never the raw manifest value — when
        constructing consensus / dedup keys.
        """
        return _norm_quant(self.manifest["quant_label"])

    @property
    def arch_family(self) -> str:
        """`arch_family` used VERBATIM (CONTRACT-1 G3 RESOLVED).

        It is `config.json["architectures"][0]` (`deriver.py:679-680`) —
        already an exact identifier, NOT a normalized family. F1 must NOT
        re-normalize it.
        """
        return self.manifest["arch_family"]

    @property
    def failure_class(self) -> None:
        """Authoritatively `None` in every `[E]` manifest (`capture.py:569`).

        F1 only SURFACES this null — `[F]` computes the real
        `failure_class` in a later STEP (F2/F3). Never expect `[E]` to have
        pre-classified.
        """
        return self.manifest["failure_class"]

    @property
    def outcome(self) -> str:
        """The RAW `[E]` interim 3-state `outcome` (`failed|partial|ok`).

        BINDING RULE (CONTRACT-1): this is `[E]`'s honest interim signal
        ONLY (`capture.py:521-534`) — it is **NOT** the §6.1 class enum.
        There is deliberately NO accessor on `FInput` that claims `outcome`
        is the §6.1 class; `[F]` derives `failure_class` itself downstream.
        Surfaced here so STEPs can read the raw value, never re-interpret.
        """
        return self.manifest["outcome"]

    # ---- CONTRACT-3 / CONTRACT-4 canonical key builders ---------------
    def consensus_key(self) -> tuple:
        """The §6.2 consensus 9-tuple (CONTRACT-3, verbatim order).

        `(model, quant_label, arch_family, topology_class,
          engine_version/pin, kv_calc_version, selected_ctx, kv_format,
          smoke_capability_set)` — normalization applied (`quant_label`
          lowercased; `arch_family` verbatim; `engine_version≡engine_pin`,
          `model≡model_id` per CONTRACT-1). A DIFFERENT key from §6.3 by
          design (Codex-r4 M2) — it validates *success* anchors, carrying
          `selected_ctx`/`kv_format`/`smoke_capability_set` and NO
          `failure_class`. `smoke_capability_set` is a tuple (hashable /
          order-stable: it ships sorted, `capture.py:259`).
        """
        m = self.manifest
        scs = m["smoke_capability_set"]
        return (
            m["model"],
            _norm_quant(m["quant_label"]),
            m["arch_family"],
            m["topology_class"],
            m["engine_pin"],
            m["kv_calc_version"],
            m["selected_ctx"],
            m["kv_format"],
            tuple(scs) if scs is not None else (),
        )

    def dedup_tuple(self) -> tuple:
        """The §6.3 dedup 7-tuple (CONTRACT-4, verbatim order).

        `(model_id, quant_label, arch_family, kv_calc_version,
          engine_version, failure_class, topology_class)` — normalized
          (`quant_label` lowercased; `arch_family` verbatim;
          `engine_version≡engine_pin`, `model_id≡model`). A DIFFERENT key
          from §6.2 by design — it dedups *failure* issues, carrying
          `failure_class` and NO ctx/KV/smoke. `failure_class` is `None`
          here on every `[E]` manifest; `[F]` fills it in a later STEP and
          re-builds this tuple — its presence IN the tuple is the §6.1
          mislabel safeguard (a misclassification yields a different tuple,
          can't silently merge with a real OOM).
        """
        m = self.manifest
        return (
            m["model"],
            _norm_quant(m["quant_label"]),
            m["arch_family"],
            m["kv_calc_version"],
            m["engine_pin"],
            m["failure_class"],
            m["topology_class"],
        )

    def dedup_hash(self) -> str:
        """`sha256("\\x1f".join(dedup_tuple))[:12]` (CONTRACT-4).

        Same `\\x1f`-join + sha256 convention as `[E]`'s
        `submission_fingerprint` (`capture.py:378-381`), truncated to 12
        hex chars — the bounded, collision-safe `loop:dedup-<hash>` label
        primitive. F5 owns the issue-tracker side; F1 owns this canonical
        deterministic serialization so every STEP hashes identically.
        """
        joined = "\x1f".join(str(p) for p in self.dedup_tuple())
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# v0.8.2 CONTRACT-1.1 — `FInputGate`: the schema==2 gate-only bundle.
#
# Parsed by the SEPARATE `read_gate_bundle()` (NOT `read_capture_bundle()`;
# the shipped schema==1 path is byte-identical). It implements
# `BaseCaptureBundle` so F2/F5 consume it through the protocol with no extra
# code. A gate-only bundle has NO pt2/3/4/5 (terminated pre-download) and a
# DEGRADED manifest (only the always-present key row guaranteed; model/arch/
# quant null pre-deriver; topology best-effort/nullable; post-C0 fields
# null). `dedup_tuple()` therefore uses `.get(k, None)` (NOT `m[k]`) —
# behaviour-neutral for schema==1 (all 22 keys present on `FInput`'s path,
# which uses its own `m[k]`), crash-safe for schema==2; `str(None)=="None"`
# makes the hash deterministic + identical to how F1 renders an absent value
# (CONTRACT-1.1 — load-bearing for the COMMON null-topology gate case,
# not an edge case).
# ---------------------------------------------------------------------------
@dataclass
class FInputGate:
    """A schema==2 capture-on-hard-block bundle (CONTRACT-1.1).

    `manifest`  — manifest.json (schema==2 hard-asserted; only the always-
                  present key row is required, `outcome=="hard-block"`,
                  `failure_class is None`).
    `pt1_gate`  — pt1-gate.json (the pre-download verdict snapshot, incl.
                  the EXACT shipped `abort_reason`).
    `pt2_download` .. `pt5_override` — ALWAYS None (terminated pre-download;
                  satisfies the `Optional[dict]` protocol slots so F2/F5's
                  existing `or {}` guards tolerate it).
    """

    manifest: dict
    pt1_gate: dict
    pt2_download: Optional[dict] = None
    pt3_boot: Optional[dict] = None
    pt4_smoke: Optional[dict] = None
    pt5_override: Optional[dict] = None
    raw_bundle_path: Optional[Path] = None
    is_gate_only: bool = True

    # ---- canonical accessors (mirror FInput; `.get`-tolerant) ----------
    @property
    def model_id(self) -> str:
        return self.manifest.get("model_id") or self.manifest.get("model")

    @property
    def engine_version(self) -> str:
        # gate-only: post-C0 -> null. Render deterministically as F1 would.
        return self.manifest.get("engine_pin")

    @property
    def quant_label(self) -> str:
        """Normalized (LOWERCASED) `quant_label`. `None` pre-deriver ->
        `_norm_quant` maps it to the literal ``"none"`` (the SAME defensive
        discipline `FInput.quant_label` uses)."""
        return _norm_quant(self.manifest.get("quant_label"))

    @property
    def arch_family(self) -> str:
        """`arch_family` VERBATIM; `None` pre-deriver (gate-only legitimately
        lacks it — `read_gate_bundle()` does NOT require it)."""
        return self.manifest.get("arch_family")

    @property
    def failure_class(self):
        """Authoritatively `None` on a gate-only bundle (`[E]`/the gate
        emitter NEVER classifies — `[F]` does, downstream)."""
        return self.manifest.get("failure_class")

    @property
    def outcome(self) -> str:
        """Always `"hard-block"` on a gate-only bundle (NOT the §6.1 class —
        same binding rule as `FInput.outcome`)."""
        return self.manifest.get("outcome")

    @property
    def abort_reason(self):
        """The EXACT shipped `res.abort_reason` sub-token (CONTRACT-1.1) —
        the new `gate_abort_reason` `_match_condition` kind keys on this
        (read off `pt1_gate`)."""
        return self.pt1_gate.get("abort_reason")

    # ---- §6.3 dedup key builders (`.get`-tolerant — CONTRACT-1.1) ------
    def dedup_tuple(self) -> tuple:
        """The §6.3 dedup 7-tuple, `.get(k, None)`-tolerant (NOT `m[k]`):
        behaviour-neutral for schema==1, crash-safe for schema==2's
        degraded manifest. `failure_class` is `None` here (gate emitter
        never classifies); `[F]` substitutes the classifier's class
        downstream (F5 `effective_dedup_tuple`).
        """
        m = self.manifest
        return (
            m.get("model", None),
            _norm_quant(m.get("quant_label", None)),
            m.get("arch_family", None),
            m.get("kv_calc_version", None),
            m.get("engine_pin", None),
            m.get("failure_class", None),
            m.get("topology_class", None),
        )

    def dedup_hash(self) -> str:
        """`sha256("\\x1f".join(str(p) for p in dedup_tuple()))[:12]` —
        byte-EXACTLY `FInput.dedup_hash`'s convention. `str(None)=="None"`
        keeps a null-topology gate bundle's hash deterministic + stable
        (acceptance asserts this)."""
        joined = "\x1f".join(str(p) for p in self.dedup_tuple())
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Normalization helpers (CONTRACT-1 binding rules — `[F]`'s job, not `[E]`'s).
# ---------------------------------------------------------------------------
def _norm_quant(value) -> str:
    """`quant_label` -> lowercased for keying (CONTRACT-1).

    Raw value is `weight_format` case-as-emitted (`capture.py:389-391`).
    A `None` (theoretically possible if the deriver surfaced no
    `weight_format`) is normalized to the literal string ``"none"`` so the
    key is always a deterministic string (same defensive `str()` discipline
    as `submission_fingerprint`, `capture.py:539`).
    """
    if value is None:
        return "none"
    return str(value).lower()


# ---------------------------------------------------------------------------
# Strict bundle validation + load.
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise CaptureBundleError(f"missing required artifact: {path.name}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CaptureBundleError(
            f"unreadable/invalid JSON in {path.name}: {exc}"
        ) from exc
    if not isinstance(obj, dict):
        raise CaptureBundleError(
            f"{path.name} must be a JSON object, got {type(obj).__name__}"
        )
    return obj


def _require_keys(obj: dict, keys: tuple, where: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise CaptureBundleError(
            f"{where} missing required key(s): {', '.join(sorted(missing))}"
        )


def read_capture_bundle(
    capture_dir,
    *,
    raw_bundle_path=None,
) -> FInput:
    """Parse + STRICTLY validate ONE `[E]` capture directory into `FInput`.

    `capture_dir` is `<repo>/.pull-captures/<sanitize_slug(slug)>/<utc_ts>/`
    (`capture.py:436-439`). Required: pt1-pt4 + manifest. Optional: pt5
    (None when absent — present iff override-accepted, `pull.py:1082`).

    `raw_bundle_path` is the OPTIONAL `report.sh --redact` attachment —
    **NOT produced by `[E]`** (CONTRACT-1 / Kimi-r1 L1); F1 never
    synthesizes it. It is only ever the explicit value the caller passes
    (default None).

    Forward-compat: future `[F]` STEPs (F3/G6-A) ADD `predicted_b_breakdown`
    to pt1 and `failure_log_excerpt` + `actual.{...}` to pt3. F1 MUST
    tolerate those being present OR absent — it validates only the
    `[E]`-shipped required shape and never rejects an additive future key.

    Raises `CaptureBundleError` on ANY schema/shape violation (schema != 1,
    a missing required artifact, a wrong `point`, malformed JSON).
    """
    cdir = Path(capture_dir)
    if not cdir.is_dir():
        raise CaptureBundleError(f"capture dir does not exist: {cdir}")

    # ---- manifest (required; schema==1 hard-asserted) ------------------
    manifest = _load_json(cdir / _MANIFEST_FILENAME)
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise CaptureBundleError(
            f"manifest schema must be {EXPECTED_SCHEMA}, "
            f"got {manifest.get('schema')!r}"
        )
    # All CONTRACT-1 first-class consensus/dedup inputs MUST be present —
    # F1 is the boundary that guarantees them to downstream STEPs.
    _require_keys(
        manifest,
        (
            "schema", "slug", "utc_ts", "submission_fingerprint", "model",
            "quant_label", "arch_family", "topology_class", "engine_pin",
            "engine_version", "kv_calc_version", "selected_ctx",
            "kv_format", "smoke_capability_set",
            "topology_summary_canonical", "model_id", "failure_class",
            "club3090_commit", "outcome", "capture_points",
        ),
        "manifest.json",
    )
    # BINDING RULE: failure_class is authoritatively null in every [E]
    # manifest (capture.py:569) — F1 enforces that invariant (a non-null
    # value means [E] wrongly classified, which it must never do).
    if manifest["failure_class"] is not None:
        raise CaptureBundleError(
            "manifest.failure_class must be null in an [E] bundle "
            f"([F] classifies, not [E]); got {manifest['failure_class']!r}"
        )

    # ---- pt1-pt4 (required); each schema/point hard-asserted -----------
    pts: dict = {}
    for fname, expected_point in _REQUIRED_POINTS.items():
        obj = _load_json(cdir / fname)
        # pt1 carries `schema` (`capture.py:443`); pt2-4 do not ship a
        # `schema` key (`capture.py:454/464/478`) — assert it ONLY where
        # `[E]` actually emits it (assert-where-present, never invent a
        # constraint `[E]` does not satisfy).
        if "schema" in obj and obj["schema"] != EXPECTED_SCHEMA:
            raise CaptureBundleError(
                f"{fname} schema must be {EXPECTED_SCHEMA}, "
                f"got {obj['schema']!r}"
            )
        if obj.get("point") != expected_point:
            raise CaptureBundleError(
                f"{fname} point must be {expected_point!r}, "
                f"got {obj.get('point')!r}"
            )
        pts[fname] = obj

    # ---- pt5 (optional — present iff override-accepted) ---------------
    pt5_path = cdir / _PT5_FILENAME
    pt5_override: Optional[dict] = None
    if pt5_path.is_file():
        pt5_override = _load_json(pt5_path)
        if pt5_override.get("point") != "override_capture":
            raise CaptureBundleError(
                f"{_PT5_FILENAME} point must be 'override_capture', "
                f"got {pt5_override.get('point')!r}"
            )

    return FInput(
        manifest=manifest,
        pt1_gate=pts["pt1-gate.json"],
        pt2_download=pts["pt2-download.json"],
        pt3_boot=pts["pt3-boot.json"],
        pt4_smoke=pts["pt4-smoke.json"],
        pt5_override=pt5_override,
        raw_bundle_path=(
            Path(raw_bundle_path) if raw_bundle_path is not None else None
        ),
    )


# v0.8.2 CONTRACT-1.1 — the schema==2 gate-only required key row. Per the
# CONTRACT-1.1 enumerated per-abort-stratum table this is the ALWAYS-PRESENT
# row (guaranteed at EVERY gate stratum). model/arch/quant/topology +
# post-C0 keys are deliberately NOT required (early deriver / profile-like /
# repo-not-found / hardware-sm-undetermined terminal legitimately lacks
# them) —
# `read_gate_bundle()` tolerates them via `.get()`, it MUST NOT reuse the
# 22-key schema-1 `_require_keys` set (that hard-`raise`s on the post-C0
# set the gate path can never populate).
_GATE_REQUIRED_KEYS = (
    "schema", "slug", "utc_ts", "club3090_commit", "outcome",
    "abort_reason", "failure_class",
)


def read_gate_bundle(
    capture_dir,
    *,
    raw_bundle_path=None,
) -> FInputGate:
    """v0.8.2 CONTRACT-1.1 — parse + validate ONE schema==2 (gate-only,
    capture-on-hard-block) bundle into `FInputGate`.

    A SEPARATE reader from `read_capture_bundle()`: the shipped schema==1
    `FInput` path is byte-identical (untouched). Required artifacts:
    `manifest.json` (schema==2) + `pt1-gate.json` (point=="gate"). pt2-5
    are NOT present on a gate-only bundle (terminated pre-download) — the
    returned `FInputGate` carries `None` for all of them.

    Validation is DELIBERATELY narrow (it MUST NOT reuse the 22-key
    schema-1 `_require_keys`): ONLY the always-present key row +
    `outcome=="hard-block"` + `failure_class is None`. Everything else
    (model/arch/quant/topology/post-C0) is `.get(k, None)`-tolerated —
    a gate-only bundle legitimately lacks those (per the CONTRACT-1.1
    per-abort-stratum table).

    Raises `CaptureBundleError` on a schema mismatch / missing required
    artifact / wrong `point` / bad `outcome` / non-null `failure_class`.
    """
    cdir = Path(capture_dir)
    if not cdir.is_dir():
        raise CaptureBundleError(f"capture dir does not exist: {cdir}")

    # ---- manifest (required; schema==2 hard-asserted) ------------------
    manifest = _load_json(cdir / _MANIFEST_FILENAME)
    if manifest.get("schema") != GATE_SCHEMA:
        raise CaptureBundleError(
            f"gate-bundle manifest schema must be {GATE_SCHEMA}, "
            f"got {manifest.get('schema')!r}"
        )
    # ONLY the always-present row — NOT the 22-key schema-1 validator
    # (CONTRACT-1.1 hard rule: gate-only legitimately cannot populate the
    # post-C0 set, so reusing `_require_keys` would wrongly reject it).
    _require_keys(manifest, _GATE_REQUIRED_KEYS, "gate manifest.json")
    if manifest.get("outcome") != "hard-block":
        raise CaptureBundleError(
            "gate-bundle manifest.outcome must be 'hard-block', "
            f"got {manifest.get('outcome')!r}"
        )
    # The gate emitter NEVER classifies (§6.1 = [F]'s job) — enforce the
    # invariant exactly as the schema-1 reader does for an [E] bundle.
    if manifest.get("failure_class") is not None:
        raise CaptureBundleError(
            "gate-bundle manifest.failure_class must be null "
            f"([F] classifies, not the gate emitter); "
            f"got {manifest['failure_class']!r}"
        )

    # ---- pt1-gate.json (required; point=="gate", schema==2) -----------
    pt1 = _load_json(cdir / "pt1-gate.json")
    if "schema" in pt1 and pt1["schema"] != GATE_SCHEMA:
        raise CaptureBundleError(
            f"pt1-gate.json schema must be {GATE_SCHEMA}, "
            f"got {pt1['schema']!r}"
        )
    if pt1.get("point") != "gate":
        raise CaptureBundleError(
            f"pt1-gate.json point must be 'gate', got {pt1.get('point')!r}"
        )

    return FInputGate(
        manifest=manifest,
        pt1_gate=pt1,
        raw_bundle_path=(
            Path(raw_bundle_path) if raw_bundle_path is not None else None
        ),
    )
