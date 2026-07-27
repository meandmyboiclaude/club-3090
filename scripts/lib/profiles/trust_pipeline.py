"""v0.8.0 Loop `[F]` — STEP F4: the §6.2 inbound-trust pipeline.

CONTRACT-3 + CONTRACT-3a (the locked v0.8.0 Loop design brief,
"CONTRACT 3" / "CONTRACT 3a"; source-of-truth design §6.2; see
docs/LOOP.md for the shipped contributor reference). F1 (`loop_input.py`) produced the
validated `FInput`; F2/F3 (`classifier.py`) produced the §6.1
`ClassificationResult`. This module CONSUMES both as a library and runs the
4-stage success-anchor trust pipeline:

    raw  ->  candidate  ->  validated  ->  tier1

`[F]` here is the CONSUMER of `[E]`'s on-disk capture bundle; it NEVER
re-redacts (artifacts arrive `[E]`-redacted, CONTRACT-3 stage-1 / G10),
NEVER re-mints `submission_fingerprint` (`[E]` mints it; F4 re-derives +
verifies it), and NEVER auto-mutates `calibration/*.yml` or runs kv-calc
(that ingestion is a separate, later, manual/maintainer concern — F4 only
*certifies* tier1-eligibility and emits the would-be calibration-row SHAPE
for a curated anchor).

This module is purely additive — it touches NO shipped code (two NEW files
only). It imports F1/F2/F3 + the shipped `compat`/`compose_registry` as a
library; it modifies none of them.

------------------------------------------------------------------------
The 4 stages (CONTRACT-3 binding mapping; design §6.2 verbatim-anchored)
------------------------------------------------------------------------

1. **raw** — the already-redacted bundle. The MACHINE key source is the
   first-class `manifest.json` keys `[E]` already scrubbed via
   `_redact_text`; `report.sh --redact` is an OPTIONAL human-triage
   Markdown attachment (G10), NOT machine-parseable and NOT required. F4
   reads keys from `FInput`/the manifest only; it never re-redacts and
   never requires the report.sh bundle. Every well-formed `FInput` is at
   least `raw` (F1 already strict-validated the schema-1 shape).

2. **candidate** — re-derive `submission_fingerprint` from the manifest
   fields with the SAME `\x1f`+sha256 8-tuple `[E]` uses
   (`capture.py:671-680`, field order quoted in `_rederive_fingerprint`)
   and reject (stop at `raw`, reason `fingerprint-mismatch`) when it does
   not equal `manifest["submission_fingerprint"]`. `submission_fingerprint`
   is CORRELATION, NOT security (§6.2 verbatim — the fields are
   user-controlled; the real trust gate is consensus + maintainer
   promotion at stage 3). Then the **capability-aware smoke gate** (closes
   Codex-r2 High#2 / the #145 class): a success anchor graduates a
   capability **iff `pt4.results[cap] == "green"`**; `unsmoked`/`red` caps
   NEVER graduate; a `partial` anchor (`pt4.partial == True`) is
   Tier-1-eligible **only for its green caps**, never the model
   wholesale. This is encoded as the `graduation_set` (a frozenset of
   green caps), NOT a bool.

3. **validated** — sanity: topology plausible (a basic bounds check on
   `topology_class` / `topology_summary_canonical` — no impossible GPU /
   VRAM); AND (**multi-submission consensus** on the FULL §6.2 9-tuple
   `FInput.consensus_key()` matching across `prior_submissions` reaching
   `consensus_n`) **OR** explicit `maintainer_promoted=True`.

   **CRITICAL CONTRACT-3 clarification (also stated in `_validate`):** the
   "predicted-vs-actual delta ≤ tolerance" check belongs to the §6.1
   *failure → kv-calc-bug* branch (pt5 / F3 Tier-1 path) — it is **NOT**
   part of success-anchor validation. A successful boot has no OOM delta;
   success validation = topology-plausible + (consensus OR maintainer
   promotion). F4 does NOT gate success anchors on any delta and requires
   no delta input at all.

4. **Tier-1** — graduates to the calibration backbone; only a
   Stage-≥validated anchor reaches it. **CONTRACT-3a — LOCKED resolution
   (b): derived anchors are classifier+dedup-ONLY this phase.** Tier-1
   calibration-backbone ingestion stays **curated-compose-only**: a
   derived/generic-dense anchor (one whose `model` slug does NOT resolve
   to a curated catalog model that has a `COMPOSE_REGISTRY` entry — the A3
   structural blocker `compat.py:1120-1139`) MUST NOT be emitted as a
   calibration row. F4 detects curated-vs-derived; for derived it stops at
   `validated` with reason `derived-tier1-deferred-v0.8.1` (it still flows
   fully through classifier+dedup elsewhere — F4 just does not push it to
   the calibration backbone). For curated it MAY reach `tier1`: F4 emits
   the would-be `calibration/<model>.yml` row SHAPE (matching the schema
   read from `compat.calibration_status` + `calibration/*.yml`), but never
   itself edits calibration YAML or runs kv-calc.

Scope/OUT honored (brief): consensus AUTOMATION (auto-promote at N≥2) is
deferred — F4 ships the consensus-key MATCHING primitive + the
`maintainer_promoted` manual hook, NOT a daemon. GGUF/derived→calibration
bridge is explicitly v0.8.1 (the `derived-tier1-deferred-v0.8.1` stop).

PURE-PYTHON, stdlib + the SAME `[F]`/profiles library imports the siblings
use. House style mirrors `classifier.py` (`class …(str, Enum)`, frozen
dataclass result, structured return — never raise for an expected
outcome).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from scripts.lib.profiles.classifier import ClassificationResult
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
from scripts.lib.profiles.loop_input import FInput

try:  # same import discipline as classifier.py / compat.py — reuse a dep.
    from scripts.lib.profiles.compat import PROFILE_ROOT, load_profiles
except Exception:  # pragma: no cover - profiles load is best-effort here
    # F4's curated-vs-derived detection degrades to COMPOSE_REGISTRY-only
    # if the full profile tree cannot load (honest: never crash the
    # offline trust pipeline; a derived verdict is the safe default).
    PROFILE_ROOT = None  # type: ignore[assignment]
    load_profiles = None  # type: ignore[assignment]


class TrustStage(str, Enum):
    """The §6.2 pipeline stage an anchor reached (CONTRACT-3, verbatim).

    Ordered raw < candidate < validated < tier1. `TrustResult.stage` is the
    HIGHEST stage reached; `TrustResult.reason` says why it stopped there.
    House style: `class …(str, Enum)` (mirrors `classifier.FailureClass`).
    """

    RAW = "raw"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    TIER1 = "tier1"


# Monotone rank for "Stage-≥validated" comparisons (design §6.2: calibration
# ingests only Stage-≥validated anchors).
_STAGE_RANK = {
    TrustStage.RAW: 0,
    TrustStage.CANDIDATE: 1,
    TrustStage.VALIDATED: 2,
    TrustStage.TIER1: 3,
}


@dataclass(frozen=True)
class TrustResult:
    """The §6.2 trust-pipeline verdict (returned, never raised).

    `stage`            — the highest `TrustStage` the anchor reached.
    `reason`           — WHY it stopped at `stage` (a stable machine token,
                         e.g. `fingerprint-mismatch`, `insufficient-consensus`,
                         `derived-tier1-deferred-v0.8.1`, `topology-implausible`,
                         `tier1-curated`).
    `graduation_set`   — the frozenset of capabilities that graduated
                         (each `pt4.results[cap] == "green"`). `unsmoked`/
                         `red` caps are NEVER in it. A `partial` anchor
                         carries only its green caps here — the capability-
                         scoped success rule (the #145-class guard).
    `fingerprint_ok`   — True iff the re-derived `submission_fingerprint`
                         equals `manifest["submission_fingerprint"]`.
    `rederived_fingerprint` — what F4 computed (surfaced for diagnosis).
    `curated`          — True iff the anchor is curated-compose-backed
                         (CONTRACT-3a): its model resolves to a curated
                         catalog model that has a COMPOSE_REGISTRY entry.
                         A derived anchor is False (stops at `validated`).
    `consensus_count`  — how many submissions (incl. this one) matched the
                         full §6.2 9-tuple consensus key.
    `consensus_reached`— True iff `consensus_count >= consensus_n`.
    `maintainer_promoted` — echo of the manual-promotion hook input.
    `calibration_row`  — the would-be `calibration/<model>.yml` row SHAPE,
                         ONLY for a curated anchor that reached `tier1`;
                         else None. F4 NEVER writes this to YAML — it only
                         certifies the shape for a later manual/maintainer
                         (or test) consumer.
    `classification`   — the F2/F3 `ClassificationResult` F4 was handed
                         (surfaced so a caller never re-runs the classifier
                         to learn routing).
    `notes`            — human-readable trace of every stage decision.
    """

    stage: TrustStage
    reason: str
    graduation_set: frozenset
    fingerprint_ok: bool
    rederived_fingerprint: str
    curated: bool
    consensus_count: int
    consensus_reached: bool
    maintainer_promoted: bool
    classification: Optional[ClassificationResult] = None
    calibration_row: Optional[dict] = None
    notes: tuple = field(default_factory=tuple)

    def at_least(self, stage: TrustStage) -> bool:
        """`stage`-monotone helper (design §6.2: calibration ingests only
        Stage-≥validated). `tr.at_least(TrustStage.VALIDATED)` is the
        gate a downstream calibration ingester would check."""
        return _STAGE_RANK[self.stage] >= _STAGE_RANK[stage]


# ---------------------------------------------------------------------------
# Stage 2 — submission_fingerprint re-derivation (CONTRACT-3 candidate).
# ---------------------------------------------------------------------------
#
# `[E]` mints the fingerprint at `capture.py:671-680` (Claude-verified):
#
#     submission_fingerprint = _fingerprint([
#         model,                          # == einput.slug == manifest["model"]
#         einput.club3090_commit,         # == manifest["club3090_commit"]
#         topology_summary_canonical,     # == manifest["topology_summary_canonical"]
#         str(quant_label),               # == str(manifest["quant_label"])
#         kv_calc_version,                # == manifest["kv_calc_version"]
#         str(engine_version),            # == str(manifest["engine_pin"]/["engine_version"])
#         stamp,                          # == manifest["utc_ts"]
#         outcome,                        # == manifest["outcome"]
#     ])
#
# where `_fingerprint(parts)` = sha256("\x1f".join(str(p) for p in parts))
# (HEXDIGEST — full 64-hex, NOT truncated; capture.py:457-460). EVERY part
# is a first-class manifest key (CONTRACT-1), so F4 re-derives purely from
# `manifest` with the byte-exact join + the same `str()` coercion on the
# two parts `[E]` wraps (`str(quant_label)`, `str(engine_version)`) —
# matching `[E]`'s serialization exactly.
#
# This is CORRELATION, NOT security (§6.2 verbatim): the inputs are
# user-controlled, so a matching fingerprint only means the bundle's key
# fields are internally consistent (no accidental cross-wiring) — the real
# trust gate is consensus + maintainer promotion (stage 3). No crypto /
# threat-model work here by design (brief Scope/OUT).
_FP_SEP = "\x1f"


def _rederive_fingerprint(manifest: dict) -> str:
    """Re-derive `submission_fingerprint` from manifest fields, byte-exactly
    matching `[E]`'s `capture.py:671-680` 8-tuple + `capture.py:457-460`
    `\\x1f`-join sha256 HEXDIGEST. `str()` is applied to exactly the two
    parts `[E]` wraps (quant_label, engine_version); the rest are joined as
    `[E]` passes them (its `_fingerprint` `str()`s every part anyway, so the
    result is identical — we mirror `[E]`'s explicit `str()` on those two
    for readability/parity).
    """
    parts = [
        manifest["model"],
        manifest["club3090_commit"],
        manifest["topology_summary_canonical"],
        str(manifest["quant_label"]),
        manifest["kv_calc_version"],
        str(manifest["engine_version"]),
        manifest["utc_ts"],
        manifest["outcome"],
    ]
    h = hashlib.sha256()
    h.update(_FP_SEP.join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Stage 2 — capability-aware smoke gate (CONTRACT-3 / the #145-class guard).
# ---------------------------------------------------------------------------
def _graduation_set(finput: FInput) -> frozenset:
    """The set of capabilities that graduate = exactly those with
    `pt4.results[cap] == "green"`.

    `unsmoked` / `red` caps are NEVER in the set (the #145-class guard:
    a model can boot + answer plain-chat while streaming/tools/vision are
    broken — those caps must not graduate). A `partial` anchor
    (`pt4.partial == True`) therefore yields ONLY its green caps here,
    never the model wholesale. Encoded as a set, NOT a bool — design §6.2:
    "Tier-1-eligible only for its green caps".
    """
    pt4 = finput.pt4_smoke or {}
    results = pt4.get("results") or {}
    return frozenset(
        cap for cap, verdict in results.items() if verdict == "green"
    )


# ---------------------------------------------------------------------------
# Stage 3 — topology plausibility (CONTRACT-3 validated, basic bounds).
# ---------------------------------------------------------------------------
def _topology_plausible(manifest: dict) -> bool:
    """A BASIC bounds check (design §6.2: "topology plausible (no
    impossible GPU/VRAM combos)") — NOT a hardware model. `topology_class`
    is `[E]`'s coarse `"{N}x{VRAM}MiB"` (`capture.py:473-478`). Reject the
    structurally-impossible: zero/negative GPU count, zero/negative VRAM, a
    malformed class, or a `topology_summary_canonical` that is not a
    non-empty string. This is the floor that stops a corrupt/garbage
    topology from validating; it deliberately does NOT try to know every
    real SKU (that is out of scope and would be confidently-wrong).
    """
    tsc = manifest.get("topology_summary_canonical")
    if not isinstance(tsc, str) or not tsc.strip():
        return False
    tclass = manifest.get("topology_class")
    if not isinstance(tclass, str) or "x" not in tclass:
        return False
    head, _, tail = tclass.partition("x")
    if not tail.endswith("MiB"):
        return False
    try:
        n_gpu = int(head)
        vram_mib = int(tail[: -len("MiB")])
    except (TypeError, ValueError):
        return False
    if n_gpu <= 0 or vram_mib <= 0:
        return False
    # An obviously-impossible single-card VRAM (e.g. > 1 TiB) is garbage.
    if vram_mib > 1024 * 1024:
        return False
    return True


# ---------------------------------------------------------------------------
# Stage 3 — multi-submission consensus on the FULL §6.2 9-tuple.
# ---------------------------------------------------------------------------
def _consensus_count(
    finput: FInput,
    prior_submissions,
) -> int:
    """Count submissions (THIS one + every prior) whose FULL §6.2 9-tuple
    `FInput.consensus_key()` equals this anchor's.

    "within tolerance" (design §6.2) for the consensus KEY is exact-tuple
    equality: every one of the 9 dimensions is a discrete identifier
    (model / quant / arch / topology_class / engine_pin / kv_calc_version /
    selected_ctx / kv_format / smoke_capability_set), so two materially-
    different runs cannot accidentally agree (Codex-r3 Medium#2 — the
    narrow-key accident this full key closes). A `prior_submissions`
    element may be an `FInput` or a raw consensus-key tuple/list (the
    matching primitive is decoupled from storage — F4 ships the primitive,
    NOT a daemon; brief Scope/OUT).
    """
    target = finput.consensus_key()
    count = 1  # this submission itself
    for prior in prior_submissions or ():
        if isinstance(prior, FInput):
            key = prior.consensus_key()
        elif isinstance(prior, (tuple, list)):
            key = tuple(prior)
        else:  # unknown shape — never silently count it as agreement.
            continue
        if key == target:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Stage 4 — CONTRACT-3a curated-vs-derived detection (the A3 structural
# blocker `compat.py:1120-1139`).
# ---------------------------------------------------------------------------
_PROFILES_CACHE = None
_PROFILES_TRIED = False


def _profiles():
    """Load the profile tree ONCE (best-effort, cached). On any failure
    return None — the curated-vs-derived check then degrades to
    COMPOSE_REGISTRY-only and the safe default (derived) so F4 never
    crashes the offline trust pipeline and never over-promotes."""
    global _PROFILES_CACHE, _PROFILES_TRIED
    if _PROFILES_TRIED:
        return _PROFILES_CACHE
    _PROFILES_TRIED = True
    if load_profiles is None or PROFILE_ROOT is None:
        _PROFILES_CACHE = None
        return None
    try:
        _PROFILES_CACHE = load_profiles(PROFILE_ROOT)
    except Exception:  # pragma: no cover - degraded path
        _PROFILES_CACHE = None
    return _PROFILES_CACHE


def _registry_models() -> frozenset:
    """The set of curated catalog model ids that HAVE a COMPOSE_REGISTRY
    entry (`entry["model"]`). A calibration row can only ever be keyed by a
    curated compose name (the A3 structural blocker: `compat.calibration_
    status` returns "predicted" when `compose ∉ COMPOSE_REGISTRY`), so a
    model with no registry entry can never back a calibration row."""
    return frozenset(
        e["model"] for e in COMPOSE_REGISTRY.values() if e.get("model")
    )


def _resolve_curated_model(finput: FInput) -> Optional[str]:
    """Map the anchor's HF slug (`manifest["model"]`, == `[E]`'s
    `einput.slug`) to a curated catalog model id IFF that slug is one of a
    curated model's declared `hf_repos` AND that model has a
    COMPOSE_REGISTRY entry. Returns the curated catalog model id, else
    None (= derived/generic-dense — borrowed only a `profile_like` SHAPE,
    not a curated catalog identity).

    This is the CONTRACT-3a structural detection grounded directly in the
    A3 blocker: Tier-1 calibration ingestion is curated-compose-only, and a
    compose is curated iff it is a COMPOSE_REGISTRY key whose `model` is a
    catalog model. A derived anchor's slug (e.g. `Qwen/Qwen2.5-0.5B-
    Instruct`) is in NO catalog model's `hf_repos`, so it resolves to None
    and is deferred to v0.8.1.
    """
    profiles = _profiles()
    if profiles is None:
        return None
    slug = finput.manifest.get("model")
    if not slug:
        return None
    registry_models = _registry_models()
    slug_norm = str(slug).strip()
    for model_id, model in profiles.models.items():
        if model_id not in registry_models:
            continue  # curated catalog model but no compose -> not Tier-1able
        for repos in model.all_hf_repos().values():
            for repo in repos:
                if str(repo).strip() == slug_norm:
                    return model_id
    return None


def _curated_compose_for(finput: FInput, curated_model: str) -> Optional[str]:
    """Resolve the best curated COMPOSE_REGISTRY name for a curated anchor
    so the would-be calibration row can be keyed by a real compose (the A3
    requirement: a calibration row's `compose` MUST be a COMPOSE_REGISTRY
    key). Prefer `pt1.profile_like` when it is a registry key whose
    `entry["model"]` matches the curated model; else the first registry
    entry for that model. Never invent a name."""
    pt1 = finput.pt1_gate or {}
    profile_like = pt1.get("profile_like")
    if (
        isinstance(profile_like, str)
        and profile_like in COMPOSE_REGISTRY
        and COMPOSE_REGISTRY[profile_like].get("model") == curated_model
    ):
        return profile_like
    for name, entry in COMPOSE_REGISTRY.items():
        if entry.get("model") == curated_model:
            return name
    return None


def _calibration_row_shape(
    finput: FInput, curated_model: str, compose_name: str
) -> dict:
    """Build the would-be `calibration/<model>.yml` row SHAPE for a curated
    Tier-1 anchor.

    Schema confirmed from `scripts/lib/profiles/calibration/qwen3.6-27b.yml`
    + the keys `compat.calibration_status` actually reads
    (`compat.py:1128-1138`): a row is
        {compose, vram_gb, measured_peak_gb, ctx_override,
         status, engine_pin, genesis_pin, source}
    where `compat` matches on `status == "active"`, `compose == compose_
    name`, `vram_gb == min(hw.vram_gb)`, and an optional `ctx_override`.

    F4 ONLY certifies this shape — it NEVER writes it to YAML, never runs
    kv-calc, and emits it `status: "candidate-tier1"` (NOT "active"): a
    later manual/maintainer (or a test) step is the only thing allowed to
    promote it to `active` and ingest it. `measured_peak_gb` is left None
    here (F4 is offline and never benchmarks — fabricating a peak would be
    the §1 confidently-wrong sin); the maintainer fills it from the
    bundle's measured evidence on ingest. `vram_gb` is derived from the
    coarse `topology_class` ({N}x{VRAM}MiB -> per-card GiB) so the row is
    self-consistent with how `compat` matches.
    """
    manifest = finput.manifest
    tclass = str(manifest.get("topology_class") or "")
    vram_gb = None
    if "x" in tclass and tclass.endswith("MiB"):
        try:
            mib = int(tclass.partition("x")[2][: -len("MiB")])
            vram_gb = round(mib / 1024.0, 3)
        except (TypeError, ValueError):
            vram_gb = None
    selected_ctx = manifest.get("selected_ctx")
    return {
        "compose": compose_name,
        "vram_gb": vram_gb,
        "measured_peak_gb": None,  # F4 never benchmarks — maintainer fills.
        "ctx_override": selected_ctx,
        "status": "candidate-tier1",  # NEVER "active" — F4 only certifies.
        "engine_pin": manifest.get("engine_pin"),
        "genesis_pin": None,  # not in the [E] manifest; maintainer fills.
        "source": (
            f"loop[F]/F4 inbound-trust anchor "
            f"model={manifest.get('model')} "
            f"slug-resolved={curated_model} "
            f"fingerprint={manifest.get('submission_fingerprint')} "
            f"utc_ts={manifest.get('utc_ts')}"
        ),
    }


# ---------------------------------------------------------------------------
# Public API — the §6.2 pipeline.
# ---------------------------------------------------------------------------
def promote(
    finput: FInput,
    classification: ClassificationResult,
    *,
    prior_submissions=(),
    maintainer_promoted: bool = False,
    consensus_n: int = 2,
    tolerance=None,
) -> TrustResult:
    """Run the §6.2 inbound-trust pipeline raw->candidate->validated->tier1
    on ONE success anchor (CONTRACT-3 + CONTRACT-3a).

    `finput`              — the F1-validated bundle (the anchor under test).
    `classification`      — the F2/F3 §6.1 verdict for the same bundle
                            (surfaced on the result so a caller never
                            re-runs the classifier to learn routing). F4's
                            job is the SUCCESS-anchor trust pipeline (§6.2),
                            DISTINCT from §6.1 classification — the
                            classification is carried, not re-decided.
    `prior_submissions`   — previously-seen submissions to consensus-match
                            this anchor against (each an `FInput` or a raw
                            consensus-key tuple). The consensus-key MATCHING
                            primitive only — NO automation/daemon (brief
                            Scope/OUT; consensus auto-promotion is deferred).
    `maintainer_promoted` — the manual promotion hook (§6.2 acceptance:
                            "Early phase: maintainer manual promotion is
                            acceptable"). True alone satisfies stage-3's
                            "OR maintainer promotion".
    `consensus_n`         — N independent matching submissions needed for
                            consensus (default 2 = the §6.2 N≥2 threshold).
    `tolerance`           — ACCEPTED + IGNORED on purpose. The §6.2
                            "predicted-vs-actual delta ≤ tolerance" check is
                            the §6.1 failure→kv-calc-bug branch (pt5 / F3
                            Tier-1), NOT success-anchor validation. A
                            successful boot has no OOM delta; F4 requires no
                            delta input at all and never gates a success
                            anchor on one. The parameter exists only so a
                            caller that mistakenly passes a tolerance does
                            not break — it has no effect on the verdict.

    Returns a `TrustResult` (never raises for an expected outcome — house
    style). The result's `stage` is the highest reached; `reason` is why it
    stopped there.
    """
    manifest = finput.manifest
    notes: list[str] = []

    # ===== Stage 1: raw =================================================
    # The bundle arrived already `[E]`-redacted (CONTRACT-3 stage-1 / G10):
    # the machine key source is the first-class manifest keys, NOT the
    # optional report.sh Markdown attachment. F1 already strict-validated
    # the schema-1 shape, so a well-formed FInput IS at least `raw`. F4
    # NEVER re-redacts and NEVER requires the report.sh bundle.
    notes.append("stage1 raw: [E]-redacted bundle accepted (manifest is the "
                 "machine key source; report.sh attachment NOT required)")

    # Always compute the graduation set + fingerprint so the result is
    # informative even when the pipeline stops early.
    graduation_set = _graduation_set(finput)
    rederived = _rederive_fingerprint(manifest)
    claimed = manifest.get("submission_fingerprint")
    fingerprint_ok = (rederived == claimed)

    # Curated-vs-derived is resolved up-front (used at stage 4; surfaced on
    # every result for diagnosis).
    curated_model = _resolve_curated_model(finput)
    curated = curated_model is not None

    # Consensus is computed up-front too (surfaced on every result).
    consensus_count = _consensus_count(finput, prior_submissions)
    consensus_reached = consensus_count >= int(consensus_n)

    def _result(stage: TrustStage, reason: str,
                calibration_row=None) -> TrustResult:
        return TrustResult(
            stage=stage,
            reason=reason,
            graduation_set=graduation_set,
            fingerprint_ok=fingerprint_ok,
            rederived_fingerprint=rederived,
            curated=curated,
            consensus_count=consensus_count,
            consensus_reached=consensus_reached,
            maintainer_promoted=bool(maintainer_promoted),
            classification=classification,
            calibration_row=calibration_row,
            notes=tuple(notes),
        )

    # ===== Stage 2: candidate ==========================================
    # (a) re-derive + verify submission_fingerprint (CONTRACT-3). Mismatch
    #     => corrupt/tampered correlation => STOP at raw. This is
    #     CORRELATION not security (§6.2 verbatim) — it only catches an
    #     accidentally-inconsistent bundle; the real trust gate is stage 3.
    if not fingerprint_ok:
        notes.append(
            f"stage2 candidate: submission_fingerprint MISMATCH "
            f"(re-derived {rederived[:12]}… != claimed "
            f"{str(claimed)[:12]}…) -> reject, stay raw"
        )
        return _result(TrustStage.RAW, "fingerprint-mismatch")
    notes.append("stage2 candidate: submission_fingerprint re-derived and "
                 "verified (correlation OK, not security)")

    # (b) capability-aware smoke gate (the #145-class guard). An anchor
    #     with NO green capability has nothing to graduate — it cannot be a
    #     success anchor for ANY capability, so it stops at candidate.
    if not graduation_set:
        notes.append(
            "stage2 candidate: capability-aware smoke gate — NO green "
            "capability (every cap unsmoked/red) -> nothing graduates, "
            "stop at candidate (#145-class guard)"
        )
        return _result(TrustStage.CANDIDATE, "no-green-capability")
    notes.append(
        f"stage2 candidate: capability-aware smoke gate — graduation_set="
        f"{sorted(graduation_set)} "
        f"(partial={bool((finput.pt4_smoke or {}).get('partial'))}; "
        f"only green caps graduate, never the model wholesale)"
    )

    # ===== Stage 3: validated ==========================================
    # CONTRACT-3 CRITICAL CLARIFICATION: the "predicted-vs-actual delta ≤
    # tolerance" check is the §6.1 failure→kv-calc-bug branch (pt5 / F3
    # Tier-1 path) — it is NOT part of success-anchor validation. A
    # successful boot has NO OOM delta. Success validation =
    #   topology-plausible AND (consensus OR maintainer promotion).
    # F4 deliberately requires no delta input (the `tolerance` arg is
    # accepted-and-ignored, see the docstring).
    if not _topology_plausible(manifest):
        notes.append(
            "stage3 validated: topology IMPLAUSIBLE "
            f"(topology_class={manifest.get('topology_class')!r}) -> "
            "stop at candidate"
        )
        return _result(TrustStage.CANDIDATE, "topology-implausible")
    notes.append("stage3 validated: topology plausible (basic GPU/VRAM "
                 "bounds OK)")

    promoted_by_consensus = consensus_reached
    promoted_by_maintainer = bool(maintainer_promoted)
    if not (promoted_by_consensus or promoted_by_maintainer):
        notes.append(
            f"stage3 validated: NEITHER consensus "
            f"({consensus_count}/{consensus_n}) NOR maintainer promotion -> "
            f"stop at candidate (insufficient-consensus)"
        )
        return _result(TrustStage.CANDIDATE, "insufficient-consensus")
    notes.append(
        "stage3 validated: "
        + (
            f"consensus reached ({consensus_count}/{consensus_n})"
            if promoted_by_consensus
            else "maintainer promotion"
        )
        + " + topology plausible -> VALIDATED"
        + (" (no predicted-vs-actual delta required for a success anchor)")
    )

    # ===== Stage 4: Tier-1 (CONTRACT-3a) ===============================
    # Only a Stage-≥validated anchor can reach Tier-1. CONTRACT-3a LOCKED
    # (b): derived anchors are classifier+dedup-ONLY this phase — Tier-1
    # calibration-backbone ingestion is curated-compose-only. A derived
    # anchor STOPS at validated with `derived-tier1-deferred-v0.8.1` (it
    # still flows fully through classifier+dedup elsewhere; F4 just does
    # not push it to the calibration backbone). It emits NO calibration row.
    if not curated:
        notes.append(
            "stage4 tier1: anchor is DERIVED/generic-dense (model slug "
            f"{manifest.get('model')!r} resolves to no curated catalog "
            "model with a COMPOSE_REGISTRY entry — the A3 structural "
            "blocker). CONTRACT-3a (b): derived = classifier+dedup-only "
            "this phase; calibration bridge is v0.8.1. STOP at validated; "
            "NO calibration row emitted."
        )
        return _result(
            TrustStage.VALIDATED, "derived-tier1-deferred-v0.8.1"
        )

    # Curated anchor: it MAY reach Tier-1. F4 certifies tier1-eligibility
    # and emits the would-be calibration-row SHAPE (status
    # "candidate-tier1", NEVER "active"). It does NOT edit calibration YAML
    # and does NOT run kv-calc — that ingestion is a separate, later,
    # manual/maintainer (or test) concern.
    compose_name = _curated_compose_for(finput, curated_model)
    if compose_name is None:
        # Curated catalog model but no resolvable compose key — cannot key
        # a calibration row (the A3 requirement). Honest degrade: stay
        # validated rather than emit a row keyed by an invented compose.
        notes.append(
            "stage4 tier1: curated catalog model "
            f"{curated_model!r} but no resolvable COMPOSE_REGISTRY key "
            "(cannot key a calibration row by the A3 rule) -> stay "
            "validated, no row"
        )
        return _result(
            TrustStage.VALIDATED, "curated-no-compose-key"
        )

    calibration_row = _calibration_row_shape(
        finput, curated_model, compose_name
    )
    notes.append(
        f"stage4 tier1: CURATED anchor (model->{curated_model}, "
        f"compose={compose_name}) -> TIER-1 eligible. Emitting would-be "
        f"calibration row SHAPE (status=candidate-tier1; F4 NEVER writes "
        f"YAML / runs kv-calc — separate manual/maintainer concern)."
    )
    return _result(
        TrustStage.TIER1, "tier1-curated", calibration_row=calibration_row
    )
