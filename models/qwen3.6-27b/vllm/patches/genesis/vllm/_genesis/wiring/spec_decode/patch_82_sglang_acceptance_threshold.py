# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 82 — SGLang-style threshold_single OR-clause acceptance.

Backport of the per-token acceptance rule from SGLang
(`sgl-kernel/csrc/speculative/speculative_sampling.cuh` ~line 107):

    if (coin <= prob_acc / threshold_acc || target_prob_single >= threshold_single):
        accept

vs vLLM's vanilla rule (`vllm/v1/sample/rejection_sampler.py:797`):

    accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob

P82 inserts the OR-clause: accept if EITHER vanilla rejection passes OR
the target's confidence in the drafted token meets a threshold. Targets
the structural ceiling `clean_rate ≈ accept_rate^num_spec` identified in
the v7.13 strict-ngram analysis.

================================================================
TRADE-OFF — READ THIS BEFORE ENABLING
================================================================

The threshold rule is **biased** — it loses the unbiased-sampling guarantee
of canonical rejection sampling. SGLang accepts this trade-off explicitly.
For greedy / low-temperature tool-call workloads (our case), the bias
short-circuits in favor of higher-prob target tokens, which is the right
direction. For temperature ≥ 1.0 creative-writing workloads the bias
could compress diversity. WE DO NOT SHIP THIS WITHOUT EMPIRICAL
VALIDATION (`genesis_quality_harness.py` ≥ 30/31 + `genesis_bench_v3.py`
TPS sweep).

================================================================
DESIGN
================================================================

- Text-patch on `vllm/v1/sample/rejection_sampler.py` inside the random
  sampling Triton kernel `rejection_random_sample_kernel`.
- The threshold is baked as a fp32 LITERAL at apply() time from env
  `GENESIS_P82_THRESHOLD_SINGLE` (default 0.3 — SGLang's typical default).
  Changing the threshold requires server restart.
- Greedy path is untouched (greedy already accepts on argmax-match;
  threshold doesn't apply to T=0).
- Synthetic mode is untouched (synthetic acceptance has its own rule).

================================================================
SAFETY MODEL
================================================================

- If env GENESIS_ENABLE_P82 is unset/0 → patch is SKIPPED, source stays
  vanilla. No runtime fall-through path needed.
- If anchor missing (upstream rewrote the line) → SKIPPED with clear
  reason; server boots on vanilla rule.
- Drift markers catch upstream's own threshold patch if/when it lands.

Status: opt-in via `GENESIS_ENABLE_P82=1`. Default OFF.

================================================================
P82 HAS NEVER APPLIED — READ BEFORE THE NEXT BOOT
================================================================

`GENESIS_ENABLE_P82=1` has been set in `compose/single/tcbench8021.yml`
(line 386), the dispatcher has logged `APPLY P82` every boot, and the patch
has written NOTHING, for two independent reasons — both fixed 2026-07-26:

  1. `sample_recovered_tokens_kernel` was listed as an upstream drift
     marker. It is the ORIGINAL vLLM kernel (since #14930), not evidence of
     anything, so the drift check fired on every image ever built and the
     boot log read "upstream may have absorbed this fix". Removed.
  2. The anchor was the pre-#45369 DIVISION form. The build line carries
     `01d50ae77 Cherry #45369`, which rewrote it to a multiplication, so the
     anchor was count == 0 on the boot pin. Re-anchored (see P82_ANCHOR_FORMS).

**Consequence: the first boot after this change is the first time P82 would
actually bias acceptance.** `GENESIS_ENABLE_P82=1` has been in the compose
the whole time the patch was inert, so it never encoded a decision to ship a
biased sampler — it encoded "this is broken and harmless". Turning years of
that into live behaviour on the next unattended restart is not a re-anchor,
it is a silent quality change, and this file's own SAFETY MODEL says the
OR-clause does not ship without `genesis_quality_harness.py` ≥ 30/31 plus a
TPS sweep.

So the re-anchor lands behind a SECOND, explicit acknowledgement:

    GENESIS_P82_ACK_UNVALIDATED=1

Without it P82 resolves and COUNTS its anchor (so drift is still detected and
still fails loud) and then declines to write, saying exactly that. With it,
P82 applies. One env var arms it; nothing about the fix is hidden behind it.

Tunable knobs
-------------
- `GENESIS_ENABLE_P82` (default unset/0): master switch
- `GENESIS_P82_ACK_UNVALIDATED` (default unset/0): required second gate —
  acknowledges that the biased OR-clause has never been validated on this rig
  because it has never once applied (see above)
- `GENESIS_P82_THRESHOLD_SINGLE` (default 0.3): float in [0.0, 1.0]
  - 0.0 → disables the OR clause (equivalent to OFF, but with overhead)
  - 0.2-0.3 → SGLang typical range, light bias
  - ≥0.5 → aggressive, expect quality regression on diverse outputs

Compatibility
-------------
- All draft methods (ngram, MTP/EAGLE, suffix) — affects only the
  acceptance comparison, not the draft generation.
- Cudagraph: unaffected (rejection sampler runs OUTSIDE the captured graph).
- P71 (block-verify): mutually exclusive in practice — P71 takes the
  block-verify branch BEFORE this point if eligible. P82 fires on the
  per-token fall-through path. Safe to enable both.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Reference algorithm: SGLang team (sgl-project/sglang).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p82_sglang_acceptance_threshold")


# NOTE: marker is built dynamically from the threshold so that operators
# changing GENESIS_P82_THRESHOLD_SINGLE between restarts cause apply() to
# re-patch (not silently skip on idempotency). See `_marker_for(threshold)`.
GENESIS_P82_MARKER_PREFIX = "Genesis P82 SGLang-style threshold_single OR-clause v7.63.x"


def _marker_for(threshold: float, min_draft_pos: int = 0) -> str:
    """Build the marker for a specific baked (threshold, min_draft_pos) tuple.

    v7.62.11 fix (B3 from hidden bug audit): previous marker was constant
    `"...v7.53"` regardless of `_BAKED_THRESHOLD`. Operator changes
    `GENESIS_P82_THRESHOLD_SINGLE` and restarts → marker check matches the
    OLD bake, returns IDEMPOTENT, **previously-baked threshold stays in
    source**. Threshold change silently ignored unless container fs reset.

    v7.63.x v2 (2026-04-30) extension: marker now also encodes
    `min_draft_pos`. Same forced-re-apply logic when EITHER value
    changes. Backward-compat: when `min_draft_pos == 0` the marker
    omits the `mdp=` segment entirely so existing v1 marker text in
    source is still recognized (no-op upgrade for current PROD users).
    """
    # Round to 4 decimals for marker stability (avoid 0.30000000000000004
    # vs 0.3 mismatches in apparently-equal env values)
    base = f"{GENESIS_P82_MARKER_PREFIX} thresh={float(threshold):.4f}"
    if min_draft_pos > 0:
        return f"{base} mdp={int(min_draft_pos)}"
    return base


# Back-compat alias (old constant name still imported by tests)
GENESIS_P82_MARKER = GENESIS_P82_MARKER_PREFIX


# ─── Threshold parsing (with bounds + fallback) ────────────────────────────

_DEFAULT_THRESHOLD = 0.3


def _read_threshold() -> float:
    raw = os.environ.get("GENESIS_P82_THRESHOLD_SINGLE", "").strip()
    if not raw:
        return _DEFAULT_THRESHOLD
    try:
        v = float(raw)
    except ValueError:
        log.warning(
            "[P82] GENESIS_P82_THRESHOLD_SINGLE=%r not parseable as float; using default %.2f",
            raw, _DEFAULT_THRESHOLD,
        )
        return _DEFAULT_THRESHOLD
    if not (0.0 <= v <= 1.0):
        log.warning(
            "[P82] threshold %.4f out of [0.0, 1.0]; clamping",
            v,
        )
        v = max(0.0, min(1.0, v))
    return v


# ─── v2: min draft-position guard (opt-in) ─────────────────────────────────

_DEFAULT_MIN_DRAFT_POS = 0  # 0 = current behavior (OR-clause fires at all positions)


def _read_min_draft_pos() -> int:
    """v2 (2026-04-30): operator can restrict the OR-clause to draft
    positions >= N. Earlier positions cascade-affect more output tokens,
    so biasing later positions is "safer" if quality drift is observed.

    Default 0 = current behavior (clause fires at every position).
    Recommended for ngram with low `prompt_lookup_min`: try =1 or =2 to
    reduce cascade impact while keeping the OR-clause's TPS win at later
    positions.

    Bounds: clamped to [0, MAX_SPEC_LEN-1] = [0, 127] so the clause can
    always fire on at least one position.
    """
    raw = os.environ.get("GENESIS_P82_MIN_DRAFT_POS", "").strip()
    if not raw:
        return _DEFAULT_MIN_DRAFT_POS
    try:
        v = int(raw)
    except ValueError:
        log.warning(
            "[P82] GENESIS_P82_MIN_DRAFT_POS=%r not parseable as int; "
            "using default %d", raw, _DEFAULT_MIN_DRAFT_POS,
        )
        return _DEFAULT_MIN_DRAFT_POS
    if v < 0:
        log.warning("[P82] min_draft_pos %d negative; clamping to 0", v)
        v = 0
    if v > 127:  # MAX_SPEC_LEN constant in upstream rejection_sampler
        log.warning(
            "[P82] min_draft_pos %d exceeds MAX_SPEC_LEN-1=127; clamping",
            v,
        )
        v = 127
    return v


# ─── v2: numerical-stability epsilon for draft_prob guard ──────────────────

# fp32 normal-range minimum is ~1e-38; we use 1e-20 as a safety margin
# that still covers ~99.999...% of realistic softmax outputs while
# guarding against denormal-zone instability in `target_prob / draft_prob`.
GENESIS_P82_DRAFT_PROB_EPS = 1e-20


# ─── Second gate: acknowledge the OR-clause has never been validated here ──

_ENV_ACK = "GENESIS_P82_ACK_UNVALIDATED"


def _ack_unvalidated() -> bool:
    return os.environ.get(_ENV_ACK, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _select_anchor_label(patcher) -> str | None:
    """Which anchor generation the built patcher ended up carrying."""
    for sp in patcher.sub_patches:
        for anchor, _v, label in P82_ANCHOR_FORMS:
            if sp.anchor == anchor:
                return label
    return None


# ─── Anchor: 3-line block including upstream NOTE comment for uniqueness ───
#
# TWO generations, newest first. Selected by content sniff in
# `_select_anchor()`; the chosen form is asserted count == 1 before use.
#
# v3 (2026-07-26 re-anchor): the build line carries `01d50ae77 Cherry #45369:
# avoid materializing target probs in spec rejection sampling`, which
# rewrote the acceptance test from a DIVISION to a MULTIPLICATION:
#     target_prob / draft_prob >= uniform_prob
#   → target_prob >= uniform_prob * draft_prob
# P82_OLD_V2 is therefore count == 0 on the boot pin
# dev1474cherrymax-1757-20260725 (counted in both the pristine image and the
# live post-boot container). P82 has never applied on this pin.
#
# The v3 vanilla clause keeps upstream's `draft_prob > 0` rather than v2's
# `>= 1e-20` tightening: that epsilon existed solely to keep the DIVISION out
# of the fp32 denormal zone, and #45369 removed the division. Re-introducing
# the epsilon on the multiply form would reject draft_prob in (0, 1e-20) for
# no numerical reason — a silent behaviour delta versus vanilla.

P82_OLD_V3 = (
    "                # NOTE(woosuk): While the draft probability should never be 0,\n"
    "                # we check it to avoid NaNs. If it happens to be 0, we reject.\n"
    "                accepted = draft_prob > 0 and target_prob >= uniform_prob * draft_prob\n"
)

P82_OLD_V2 = (
    "                # NOTE(woosuk): While the draft probability should never be 0,\n"
    "                # we check it to avoid NaNs. If it happens to be 0, we reject.\n"
    "                accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob\n"
)

# (anchor, vanilla-clause expression, generation label). Order = preference.
P82_ANCHOR_FORMS = (
    (P82_OLD_V3, "draft_prob > 0 and target_prob >= uniform_prob * draft_prob", "v3/#45369-multiply"),
    (P82_OLD_V2, "draft_prob >= {eps} and target_prob / draft_prob >= uniform_prob", "v2/pre-#45369-divide"),
)

# Kept for callers/tests that still import the old name.
P82_OLD = P82_OLD_V2


def _select_anchor(content: str) -> tuple[str, str, str] | None:
    """Pick the anchor generation present in `content`, COUNTED.

    Returns (anchor, vanilla_expr, label) or None. A form that appears more
    than once is refused outright rather than patched at an arbitrary site —
    counting is the whole point of this function.
    """
    for anchor, vanilla, label in P82_ANCHOR_FORMS:
        n = content.count(anchor)
        if n == 1:
            return anchor, vanilla, label
        if n > 1:
            log.error(
                "[P82] anchor form %s appears %d times — refusing to patch "
                "an ambiguous site", label, n,
            )
            return None
    return None


def _build_replacement(threshold: float, min_draft_pos: int = 0,
                       vanilla_expr: str | None = None) -> str:
    """Build the Triton-side replacement block.

    v2 improvements (2026-04-30):
    - Numerical-stability guard: replaces `draft_prob > 0` with
      `draft_prob >= 1e-20` to prevent fp32 denormal-zone overflow in
      `target_prob / draft_prob` (denormals can produce inf/NaN even
      though the value is "non-zero").
    - Defensive `target_prob > 0` check on the threshold-clause side:
      guards against malformed input where target softmax somehow
      produces a non-positive probability (impossible in practice but
      defensive).
    - `min_draft_pos` runtime guard: when `min_draft_pos > 0`, the
      OR-clause fires only on positions `pos >= min_draft_pos`. Earlier
      positions cascade-affect more output tokens; restricting the bias
      to later positions reduces quality drift while keeping the TPS
      win where it matters most. Default 0 = current behavior.

    All v2 changes preserve bit-equivalence with v1 when invoked with
    default args (threshold > 0, min_draft_pos = 0): the new guards
    only EXCLUDE acceptances v1 would have made on numerically-degenerate
    input, which is the desired safety direction.
    """
    # Bake threshold as a fp32-precision literal (Python repr of float is
    # round-trip safe, sufficient for Triton constexpr coercion).
    threshold_literal = repr(float(threshold))
    eps_literal = repr(float(GENESIS_P82_DRAFT_PROB_EPS))
    # The vanilla half must be whatever this pin's kernel actually says, so
    # P82 only ever ADDS the OR-clause and never silently reverts an upstream
    # rewrite of the rejection test itself.
    if vanilla_expr is None:
        vanilla_expr = P82_ANCHOR_FORMS[1][1]
    vanilla_clause = vanilla_expr.format(eps=eps_literal)
    eps_note = (
        f"                #   - draft_prob guard tightened: > 0  →  >= {eps_literal}\n"
        "                #     (fp32 denormal-zone protection; prevents inf/NaN ratio)\n"
        if "{eps}" in vanilla_expr or eps_literal in vanilla_clause else
        "                #   - vanilla clause taken verbatim from this pin's kernel\n"
        "                #     (upstream #45369 multiply form — no division to guard)\n"
    )

    # Build the threshold-clause guard. When min_draft_pos == 0 (default)
    # we omit the position guard entirely so the kernel disasm stays
    # identical to v1 — important because operators may have validated
    # P82 v1 on PROD and we don't want to introduce silent reordering.
    if min_draft_pos > 0:
        position_guard = f" and pos >= {min_draft_pos}"
        position_doc = (
            f"                # [Genesis P82 v2] OR-clause restricted to "
            f"draft pos >= {min_draft_pos} (min_draft_pos guard); earlier\n"
            f"                # positions use vanilla rule only. Cascade-impact "
            f"reduction.\n"
        )
    else:
        position_guard = ""
        position_doc = ""

    return (
        "                # NOTE(woosuk): While the draft probability should never be 0,\n"
        "                # we check it to avoid NaNs. If it happens to be 0, we reject.\n"
        "                # ════════════════════════════════════════════════════════════════\n"
        "                # [Genesis P82 v2 SGLang-style] threshold_single OR-clause\n"
        "                # accept if EITHER vanilla rejection passes OR target's confidence\n"
        "                # in the drafted token meets the configured threshold. Bias trade-off:\n"
        "                # loses unbiased-sampling guarantee; chosen for low-temp tool-call.\n"
        "                # Threshold baked from env GENESIS_P82_THRESHOLD_SINGLE at server start.\n"
        "                #\n"
        "                # v2 (2026-04-30) hardening:\n"
        + eps_note
        + "                #   - target_prob > 0 explicit (defensive vs malformed softmax)\n"
        + (f"                #   - min_draft_pos = {min_draft_pos} (OR-clause restricted)\n"
           if min_draft_pos > 0 else "")
        + "                # ════════════════════════════════════════════════════════════════\n"
        + position_doc
        + "                _genesis_p82_vanilla = (\n"
        f"                    {vanilla_clause}\n"
        "                )\n"
        f"                _genesis_p82_threshold = (\n"
        f"                    target_prob > 0 and target_prob >= {threshold_literal}{position_guard}\n"
        f"                )\n"
        "                accepted = _genesis_p82_vanilla or _genesis_p82_threshold\n"
    )


def _make_patcher(threshold: float, min_draft_pos: int = 0) -> TextPatcher | None:
    target = resolve_vllm_file("v1/sample/rejection_sampler.py")
    if target is None:
        return None
    try:
        with open(target, encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        log.error("[P82] cannot read %s: %s", target, e)
        return None
    picked = _select_anchor(content)
    if picked is None:
        log.error(
            "[P82] NO anchor generation matches %s — the acceptance test has "
            "been rewritten again. P82 is INERT this boot; re-derive the "
            "anchor before claiming the OR-clause is in force.", target,
        )
        return None
    anchor, vanilla_expr, label = picked
    log.info("[P82] anchor generation %s selected (count == 1)", label)
    return TextPatcher(
        patch_name=(
            "P82 v2 v1/sample/rejection_sampler.py — SGLang threshold_single "
            f"OR-clause (threshold={threshold:.4f}, min_draft_pos={min_draft_pos}, "
            f"anchor={label})"
        ),
        target_file=str(target),
        # B3 fix: marker now embeds threshold + min_draft_pos so a config
        # change forces re-apply instead of silently passing IDEMPOTENT.
        marker=_marker_for(threshold, min_draft_pos),
        sub_patches=[
            TextPatch(
                name="p82_threshold_or_clause",
                anchor=anchor,
                replacement=_build_replacement(threshold, min_draft_pos,
                                               vanilla_expr),
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P82",
            "_genesis_p82_threshold",
            # Upstream-side markers: if vLLM ever ships its own threshold_single
            # arg in this kernel, we should bow out and let upstream handle it.
            "threshold_single",
            "speculative-accept-threshold",
            # PR #40819 markers (block-wise verification rule, OPEN as of
            # 2026-04-29). #40819 inserts a separate `if use_block_verify:`
            # branch BEFORE the per-token kernel; control flow only reaches
            # P82's anchor when verify_method != "block", so the two are
            # complementary at runtime. We still drift-detect the merge so
            # we know upstream now has the canonical SGLang block-verify
            # rule for ≥3 spec tokens + real draft probs (P82 keeps the
            # OR-clause for ngram + short-spec paths). Authors:
            # masterFoad / z00918512.
            "use_block_verify",
            "verify_method",
            "_BLOCK_VERIFY_VOCAB_BLOCK",
            "_block_verify_kernel",
            # vllm/config/speculative.py marker for the new SpecVerifyMethod
            # field added by #40819:
            "SpecVerifyMethod",
            # PR #41258 (masterFoad): "Lazy recovery evaluation for spec
            # rejection sampling" — computes recovered tokens lazily inside
            # `rejection_random_sample_kernel` instead of the eager
            # full-vocab `sample_recovered_tokens_kernel` pass.
            #
            # [2026-07-26] `sample_recovered_tokens_kernel` USED TO BE LISTED
            # HERE and it is the single reason P82 has never applied on any
            # pin. It is not a signal that #41258 landed — it is the ORIGINAL
            # vLLM kernel, present since 99abb8b65 "[V1][Spec Decode]
            # Optimize Rejection Sampler with Triton Kernels (#14930)". A
            # drift marker whose presence means "nothing has changed" fires
            # on every image forever: the boot logged
            #   "upstream drift marker 'sample_recovered_tokens_kernel' ...
            #    upstream may have absorbed this fix"
            # while upstream had absorbed nothing. The kernel is still
            # present (count == 2) on the boot pin AFTER #41258's lazy path
            # landed as build-line cherry 81cdcb55e, which is the proof the
            # marker cannot discriminate. The real #41258 signals are the two
            # below; they stay.
            "_lazy_recovered_token",
            "lazy_recovery",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P82 — SGLang threshold_single OR-clause acceptance."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P82")
    log_decision("P82", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    threshold = _read_threshold()
    if threshold == 0.0:
        # Equivalent to OFF (OR-clause never fires) but with patch overhead;
        # explicitly skip to keep the source vanilla.
        return "skipped", (
            "GENESIS_P82_THRESHOLD_SINGLE=0.0 — OR clause would never fire; "
            "skipping patch to keep source vanilla"
        )

    # v2 (2026-04-30): explicit skip when threshold is 1.0 — operator UX.
    # `target_prob >= 1.0` only fires for argmax-tier confidence which is
    # essentially never. Patch overhead would add no value.
    if threshold >= 1.0:
        return "skipped", (
            "GENESIS_P82_THRESHOLD_SINGLE=1.0 — OR clause would only fire on "
            "100%-confident target prob (argmax cases). Effectively a no-op; "
            "skipping patch. Set threshold to 0.7-0.95 for meaningful TPS gain."
        )

    min_draft_pos = _read_min_draft_pos()

    patcher = _make_patcher(threshold, min_draft_pos)
    if patcher is None:
        # _make_patcher logs the discriminating reason (file missing vs no
        # anchor generation vs ambiguous anchor). Never report a bare
        # "not found" for an anchor problem — that is how P82 spent months
        # reading as an upstream absorption.
        return "skipped", (
            "P82 could not be built: rejection_sampler.py missing, or NO "
            "anchor generation matched (see the [P82] log line above). The "
            "OR-clause is NOT in force this boot."
        )

    # Second gate. Deliberately AFTER _make_patcher, so the anchor is still
    # resolved and counted (drift still fails loud) even when we decline to
    # write. See "P82 HAS NEVER APPLIED" in the module docstring.
    if not _ack_unvalidated():
        log.warning(
            "[P82] anchor RESOLVED and unique, but NOT writing: %s is unset. "
            "P82 has never applied on any pin, so its biased OR-clause is "
            "unvalidated on this rig; set %s=1 alongside GENESIS_ENABLE_P82=1 "
            "to arm it.", _ENV_ACK, _ENV_ACK,
        )
        return "skipped", (
            f"anchor resolved + counted, write withheld — {_ENV_ACK} not set. "
            f"P82 has NEVER applied on any pin (drift marker + stale anchor, "
            f"both fixed 2026-07-26), so enabling it is a first-ever, "
            f"unvalidated change to acceptance sampling. Set {_ENV_ACK}=1 to "
            f"arm the OR-clause."
        )

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P82] marker present (current threshold) — skip (idempotent)")
        return "applied", "idempotent (marker present, threshold unchanged)"
    # B3 fix: detect stale P82 marker from a different threshold bake.
    # If a different P82 prefix marker is present, the source has the OLD
    # threshold baked. We can't safely re-patch because the original anchor
    # is now consumed. Operator must `docker compose down && up -d` to reset
    # the container fs first. Surface this clearly instead of silent skip.
    if GENESIS_P82_MARKER_PREFIX in content:
        return (
            "skipped",
            f"P82 stale marker present (different threshold). Container fs has a previous "
            f"P82 bake; current threshold={threshold:.4f} cannot be applied without resetting "
            f"the source. Reset via `docker compose down && up -d` (NOT just stop/start)."
        )
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P82" and m in content:
            continue  # our marker; handled above
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed this fix or independent threshold patch",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: route SKIPPED to "skipped" (was masked as "applied")
    # via the centralized helper that already lives in text_patch.py.
    from vllm._genesis.wiring.text_patch import result_to_wiring_status
    pos_note = (
        f", min_draft_pos={min_draft_pos}"
        if min_draft_pos > 0
        else ""
    )
    # The hardening line must describe the generation that was actually baked.
    # On the #45369 multiply form there is no division and hence no denormal
    # guard, and printing one would be the same class of false log line that
    # kept P82 unread for months.
    anchor_gen = _select_anchor_label(patcher)
    hardening = (
        "vanilla clause taken verbatim from this pin's kernel "
        "(#45369 multiply form — no division, so no denormal guard), "
        "explicit target_prob > 0 check"
        if anchor_gen and anchor_gen.startswith("v3")
        else "v2 hardening: fp32 denormal guard (draft_prob >= 1e-20), "
             "explicit target_prob > 0 check"
    )
    applied_msg = (
        f"P82 applied [anchor={anchor_gen}]: SGLang threshold_single OR-clause "
        f"installed at threshold={threshold:.4f}{pos_note}. {hardening}"
        + (f", OR-clause restricted to draft pos >= {min_draft_pos}"
           if min_draft_pos > 0 else "")
        + ". Activates on random-sample path (greedy / synthetic untouched). "
        f"BIASED rule, armed via {_ENV_ACK}=1 — validate with "
        "genesis_quality_harness before prod."
    )
    return result_to_wiring_status(
        result, failure,
        applied_message=applied_msg,
        patch_name=patcher.patch_name,
    )
