# SPDX-License-Identifier: Apache-2.0
"""PN282 — Spec-decode acceptance proxy metric (rejection_sample wrap).

Sibling of PN248 (acceptance log trace). PN282 wraps the same Python
orchestration function — ``vllm.v1.sample.rejection_sampler.rejection_sample``
— but instead of writing a heavyweight log file it increments three
Prometheus counters / gauge defined in
``sndr.observability.spec_decode_metrics``:

  sndr_spec_decode_accepted_per_call_total{k, profile}
  sndr_spec_decode_calls_total{profile}
  sndr_spec_decode_max_spec_len{profile}

The metric series register into the same ``prometheus_client.REGISTRY``
that vllm already exposes via its built-in /metrics endpoint, so no
new server is started and no gateway aggregation is needed — Grafana /
Prometheus scrape jobs that already pull ``vllm:*`` series pick up the
``sndr_*`` series automatically.

Coexistence with PN248:

  * Idempotency markers are independent (``_genesis_pn282_wrapped`` vs.
    ``_genesis_pn248_wrapped``); each wrap installs at most once.
  * Apply order is non-critical: both wraps fall through to the wrapped
    callable; double-wrap simply stacks layers. PN282 should preferably
    apply BEFORE PN248 so PN248's log-writer is the outer layer and any
    exception in the metric path doesn't lose the trace, but the inverse
    order is also safe (PN248 never raises into PN282 either).
  * No double-counting: each PN282 installation wraps exactly once.

Default OFF — opt in via:

    SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC=1

Legacy alias (warns once):

    GENESIS_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC=1

Author: Sandermage; PN282 / 2026-05-20.
"""
from __future__ import annotations

import logging

log = logging.getLogger("genesis.observability.pn282_spec_decode_acceptance_metric")

GENESIS_PN282_MARKER = "Genesis PN282 — spec-decode acceptance proxy metric"

_APPLIED = False
_ORIGINAL_REJECTION_SAMPLE = None


def _placeholder_token_id() -> int:
    try:
        from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
        return int(PLACEHOLDER_TOKEN_ID)
    except Exception:  # noqa: BLE001
        return -1


def install_runtime() -> tuple[str, str]:
    """Install the rejection_sample wrap. Idempotent.

    Returns:
        (status, reason) where status in {"applied", "skipped"}.
    """
    global _APPLIED, _ORIGINAL_REJECTION_SAMPLE

    from sndr.observability.spec_decode_metrics import is_enabled

    if _APPLIED:
        return "applied", "PN282 already installed (idempotent)"
    if not is_enabled():
        return "skipped", (
            "PN282 disabled (set SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC=1)"
        )

    try:
        from vllm.v1.sample import rejection_sampler as rs
    except ImportError as e:
        return "skipped", f"rejection_sampler not importable: {e}"

    original = rs.rejection_sample
    if getattr(original, "_genesis_pn282_wrapped", False):
        _APPLIED = True
        return "applied", "PN282 already wrapped (idempotent)"
    _ORIGINAL_REJECTION_SAMPLE = original

    placeholder = _placeholder_token_id()

    import inspect as _inspect
    try:
        _orig_sig = _inspect.signature(original)
    except (ValueError, TypeError):
        _orig_sig = None

    def wrapped(*args, **kwargs):
        # Forward TRANSPARENTLY. rejection_sample's signature drifts across
        # pins (dev491 added use_fp64_gumbel after synthetic_conditional_rates
        # per vllm#43150; more kwargs may follow). Passing *args/**kwargs
        # verbatim is forward-proof — this metric is a pure side-channel and
        # must NEVER alter the forwarded call. max_spec_len is read back via
        # signature binding only. 2026-06-16 dev491 drift fix.
        result = original(*args, **kwargs)

        try:
            # max_spec_len is read back ONLY to update the gauge for
            # dashboard awareness; it is read best-effort and MUST NOT
            # gate the counter emission. The accepted-per-request counts
            # are derived purely from ``result`` + the placeholder
            # constant, so they stay correct even when the bound
            # signature does not surface ``max_spec_len`` by name (e.g.
            # a *args-only wrapper, or a pin where the param was
            # renamed/reordered such that bind() can't resolve it). When
            # we cannot resolve it, default to 0 and still emit counters.
            max_spec_len = 0
            if _orig_sig is not None:
                try:
                    _bound = _orig_sig.bind(*args, **kwargs)
                    _bound.apply_defaults()
                    max_spec_len = _bound.arguments.get("max_spec_len", 0)
                except (TypeError, ValueError):
                    max_spec_len = 0
            # result shape: [B, max_spec_len + 1]
            # row[0] = bonus / recovered token
            # row[1:] = accepted draft IDs or PLACEHOLDER for rejected
            rows = result.detach().cpu().tolist()
            accepted_per_req = [
                sum(1 for x in row[1:] if x != placeholder)
                for row in rows
            ]
            from sndr.observability.spec_decode_metrics import (
                record_acceptance,
            )
            record_acceptance(accepted_per_req, int(max_spec_len))
        except Exception as e:  # noqa: BLE001
            log.debug(
                "[PN282] metric emission suppressed: %s",
                e,
            )

        return result

    wrapped._genesis_pn282_wrapped = True  # type: ignore[attr-defined]
    rs.rejection_sample = wrapped
    _APPLIED = True
    log.info(
        "[PN282] rejection_sample wrapped — sndr_spec_decode_* "
        "metrics live on /metrics."
    )
    return "applied", "PN282 acceptance metric installed"


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Test-only: restore the original rejection_sample.

    Returns True if a wrap was removed, False if no wrap was present.
    """
    global _APPLIED, _ORIGINAL_REJECTION_SAMPLE
    if not _APPLIED or _ORIGINAL_REJECTION_SAMPLE is None:
        return False
    try:
        from vllm.v1.sample import rejection_sampler as rs
        rs.rejection_sample = _ORIGINAL_REJECTION_SAMPLE
    except Exception:  # noqa: BLE001
        return False
    _APPLIED = False
    _ORIGINAL_REJECTION_SAMPLE = None
    return True


__all__ = ["apply", "is_applied", "revert", "GENESIS_PN282_MARKER"]


# ── exec-survival (2026-07-26) ────────────────────────────────────────────
# PN282 rebinds rejection_sample with a plain setattr, and `apply_all` -- which is where
# lane 2 runs, inside its main() -- is the process the entrypoint replaces
# with `exec vllm serve`. So the setattr alone reaches nothing: the row would
# announce APPLY on every boot and the served process would never see the
# wrapper. This is the P39a class, and wiring a registry row without fixing it
# would just add another announce-and-do-nothing entry.
#
# PN282_SELF_INSTALL_ANCHOR is the END of v1/sample/rejection_sampler.py, checked for count==1
# against POST-BOOT bytes (siblings rewrite these files, so pristine-image
# counts answer the wrong question). The bare tail line of
# llm_base_proposer.py is count==2, which is why the anchor carries the line
# above it.
from sndr.engines.vllm.patches.spec_decode.probes.self_install import (
    make_self_install_patcher,
)

PN282_SELF_INSTALL_TARGET = "v1/sample/rejection_sampler.py"
PN282_SELF_INSTALL_MARKER = (
    "Genesis PN282 self-install hook (exec-survival, P39a class) v1"
)
PN282_SELF_INSTALL_ANCHOR = (
    "    recovered_id = tl.minimum(recovered_id, vocab_size - 1)\n"
    "    tl.store(output_token_ids_ptr + token_idx, recovered_id)\n"
)


def install_self_install_hook() -> tuple[str, str]:
    """Text-patch the import-time hook so the wrapper survives `exec`."""
    patcher = make_self_install_patcher(
        target_rel=PN282_SELF_INSTALL_TARGET,
        anchor=PN282_SELF_INSTALL_ANCHOR,
        patch_id="PN282",
        env_flag="SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC",
        install_module="sndr.engines.vllm.patches.observability.pn282_spec_decode_acceptance_metric",
        marker=PN282_SELF_INSTALL_MARKER,
    )
    if patcher is None:
        return "skipped", (
            "PN282 self-install: v1/sample/rejection_sampler.py unresolvable or the tail anchor is "
            "not uniquely present — refusing to guess at the insertion point"
        )
    from sndr.kernel.text_patch import TextPatchResult

    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return "failed", (failure.reason if failure else "unknown failure")
    if result == TextPatchResult.SKIPPED:
        return "skipped", (failure.reason if failure else "anchor drift")
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN282 self-install hook already present"
    return "applied", (
        "PN282 self-install hook written into v1/sample/rejection_sampler.py — the wrapper is "
        "re-installed on every fresh import, so it survives `exec vllm serve`"
    )


def apply() -> tuple[str, str]:
    """Dispatcher entry: install for THIS process, then make it survive `exec`.

    Both halves, always. The setattr half is what a `--dry-run`/in-process
    caller sees; the text half is the only one the served process will ever
    see, because the entrypoint replaces this process. Reporting "applied" off
    the setattr alone is the announce-and-do-nothing shape the ledger exists
    to catch, so a failure of the text half is surfaced in the reason string
    rather than swallowed.
    """
    status, reason = install_runtime()
    if status == "skipped":
        return status, reason
    hook_status, hook_reason = install_self_install_hook()
    if hook_status != "applied":
        return "failed", (
            f"{reason} IN THIS PROCESS ONLY — the exec-survival hook did not "
            f"land ({hook_reason}), so the served process will not see "
            f"PN282"
        )
    return "applied", f"{reason}; {hook_reason}"
