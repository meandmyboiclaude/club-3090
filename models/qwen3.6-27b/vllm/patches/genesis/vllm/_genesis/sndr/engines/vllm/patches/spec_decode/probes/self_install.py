# SPDX-License-Identifier: Apache-2.0
"""Shared exec-survival hook for lane-2 setattr patches.

Why this exists
---------------
`run_lane2()` is called from inside `apply_all`'s `main()`
(`patches/apply_all.py` -> `patches/sndr_lane.py:run_lane2`), which is the
standalone process the compose entrypoint then replaces with
`exec vllm serve`. `exec` REPLACES the process image, so every `setattr` made
during the patch pass is gone before the first token is served. Lane 2 is not
exempt from this: it is a function call at the tail of lane 1, not a
subprocess and not a plugin. Confirmed on the boot pin — no genesis/sndr entry
point in `vllm.general_plugins`, no dist-info, no `.pth`, stock
`sitecustomize`.

So a module whose `apply()` only rebinds an attribute announces success every
boot and reaches nothing. That is the P39a class (months lost) and the BUG-122
shape. The cure is the P103 / P39a one: TEXT-PATCH a hook into the END of the
target module, so every fresh import in every process re-runs the install.

Appending at the END is load-bearing. The hook imports a sndr module that in
turn imports the very module being patched; at end-of-module the class or
function it needs is already bound in the partially-initialised module object
in `sys.modules`, so the circular import resolves. A hook at the top would
not.

Wire this BEFORE enabling any of these
--------------------------------------
The 2026-07-26 exec-discard triage found these lane-2 rows to be setattr-only
with their gating flag currently UNSET. Nothing is lost while they are off —
which is exactly why none of them was given a hook. Turning one on without
wiring it first buys a `RESULT applied` line and no capability, so the flag
flip and the hook belong in the same change:

    PN72   spec_decode/pn72_frequency_ngram_drafter.py   NgramProposer.propose
    PN77   quantization/pn77_fp8_lm_head.py              (+ its kernels_legacy
           helper lm_head_fp8_compressor.py, which is imported, not dispatched)
    PN302  detection/pn302_model_profile_init.py         — doubly dead: its job
           is to stamp GENESIS_MODEL_* into os.environ, and apply_all is a
           separate process from the shell that execs `vllm serve`, so the
           stamps do not reach the server either way.
    PN352B moe/pn352b_marlin_moe_sum.py                  MarlinExpertsBase.moe_sum
           — its parked sibling PN352 is already a text patch; copy that shape.
           Wire it before the larger-M A/B its registry note asks for, or the
           A/B measures nothing.
    PN520  model_compat/qwen3_5/pn520_..._47058_revert.py
    SNDR_EAGLE3_AUX_HIDDEN_001  spec_decode/sndr_eagle3_aux_hidden_001.py

Not on the list and not fixable here: the lane-2 copies of the shared ids
P14 / P22 / P31 / P38 / P40 / P28(P73) / PN26 / PN61 / PN62 never run at all —
`sndr_lane.apply_policy()` hands those ids to lane 1 and injects
GENESIS_DISABLE_<bare>, so the boot reports them "explicitly disabled by
operator". A hook on the lane-2 copy would be dead code; the live form is
lane 1's, and so is the fix.

Usage
-----
    from ..probes.self_install import make_self_install_patcher

    def apply():
        patcher = make_self_install_patcher(
            target_rel="v1/sample/rejection_sampler.py",
            anchor=_TAIL_ANCHOR,
            patch_id="PN282",
            env_flag="SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC",
            install_module=__name__,
            marker="Genesis PN282 self-install hook (exec-survival) v1",
        )
"""
from __future__ import annotations

from typing import Any

_TRUTHY = ("1", "true", "yes", "on")


def _render_gate(low: str, flags: tuple[str, ...]) -> str:
    """The `if` line(s) of the hook's env gate.

    One flag renders byte-identically to the original single-flag form, so
    PN241/PN258/PN282's shipped hooks are unchanged. Two or more render an
    ALL-must-be-truthy genexp — needed when a patch is only lane-2's to
    install under a second condition the served process cannot otherwise
    reconstruct (PN65: `apply_policy` hands the shared id over on
    GENESIS_SNDR_OWNS_PN65, and its GENESIS_DISABLE_ injection lives only in
    the apply_all process's environ).
    """
    if len(flags) == 1:
        return (
            f"    if _genesis_{low}_os.environ.get(\n"
            f"        \"{flags[0]}\", \"\"\n"
            "    ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\"):\n"
        )
    names = ", ".join(f"\"{f}\"" for f in flags)
    return (
        f"    if all(\n"
        f"        _genesis_{low}_os.environ.get(_f, \"\").strip().lower()\n"
        "        in (\"1\", \"true\", \"yes\", \"on\")\n"
        f"        for _f in ({names},)\n"
        "    ):\n"
    )


def render_hook(patch_id: str, env_flag: str, install_module: str,
                marker: str, also_require: tuple[str, ...] = ()) -> str:
    """The block appended to the target module. Never raises on import."""
    low = patch_id.lower()
    return (
        "\n\n"
        "# ============================================================\n"
        f"# [{marker}]\n"
        "# ============================================================\n"
        f"# `apply_all` runs in its own process and the entrypoint then does\n"
        f"# `exec vllm serve`, which replaces it. A setattr made during the\n"
        f"# patch pass therefore never reaches the served process ({patch_id}\n"
        "# is lane-2, and lane 2 runs inside apply_all's own main() -- it is\n"
        "# not exempt). This module-import-time hook re-installs the wrapper\n"
        "# in EVERY process that imports this file, which is how the P103 /\n"
        "# P39a exec-survival pattern works.\n"
        "#\n"
        "# Placed at end-of-module on purpose: the import below reaches back\n"
        "# into this module, and by here its names are already bound in the\n"
        "# partially-initialised module object in sys.modules.\n"
        "#\n"
        "# Opt-in and fail-quiet: with the flag unset this costs one env read,\n"
        "# and any failure leaves the file importable. A probe must never be\n"
        "# able to stop the server from booting.\n"
        "try:\n"
        f"    import os as _genesis_{low}_os\n"
        + _render_gate(low, (env_flag,) + tuple(also_require)) +
        f"        from {install_module} import (\n"
        f"            install_runtime as _genesis_{low}_install,\n"
        f"        )\n"
        f"        _genesis_{low}_install()\n"
        "except Exception:  # noqa: BLE001\n"
        "    pass\n"
    )


def make_self_install_patcher(
    target_rel: str,
    anchor: str,
    patch_id: str,
    env_flag: str,
    install_module: str,
    marker: str,
    also_require: tuple[str, ...] = (),
) -> Any:
    """Build the TextPatcher, or None when the anchor is not UNIQUE.

    Counted, not merely present. `llm_base_proposer.py`'s bare tail line is
    count==2 and appending after the wrong one would drop the hook in the
    middle of the module, where the circular import has nothing to resolve
    against. A non-unique anchor is refused, never guessed at.
    """
    from sndr.engines.vllm.detection.guards import resolve_vllm_file
    from sndr.kernel.text_patch import TextPatch, TextPatcher

    target = resolve_vllm_file(target_rel)
    if target is None:
        return None
    try:
        content = open(str(target), encoding="utf-8").read()
    except OSError:
        return None
    if content.count(anchor) != 1:
        return None
    return TextPatcher(
        patch_name=f"{patch_id} {target_rel} — self-install hook",
        target_file=str(target),
        marker=marker,
        sub_patches=[
            TextPatch(
                name=f"{patch_id.lower()}_self_install",
                anchor=anchor,
                replacement=anchor + render_hook(
                    patch_id, env_flag, install_module, marker, also_require),
                required=True,
            ),
        ],
        upstream_drift_markers=[marker],
    )


def anchor_count(target_rel: str, anchor: str) -> int:
    """-1 when the target is unresolvable; else the anchor's count."""
    from sndr.engines.vllm.detection.guards import resolve_vllm_file
    target = resolve_vllm_file(target_rel)
    if target is None:
        return -1
    try:
        return open(str(target), encoding="utf-8").read().count(anchor)
    except OSError:
        return -1
