#!/usr/bin/env python3
"""H119 — lens-router capture tap wiring (boot patcher).

RENAMED 2026-07-25 (patch-id collision audit): this patch was called "PN119".
That id was ALSO lane-2/sndr's `PN119` — an entirely unrelated TurboQuant k8v4
GQA head-grouping kernel (vllm#40792,
sndr/engines/vllm/patches/attention/turboquant/pn119_tq_gqa_grouping.py, enabled
by GENESIS_ENABLE_PN119=1 in five composes). One identifier, two subsystems; it
mis-routed real work on 2026-07-25. The lens router is the newer/shallower of
the two, so IT moved: house lane id `H119` (H = house lane, per
~/shared/PATCH-ID-COLLISION-AUDIT-20260725.md §2). The TurboQuant kernel keeps
`PN119` unchanged.

Env flags:
  GENESIS_ENABLE_H119_LENS_ROUTER  — canonical, use this.
  GENESIS_ENABLE_PN119_ROUTER      — BACKWARD-COMPATIBLE ALIAS, still honored.
The sidecar (/fixes/pn119_router.py) still reads the PN119_ROUTER name; the
compose entrypoint resolves the canonical name onto it:
    export GENESIS_ENABLE_PN119_ROUTER="${GENESIS_ENABLE_PN119_ROUTER:-\\
                                         ${GENESIS_ENABLE_H119_LENS_ROUTER:-}}"
so an operator's existing compose keeps working untouched. The sidecar module
name (vllm/_genesis_pn119.py), its class (PN119Router) and its PN119_* tuning
knobs are deliberately NOT renamed here — see the audit §3.1 "deliberately NOT
touched".

Installs /fixes/pn119_router.py as vllm/_genesis_pn119.py and text-patches
gpu_model_runner.py at three sites:
  A) load_model tail  -> PN119Router.maybe_create(self) (enables aux hidden
     states for layers (42,47,51) BEFORE compile/cudagraph capture).
  B) execute_model postprocess (aux unpack site) -> router.observe(...).
  C) _update_states finished-request removal -> router.on_finish(...)
     (the v2 self-training sink's generated-token label line).

Inert without GENESIS_ENABLE_H119_LENS_ROUTER=1 (maybe_create returns None and
sites B/C guard on the attribute). Anchors target nightly-0ba2aa35 /
club-dev1474-cherry; on older pins the anchors soft-skip (return 0) — H119
is a NEW capability, not a restore, so absence on old pins is declared-fine.
"""
import pathlib
import shutil
import sys

LOG = "[h119-lens-router]"
VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
RUNNER = VLLM / "v1/worker/gpu_model_runner.py"
# NOT renamed (owned by the sidecar, see module docstring): the source file,
# the installed module name and the PN119Router class keep their PN119 spelling.
SIDECAR_SRC = pathlib.Path("/fixes/pn119_router.py")
SIDECAR_DST = VLLM / "_genesis_pn119.py"
MARKER = "# H119:"

A_OLD = (
    "        if not load_dummy_weights:\n"
    "            prepare_communication_buffer_for_model(self.model)\n"
)
A_NEW = (
    "        # H119: lens-router init — must run BEFORE compile/cudagraph\n"
    "        # capture so the aux-hidden-state model output shape is baked in.\n"
    "        try:\n"
    "            from vllm._genesis_pn119 import PN119Router as _H119Router\n"
    "            self._h119 = _H119Router.maybe_create(self)\n"
    "        except Exception as _h119_e:\n"
    "            import logging as _h119_logging\n"
    "            _h119_logging.getLogger(\"genesis.h119\").warning(\n"
    "                \"[H119] wiring failed: %s — router disabled\", _h119_e)\n"
    "            self._h119 = None\n"
    "        if not load_dummy_weights:\n"
    "            prepare_communication_buffer_for_model(self.model)\n"
)

B_OLD = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                # True when EAGLE 3 is used.\n"
    "                hidden_states, aux_hidden_states = model_output\n"
    "            else:\n"
    "                # Common case.\n"
    "                hidden_states = model_output\n"
    "                aux_hidden_states = None\n"
)
B_NEW = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                # True when EAGLE 3 is used.\n"
    "                hidden_states, aux_hidden_states = model_output\n"
    "            elif isinstance(model_output, tuple):\n"
    "                # H119: aux capture active WITHOUT the eagle3 flag — the\n"
    "                # model returns (hidden, aux); drafter paths stay flag-off\n"
    "                # stock (feeding aux concat to MTP crashes the proposer).\n"
    "                hidden_states, aux_hidden_states = model_output\n"
    "            else:\n"
    "                # Common case.\n"
    "                hidden_states = model_output\n"
    "                aux_hidden_states = None\n"
    "            # H119: per-request prefill pooling + score (shadow/enforce).\n"
    "            _h119 = getattr(self, \"_h119\", None)\n"
    "            if _h119 is not None and aux_hidden_states is not None:\n"
    "                _h119.observe(scheduler_output, aux_hidden_states)\n"
)

C_OLD = (
    "        for req_id in scheduler_output.finished_req_ids:\n"
    "            req_state = self.requests.pop(req_id, None)\n"
    "            self._on_request_state_removed(req_id, req_state)\n"
)
C_NEW = (
    "        for req_id in scheduler_output.finished_req_ids:\n"
    "            req_state = self.requests.pop(req_id, None)\n"
    "            # H119: v2 sink — generated-token label line at finish.\n"
    "            _h119 = getattr(self, \"_h119\", None)\n"
    "            if _h119 is not None:\n"
    "                _h119.on_finish(req_id, req_state)\n"
    "            self._on_request_state_removed(req_id, req_state)\n"
)

# Site D — the dummy-run/profiling unpack must also tolerate the tuple
# (capture + memory profiling run through _dummy_run).
D_OLD = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                hidden_states, _ = outputs\n"
    "            else:\n"
    "                hidden_states = outputs\n"
)
D_NEW = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                hidden_states, _ = outputs\n"
    "            elif isinstance(outputs, tuple):\n"
    "                # H119: aux capture active without the eagle3 flag.\n"
    "                hidden_states, _ = outputs\n"
    "            else:\n"
    "                hidden_states = outputs\n"
)


def main() -> int:
    if not RUNNER.exists():
        print(f"{LOG} FATAL: {RUNNER} not present", file=sys.stderr)
        return 1
    try:
        shutil.copy2(SIDECAR_SRC, SIDECAR_DST)
    except OSError as e:
        print(f"{LOG} FATAL: sidecar install failed: {e}", file=sys.stderr)
        return 1
    text = RUNNER.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"{LOG} already applied (idempotent)")
        return 0
    # ALL-OR-NOTHING with soft failure: H119 is a NEW capability, never a
    # restore — on ANY anchor mismatch we ship the file UNTOUCHED and return 0
    # (the entrypoint runs under `set -e`; a hard failure would brick boots on
    # pins whose runner drifted). A and C without B would be inert-but-harmless;
    # B without A would crash at runtime — hence all-or-nothing.
    sites = (("A-load", A_OLD, A_NEW), ("B-observe", B_OLD, B_NEW),
             ("C-finish", C_OLD, C_NEW), ("D-dummy", D_OLD, D_NEW))
    missing = [name for name, old, _ in sites if text.count(old) != 1]
    if missing:
        print(f"{LOG} soft-skip: anchor(s) {missing} not unique on this pin — "
              f"router NOT wired (new capability; serving unaffected)",
              file=sys.stderr)
        return 0
    for _name, old, new in sites:
        text = text.replace(old, new, 1)
    RUNNER.write_text(text, encoding="utf-8")
    print(f"{LOG} applied 4/4 sites (A=init B=observe C=finish D=dummy); "
          f"inert unless GENESIS_ENABLE_H119_LENS_ROUTER=1 "
          f"(alias: GENESIS_ENABLE_PN119_ROUTER=1)")
    return 0


sys.exit(main())
