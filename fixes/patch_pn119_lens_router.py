#!/usr/bin/env python3
"""PN119 — lens-router capture tap wiring (boot patcher).

Installs /fixes/pn119_router.py as vllm/_genesis_pn119.py and text-patches
gpu_model_runner.py at three sites:
  A) load_model tail  -> PN119Router.maybe_create(self) (enables aux hidden
     states for layers (42,47,51) BEFORE compile/cudagraph capture).
  B) execute_model postprocess (aux unpack site) -> router.observe(...).
  C) _update_states finished-request removal -> router.on_finish(...)
     (the v2 self-training sink's generated-token label line).

Inert without GENESIS_ENABLE_PN119_ROUTER=1 (maybe_create returns None and
sites B/C guard on the attribute). Anchors target nightly-0ba2aa35 /
club-dev1474-cherry; on older pins the anchors soft-skip (return 0) — PN119
is a NEW capability, not a restore, so absence on old pins is declared-fine.
"""
import pathlib
import shutil
import sys

LOG = "[pn119-lens-router]"
VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
RUNNER = VLLM / "v1/worker/gpu_model_runner.py"
SIDECAR_SRC = pathlib.Path("/fixes/pn119_router.py")
SIDECAR_DST = VLLM / "_genesis_pn119.py"
MARKER = "# PN119:"

A_OLD = (
    "        if not load_dummy_weights:\n"
    "            prepare_communication_buffer_for_model(self.model)\n"
)
A_NEW = (
    "        # PN119: lens-router init — must run BEFORE compile/cudagraph\n"
    "        # capture so the aux-hidden-state model output shape is baked in.\n"
    "        try:\n"
    "            from vllm._genesis_pn119 import PN119Router as _PN119Router\n"
    "            self._pn119 = _PN119Router.maybe_create(self)\n"
    "        except Exception as _pn119_e:\n"
    "            import logging as _pn119_logging\n"
    "            _pn119_logging.getLogger(\"genesis.pn119\").warning(\n"
    "                \"[PN119] wiring failed: %s — router disabled\", _pn119_e)\n"
    "            self._pn119 = None\n"
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
    "            else:\n"
    "                # Common case.\n"
    "                hidden_states = model_output\n"
    "                aux_hidden_states = None\n"
    "            # PN119: per-request prefill pooling + score (shadow/enforce).\n"
    "            _pn119 = getattr(self, \"_pn119\", None)\n"
    "            if _pn119 is not None and aux_hidden_states is not None:\n"
    "                _pn119.observe(scheduler_output, aux_hidden_states)\n"
)

C_OLD = (
    "        for req_id in scheduler_output.finished_req_ids:\n"
    "            req_state = self.requests.pop(req_id, None)\n"
    "            self._on_request_state_removed(req_id, req_state)\n"
)
C_NEW = (
    "        for req_id in scheduler_output.finished_req_ids:\n"
    "            req_state = self.requests.pop(req_id, None)\n"
    "            # PN119: v2 sink — generated-token label line at finish.\n"
    "            _pn119 = getattr(self, \"_pn119\", None)\n"
    "            if _pn119 is not None:\n"
    "                _pn119.on_finish(req_id, req_state)\n"
    "            self._on_request_state_removed(req_id, req_state)\n"
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
    # ALL-OR-NOTHING with soft failure: PN119 is a NEW capability, never a
    # restore — on ANY anchor mismatch we ship the file UNTOUCHED and return 0
    # (the entrypoint runs under `set -e`; a hard failure would brick boots on
    # pins whose runner drifted). A and C without B would be inert-but-harmless;
    # B without A would crash at runtime — hence all-or-nothing.
    sites = (("A-load", A_OLD, A_NEW), ("B-observe", B_OLD, B_NEW),
             ("C-finish", C_OLD, C_NEW))
    missing = [name for name, old, _ in sites if text.count(old) != 1]
    if missing:
        print(f"{LOG} soft-skip: anchor(s) {missing} not unique on this pin — "
              f"router NOT wired (new capability; serving unaffected)",
              file=sys.stderr)
        return 0
    for _name, old, new in sites:
        text = text.replace(old, new, 1)
    RUNNER.write_text(text, encoding="utf-8")
    print(f"{LOG} applied 3/3 sites (A=init B=observe C=finish-sink); "
          f"inert unless GENESIS_ENABLE_PN119_ROUTER=1")
    return 0


sys.exit(main())
