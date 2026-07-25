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

and text-patches thinking_budget_state.py at three more (the ENFORCE ROUTE
CONSUMER, added 2026-07-25):
  E) module head    -> the two lazy, never-raising shims into the sidecar.
  F) sync_batch add loop -> h119_on_batch_add(): a request the CALLER left
     unbudgeted gets a PROVISIONAL deep-budget state entry, so the holder is
     tracking it by the time _make_sampling_metadata() reads
     has_tracked_requests(). An explicit caller budget always wins and is left
     entirely alone.
  G) update_state head   -> h119_resolve_routes(): rewrites the provisional
     budget to the routed one (deep/lean) once ROUTES has the decision, which
     is BEFORE the sampler that emits the request's first token.

Inert without GENESIS_ENABLE_H119_LENS_ROUTER=1 (maybe_create returns None and
sites B/C guard on the attribute). The E/F/G consumer needs its OWN flag on top
of that — GENESIS_ENABLE_H119_ROUTE_BUDGET=1, default OFF — plus the router in
PN119_MODE=enforce; with either missing the shims return immediately and the
holder behaves exactly as upstream. Anchors target nightly-0ba2aa35 /
club-dev1474-cherry; on older pins the anchors soft-skip (return 0) — H119
is a NEW capability, not a restore, so absence on old pins is declared-fine.
The two site groups are INDEPENDENT: the runner group (A-D) and the holder
group (E-G) each soft-skip on their own, so a drift in one never disables the
other, and a pin with no thinking_budget_state.py at all skips E-G cleanly.
"""
import pathlib
import shutil
import sys

LOG = "[h119-lens-router]"
VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
RUNNER = VLLM / "v1/worker/gpu_model_runner.py"
TBS = VLLM / "v1/sample/thinking_budget_state.py"
# NOT renamed (owned by the sidecar, see module docstring): the source file,
# the installed module name and the PN119Router class keep their PN119 spelling.
SIDECAR_SRC = pathlib.Path("/fixes/pn119_router.py")
SIDECAR_DST = VLLM / "_genesis_pn119.py"
MARKER = "# H119:"
TBS_MARKER = "# H119-BUDGET:"

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


# ═══════════════════════════════════════════════════════════════════════════
# Sites E/F/G — vllm/v1/sample/thinking_budget_state.py (the route CONSUMER)
# ═══════════════════════════════════════════════════════════════════════════
# Verified byte-identical on both pinned images (extracted with
# `podman run --rm --entrypoint cat ... thinking_budget_state.py`):
#   dev1060cherry-20260713          588 lines
#   dev1474cherry-1711-20260725     582 lines
# The two files differ ONLY inside update_state's spec-suffix strip (an old
# draft-suffix trim the newer pin dropped), which none of these anchors span —
# so E/F/G are literal on both pins and no content sniff is needed for them.
# The sniff that IS needed is existence + anchor uniqueness, both checked below,
# because a future pin may drift or drop the file entirely.

E_OLD = (
    "if TYPE_CHECKING:\n"
    "    from vllm.config.reasoning import ReasoningConfig\n"
)
E_NEW = (
    "if TYPE_CHECKING:\n"
    "    from vllm.config.reasoning import ReasoningConfig\n"
    "\n"
    "\n"
    "# H119-BUDGET: enforce-route consumer shims. The lens router publishes a\n"
    "# per-request deep/lean decision into vllm._genesis_pn119.ROUTES from the\n"
    "# SAME process and the SAME req_id namespace as this holder; these two\n"
    "# calls are the only place that decision is allowed to touch a request.\n"
    "# Resolution is cached once: False = sidecar absent, never retried, so a\n"
    "# boot without the H119 sidecar pays one ImportError total, not one per\n"
    "# sampling step. Neither shim can raise into sampling.\n"
    "_H119_MOD: Any = None\n"
    "\n"
    "\n"
    "def _h119() -> Any:\n"
    "    global _H119_MOD\n"
    "    if _H119_MOD is None:\n"
    "        try:\n"
    "            from vllm import _genesis_pn119 as _mod\n"
    "\n"
    "            _H119_MOD = _mod\n"
    "        except Exception:\n"
    "            _H119_MOD = False\n"
    "    return _H119_MOD or None\n"
    "\n"
    "\n"
    "def _h119_on_add(holder, index, params, prompt_tok_ids,\n"
    "                 output_tok_ids) -> bool:\n"
    "    \"\"\"True iff H119 installed a provisional entry (caller must not pop).\"\"\"\n"
    "    mod = _h119()\n"
    "    if mod is None:\n"
    "        return False\n"
    "    try:\n"
    "        return bool(mod.h119_on_batch_add(holder, index, params,\n"
    "                                          prompt_tok_ids, output_tok_ids))\n"
    "    except Exception:\n"
    "        return False\n"
    "\n"
    "\n"
    "def _h119_resolve(holder) -> None:\n"
    "    mod = _h119()\n"
    "    if mod is None:\n"
    "        return\n"
    "    try:\n"
    "        mod.h119_resolve_routes(holder)\n"
    "    except Exception:\n"
    "        pass\n"
)

# Site F — sync_batch's add loop. The ONLY change to stock semantics is that
# the unconditional `self._state.pop(index, None)` on the no-caller-budget path
# now runs only when H119 declined the row. With the flag off _h119_on_add()
# always returns False, so the pop always runs: behaviour identical to stock.
F_OLD = (
    "        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:\n"
    "            thinking_token_budget = params.thinking_token_budget\n"
    "            if thinking_token_budget is not None:\n"
    "                self._state[index] = self._init_state_entry(\n"
    "                    prompt_tok_ids, thinking_token_budget\n"
    "                )\n"
    "                self._state[index][\"output_tok_ids\"] = output_tok_ids\n"
    "                self._state[index][\"spec_token_ids\"] = []\n"
    "            else:\n"
    "                self._state.pop(index, None)\n"
)
F_NEW = (
    "        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:\n"
    "            thinking_token_budget = params.thinking_token_budget\n"
    "            if thinking_token_budget is not None:\n"
    "                self._state[index] = self._init_state_entry(\n"
    "                    prompt_tok_ids, thinking_token_budget\n"
    "                )\n"
    "                self._state[index][\"output_tok_ids\"] = output_tok_ids\n"
    "                self._state[index][\"spec_token_ids\"] = []\n"
    "                # H119-BUDGET: an EXPLICIT caller budget outranks the\n"
    "                # router unconditionally — this call only counts it.\n"
    "                _h119_on_add(self, index, params, prompt_tok_ids,\n"
    "                             output_tok_ids)\n"
    "            elif not _h119_on_add(self, index, params, prompt_tok_ids,\n"
    "                                  output_tok_ids):\n"
    "                # H119-BUDGET: unchanged stock path. H119 returns False\n"
    "                # whenever it is off, unavailable, or declined the row.\n"
    "                self._state.pop(index, None)\n"
)

# Site G — the head of update_state, which the sampler calls once per step.
# The resolve MUST sit above the `not self._state` guard's sibling so that it
# runs before _update_think_state consumes the budget on this very step; it is
# placed immediately after the guard because a provisional entry is itself a
# _state entry, so an empty _state has nothing to resolve by construction.
G_OLD = (
    "        \"\"\"Refresh output/spec from sampling rows and recompute think state.\"\"\"\n"
    "        if not self.is_enabled or not self._state:\n"
    "            return\n"
)
G_NEW = (
    "        \"\"\"Refresh output/spec from sampling rows and recompute think state.\"\"\"\n"
    "        if not self.is_enabled or not self._state:\n"
    "            return\n"
    "        # H119-BUDGET: promote provisional budgets to their routed value.\n"
    "        # The route was written into ROUTES at the end of the PREFILL step,\n"
    "        # i.e. earlier in this same engine step, and no token has been\n"
    "        # sampled for the request yet — so the cap binds from token 0.\n"
    "        _h119_resolve(self)\n"
)


def _patch_thinking_budget_state() -> str:
    """Apply sites E/F/G. Returns a one-line status for the caller to print.

    NEVER fails the boot: a missing file, a drifted anchor or an unwritable
    target all degrade to "consumer not wired", which leaves the holder exactly
    as upstream ships it. The consumer is a new capability, not a restore.
    """
    if not TBS.exists():
        return ("soft-skip E-G: thinking_budget_state.py absent on this pin — "
                "route consumer NOT wired (holder behaviour is upstream's)")
    try:
        text = TBS.read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return f"soft-skip E-G: unreadable ({e}) — route consumer NOT wired"
    if TBS_MARKER in text:
        return "E-G already applied (idempotent)"
    sites = (("E-shims", E_OLD, E_NEW), ("F-add", F_OLD, F_NEW),
             ("G-resolve", G_OLD, G_NEW))
    missing = [name for name, old, _ in sites if text.count(old) != 1]
    if missing:
        return (f"soft-skip E-G: anchor(s) {missing} not unique on this pin — "
                f"route consumer NOT wired (holder behaviour is upstream's)")
    for _name, old, new in sites:
        text = text.replace(old, new, 1)
    try:
        TBS.write_text(text, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return f"soft-skip E-G: write failed ({e}) — route consumer NOT wired"
    return ("applied 3/3 consumer sites (E=shims F=add G=resolve); inert "
            "unless GENESIS_ENABLE_H119_ROUTE_BUDGET=1 AND PN119_MODE=enforce")


def main() -> int:
    if not RUNNER.exists():
        print(f"{LOG} FATAL: {RUNNER} not present", file=sys.stderr)
        return 1
    try:
        shutil.copy2(SIDECAR_SRC, SIDECAR_DST)
    except OSError as e:
        print(f"{LOG} FATAL: sidecar install failed: {e}", file=sys.stderr)
        return 1
    # The holder group is patched independently of the runner group so a drift
    # in one never silently disables the other.
    print(f"{LOG} {_patch_thinking_budget_state()}")
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
