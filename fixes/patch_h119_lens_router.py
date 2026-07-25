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

A SOFT-SKIP IS NOT SILENT. If the operator set GENESIS_ENABLE_H119_ROUTE_BUDGET
=1 and either group fails to install, this prints a boxed ERROR to stderr naming
the exact anchor — the 07-25 no-op arm cost a whole GPQA-30 because the only
trace was one INFO line. Verify anchors WITHOUT a boot (they are counted against
post-genesis-patch content, which is the only content that matters) with:
    python3 fixes/verify_h119_consumer_anchors.py
"""
import logging
import os
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
# ANCHORS ARE VALIDATED AGAINST BOOT-TIME CONTENT, NOT THE PRISTINE IMAGE.
# The first cut of these three sites counted 1 each in the file as extracted
# with `podman run --rm --entrypoint cat`, and still soft-skipped at boot with
# `anchor(s) ['F-add'] not unique on this pin` — a silent no-op (a GPQA-30 with
# the consumer "on" came back byte-identical to the run with it off). Two
# separate reasons, both now covered:
#   1. FIVE other genesis patches rewrite this file earlier in the same
#      entrypoint (pn108 -> pn112 -> pr44812 -> holder_syncbatch -> pn114), so
#      pristine content is the wrong thing to count against.
#   2. The pin the compose actually runs — dev1474cherrymax-1757-20260725 (634
#      lines) — was never checked; only the two older pins were. Upstream added
#      `relaxed_thinking` there and restructured sync_batch's add loop, which is
#      what actually zeroed F.
# fixes/verify_h119_consumer_anchors.py replays those five patches onto each
# pinned image's file and reports the counts the BOOT sees. Current state:
#   dev1474cherrymax-1757-20260725  634 -> 951 lines post-patch  (relaxed loop)
#   dev1474cherry-1711-20260725     582 -> 899 lines post-patch  (legacy loop)
#   dev1060cherry-20260713          588 -> 905 lines post-patch  (legacy loop)
# E and G are literal on all three. F is NOT: the add loop has two upstream
# shapes, so site F is a VARIANT SET, content-sniffed by "which variant's
# anchors count exactly 1" — the sniff that survives a pin we have not seen.

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
#
# F comes in TWO variants because the add loop has two upstream shapes:
#   F-add/legacy   — `if thinking_token_budget is not None: ... else: pop`
#                    (dev1060cherry-20260713, dev1474cherry-1711-20260725)
#   F-add/relaxed  — upstream inverted the test and added the relaxed_thinking
#                    -1-sentinel branch (dev1474cherrymax-1757-20260725, the pin
#                    the compose runs). The legacy anchor counts ZERO here,
#                    which is the soft-skip that shipped a silent no-op.
# The variant is chosen by counting, not by tag matching, so an unseen pin that
# carries either shape still wires; one that carries neither soft-skips LOUDLY.
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

# ── F variant "relaxed" — the add loop as the 1757 pin ships it ────────────
# Upstream shape there:
#     if thinking_token_budget is None:
#         if not self.relaxed_thinking:
#             <pop>; continue                 <- the untracked path we take over
#         <relaxed -1 sentinel entry>
#     else:
#         <budgeted entry>
#     <common output/spec tail>
# Two disjoint edits, both anchored on code (no comment text), each unique:
#   F1 inserts the H119 attempt as the FIRST thing on the unbudgeted path. On
#      True the shim has already installed a complete provisional entry
#      (output_tok_ids + spec_token_ids included, see pn119_router
#      h119_on_batch_add), so `continue` is exactly right — and the
#      `index in self._state` belt-and-braces means a shim that returned True
#      without installing anything falls through to stock instead of raising.
#      On False both stock branches are reached byte-for-byte as upstream.
#   F2 adds the caller-explicit COUNTING call, the mirror of the legacy
#      variant's first _h119_on_add — it never overrides the caller's budget.
F_REL1_OLD = (
    "            if thinking_token_budget is None:\n"
    "                if not self.relaxed_thinking:\n"
)
F_REL1_NEW = (
    "            if thinking_token_budget is None:\n"
    "                # H119-BUDGET: a request the CALLER left unbudgeted gets a\n"
    "                # PROVISIONAL deep-budget entry instead of being dropped, so\n"
    "                # the holder is already tracking it by the time\n"
    "                # _make_sampling_metadata() reads has_tracked_requests().\n"
    "                # Returns False whenever H119 is off, absent or declined —\n"
    "                # and then both stock branches below run untouched.\n"
    "                if _h119_on_add(self, index, params, prompt_tok_ids,\n"
    "                                output_tok_ids) and index in self._state:\n"
    "                    continue\n"
    "                if not self.relaxed_thinking:\n"
)
F_REL2_OLD = (
    "            else:\n"
    "                self._state[index] = self._init_state_entry(\n"
    "                    prompt_tok_ids, thinking_token_budget\n"
    "                )\n"
)
F_REL2_NEW = (
    "            else:\n"
    "                self._state[index] = self._init_state_entry(\n"
    "                    prompt_tok_ids, thinking_token_budget\n"
    "                )\n"
    "                # H119-BUDGET: an EXPLICIT caller budget outranks the\n"
    "                # router unconditionally — this call only counts it.\n"
    "                _h119_on_add(self, index, params, prompt_tok_ids,\n"
    "                             output_tok_ids)\n"
)

# Ordered variant set for site F. First entry whose EVERY (old, new) pair
# counts exactly 1 wins; content sniff, not tag matching.
F_VARIANTS = (
    ("F-add/relaxed", ((F_REL1_OLD, F_REL1_NEW), (F_REL2_OLD, F_REL2_NEW))),
    ("F-add/legacy", ((F_OLD, F_NEW),)),
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


def _route_budget_requested() -> bool:
    """True iff the operator ASKED for the consumer on this boot."""
    v = os.environ.get("GENESIS_ENABLE_H119_ROUTE_BUDGET", "")
    return v.strip().lower() in ("1", "true", "yes", "on")


def _resolve_consumer_sites(text: str):
    """Pick the E/F/G edit list for `text`, or explain why none fits.

    Returns (sites, problem). `sites` is a list of (name, old, new) with every
    anchor counted exactly once against THIS text; `problem` is None on success
    and a human diagnosis naming the offending anchor(s) otherwise.
    """
    counts = {"E-shims": text.count(E_OLD), "G-resolve": text.count(G_OLD)}
    fixed_bad = {n: c for n, c in counts.items() if c != 1}
    sites = [("E-shims", E_OLD, E_NEW)]
    f_detail = []
    for vname, pairs in F_VARIANTS:
        vcounts = [text.count(old) for old, _ in pairs]
        if all(c == 1 for c in vcounts):
            sites.extend((f"{vname}[{i}]", old, new)
                         for i, (old, new) in enumerate(pairs))
            break
        f_detail.append(f"{vname}={vcounts}")
    else:
        fixed_bad["F-add"] = "no variant matched (" + ", ".join(f_detail) + ")"
    sites.append(("G-resolve", G_OLD, G_NEW))
    if fixed_bad:
        return None, ", ".join(f"{n} count={c}" for n, c in fixed_bad.items())
    return sites, None


def _patch_thinking_budget_state() -> tuple[str, bool]:
    """Apply sites E/F/G. Returns (one-line status, installed?).

    NEVER fails the boot: a missing file, a drifted anchor or an unwritable
    target all degrade to "consumer not wired", which leaves the holder exactly
    as upstream ships it. The consumer is a new capability, not a restore.
    The BOOLEAN is what makes that honest — main() escalates a non-install to
    stderr/ERROR when the operator actually asked for the consumer, because a
    soft-skip buried in INFO is indistinguishable from a working no-op arm
    (2026-07-25: a GPQA-30 "with the consumer on" was byte-identical to the
    run with it off, and the only trace was one INFO line).
    """
    if not TBS.exists():
        return ("soft-skip E-G: thinking_budget_state.py absent on this pin — "
                "route consumer NOT wired (holder behaviour is upstream's)",
                False)
    try:
        text = TBS.read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return (f"soft-skip E-G: unreadable ({e}) — route consumer NOT wired",
                False)
    if TBS_MARKER in text:
        return ("E-G already applied (idempotent)", True)
    sites, problem = _resolve_consumer_sites(text)
    if problem:
        return (f"soft-skip E-G: {problem} — anchors do not fit this pin's "
                f"thinking_budget_state.py AS PATCHED BY THE EARLIER GENESIS "
                f"PATCHES; route consumer NOT wired (holder behaviour is "
                f"upstream's)", False)
    applied = []
    for name, old, new in sites:
        text = text.replace(old, new, 1)
        applied.append(name)
    try:
        TBS.write_text(text, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return (f"soft-skip E-G: write failed ({e}) — route consumer NOT wired",
                False)
    return (f"applied consumer sites {applied}; inert unless "
            f"GENESIS_ENABLE_H119_ROUTE_BUDGET=1 AND PN119_MODE=enforce", True)


def _shout(what: str, detail: str) -> None:
    """A requested-but-not-installed site group must never read as INFO."""
    bar = "=" * 72
    msg = (f"{bar}\n"
           f"{LOG} ERROR: {what} was REQUESTED but is NOT INSTALLED.\n"
           f"{LOG} ERROR: {detail}\n"
           f"{LOG} ERROR: this boot will behave EXACTLY as if the flag were "
           f"off — any A/B against it measures nothing.\n"
           f"{LOG} ERROR: re-derive the anchors against POST-PATCH content: "
           f"python3 /fixes/verify_h119_consumer_anchors.py\n"
           f"{bar}")
    print(msg, file=sys.stderr, flush=True)
    try:
        logging.getLogger("genesis.h119").error(msg.replace("\n", " | "))
    except Exception:  # noqa: BLE001 — logging must never break a boot
        pass


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
    tbs_status, tbs_ok = _patch_thinking_budget_state()
    print(f"{LOG} {tbs_status}")
    if not tbs_ok and _route_budget_requested():
        _shout("GENESIS_ENABLE_H119_ROUTE_BUDGET=1 (sites E/F/G)", tbs_status)
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
        # The consumer reads decisions the RUNNER group publishes; without A-D
        # there are no decisions, so a requested consumer is a no-op too.
        if _route_budget_requested():
            _shout("GENESIS_ENABLE_H119_ROUTE_BUDGET=1 (runner sites A-D)",
                   f"anchor(s) {missing} not unique in gpu_model_runner.py — "
                   f"no routes are published, so E/F/G have nothing to consume")
        return 0
    for _name, old, new in sites:
        text = text.replace(old, new, 1)
    RUNNER.write_text(text, encoding="utf-8")
    print(f"{LOG} applied 4/4 sites (A=init B=observe C=finish D=dummy); "
          f"inert unless GENESIS_ENABLE_H119_LENS_ROUTER=1 "
          f"(alias: GENESIS_ENABLE_PN119_ROUTER=1)")
    return 0


sys.exit(main())
