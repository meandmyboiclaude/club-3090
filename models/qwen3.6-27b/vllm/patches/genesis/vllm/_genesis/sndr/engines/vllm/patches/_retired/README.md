# `_retired/` — archived patch wirings

Genesis patch lifecycle has 4 active phases (`experimental`, `validated`,
`legacy`, `community-experimental`) and 1 terminal phase: `retired`.
Once a patch is retired, its registry entry stays as-is for audit trail,
but the on-disk wiring module moves here.

## Policy

A patch transitions to `_retired/` when **one** of these conditions holds:

1. **Upstream merged the same fix.** The patch is now redundant after a
   specific vLLM pin bump (e.g. `vllm_version_range: ">=0.20.2rc1.dev209,<inf"`).
2. **Hypothesis disproven empirically.** A research-track patch was
   benchmarked and shown to not provide the expected gain (or to regress).
   Registry retains `retired_waiver: True` + brief explanation.
3. **Duplicate of another active patch.** Two patches with overlapping
   functionality where one is consolidated into the other. Registry
   retains `superseded_by: "<other_pid>"`.
4. **Deprecated mechanism.** Patch is replaced by a new, more robust
   approach (e.g. `P65 → P67` workaround→root-cause).

## Registry contract for retired patches

Every retired entry must have:

- `lifecycle: "retired"` — drives dispatcher skip + audit gates
- At least one of:
  - `superseded_by: "<other_pid>"` — names the replacement
  - `retired_waiver: True` + `credit` text — explains why retired without replacement
  - `vllm_version_range: ">=X,<Y"` — version window where active (for drift safety)
- `apply_module` — pointing to `sndr.engines.vllm.patches._retired.<file>`
  (so `audit_registry_contract.py` can validate import path).

## Why keep registry entries (vs delete)?

1. **Audit trail** — boot logs show "patch X skipped (retired since pin Y)"
2. **Drift detection** — operator running against an old pin gets explicit
   warning that retired patch was last active on different pin.
3. **Cross-rig community evidence** — patches retired locally may still
   be active on community forks; registry IDs stay stable for issue ref.

## Current contents

(Filled by Phase 2 cleanup 2026-05-14)

| Patch ID | Moved | Reason |
|---|---|---|
| (see registry.py `lifecycle: retired` entries) | | |

### 2026-07-26 orphan triage

These had `apply()` and NO registry row, so the dispatcher could never
reach them. They therefore have no registry entry to carry the audit
trail and the reason is recorded here instead. Each was checked for
inbound imports across the whole lane-2 tree before moving; all nine are
zero-reference (the two `pn248_acceptance_trace` hits in
`observability/` are a `/tmp/...log` PATH string, not an import).

| Module | Ground | Evidence |
|---|---|---|
| `p71_block_verify.py` | duplicate | `p71_pn369_rejection_sampler_consolidated.py:109` carries the branch "VERBATIM from p71_block_verify.py" and the P71 registry row's `apply_module` points at the consolidated module. `spec_decode/__init__.__all__` entry commented out in the same commit. |
| `pn248_acceptance_trace.py` | duplicate | `registry.py:2091` records it: "PN248 was a debug-log trace predecessor... never promoted to a registry entry (K.1.R.R.1 cleanup 2026-05-28)". PN282 is the production counterpart on the same `rejection_sample` hook. |
| `pn266_propose_trace.py` | investigation closed | self-declared "G4_77 design probe"; G4_77 has no registry row in EITHER lane — the design never shipped. |
| `pn267_kv_bridge_trace.py` | investigation closed | "G4_78-0 probe"; `registry.py:11140` G4_78 "RETIRED 2026-05-21... not enabled by any V2 profile". Hardcodes Gemma-4-31B layers 58/59. |
| `pn268_drafter_blocks_origin.py` | investigation closed | same G4_78 cycle; keys on the Gemma-4 MTP `draft_model.` prefix. |
| `pn269_a0_block_table_trace.py` | investigation closed | marker reads "G4_78-A0 trace"; hardcodes `.layers.58/.59.self_attn.attn`. |
| `pn270_drafter_kv_proj_audit.py` | investigation closed | same G4_78 cycle, drafter K/V-projection audit. |
| `pn272_gemma4_drafter_input_probe.py` | useless here + closed | `model_family: gemma4`; walks `gemma4_mtp.py`'s `Gemma4MultiTokenPredictor.forward`. This rig serves Qwen3.6. |
| `pn274_install.py` | useless here | its guard is a no-op on Qwen: `spec_decode/mapping/registry.py:24` has `PROVIDERS = [Gemma4MappingProvider()]` as the only provider, so `find_provider_for_config()` returns None and the module unconditionally ALLOWs — its own matrix says so. The PN274 registry row is a `lifecycle: coordinator` entry that registers only the consent env and does NOT point at this file. |

NOT moved, against an earlier reading that said retire:
`attention/turboquant/pn34_workspace_lock_runtime_relax.py`. Its registry
row is retired and points at nothing, but `apply/_per_patch_dispatch.py:4245`
IMPORTS AND CALLS it directly — there is a second dispatch path besides the
registry, and moving the file would have broken that import at boot.

## How to add a patch here

1. `git mv sndr/engines/vllm/patches/<family>/<file>.py sndr/engines/vllm/patches/_retired/<file>.py`
2. Update registry entry:
   ```python
   "lifecycle": "retired",
   "apply_module": "sndr.engines.vllm.patches._retired.<file>",
   "superseded_by": "<other_pid>",  # or retired_waiver: True
   "vllm_version_range": ">=X,<Y",  # version window where active
   ```
3. Run `python3 scripts/audit_registry_contract.py` — should stay green
4. Run `make evidence` — should stay 40/40
