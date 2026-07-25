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

### 2026-07-26 JIT-warmup family — exec-discard triage

Six modules, one mechanism, one verdict. Every one installs by plain
`setattr` (five on `Worker.compile_or_warm_up_model`, PN522 on
`model_executor.warmup.kernel_warmup`) from inside `apply_all`'s own
`main()` — the process the compose entrypoint then replaces with
`exec vllm serve`. `exec` replaces the process image, so the wrapper is
gone before the engine starts. There is no survival hatch on boot pin
`dev1474cherrymax-1757-20260725`: no genesis/sndr entry point in
`vllm.general_plugins`, no dist-info, no `.pth`, stock `sitecustomize`.
All six have their enable flag SET in the live compose, all six print
`RESULT applied` on every boot, and not one has ever run.

The restore is known and cheap in mechanism — the P103/P39a self-install
hook, packaged in `spec_decode/probes/self_install.py`, text-patched into
`v1/worker/gpu_worker.py`. It was priced and declined on VALUE, not cost:

* **The payoff is first-request TTFT and nothing else.** The registry's own
  notes say so: "no effect on steady-state wall_TPS in mean" (PN364),
  "expected TTFT CV drop 30%→~15%" (PN126). No benchmark on this rig can
  see it — the house protocol is 3 warm + 5 measured runs, and the warm
  runs absorb the first-request JIT by construction.
* **The Triton cache is host-mounted and persists**, so the cost being
  removed is paid ONCE per pin/patch change, on the first cold-cache boot,
  by whoever's first request lands — not per boot and not per request.
* **The hook site is the worst moment of the boot to add allocations.**
  `compile_or_warm_up_model` runs AFTER KV cache allocation
  (`v1/engine/core.py:136 _initialize_kv_caches` → `:286`
  `determine_available_memory` → `:297` `get_kv_cache_configs`, then
  `gpu_worker.py:663 compile_or_warm_up_model`). PN126 Pass 2 issues
  `_dummy_run(..., cudagraph_runtime_mode=FULL)` there. On a rig where KV
  is the residual and the measured util ceiling is 0.935, a restore adds a
  new late-boot allocation surface to buy a latency spike nobody measures.

**It does not address BUG-128, and that was the question that decided it.**
BUG-128 is cold-Triton-JIT boot fragility: the first boot after a pin or
patch change pays JIT during the memory-profile run, the transient peak
(~+2.3 GiB) makes the KV check fail ("1.0 GiB available vs 1.6 needed"),
the engine dies, and podman's restart policy then passes on the now-warm
cache — a stale degraded boot that looks healthy. That death happens inside
`get_kv_cache_configs`, which is called at `core.py:297`, strictly BEFORE
`compile_or_warm_up_model` is ever reached. A boot that hits BUG-128 dies
before any of these six could run; a boot that does not hit it did not need
them. Restoring the family would leave BUG-128 exactly where it is (it is
handled operationally by the `tcbench-boot-guard.sh` ExecStartPost recycle,
commit `cd9b3157`).

The BUG-128-shaped lever, for whoever picks it up, is a DIFFERENT patch:
warm the hot kernels ahead of `determine_available_memory` and release the
compile scratch before the profile run measures. That is new work against a
different anchor, needs a GPU to validate, and none of these six is a
starting point for it — they all hook the wrong side of the profile.

| Module | Install site | Ground |
|---|---|---|
| `pn126_v1_decode_kernel_warmup.py` | `Worker.compile_or_warm_up_model` (setattr, `pn126:...`) | Orchestrator of the family; 2 extra `_dummy_run` passes. TTFT-only by its own registry note, and Pass 2's FULL-cudagraph run is the one that lands post-KV-alloc. |
| `pn128_spec_decode_helper_warmup.py` | same | Backport of OPEN vllm#41481; 4 eagle helper kernels. Same exec-discard, same TTFT-only payoff. |
| `pn129_slot_mapping_warmup.py` | same | Backport of OPEN vllm#42165. Its own registry row already carried a RETIRE-ON-MERGE WATCH against the cleaner OPEN vllm#46446; retiring now for the exec reason costs nothing that watch was not already going to spend. |
| `pn130_turboquant_decode_warmup.py` | same | Backport of OPEN vllm#42215, one TQ decode kernel. Also `applies_to.vllm_version_range` `<0.24.0`, which the cherrymax pin is at the edge of. |
| `pn364_hybrid_gdn_mamba_warmup.py` | same (`worker_cls.compile_or_warm_up_model`) | Vendor of OPEN vllm#43642. Chains AFTER PN126's wrapper, so it is the smallest share of an already-small payoff, and it cannot install at all once PN126 is gone from the chain. |
| `pn522_tq_raw_tail_kernel_warmup.py` | `model_executor.warmup.kernel_warmup` (setattr) | Same class, one pin later. Its own credit text says the kernel "JIT-compiles on the FIRST MTP verify request" — first-request-only by construction. |

Inbound imports were checked for all six before moving, and the check
changed the work: `apply/_per_patch_dispatch.py` has a second, non-registry
dispatch path with function-local imports for PN126/PN128/PN129/PN130/PN364
(the `pn34_workspace_lock_runtime_relax` shape). Those imports were
repointed to `_retired` in the same commit rather than left to break. That
path also bypasses `decision._check_lifecycle_gate` — it calls
`_wiring.apply()` directly — so each module's own `apply()` now returns a
`skipped` naming the retirement, honouring `GENESIS_ALLOW_RETIRED=1`. The
full source is kept in place (not stubbed as `g4_05` was) precisely so an
un-retire is a guard deletion, not an archaeology exercise.

**Operator follow-up:** the compose still sets
`GENESIS_ENABLE_PN126_V1_DECODE_WARMUP`, `..._PN128_SPEC_DECODE_WARMUP`,
`..._PN129_SLOT_MAPPING_WARMUP`, `..._PN130_TQ_DECODE_WARMUP`,
`..._PN364_HYBRID_GDN_WARMUP` and `..._PN522_TQ_RAW_TAIL_WARMUP` to `1`.
They are now inert-and-labelled rather than inert-and-silent, but the lines
are dead and can be dropped from the compose at the next edit.

**Same class, NOT retired:** `attention/turboquant/g4_62_tq_kernel_warmup.py`
is the same setattr-on-`kernel_warmup` shape, has
`GENESIS_ENABLE_G4_62_TQ_KERNEL_WARMUP=1` in the live compose, and reports
applied every boot while reaching nothing. It is left alone on purpose: it
is a `g4_`-family patch and this repo also serves `models/gemma-4-31b`, so
the "worth it on THIS rig" judgement above is not the whole question for it.

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
