# Patch Attribution Profiles

v0.8.0 Phase A adds three data-only profile files:

- `scripts/lib/profiles/patches.yml` records every local patch bundle and Genesis env-gated patch, what it fixes, how it is delivered, and any known compose coverage gaps.
- `scripts/lib/profiles/arch_patches.yml` is the declarative input for the locked C0 engine-support gate: arch, loadable engine pins, required patches, valid TP flags, trust-remote-code tri-state, and kernel constraints.
- `scripts/lib/profiles/calibration_seed.yml` seeds the calibration backbone from directly measured `BENCHMARKS.md` rows.

These files support the v0.8.x scope: evaluate any safetensors HF repo; pull only vLLM-loadable supported ones, and only when the gates pass (or an explicit override is accepted).

When adding a local patch, add a `patches.yml` entry in the same change. Declare all delivery channels: Dockerfile bake, compose mount/invoke, or Genesis env gate. If the patch gates an architecture or engine path, add or extend the matching `arch_patches.yml` row. If a patch is known to be load-bearing but cannot reach a compose today, record it under `delivery_gaps` rather than silently fixing runtime files in the audit change.

When adding measured results that should seed the cold-start predictor, add a `calibration_seed.yml` anchor only for directly measured configs. `confidence: exact` applies only to capabilities listed in `smoked_capabilities`; everything else stays in `unsmoked_capabilities`.

If no seeded or community calibration anchor matches a future pull target, the verdict must say: `no calibration data for this config class -- prediction is an unvalidated lower bound`. Also surface the boot-fit caveat: static fit does not guarantee stability under accumulated-context workloads; run or request a continuous soak before treating a config as production-stable.

## v0.8.0 Phase A-prime fold-ins (#359 / PR #147)

Phase A-prime is a discrete prerequisite commit that enriches the Phase A
data for the #141 compose generator. It does **not** reopen Phase A and keeps
`test-patch-attribution.sh` green.

### Storage choice — `scripts/lib/profiles/profile_runtime.yml`

The per-profile captured template, the `genesis_equipped` discriminator, the
arch→model-slug cross-reference, and per-model trust-remote-code evidence are
stored in a **new `scripts/lib/profiles/profile_runtime.yml`**, not in
`COMPOSE_REGISTRY` fields and not as new top-level keys on `arch_patches.yml`.

Rationale:

- `arch_patches.yml` has a **strict closed key-set** enforced by
  `test-patch-attribution.sh` (`arch_allowed_keys`). The brief scopes
  `model_slugs` "onto arch rows", but adding a new top-level arch key would
  trip the "unknown keys" guard, and **the test is the contract** (it is not
  edited in this commit). The fold-in therefore lives in
  `profile_runtime.yml` under `arch_model_xref`, keyed by the same `arch:`
  string, so the arch schema stays clean and the existing test stays green.
  `arch_patches.yml` remains the C0 authority for
  `requires_trust_remote_code`; `arch_model_xref` is the evidence ledger the
  generator's trc gate reads. The two MUST agree for in-scope arches (they
  do: the four on-stack arches were set to evidence-cited `false`).
- `COMPOSE_REGISTRY` is a thin param-value bridge imported as a Python module
  by the test; the captured templates are large structured blocks, so keeping
  them out of the registry avoids import-surface churn.

### `profile_runtime.yml` contents

- `profiles.<name>` — for every in-scope vLLM profile (40): `compose_path`,
  `genesis_equipped` (locked v6 discriminator: compose contains
  `_genesis`/`GENESIS_PIN`/`GENESIS_ENABLE` OR `kv_format` starts
  `turboquant`) + its evidence, and `compose_service_template` — the token
  classification the generator applies to the whole shipped service
  definition: `param_slots` (env-substituted from the registry),
  `governed_slots` (`--trust-remote-code` — captured, flagged governed,
  **never blind-passthrough**, per locked design v6 §88), `constants`
  (verbatim — **including the `${VLLM_IMAGE:-…}` expression, which is NOT
  substituted**), and the two named `insertion_points` (`volumes:`,
  `entrypoint:`). `extends:`-based profiles record `extends_base` and their
  anchors are `inherited-from-extends-base`.
- `arch_model_xref.<arch>` — `model_slugs` (compose_registry models served by
  that arch), `trust_remote_code`, and `trust_remote_code_evidence` derived
  from each model's **own** `config.json` on this stack.

### Per-load-bearing-patch delivery metadata (`patches.yml`)

Only the ~10 patches with a non-empty `load_bearing_when` carry a real
`delivery_mechanism` ∈ `python_sidecar | site_package_overlay |
install_script`, plus `delivery_spec`, a mandatory `drift_guard`,
`capability`, and `foundational`. Diagnostics / negative-local-result /
Genesis-env patches are `delivery_mechanism: none`. `foundational: true`
patches hard-refuse on drift-guard failure (weights won't load/boot without
them); capability-scoped patches degrade (omit + DEGRADED).

The legacy `delivery:` boolean block
(`dockerfile_bake`/`entrypoint_invoke`/`genesis`) is **DEPRECATED and
READ-ONLY**: it is retained verbatim only because
`test-patch-attribution.sh` still reads it. The generator and all new tooling
MUST read `delivery_mechanism`/`delivery_spec` instead — do not add new
compose-wiring decisions to the boolean block.

### Drafter fold-ins (`drafters/*.yml`)

Each drafter carries `speculative_config_template` (the verbatim
`--speculative-config` JSON form from the shipped composes, with `{N}` /
`{LOCAL_MODEL_PATH}` substitution points) and `local_model_path` (the
container path the compose passes; `null` for the built-in MTP head and for
the llama.cpp-scoped GGUF drafter, which the vLLM generator never selects).

### Citations

`BENCHMARKS.md` carries explicit `<a id="…">` anchors
(`#gemma-4-31b-community-experimental`, `#moe-models`) so the patch/profile
citations resolve on GitHub even though the rendered heading slug differs
(`## MoE models (v0.7.3 — preview track)` would otherwise slug to
`#moe-models-v073-preview-track`).

Run the audit with:

```bash
bash scripts/tests/test-patch-attribution.sh
```

### Reusable core: `scripts/lib/profiles/patch_attribution.py` (v0.8.0 STEP 2)

The attribution logic that used to live embedded inside
`test-patch-attribution.sh` is extracted into
`scripts/lib/profiles/patch_attribution.py` so the #141 generator (STEP 3)
and the test share one implementation. The test now imports it
(`load`, `compose_text`, `gap_declared`, `reaches`, `c0_state`, the
schema/coverage helpers, and the schema key-sets) and asserts identically
— same `PASS: 61 patch entries, 11 arch rows, 18 calibration seeds`
summary and the same known-delivery-gaps list.

`reaches(root, patch, name_or_abs_path)` is now **sound** (brief v9
correction #4): it probes the **comment-stripped service body only**
(everything from the top-level `services:` key onward, with whole-line
and inline `#` comments removed) and validates the patch's **actual
`delivery_spec` wiring** — the declared volume-mount target(s) and/or
entrypoint invoke at the `wired_at` insertion point(s) — instead of a
bare `patch["id"] in text` substring. A patch ID merely named in the
file-header banner or the generator's own header WARNING block can no
longer register as reached. It accepts a `COMPOSE_REGISTRY` profile name
**or** an arbitrary absolute path to a compose file.
