# Architecture — how this stack thinks about LLM serving on 24 GB

A short orientation to the design choices in this repo. Not a deep technical doc — just the mental model.

---

## What the stack assumes

You have **1 or 2 RTX 3090s** (or compatible 24 GB Ampere-class cards). You want to serve a modern LLM **locally** for chat / coding / agents / RAG. You're OK with a bit of setup but you don't want to fork engines or write CUDA kernels.

---

## Two ways in: curated catalog + universal `pull`

The stack has two entry paths, and the mental model is "the curated catalog is the *measured backbone*; `pull` is the *general front door*."

- **Curated catalog** (`models/<model-name>/`) — models we have measured: real benchmark numbers, validated composes, KV-math calibration anchors, per-model gotchas. This is the high-confidence path and the source of truth the math is calibrated against.
- **Universal `pull`** (`scripts/pull.sh <hf-repo> --profile-like <catalog-key>`) — point it at *any* safetensors HF repo. It derives a spec from the repo's own files, runs it through the gate, and only proceeds for a vLLM-loadable, supported model whose fit math passes (or where you accept an explicit override). It is **honest about confidence and never silently gate-passes** — every non-pass outcome hard-stops with a precise reason. Headline: *"evaluate any safetensors HF repo; pull only vLLM-loadable supported ones, and only when the gates pass (or an explicit override is accepted)."*

The pipeline behind `pull`:

```
pull → derive (read repo config/safetensors)
     → gate   ([C0] engine-support · [C2a] disk · [B] kv-calc fit verdict
                · [C1] confidence × verdict → terminal action)
     → emit   (generate a minimal derived compose for an in-scope model)
     → boot   (gated download → derived-compose boot → capture artifacts)
     → loop   (classify failures · trust pipeline · dedup · feed the
                calibration backbone)
```

**Pull-gate surfaces (current):**
- `scripts/pull.sh … --recommend` — appends an honest one-line fit verdict (FITS / FITS-but-not-yet-accepted / DOES-NOT-FIT) over the same gate result; presentation only, never changes the verdict, carries the boot-fit≠runtime caveat.
- **Failure on-ramp** — every hard-block leaves a redacted `.pull-captures/<slug>/<ts>/` bundle; `scripts/pull.sh --submit-last` (or `--submit <dir>`) is a separate, consented, user-invoked step that submits it (works with *or without* `gh`; no telemetry, no auto-send).
- **Broadened arch registry** — materially more safetensors architectures pass `[C0]` without `--experimental-arch`; native built-ins reach a clean serve verdict, per-repo remote-code stays fail-closed (zero false-pass).
- **Optional non-NVIDIA `hwdetect`** — bounded subprocess augmenting hardware detection where `nvidia-smi` doesn't apply (AMD/Apple); absent → graceful degrade, never feeds kv-calc.

**You don't need to know any of these stage names to use it** — you run one command (`scripts/pull.sh`); the `[C0]`/`[B]`/`[C1]`/… taxonomy above is internal flow for contributors. Start at the user guide: [`docs/PULL.md`](PULL.md) (Quickstart at the top). Contributor depth, in pipeline order: [`COMPOSE_GENERATOR.md`](COMPOSE_GENERATOR.md) (emit) → [`PULL_GATE.md`](PULL_GATE.md) (gate) → [`PULL_EMIT_DERIVED.md`](PULL_EMIT_DERIVED.md) (boot+capture) → [`LOOP.md`](LOOP.md) (classify+trust+dedup). The curated path still works exactly as before — `pull` is additive, the front door for anything not in the catalog.

A boot-fit pass is a *static* check. It is necessary-not-sufficient: a `fits-clean` config can still degrade under accumulated-context agent workloads (the Cliff 2 / prefill-cliff failure modes — see [`CLIFFS.md`](CLIFFS.md)). The gate verdict says so explicitly and points at soak-continuous validation; trust the caveat, not just the green.

---

## How the repo is organized

The mental model: **engines are general; models are specific; hardware is fixed.**

```
docs/                         engine + hardware docs (general, model-agnostic)
  engines/                      vLLM / llama.cpp / SGLang — comparison + deep dives
  HARDWARE.md                   Ampere SM 8.6+, 24 GB, no NVLink
  GLOSSARY.md                   plain-language definitions
  img/                          chart sources + PNG exports (performance, vram-budget)

models/<model-name>/          everything specific to a model
  README.md                     model overview + quants + Genesis surface
  INTERNALS.md                  this model's quirks (architecture, bugs, fixes)
  CHANGELOG.md                  this model's dated history
  vllm/                         vLLM-specific configs for this model
    compose/                      docker-compose files
    patches/                      model+engine patches
    README.md                     "vLLM recipes for this model"
  llama-cpp/                    llama.cpp recipes for this model
    recipes/                      shell scripts
    README.md                     "llama.cpp recipes for this model"
  sglang/                       SGLang status / TBD recipes

scripts/                      shared, model-aware
  setup.sh <model>              downloads + verifies model + clones patches
  verify.sh                     quick smoke (~10 sec)
  verify-full.sh                fast functional test, 8 checks (~1-2 min)
  verify-stress.sh              boundary-case stress test, 2 checks (~5-10 min)
  bench.sh                      canonical TPS bench
  pull.sh <hf-repo>             universal evaluate→gate→emit→boot front door
                                  (+ --recommend verdict; --submit-last/--submit on-ramp)
  generate-compose.sh           emit a minimal compose for an in-scope profile
  lib/profiles/                 the gate/derive/classify/trust pipeline (engine)
  tests/                        executable specs (test-pull, test-classifier, …)

.pull-captures/<slug>/<ts>/   runtime: per-pull capture bundle the loop consumes
                              (pt1–5 + manifest; redacted; created by pull)
```

---

## Why this layout

### Why "models/" isn't at the top

If "qwen3.6-27b" were the top-level partition, every cross-model concept (engines, hardware, scripts) would either be duplicated or live awkwardly in some shared subdir. By putting models inside `models/`, the top-level becomes *infrastructure* (engines, hardware, glossary, scripts) and `models/<m>/` becomes *content*. This scales: when we add Qwen3.5-27B / GLM-4.6 / Llama-3.x, they get a new subdir under `models/` with the same internal pattern, and the top-level docs stay relevant.

### Why engines are general docs

vLLM behaves the same way regardless of whether you're serving Qwen, GLM, or Llama. The tuning levers (mem-util, KV type, spec-decode config, power cap) are model-agnostic. So `docs/engines/VLLM.md` covers vLLM-the-engine, not vLLM-on-Qwen. Per-model engine recipes live under `models/<m>/<engine>/`.

### Why patches are per-model-per-engine

A patch like `patch_tolist_cudagraph.py` fixes a bug that hits Qwen3-Next + vLLM + TurboQuant + spec-decode together. It wouldn't apply to a different model with different attention layout. So patches live at the most specific level: `models/<m>/<engine>/patches/`.

If a patch is general (across engines or models), it bubbles up to `docs/engines/<engine>.md` notes or — rarely — into a top-level `patches/` (none today).

### Why scripts are top-level but model-aware

`bash scripts/setup.sh qwen3.6-27b` is the model-aware form. The script reads the model name and does the right downloads / SHA verification / patch fetching. We keep the script set in one place because the *operation* (download, verify, boot, test, bench) is the same shape across models.

---

## How information flows

**A user comes in cold:**
1. Lands on top-level [README](../README.md) → understands what the stack is, picks their model.
2. Goes to `models/<m>/README.md` → sees recommended config + quick start for their card count.
3. Boots; tests with `verify-full.sh` (fast, 8 checks); for boundary cases (KV-cache pressure, prefill OOM) runs `verify-stress.sh`; benches with `bench.sh`.

**A user hits a problem:**
1. Checks `docs/SINGLE_CARD.md` or `docs/DUAL_CARD.md` for workload-specific gotchas matching their hardware.
2. Checks `docs/FAQ.md` "Troubleshooting" section for the specific failure mode.
3. If still stuck: `models/<m>/INTERNALS.md` for engineering depth.
4. If engine-related: `docs/engines/<engine>.md` for general engine tuning.
5. Files an issue with logs.

**A power user wants to push limits:**
1. `docs/engines/<engine>.md` — engine tuning levers.
2. `models/<m>/INTERNALS.md` — model-specific knobs.
3. `models/<m>/<engine>/README.md` — recipe-specific tips.

---

## Design rules

A few principles the repo follows:

1. **No tutorials disguised as configs.** Composes are working configs, not pedagogy. Configs reference docs for the "why."
2. **Honest framing always.** If a config has a known cliff, the cliff is documented at the top of the relevant doc, not buried in a footnote. Users discovering issues at boot should already have read the warning.
3. **Cross-rig data welcome.** TPS numbers are run-to-run variable; we publish ours and welcome PRs adding "your rig" rows.
4. **Patches stay surgical.** We don't fork engines. Disk-edits at boot, runtime monkey-patches, or volume-mounts of patched source. When upstream lands a fix, the patch becomes a no-op (anchor doesn't match) and we drop it cleanly.
5. **Verification gates production.** `verify-full.sh` runs 8 fast functional checks; `verify-stress.sh` runs the heavy boundary-case tests (long-context needle ladder, ~25K-token tool-response prefill OOM detection). We don't claim a config works until both are green.
6. **Document the negative results too.** Probes that didn't pan out (PR #40798 backport, `--enforce-eager` mode) are documented so future-us doesn't redo the experiments.

---

## Things this stack is NOT

- **A vLLM fork.** All vLLM patches are mounted at boot, not forked into a custom build.
- **A model card / training recipe.** We use pre-quantized weights as-is. For training/quantization details, see the model authors' (Lorbus, Qwen) repos.
- **A general benchmarking suite.** `bench.sh` is the minimum needed to verify your setup matches ours. For rigorous A/B comparisons use [vllm-project/bench](https://github.com/vllm-project/bench) or similar.
- **A cloud-replacement service.** It's a recipe for running locally. Wrap it in your own auth/queueing/quota/etc. for production.
