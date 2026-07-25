# Nightly / image pin bump runbook

This is the procedure for moving a club-3090 compose to a newer engine image
(vLLM nightly hash, llama.cpp tag, SGLang variant, etc.) — and especially for
**collapsing** pin drift, where one engine has multiple distinct tags pinned
across composes.

It exists because:

- Each pinned image is **~30 GB on disk** for vLLM nightly builds (and several
  GB for other engines). Three distinct vLLM nightly hashes = ~100 GB of cached
  images on this rig before any sandbox / one-off images.
- Each compose mounts **patches** that target specific files inside the engine
  image. Patches written against nightly A may break against nightly B — files
  move, function signatures change, hunks fail to apply.
- **Bumping blindly** can succeed at boot but produce incorrect output at
  runtime (e.g., a spec-decode patch silently degrading), so we don't.
- We bump by **process**, with a verify gate, so we catch regressions before
  they ship.

When to consider a bump:

- A new vLLM nightly fixes a bug we depend on (track via `docs/UPSTREAM.md`).
- We have **pin drift** — `scripts/maintenance/list-image-pins.sh` shows >1 tag
  for the same engine. Goal: collapse to one tag per engine.
- Disk pressure on the docker partition (`docker system df` shows multiple
  large nightlies cached with low reclaimable).

When NOT to bump:

- During an active bench / paper / cross-rig validation cycle. Wait until the
  cycle ends.
- If the current pin is a stable production target with known TPS / quality
  numbers and no upstream signal demanding the move.

---

## MANDATORY GATE — the capability ledger

**Read this before step 0. It is not optional and it is not a formality.**

Five separate bumps have silently dropped work we had already done, and each
time it was found weeks later by accident. The ledger exists so a bump that
drops a capability **fails loudly at the gate** instead of being discovered
five sessions later.

Tooling lives in `ops/vllm-capability-ledger/` (stdlib-only python3, no pip
deps, runs inside a bare container):

| file | what it is |
|---|---|
| `vllm-capability-inventory.json` | every capability across all 3 lanes, with its EFFECT handle |
| `vllm-commit-ledger.json` | every commit we own, reduced to content fingerprints |
| `vllm-capability-extract.py` | regenerates the inventory from the repo |
| `vllm-commit-fingerprint.py` | regenerates the commit ledger from git history |
| `vllm-capability-verify.py` | reports LIVE / MISSING / DEGRADED against a container or an image |

### The three rules the ledger encodes (all learned the hard way)

**1. "Applied" in the boot log does NOT mean effective.**
On 2026-07-25 alone four patches reported applied while doing nothing:
BUG-122 (SPN71/73/92 announced APPLY, module gate disagreed, targets had ZERO
markers, and the record DB said "applied" for weeks); P89 (silently inert on
EVERY pin — and the thing that looked like proof it worked, populated
`reasoning_tokens` in bench rows, was computed CLIENT-side); P39a; PN346B
(landed one sub-patch, soft-skipped the other half). The existing recorder
counts APPLY announcements and "N applied" tallies and cannot see any of it.
**Never accept a log line as evidence.** The verifier only checks markers in
files, modules on disk, symbols in the wheel, and env as the container got it.

**2. `exec` discards setattr.**
The entrypoint runs `python3 -m vllm._genesis.patches.apply_all` and then
`exec vllm serve "$@"`. `exec` **replaces** the process, so every `setattr` and
monkey-patch made by `apply_all` is gone before a token is served. Only TEXT
patches — the ones that write bytes into files under
`/usr/local/lib/python3.12/dist-packages/vllm/` — survive. P39a logged
"applied" every boot for months and did nothing. A monkey-patch that must take
effect needs a **self-install hook** text-patched into the target module (the
P103/P39a pattern: append `_genesis_<id>_install_at_import(globals())` so every
fresh import in every process re-installs the wrapper). The verifier reports
lane-1 setattr-only patches as `INERT` on purpose.

**3. Dual-pin: anchors are content-sniffed, never rewritten.**
Older images must keep booting. A patch is re-anchored by sniffing the file for
whichever shape is present and splicing onto that, wrapped in
`except Exception` with a skip on drift — **not** by rewriting the anchor to
match only the new pin. `resolve_vllm_file()` returns a `str`, not a `Path`;
treating it as a `Path` cost a boot cycle twice (`92647851` → `748d2a0b`).
Verify a re-anchor by extracting the anchor from every live image pin and
splicing for real, before any boot.

### GATE A — capture the BEFORE state (do this first, on the OLD pin)

```bash
cd ~/club-3090/ops/vllm-capability-ledger
python3 vllm-capability-extract.py                # refresh the inventory
python3 vllm-commit-fingerprint.py                # refresh the commit ledger
python3 vllm-capability-verify.py --container <running-old-container> \
        --json /var/tmp/cap-before-$(date +%Y%m%d).json
git add -A && git commit -m "ledger: baseline before <pin> bump"
```

Commit the baseline. If you skip this you have nothing to diff against and the
gate degrades to a guess.

### GATE B — verify the wheel offline, before you boot anything

```bash
python3 vllm-capability-verify.py --image <new-image-ref> --json /var/tmp/cap-image.json
```

Safe by construction: short-lived CPU container, `--network=none`, no GPU, no
server. An image holds the **pristine** vllm — boot-time text patches have not
run — so this answers "did the wheel keep its cherry-picks and are the lane
sources present", not "are the patches applied". Boot markers reporting `N/A`
here is correct and expected.

### GATE C — after the bump boots, diff against the baseline

```bash
python3 vllm-capability-verify.py --container <new-container> \
        --baseline /var/tmp/cap-before-<date>.json \
        --json /var/tmp/cap-after.json
```

Exit code 1 = **the bump dropped work that was in effect before**. The output
names, per capability and per commit, what regressed.

**A non-zero gate blocks the bump.** Every regression gets one of three
dispositions, written down:

- **re-anchor** — the target moved or the anchor drifted; fix and re-run.
- **retired with evidence** — upstream absorbed it. The evidence is a named
  upstream commit/PR present in the new base, plus a `docs/UPSTREAM.md` row.
  "It looks merged" is not evidence.
- **accepted loss** — deliberate, with the reason recorded in
  `ops/vllm-capability-ledger/commit-verdict-overrides.json` so the next run
  does not re-raise it.

### Why the COMMIT sweep matters more than the capability sweep

A capability check asks "is patch X applied?". That is not enough, and it is
probably the mechanism behind several of the five losses: a bump can
legitimately restore X in its **upstream/original form** while silently
dropping the three later commits in which we *fixed* X — and the capability
check passes.

So the unit of tracking is the **commit**. Commit hashes do not survive
rebases, cherry-pick storms or squashes (we do all three), so each commit is
reduced to 2-3 independent **content fingerprints** — a marker string, a flag
name, a distinctive identifier, a whole file it added. `--baseline` fails the
gate when a commit that was in effect before is no longer in effect after.

### After a wheel rebuild specifically

The cherry-pick set for the fork lives in `~/engines/vllm-build`; the rebuild
recipe (replay order, hand-adapted picks, the CI dispatch, and the
`torch_version=` / `-ubuntu-2204` traps) is in that repo's `BUILD-NOTES.md`.
Two ledger-specific notes:

- Any commit touching `turboquant_attn.py`, `triton_turboquant_decode.py` or
  `triton_turboquant_store.py` threads new kernel arguments through the TQ
  decode entry point, which breaks every anchor spanning those calls. That is
  exactly how our own KVQ squash broke P101, P89, PN119 and P18B_TEXT. Re-run
  GATE B/C before booting such a wheel, and re-verify those four by name.
- `--image` on the new wheel plus `--image` on the previous wheel, diffed, is
  the cheapest way to see what a rebuild changed.

---

## The procedure

### 0. Identify scope

```bash
bash scripts/maintenance/list-image-pins.sh
```

This prints the pin distribution + flags drift. Decide:

- **Which pin do you want to retire?** (e.g., the least-used one)
- **Which pin do you want to consolidate to?** (usually the newest, or the one
  required by an active bug fix)
- **Which composes are affected?** (the script lists them per pin)
- **What patches do those composes mount?** (drives migration cost)

Order the work from **lowest patch surface first** — composes with `patches=none`
or `patches=vllm-marlin-pad` (small) are quick wins; composes with `patches=
vllm-gemma4-dflash-int8` (~13 patched files) are higher-risk.

### 1. Branch + bump one compose

```bash
git checkout -b bump-<engine>-<short-new-tag>-<compose-name>
```

Edit the compose's `image:` line to the new tag. Don't bump multiple composes
in one branch — keep blast radius small.

### 2. Patch survival check

For each patch the compose mounts:

```bash
docker pull <new-image>
# For each patched file, fetch the upstream version and diff against our patch
docker run --rm <new-image> cat /path/to/patched/file > /tmp/upstream-new.py
diff -u /tmp/upstream-new.py <repo>/models/<model>/<engine>/patches/<patch-dir>/<file>
```

Verdict per patch:

- **Hunks land cleanly** → no rebase needed.
- **Context shifted, hunks land with offset** → no rebase needed but verify
  carefully (line numbers in our patch may now be off if anyone reads them).
- **Hunks fail / structural change** → manual rebase. Use Codex / agent help if
  the patch is large (e.g., DFlash with 13 files touching `gpu_model_runner`).

If any patch fails to rebase, **stop**: file an issue documenting which file
changed in upstream, why our patch broke, and link the upstream commit. Decide
whether to:

- Wait for upstream to absorb the patch (preferred — see `docs/UPSTREAM.md`).
- Rebase the patch against the new internals (effort cost).
- Stay on the old pin (status quo — document the reason in the tracker).

### 3. Boot the new pin

```bash
gpu-mode <appropriate-mode>
# or, manually for one compose:
sudo docker compose --env-file <repo>/.env -f <compose-file> up -d
docker logs -f <container-name>
```

Watch for tracebacks during model load. If anything looks off, **stop** and
revert the compose's `image:` line.

### 4. Verify gate

In order:

```bash
bash scripts/verify-full.sh        # ~1-2 min — reachability, tools, streaming, MTP AL
bash scripts/verify-stress.sh      # ~5-10 min — longctx needle ladder, tool-prefill OOM
```

Both must be GREEN. If either fails, **stop**: revert the bump, file an issue
linking the failing check + the new pin, document in `docs/UPSTREAM.md`.

### 5. Bench delta

Run the canonical 800-word essay bench and confirm TPS within ~5% of the
pre-bump pin:

```bash
bash scripts/bench.sh > /tmp/bench-after-bump.txt
# Compare to the row in BENCHMARKS.md for the same compose
```

If TPS drops >5% with no obvious cause (e.g., new nightly removed a Marlin
optimization), file an upstream issue and decide if the bump is still worth it.

### 6. Land the bump

```bash
git add <compose-file>
git commit -m "bump <compose-name> to <engine>:<new-tag>"
# Open PR — link to verify-full + verify-stress run logs in description
```

### 7. Pin retirement (when ALL composes for one tag have moved)

When `list-image-pins.sh` shows that the old pin has zero composes:

1. Delete the now-cached image from local docker:
   ```bash
   docker image rm <old-image>:<old-tag>
   ```
2. Add an entry to `docs/UPSTREAM.md` under "Retired pins" documenting the
   retirement date + the reason (e.g., "consolidated to nightly-X after PR
   #41745 merged").

This is the "free 30 GB of disk" moment.

---

## Anti-patterns

- **Bumping all composes in one PR.** If anything breaks, you don't know which
  one. Bump one at a time.
- **Skipping the verify gate** because "it booted, looks fine." Spec-decode
  regressions are silent. Always run `verify-full.sh + verify-stress.sh`.
- **Bumping during an active bench cycle.** TPS deltas pollute your numbers.
- **Forgetting to update `docs/UPSTREAM.md`.** Future you will thank present
  you for documenting which pin was retired and why.

---

## Engine-specific notes

### vLLM

- Nightly tag format: `vllm/vllm-openai:nightly-<8-char-commit-hash>`.
- Most volatile internal: `vllm/v1/worker/gpu_model_runner.py`. DFlash and
  speculator patches both touch it.
- Marlin kernel (`vllm/model_executor/kernels/linear/mixed_precision/`) changes
  rarely but breaks completely when it does.
- Bumping a vLLM nightly often requires Genesis pin re-evaluation too — check
  `docs/UPSTREAM.md` for the current Genesis pin compatibility.

### llama.cpp

- Stable tag `ghcr.io/ggml-org/llama.cpp:server-cuda` shifts under us — no
  hash. To get a deterministic pin, capture the digest:
  ```bash
  docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
  docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/ggml-org/llama.cpp:server-cuda
  ```
  Use the `@sha256:...` form in the compose if reproducibility matters.

### SGLang

- Variant tags (`cu13`, `cu13-gemma4`, etc.) — different from version tags.
  Variant naming changes more often than version numbers. Document the variant
  reason in the compose header.

### Luce DFlash / xtransformers / future engines

- When adding a new engine, add an "Engine-specific notes" subsection here
  documenting its pinning convention + which files in its image are most
  volatile across versions.
