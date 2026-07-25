# vLLM capability ledger

**Purpose: stop losing our vLLM work on every upstream bump.** It has happened
five times. Each time the loss was found weeks later, by accident.

This directory is the machine-readable record of everything we have built or
vendored on top of vLLM, plus the tooling that proves — by EFFECT, not by a log
line — which of it is actually in force in a given container or image.

```
vllm_ledger_lib.py             shared helpers (tree / container / image readers)
vllm-capability-extract.py     regenerates the inventory from the repo
vllm-commit-fingerprint.py     regenerates the commit ledger from git history
vllm-capability-verify.py      LIVE / MISSING / DEGRADED against a target
vllm-capability-inventory.json the inventory (generated; committed)
vllm-commit-ledger.json        the commit ledger (generated; committed)
inventory-overlay.json         hand-written per-capability facts, merged on top
commit-verdict-overrides.json  hand verdicts on commits; survive regeneration
```

Everything is **stdlib-only python3**. This box's venv ships neither sklearn
nor scipy and PyYAML is absent, and the verifier has to run inside a bare
container, so there are no dependencies at all.

---

## The one idea

> **"Applied" in the boot log does NOT mean effective.**

2026-07-25 alone produced four independent patches that reported applied while
doing nothing:

| case | what happened |
|---|---|
| **BUG-122** | SPN71/73/92 announced APPLY, the module-level gate disagreed, the targets had **zero** markers. The record DB said "applied" for weeks. |
| **P89** | Silently inert on **every** pin — three drifted anchors, nobody noticed. What looked like proof it worked (populated `reasoning_tokens` in bench rows) was computed **client-side**. |
| **P39a** | `apply_all` runs in a standalone process, then the entrypoint does `exec vllm serve`. `exec` **replaces** the process, so setattr / monkey-patch effects never reach the server. Only TEXT patches (which write files) survive. It logged "applied" every boot for months and did nothing. |
| **PN346B** | Landed one sub-patch, silently soft-skipped the other half. |

The existing recorder counts APPLY announcements and "N applied" tallies. It
cannot see any of this.

So **every verification here is an effect check**:

| handle | what it proves |
|---|---|
| `marker` | the literal string a text patch writes INTO the installed vllm file |
| `file` | a source module physically present at its container path |
| `symbol` | an identifier compiled into the wheel |
| `flag` | the gating env var **as the container actually received it** |

Never a log line.

---

## The second idea (and the more important one)

> **The unit of tracking is the COMMIT, not the patch.**

A capability check asks "is patch X applied?". That is not enough. A bump can
legitimately restore X in its **upstream / original form** while silently
dropping the three later commits in which we *fixed* X — and the capability
check passes. This is very likely one of the mechanisms behind the five losses.

Commit **hashes** are useless as a test: they do not survive rebases,
cherry-pick storms or squashes, and we do all three. So each commit is reduced
to 2-3 independent **content fingerprints** derived from what it ADDED:

- marker / log strings (`"Genesis P101 …"`, `"# PN96:"`, `"[pn100-auto-…]"`)
- env flag names (`GENESIS_ENABLE_*`, `SNDR_*`)
- new identifiers (`_genesis_*`, `def …`, `class …`, module constants)
- whole files the commit added

Candidates are ranked by **rarity in the current tree** (a token that occurs
once is a perfect fingerprint) then length. Multiple fingerprints per commit
mean one incidental deletion cannot raise a false alarm.

Verdicts: `intact` · **`LOST`** · **`PARTIAL-LOSS`** · `config-drift`
(compose/yaml values that are *meant* to be retuned) · `doc-only` · `removal` ·
`unfingerprintable`. Only the two loss verdicts are alarms, and each one needs
a human verdict of *really lost* vs *superseded with evidence* — recorded in
`commit-verdict-overrides.json` so it is not re-raised every run.

`PARTIAL-LOSS` exists because a whole file the commit ADDED is decisive: if
that module is gone the commit is not intact no matter how many incidental
call-site tokens survive. Without that rule KVQ-2 `f785a3a5f` read "intact" off
two generic Triton kernel params (`stride_slot`, `stride_head`) while its
entire new `sink.py` was absent from the tree that gets built.

Note the choice of probe root: `kvq2-sink-runtime` is fingerprinted against
`~/engines/vllm-build`, **not** against its own worktree. Probing a branch
against its own checkout can only ever say "intact" and tells you nothing. What
we need to know is whether the work reached the line that gets BUILT.

---

## The three lanes

| lane | lives in | how it applies | how it is verified |
|---|---|---|---|
| **lane-1 genesis** | `models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis/{wiring,kernels,middleware}` | `python3 -m vllm._genesis.patches.apply_all` in its own process | marker in the patched vllm file; module presence for kernels/middleware |
| **lane-2 sndr** | `…/_genesis/sndr/engines/vllm/patches/**` (vendored) | bridged from lane-1 via `patches/sndr_lane.py` → `run_lane2()`, same process | marker (288 of 329 registry rows carry one) |
| **/fixes** | `~/club-3090/fixes/*.py` | one `python3 /fixes/X.py` per line in the compose entrypoint | marker; plus "is the line actually uncommented in the compose" |

Both lanes ship a **data-only `PATCH_REGISTRY`** — `_genesis/dispatcher.py`
(126 rows) and `_genesis/sndr/dispatcher/registry.py` (329 rows). The extractor
`runpy`s them rather than grepping, so `lifecycle`, `default_on`,
`superseded_by`, `conflicts_with` and `apply_module` come straight from the
source of truth.

### Structural facts the verifier has to know (or it lies)

1. **`exec` discards setattr.** A lane-1 patch that only does `setattr` cannot
   reach the exec'd server. Reported as `INERT`, not LIVE. The fix pattern is a
   **self-install hook** text-patched into the target module
   (`_genesis_<id>_install_at_import(globals())`), so every fresh import in
   every process re-installs the wrapper — see P103 and P39a.
2. **Shared-id suppression.** `sndr_lane.py` exports `GENESIS_DISABLE_<id>` for
   every id present in *both* registries; lane-1 owns it and lane-2's copy
   deliberately no-ops. Its boot line reads "explicitly disabled by operator",
   which looks like operator action but is self-inflicted by design. 157 rows.
   Without this the verifier calls all of them lost.
3. **`GENESIS_SNDR_TRUST_DEFAULT_ON=1`** is set in both composes, so lane-2
   rows with `default_on=True` engage with **no explicit flag in the compose**.
   An inventory keyed on "flags present in the compose" misses them entirely.
4. **Model gating.** 93 rows are Gemma-4 family; on a Qwen rig they are inert
   **by design**. Reported `N/A`. Override with `VLLM_LEDGER_MODEL_FAMILY`.
5. **Path relocation.** Upstream moved `model_executor/layers/fla/` to
   `third_party/flash_linear_attention/`. The patch modules cope at runtime via
   `resolve_vllm_file()`, but a static read of the source only sees the literal
   it was written with, so the verifier carries an alias table.
6. **Id collisions are real.** 17 ids are claimed by both `/fixes` and lane-2
   (PN71, PN72, PN73, PN80, PN82, PN90, PN91, PN92, PN95, PN96, PN102, PN104,
   PN105, PN106, PN108, PN118, PN119). Nothing here may be keyed on bare id —
   entries carry a `key` of `<id>@<lane>:<module>` when the id is ambiguous.
7. **The lanes are read-only bind mounts.** Tree presence == container presence
   for lane *source* files. The interesting container-only signal is the marker
   written into the *installed* vllm at boot.

---

## Modes

```bash
# the real answer — boot-time text patches HAVE run
python3 vllm-capability-verify.py --container vllm-tcbench-8021

# wheel-level, offline, before booting anything.  Short-lived CPU container,
# --network=none, no GPU, no server.  The installed vllm is PRISTINE here, so
# boot markers reporting N/A is correct.
python3 vllm-capability-verify.py --image localhost/vllm-qwen36-endgame:<tag>

# host tree only
python3 vllm-capability-verify.py --tree

# THE BUMP GATE
python3 vllm-capability-verify.py --container <new> --baseline cap-before.json
```

`--baseline` exits non-zero when a capability or a commit that was in effect
before is no longer in effect after. That is the whole point: a bump that
silently drops capabilities **fails loudly** instead of being discovered five
sessions later.

Wired into `docs/NIGHTLY_BUMP_RUNBOOK.md` as gates A / B / C.

---

## Result states

| state | meaning |
|---|---|
| `LIVE` | marker present in the target / module present at its container path |
| `DARK` | present but its gate is off, or the compose does not invoke it — **intentional, not a loss** |
| `N/A` | not applicable here (retired, wrong model family, boot marker checked against a pristine image) |
| `INERT` | structurally cannot take effect — lane-1 setattr-only, the P39a class |
| `UNVERIFIABLE` | **no effect handle exists at all.** A blind spot: it still announces an APPLY decision. Fix by giving it a marker. |
| `DEGRADED` | some sub-patches landed, others did not — the PN346B class |
| `MISSING` | it should be in force and it is not. **Act on these.** |

`MISSING` splits into three actionable shapes, named in the `why` field:

- *announced-but-inert (BUG-122 class)* — the gate says on, the target exists,
  zero markers. Real inertness.
- *target path no longer exists upstream* — a re-anchor item.
- *module has apply() but NO registry row* — orphaned work the dispatcher can
  never reach.

---

## Regenerating

```bash
python3 vllm-capability-extract.py      # ~2 s
python3 vllm-commit-fingerprint.py      # ~10 min (one grep pass per repo)
```

Both are idempotent and their **diff is itself a signal** — re-run them as step
0 of every bump and review what changed.

Hand-written facts go in the two overlay files, never in the generated JSON:

- `inventory-overlay.json` — keyed by capability `key` (or bare `id`); merged
  over the extracted record. Use it for descriptions, upstream evidence,
  retirement rationale, and non-marker probes.
- `commit-verdict-overrides.json` — keyed by short sha:
  `{"5192458dcc30": {"verdict": "superseded", "note": "…"}}`.

### A trap worth knowing

`git rev-list --all` in `~/engines/vllm-build` is a trap: that repo carries ~28
`pr*` branches which are **upstream PR heads** kept as cherry-pick sources, so
`--all ^origin/main` yields 2265 commits of other people's work instead of our
39. The script pins an explicit branch list (`OUR_BUILD_BRANCHES`).
