# IMPORT-NOTES — sndr_core_engine v12 wholesale import (branch `sndr-v12-import`)

Date: 2026-07-13 · Source: `Sandermage/sndr_core_engine` @ `34e26930`
(v12.1.0-129-g34e2693, fetched as remote `sndr`) · Target pin:
`0.23.1rc1.dev1060+g9e57de719` (image `nightly-9e57de7197f2...`).

User mandate: import Sander's entire engine-patch surface — all 209 net-new
registry entries (incl. the Gemma4 `G4_*` series), the newer forms of the 120
shared patches, and the pins/anchor tooling. Excluded by user decision:
`sndr/memory`, `gui/`, `cli/`, `plugins/` (agent-platform, not serving path).

---

## 1. Integration decision — TWO-LANE (variant of option B, "transplant into
##    our layout, his architecture intact")

**Chosen:** vendor his `sndr` package **byte-identical** under
`vllm/_genesis/sndr/`, run it as a second apply lane behind our existing
monolith, with a thin policy overlay applied **in-memory at boot** (his files
stay unmodified except one appended pin hunk — see §6).

Why not option A (his layout at repo root + `vllm/_genesis` compat shim):
the live compose mounts **only** `vllm/_genesis` into the container
(`../../patches/genesis/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro`).
A top-level `sndr/` would simply not exist inside the container, and symlinks
out of a bind mount dangle. The compose contract is byte-unchanged only if
everything lives under `vllm/_genesis/`.

Why two lanes instead of replacing our apply path with his dispatcher:

- **His registry marks 7 of our currently-applied patches `lifecycle=retired`**
  (P64, P78, P83, P94, PN54, PN67, PN8) and consolidated our applied P59 into
  his retired P61b. His dispatcher hard-skips retired patches. Routing our
  live flags through his engine would silently drop 8 of today's 31 applied
  patches — an invariant violation. (He retired them against dev748 where the
  upstream parser refactor landed; on our dev1060 image the files still exist
  and our forms still apply — verified live.)
- **His shared-patch forms are dev748-anchored**; our 4 dev1060 re-anchors
  (PN8/PN12/P34/P83, commits d5e5604..aef8ce9) are newer. The dev1060 anchor
  manifest (§4) confirms his PN12/P34 forms genuinely drift and his PN8/P83
  anchors are gone on dev1060 — ours win in all 4 cases, test-verified.
- Lane-1 byte-identical ⇒ the no-regression invariant holds **by
  construction**, not by hope.

Both goals from the task are met: (i) compose contract untouched, (ii) his
one-file-per-patch + pins architecture is intact and graft-friendly — future
updates land with `git read-tree --prefix=vllm/_genesis/sndr/ sndr/main:sndr`
(plus re-applying the single pin hunk if it conflicts).

### Mechanics

- `vllm/_genesis/patches/sndr_lane.py` — the bridge. Registers
  `vllm/_genesis/sndr` as top-level `sndr` via an explicit importlib spec
  (his modules use absolute `from sndr.x import y`), applies the policy
  overlay, then calls `sndr.apply.run(apply=...)`.
- `vllm/_genesis/patches/apply_all.py` — after lane-1 finishes, invokes
  lane-2 and merges exit codes. **Kill-switch: `GENESIS_SNDR_LANE=0`**
  restores pre-import behavior exactly.
- `SNDR_APPLY_VIA_SPECS=1` is forced for lane-2. His default "legacy loop"
  calls ~95 hand-written apply functions **unconditionally** (measured: 29
  patches text-applied with no env flag set, including re-patching shared
  P15/P26 on top of lane-1). The spec-driven path routes every patch through
  `should_apply`, where the policy overlay binds. Load-bearing; do not remove.

### Policy overlay (in-memory, every boot — `sndr_lane.apply_policy()`)

1. **Shared suppression** — every id present in both registries (120) gets
   `GENESIS_DISABLE_<bare>=1` injected process-locally. Lane-1 owns them.
   The "both ENABLE and DISABLE set" warnings in lane-2 logs are this,
   working as designed.
2. **Net-new default-off** — his 28 net-new `default_on=True` entries
   (13 Gemma4 `G4_*` + PN523, PN525, PN96b, PN252, PN286, PN367, PN346,
   PN346B, P108, P109, PN116, PN118, PN119, P18B_TEXT, PN377) are forced
   `default_on=False`. Nothing changes live behavior until an enable-wave
   sets an explicit `GENESIS_ENABLE_*` flag. (The G4 entries would also
   self-skip on model detection; we don't rely on that alone.)
3. **S-prefix aliases** — see §3.

### Migration lever (for enable-waves)

`GENESIS_SNDR_OWNS_<bare>=1` hands ONE shared patch to lane-2: lane-1's
`should_apply` skips it ("delegated to sndr lane-2") and lane-2 skips the
DISABLE injection. This is the A/B path for adopting his newer shared forms
one patch at a time.

---

## 2. Import stats

| | before | after |
|---|---|---|
| registry entries | 126 (ours) | 126 (lane-1) + 329 (lane-2, his) — 120 shared, **209 net-new** |
| patch impl files | monolith + wiring/ | + 381 one-file-per-patch modules under `sndr/engines/vllm/patches/` |
| vendored files | — | 741 files, ~236K lines (`vllm/_genesis/sndr/`) |
| pins | ours: KNOWN_GOOD list | + his 8 pin dirs + **new `pins/0.23.1_9e57de719/`** (dev1060) |

Vendored dirs: root `*.py` + `apply/ assets/ bundles/ cache/ compat/
detection/ dispatcher/ engines/ kernel/ model_configs/ observability/
runtime/` (the transitive import closure of the engine surface — verified by
AST walk; none of the excluded dirs are imported). `scripts/anchor_sot/`
imported at repo level.

Consolidation map (recon-confirmed in his registry):

| our id(s) | his entry | his env_flag | alias that keeps our flag working |
|---|---|---|---|
| P59 + PN51 | P61b (retired) | `GENESIS_ENABLE_P61B_STREAMING_OVERLAP` | `env_flag_aliases` carries `..._P59_QWEN3_TOOL_RECOVERY` + `..._PN51_...` |
| P64 + P61c + PN56 | P64 (retired) | `GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING` | aliases carry P61C/PN56 flags |
| PN29 | PN298 | `GENESIS_ENABLE_PN298_FLA_CHUNK_O_ARCH_WARPS` | alias carries `..._PN29_GDN_SCALE_FOLD` |
| (P71 sibling) | P71 | `GENESIS_ENABLE_P71_BLOCK_VERIFY` | alias carries `..._PN369_RELAXED_ACCEPTANCE` |
| P56 / P57 | retired research entries | — | not enabled anywhere; lane-1 forms remain |

Our compose flag `GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY` resolves in BOTH
lanes: lane-1 applies our P59 (it did today and still does); lane-2 sees it
as an alias of P61b, which is (a) shared-suppressed and (b) retired — skip.
No double-apply, no dropped flag.

---

## 3. Flag / ID collision map

**Exact env-var collisions: ZERO.** Cross-checked: 35 live compose
`GENESIS_*` vars + 7 `GENESIS_*` names read by `/fixes/*.py` × 949
`GENESIS_*` names in his surface. The only shared names
(`GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING`,
`GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER`) are the *same* patches
(fixes-side reads them for coordination — same semantics, not collisions).
Near-misses are distinct: ours `GENESIS_P66_TARGET` vs his
`GENESIS_P66_MARKER`; ours `GENESIS_PN81_RERANK/BATCH/PACKED`,
`GENESIS_PN83_CHAT_SEQS` have no counterpart in his tree.

**ID/log-tag collisions (same PN number, different patch):** our `/fixes`
house series vs his in-registry entries — PN71, PN72, PN73, PN79, PN80,
PN82, PN90, PN91(vs our 91g), PN92:

| # | ours (/fixes, entrypoint-applied) | his (lane-2 registry, opt-in) |
|---|---|---|
| PN71 | reasoning alias | thinking-tag normalize (`GENESIS_ENABLE_PN71_THINKING_TAG_NORMALIZE`) |
| PN72 | streaming toolcall content recover | frequency-ngram drafter |
| PN73 | vendor legacy parsers (superseded) | tool-args safe normalize |
| PN79 | TQ decode scratch cudagraph-safe | in-place SSM state (+V2 variants) |
| PN80 | TQ v1-runner cache-dtype preserve | LoRA tensorizer device |
| PN82 | bonus-logprobs full-vocab guard | mamba cudagraph prefill zero |
| PN90 | MTP quant-leak guard (#47828) | probabilistic draft |
| PN91g | GDN spec state-index clamp (#48475) | developer role (PN91) |
| PN92 | explicit spec-method guard (#47490) | NIXL EP trial import |

Mitigation: the policy overlay registers an **S-prefix alias** for each of
his colliding entries (e.g. `GENESIS_ENABLE_SPN71_THINKING_TAG_NORMALIZE`).
**Enable-waves must use the S-form for these nine ids** so compose greps and
log forensics stay unambiguous. Both lanes log under `genesis.*` logger
names; the `[Genesis lane-2/sndr]` banner demarcates lane-2 output.

Shared `default_on` deltas (his registry vs ours, informational): P4, P6,
P20, P36 are default-on in ours / off in his — all four are skip-gated on
this pin either way; lane-1 semantics (ours) remain live.

---

## 4. dev1060 anchor manifest (`sndr/engines/vllm/pins/0.23.1_9e57de719/`)

Generated with his pipeline (`scripts/anchor_sot/`: discover in a
GPU+live-env container → pristine dump from the bare image → classify +
round-trip verify), podman, image `nightly-9e57de7197f2...`:

```
discovered=212 anchors → ok=118 (44 files) + rejected=94
counts: retired=46 · upstream_merged=8 · version_gated=7 ·
        optional_absent=23 · anchor_drift=10        roundtrip_fail=0
```

**Clean: 118 anchors · Re-anchored: 2 (PN385, PN525) · Absorbed-native: 3 (PN38, PN40, P85) · BLOCKED: 0.**
*(2026-07-14 resolution pass — every former BLOCKED entry verified against the
installed dev1060 source; verdicts below. Nothing dropped.)*

Genuine drift dispositions (nothing silently dropped):

| patch | sub-anchor(s) | disposition |
|---|---|---|
| PN12 (shared) | `pN12_silu_and_mul_pool` | lane-1 covered — our dev1060 re-anchor 3da5370 is live; his dev748 form drifts (SiluAndMul gained forward_cpu). Ours wins, test-verified. |
| P34 (shared) | `p34_deadlock_guard` | lane-1 covered — our re-anchor d1a01cb (_mamba_block_aligned_split rewrite). Ours wins. |
| P3 (shared) | `p3_bf16_fp8_cast` | lane-1 covered — our P3 form is the live one (currently skip-gated on this profile anyway). |
| **PN525** (net-new) | `pn525_no_toolcall_cleaned_content` + `_dev1060` | **RE-ANCHORED (2026-07-14)** — NOT absorbed: dev1060's expanded else-branch adds a required/named empty-content guard (a *different* fix); the closing `return None, content` still ignores `tool_call_info.content` on the auto path. New dev1060 sub-patch keys on the unique guard-tail+raw-return pair (count==1 byte-verified). Mutually exclusive with the historical anchor. |
| **P85** (net-new) | `p85_mamba_cache_blocks_shadow` | **ABSORBED-NATIVE (verified 2026-07-14)** — dev1060 (481e481b) `single_type_kv_cache_manager.py` natively implements the fine-grained mechanism (`hash_block_size` mode in `find_longest_cache_hit`, `scale_factor` walk, partial-tail CoW `_partial_hit_reqs`); `MambaManager` inherits it. Applying P85 would double-register shadow keys. Retire-on-pin (version-gate ≤ pre-481e481b). |
| **PN38** (net-new) | `pN38_a_qkv_proj_call`, `pN38_c_conditional_fused_kv` | **ABSORBED-NATIVE (verified 2026-07-14)** — dev1060 `qwen3_dflash.py:224` is literally PN38's replacement (`qkv, _ = self.qkv_proj(hidden_states)`); fused-KV native (`_build_context_kv_buffers`:413, `_fused_kv_weight`:422, fused GEMM :494). Version-gated ≤dev748. |
| **PN40** (net-new) | `pN40_a_fused_k_norm` | **ABSORBED-NATIVE (verified 2026-07-14)** — `_normalize_context_k` (:509-518) is the fused single `ops.rms_norm` over stacked `_k_norm_weights` PN40 intended; the per-layer loop anchor is gone. Version-gated ≤dev748. |
| **PN385** (net-new) | `pn385_*_forced_named` + `_dev1060` pair | **RE-ANCHORED + LIVE (shipped 2026-07-13)** — the dev1060 sub-patch pair against the namespace-tool helpers is applied in the running container (`tool_parsers/utils.py:404-408, 421-425`) with `GENESIS_ENABLE_PN385_FORCED_NAMED_EMPTY_PARAMS=1`. This row was stale. |

Also from the manifest: PN399's dependency on retired PN353A is
HIGH-but-MITIGATED (PN399 has a native-form fallback anchor that classifies
ok on dev1060). 46 retired-patch anchors gone **as expected**; 8
upstream-merged (PN524, P87×6, P26 `cu_2`); 7 version-gated (P77, PN50,
PN288×2, P7, PN66×2).

**Our 4 dev1060 re-anchors (PN8/PN12/P34/P83) all survive** — lane-1 applies
our forms (in today's applied set where enabled); manifest proves his dev748
forms do not anchor cleanly (PN8/P83 anchors absent, PN12/P34 genuine drift).
Our `guards.py` KNOWN_GOOD pins are untouched (lane-1 gate) and mirrored into
his `sndr/engines/vllm/detection/guards.py` KNOWN_GOOD list (lane-2 gate).

---

## 5. No-regression proof (verification runs, image `nightly-9e57de71…`)

Baseline (pre-import tree, this branch's parent `930c3ff`):

```
CPU:  Patches:  81 total  →  31 APPLY  |  50 SKIP    (0 failed)
GPU:  Patches:  83 total  →  31 APPLY  |  52 SKIP    (0 failed)
live :8020 boot log (same image):    83 total → 31 APPLY | 52 SKIP
```

(The task briefing said "34 applied"; the live present-tense number is 31 —
confirmed both in the running container's boot log and reproduced in
throwaway containers. The /fixes chain adds its own patches outside Genesis.)

New tree (lane-1 + lane-2), same env-file extracted from the live compose:

```
(a) CPU:  lane-1: 81 total → 31 APPLY | 50 SKIP · 0 failed
          lane-2: 0 applied / 334 skipped / 0 failed
          applied-name-set diff vs baseline: IDENTICAL
(c) GPU:  lane-1: 83 total → 31 APPLY | 52 SKIP · 0 failed
          lane-2: 0 applied / 334 skipped / 0 failed
          applied-name-set diff vs baseline: IDENTICAL
(b) GPU + full-cudagraph wave-1 flags
    (PN125/126/128/129/130/364/522 enable + PN358 enable + GENESIS_PN358_MODE=detect):
          lane-1: unchanged (31 APPLY, 0 failed)
          lane-2: 7 applied / 0 failed —
            PN125, PN126, PN128, PN129, PN130, PN364, PN358 (3/3 sub-patches,
            detect mode) all APPLY;
            PN522 self-skips by design ("PN521 raw-tail verify not enabled —
            nothing to warm") and was verified to APPLY (0 failed) when
            GENESIS_ENABLE_PN521_TQ_RAW_TAIL_VERIFY=1 is added. Wave-1 must
            include PN521's flag if PN522's warmup is wanted.
```

Lane-2's constant "18 partial-apply warnings" accompany 0 applied / 0 failed
— they are warning-classed *skip* reasons counted by his PatchStats, not
partial applies; benign, noted for log readers.

Nothing running was touched: the :8020 service, its compose, and `/fixes`
are unmodified. All runs were `--rm` throwaway containers.

---

## 6. Byte-identity & graft notes

The vendored tree is byte-identical to `sndr@34e26930` except **one appended
hunk**: club-3090 validated pins (dev178/dev424/dev799/dev1060) added to
`sndr/engines/vllm/detection/guards.py::KNOWN_GOOD_VLLM_PINS`, plus the new
generated `pins/0.23.1_9e57de719/` directory. Everything else (registry
default_on flips, shared suppression, S-aliases) happens in-memory at boot
from `sndr_lane.py`, so upstream grafts stay conflict-free.

## 7. Commits (this import)

- `4fde3cc` import(sndr): vendor sndr_core_engine v12 engine surface under vllm/_genesis/sndr
- `e4ef2a3` import(sndr): two-lane bridge — lane-2 sndr dispatcher + policy overlay
- `2ffcd98` fix(sndr-lane): force SNDR_APPLY_VIA_SPECS=1 — legacy loop bypasses gating
- `392935a` import(sndr): dev1060 anchor manifest + anchor-SoT tooling
- (this file + CHANGELOG entry)

## 8. Enable-wave quick reference

- Net-new patch: set its `GENESIS_ENABLE_*` flag (S-form for the nine
  colliding PN ids in §3) in the compose. Lane-2 applies it.
- Adopt his newer form of a shared patch: set `GENESIS_SNDR_OWNS_<bare>=1`
  (lane-1 skips, lane-2 engages) — one patch per A/B.
- His retired entries stay hard-gated; diagnostics only via
  `GENESIS_ALLOW_RETIRED=1`. Blocked list in §4 needs review before any of
  those five are waved in.
- Instant full rollback of the import at runtime: `GENESIS_SNDR_LANE=0`.
