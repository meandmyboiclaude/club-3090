# vllm-patch-guard — patch-set accounting that survives us

Three failures inside one week (2026-07-13..19), each costing an afternoon:

1. **Forgetting.** The expected patch count lived in chat scrollback. Every
   retelling produced a different number — 57, 88, 93, 122, 155 — for a true
   **134**. Nobody could say what "correct" was, so nobody could tell when it broke.
2. **Bad data.** The numbers came from ad-hoc greps that silently measured the
   wrong thing, and each wrong number was then quoted downstream as fact.
3. **Silent loss.** Touching vLLM — image bump, sndr rebase, compose edit —
   drops patches whose anchors drifted. Nothing noticed until behaviour changed
   weeks later, by which point the cause was untraceable.

## The three counting traps (why every hand-count was wrong)

- **`Genesis Results` prints once PER LANE.** There are two — legacy
  `genesis.apply_all` (46) and the sndr v12 registry (57). Quoting one line
  undercounts by nearly half. This single mistake produced most of the bad numbers.
- **`journalctl CONTAINER_NAME=` returns EVERY incarnation of a name.** A
  container recreated three times yields three concatenated boots, so every count
  comes out 3× inflated (162 sub-patches read as 486; 4 DRIFT read as 12). Scope
  on the container's current `StartedAt`.
- **`podman logs` truncates under the journald driver.** It reported *zero*
  `PN100:` lines, and had lost the boot-time patch lines entirely, for a container
  that was demonstrably running the patch. Always read via `journalctl`.

A fourth, subtler one: `APPLY X` is the dispatcher *announcing an attempt*, not
confirming success. A patch can announce and then fail to anchor, so the applied
set is announced **minus** drifted — otherwise the name list (106) contradicts
the dispatcher's own tally (103).

## What runs

`vllm-patch-watcher.service` — a **root systemd service watching `podman events`**,
deliberately *not* an `ExecStartPost` on the vLLM units. A hook living inside the
thing it audits shares that thing's fate: the next compose rewrite or unit
regeneration removes it, and auditing stops exactly when config churn makes it
most necessary. Watching podman's event stream means it fires on every start path
— systemd, podman-compose, a manual `podman run`, an auto-restart after a crash —
and nothing in the vLLM config can disable it by accident.

On each start it settles 150s, archives the full boot log to
`~/shared/vllm-boot-logs/`, records the patch set to Postgres, then diffs against
the previous boot. If patches vanished it writes `~/shared/PATCH-REGRESSION-ALERT.md`
naming them; when green again it removes that file.

**Untrustworthy input is refused, never reported with a caveat** — an empty log,
a log with no `Genesis Results`, or one spanning multiple boots aborts the record.
A number with an asterisk gets quoted later without the asterisk.

## Database — `vllmops` on the local Postgres

`boots` — one row per container boot: counts per lane, sub-patches, hunks, drift,
failures, KV tokens, image.
`boot_patches` — one row per patch per boot: `(patch, lane, status)` where lane is
`dispatcher|house` and status is `applied|drift`.

```bash
vllm-patch-record.py record  <container>   # usually the watcher's job
vllm-patch-record.py history <container>   # totals over time
vllm-patch-record.py diff    <container>   # vs previous boot; exit 1 on loss
vllm-patch-record.py missing <container>   # ever-applied but absent now
```

`missing` is the slow-drift detector: a patch lost three image bumps ago never
shows up in a pairwise diff, but it shows up here.

```sql
-- when did a specific patch last apply?
SELECT b.started_at FROM boot_patches p JOIN boots b ON b.id=p.boot_id
WHERE p.patch='PN96' AND p.status='applied' ORDER BY 1 DESC LIMIT 1;

-- total over time
SELECT started_at, total_active, drift_skipped FROM boots ORDER BY 1 DESC LIMIT 20;
```

## Known-good baseline (2026-07-19, thinkingcap-gptq-pro-v2)

| | |
|---|---|
| dispatcher lanes | 2 (legacy 46 + sndr 57 = 103) |
| house `/fixes` | 31 of 36 |
| **total active** | **134** |
| sub-patch applications | 162 |
| direct file hunks | 10 |
| DRIFT | 4 — P3, P62, P67, P87 |
| hard failures | 0 |

DRIFT is expected, not breakage: P62 is superseded by PN96 (applied), P87/P3/P67
are upstream anchor drift. Also deliberately off: PN54 (cudaErrorAssert crash),
P67 (BUG-028 asserts), P78 (needs cudagraphs), P82 (research-only), PN61
(`qwen3_vl` only — model is `qwen3_5`), P60/P60B (ngram target files absent; we
run MTP). PN19 is `upstream_merged` — retired by success.
