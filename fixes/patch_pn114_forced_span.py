#!/usr/bin/env python3
"""PN114 forced-span grafts (2026-07-23) — apply AFTER patch_pr44812_tool_guard.py.

Generalizes the holder's </think> end-forcing machine to force an ARBITRARY
per-seq token sequence: every state-machine reference to
`self.think_end_token_ids` becomes `(state.get("force_seq") or
self.think_end_token_ids)`. With no force_seq set, byte-for-byte behavior is
IDENTICAL to stock (the fallback is the original attribute) — the machinery
is inert unless _genesis/plateau/pn114.py arms a span (env-gated:
GENESIS_ENABLE_PN114_PROBE / GENESIS_PN112_WRAPUP / GENESIS_PN112_CONFIRM).

Grafts:
  A) natural-abort guard: skip the "end token not in new_tokens -> revert to
     think" check while a PN114 span is armed (we set in_end BEFORE forcing).
  B-D) force_seq lookups in the spec-advancement + completion machinery.
  E) completion divert: probe phases return to THINK mode (pn114.on_force_
     complete) instead of the answer-mode reset; wrap-up/normal fall through.
  F) forcing-loop lookups (_apply_forcing_to_logits).
  G) pn114.observe_state call after pn112's observe.

Idempotent by marker; anchor drift FATAL exit 1 (loud bad boot); stale-pyc
drop — house style (patch_pn108/pn112/pr44812 lineage).
"""
import pathlib
import sys

LOG = "[patch_pn114_forced_span]"
HOLDER = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/"
    "thinking_budget_state.py"
)

FSEQ = '(state.get("force_seq") or self.think_end_token_ids)'

GRAFTS: list[tuple[str, str, str, str]] = []

# A: natural-abort guard + force_seq in the membership check
GRAFTS.append((
    "# PN114 A",
    '''        if state["in_end"] and state["end_count"] == 0:
            new_tokens = output[prev_length:]
            stopping_thinking = (
                self.think_end_token_ids[state["end_count"]] in new_tokens
            )
            if not stopping_thinking:
''',
    '''        if state["in_end"] and state["end_count"] == 0:
            # PN114 A: an armed span sets in_end BEFORE its first forced token
            # exists in the output — skip the natural-abort check then.
            new_tokens = output[prev_length:]
            stopping_thinking = (
                (state.get("_pn114") or {}).get("phase") is not None
                or ''' + FSEQ + '''[state["end_count"]] in new_tokens
            )
            if not stopping_thinking:
''',
    "A natural-abort guard",
))

# B: spec-advancement bound + token compare
GRAFTS.append((
    "# PN114 B",
    '''                for i, token_id in enumerate(state["spec_token_ids"]):
                    if state["end_count"] + 1 < len(self.think_end_token_ids):
                        if token_id == self.think_end_token_ids[state["end_count"] + 1]:
''',
    '''                for i, token_id in enumerate(state["spec_token_ids"]):  # PN114 B
                    if state["end_count"] + 1 < len(''' + FSEQ + '''):
                        if token_id == ''' + FSEQ + '''[state["end_count"] + 1]:
''',
    "B spec advancement",
))

# C+E: completion check + divert
GRAFTS.append((
    "# PN114 E",
    '''            if state["end_count"] >= len(self.think_end_token_ids):
                state.update(
                    {
                        "in_end": False,
                        "end_count": 0,
                        "check_count_down": state["thinking_token_budget"],
                        "start_thinking": -1,
                        "end_thinking": -1,
                        "think_count": 0,
                        "continue_thinking": False,
                        "scan_offset": len(state.get("output_tok_ids", [])),
                    }
                )
''',
    '''            if state["end_count"] >= len(''' + FSEQ + '''):
                # PN114 E: probe phases resume THINK mode instead of the
                # answer-mode reset; wrap-up/normal closes fall through.
                _pn114_handled = False
                try:
                    from vllm._genesis.plateau import pn114 as _pn114_mod
                    _pn114_handled = _pn114_mod.on_force_complete(state)
                except Exception:
                    import logging as _pn114_log
                    _pn114_log.getLogger(
                        "vllm.genesis.pn114"
                    ).warning("PN114 on_force_complete raised", exc_info=True)
                if not _pn114_handled:
                    state.pop("force_seq", None)
                    state.update(
                        {
                            "in_end": False,
                            "end_count": 0,
                            "check_count_down": state["thinking_token_budget"],
                            "start_thinking": -1,
                            "end_thinking": -1,
                            "think_count": 0,
                            "continue_thinking": False,
                            "scan_offset": len(state.get("output_tok_ids", [])),
                        }
                    )
''',
    "E completion divert",
))

# F: forcing loop lookups
GRAFTS.append((
    "# PN114 F",
    '''                    end_count = state.get("end_count", 0)
                    for force_idx in force_index:
                        if end_count < len(self.think_end_token_ids):
''',
    '''                    end_count = state.get("end_count", 0)  # PN114 F
                    _pn114_fseq = ''' + FSEQ + '''
                    for force_idx in force_index:
                        if end_count < len(_pn114_fseq):
''',
    "F forcing-loop bound",
))
GRAFTS.append((
    "# PN114 F2",
    '''                                active_indices_cpu.append(mask_idx)
                                force_tokens_cpu.append(
                                    self.think_end_token_ids[end_count]
                                )
''',
    '''                                active_indices_cpu.append(mask_idx)  # PN114 F2
                                force_tokens_cpu.append(
                                    _pn114_fseq[end_count]
                                )
''',
    "F2 forcing-loop token",
))

# G: observe call after pn112's (anchors on pn112 graft B's inserted tail)
GRAFTS.append((
    "# PN114 G",
    '''                ).warning('PN112 observe raised', exc_info=True)
            self._update_think_state(state)
''',
    '''                ).warning('PN112 observe raised', exc_info=True)
            # PN114 G: forced-span probes / wrap-up / confirm-at-fire.
            try:
                from vllm._genesis.plateau import pn114 as _pn114
                if _pn114.any_enabled():
                    _pn114.observe_state(
                        state, len(self.think_start_token_ids), seq_idx,
                        conf=getattr(
                            self, '_genesis_pn112_conf', {}
                        ).get(seq_idx),
                        req_id=getattr(
                            self, '_genesis_req_id_by_index', {}
                        ).get(seq_idx),
                    )
            except Exception:
                import logging as _pn114_olog
                _pn114_olog.getLogger(
                    'vllm.genesis.pn114'
                ).warning('PN114 observe raised', exc_info=True)
            self._update_think_state(state)
''',
    "G observe call",
))


def _apply(marker: str, anchor: str, repl: str, what: str) -> int:
    if not HOLDER.exists():
        print(f"{LOG} FATAL: holder missing: {HOLDER}", flush=True)
        return 1
    src = HOLDER.read_text(encoding="utf-8")
    if marker in src:
        print(f"{LOG} SKIP (already applied): {what}", flush=True)
        return 0
    count = src.count(anchor)
    if count != 1:
        print(f"{LOG} FATAL: anchor occurs {count}x (need 1) for {what}",
              flush=True)
        return 1
    assert marker in repl, f"marker missing from replacement: {what}"
    HOLDER.write_text(src.replace(anchor, repl, 1), encoding="utf-8")
    print(f"{LOG} applied: {what}", flush=True)
    return 0


def _drop_stale_pyc() -> None:
    cache = HOLDER.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob(HOLDER.stem + ".*.pyc"):
        try:
            pyc.unlink()
            print(f"{LOG} dropped stale pyc {pyc.name}", flush=True)
        except OSError as exc:
            print(f"{LOG} WARN: could not drop {pyc.name}: {exc}", flush=True)


def main() -> int:
    rc = 0
    f_ok = True
    for marker, anchor, repl, what in GRAFTS:
        if what.startswith("F2") and not f_ok:
            # F2 references _pn114_fseq defined by F — applying it alone
            # would NameError at runtime. Fail loudly instead.
            print(f"{LOG} FATAL: skipping F2 because F failed", flush=True)
            rc |= 1
            continue
        r = _apply(marker, anchor, repl, what)
        if what.startswith("F ") or what == "F forcing-loop bound":
            f_ok = r == 0
        rc |= r
    if rc == 0:
        _drop_stale_pyc()
    return rc


if __name__ == "__main__":
    sys.exit(main())
