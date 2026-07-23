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
  B) span-sound in_end walk (MTP soundness redesign, ultra-review #2): for an
     armed force_seq, end_count is RECOMPUTED each step as the positional
     match of the landed output (from state["force_seq_base"]) against the
     span — drafts are never credited, so an MTP rejection cannot desync and
     there is no pre-increment (no sacrificial prepend needed). Stock
     </think> walk byte-identical when no span armed (fseq-fallback kept as
     the fail-open path).
  E) completion divert: probe phases return to THINK mode (pn114.on_force_
     complete) instead of the answer-mode reset; wrap-up/normal fall through.
  S) span-sound forcing (_apply_forcing_to_logits): a span seq masks EVERY
     row it owns this step with force_seq[end_count + k] (bonus row gets
     end_count + n_spec), clamped at span end. Draft verification becomes
     forced-vs-forced; a rejection's recovery token IS the forced token.
  F) forcing-loop lookups (stock loop; fail-open fallback for spans).
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
            # exists in the output — skip the natural-abort check then (span
            # rejection recovery is site B's positional walk, not this).
            new_tokens = output[prev_length:]
            stopping_thinking = (
                (state.get("_pn114") or {}).get("phase") is not None
                or state.get("force_seq") is not None
                or self.think_end_token_ids[state["end_count"]] in new_tokens
            )
            if not stopping_thinking:
''',
    "A natural-abort guard",
))

# B: span-sound in_end walk. Anchor is the ENTIRE stock walk block; the
# replacement prepends the span path (position authority = landed output,
# positional from force_seq_base) and keeps the stock walk (with the fseq
# fallback lookups, so a raising span path fails open to old behavior).
GRAFTS.append((
    "# PN114 B",
    '''            state["force_index"] = []
            if len(state["spec_token_ids"]) > 0:
                for i, token_id in enumerate(state["spec_token_ids"]):
                    if state["end_count"] + 1 < len(self.think_end_token_ids):
                        if token_id == self.think_end_token_ids[state["end_count"] + 1]:
                            state["end_count"] += 1
                        else:
                            state["end_count"] += 1
                            state["force_index"] = [i]
                            break
                    else:
                        state["end_count"] += 1
                if len(state["force_index"]) == 0:
                    state["end_count"] += 1
                    state["force_index"] = [len(state["spec_token_ids"])]
            else:
                state["end_count"] += 1
                state["force_index"] = [0]
''',
    '''            # PN114 B: span-sound walk. For an armed force_seq the
            # position authority is the LANDED output (positional match from
            # force_seq_base) — drafts are NEVER credited, so an MTP
            # rejection cannot desync the span, and there is no
            # pre-increment (index 0 forces intact). Stock </think> walk
            # runs byte-identically when no span is armed.
            _pn114_span_ok = False
            if state.get("force_seq"):
                try:
                    _pn114_fs = state["force_seq"]
                    _pn114_out = state.get("output_tok_ids", [])
                    _pn114_base = state.get("force_seq_base")
                    if _pn114_base is None or _pn114_base > len(_pn114_out):
                        # armer records this at arm time; fail-safe here
                        _pn114_base = len(_pn114_out)
                        state["force_seq_base"] = _pn114_base
                    _pn114_emitted = 0
                    while (_pn114_emitted < len(_pn114_fs)
                           and _pn114_base + _pn114_emitted < len(_pn114_out)
                           and (_pn114_fs[_pn114_emitted] is None
                                or _pn114_out[_pn114_base + _pn114_emitted]
                                == _pn114_fs[_pn114_emitted])):
                        # None = free HOLE in the span (wildcard: any landed
                        # token advances it — the probe's captured letter)
                        _pn114_emitted += 1
                    if (_pn114_emitted < len(_pn114_fs)
                            and len(_pn114_out) - _pn114_base
                            > len(_pn114_fs) + 32):
                        # runaway guard: span tokens are not landing (mask
                        # writes failing?) — declare done, fail open.
                        import logging as _pn114_blog
                        _pn114_blog.getLogger(
                            "vllm.genesis.pn114"
                        ).warning(
                            "PN114 span runaway (emitted=%d/%d grew=%d) — "
                            "failing open",
                            _pn114_emitted, len(_pn114_fs),
                            len(_pn114_out) - _pn114_base,
                        )
                        _pn114_emitted = len(_pn114_fs)
                    state["end_count"] = _pn114_emitted
                    state["force_index"] = []
                    _pn114_span_ok = True
                except Exception:
                    import logging as _pn114_blog2
                    _pn114_blog2.getLogger(
                        "vllm.genesis.pn114"
                    ).warning(
                        "PN114 span walk raised — stock fallback",
                        exc_info=True,
                    )
            if not _pn114_span_ok:
                state["force_index"] = []
                if len(state["spec_token_ids"]) > 0:
                    for i, token_id in enumerate(state["spec_token_ids"]):
                        if state["end_count"] + 1 < len(''' + FSEQ + '''):
                            if token_id == ''' + FSEQ + '''[state["end_count"] + 1]:
                                state["end_count"] += 1
                            else:
                                state["end_count"] += 1
                                state["force_index"] = [i]
                                break
                        else:
                            state["end_count"] += 1
                    if len(state["force_index"]) == 0:
                        state["end_count"] += 1
                        state["force_index"] = [len(state["spec_token_ids"])]
                else:
                    state["end_count"] += 1
                    state["force_index"] = [0]
''',
    "B span-sound walk",
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
                    state.pop("force_seq_base", None)
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

# S: span-sound forcing — a span seq masks EVERY row it owns this step with
# the positionally-correct span token, so draft verification is
# forced-vs-forced and a rejection's recovery token IS the forced token
# (mid-span rejection cannot desync). Stock loop (F/F2 below) is untouched
# and remains the fail-open fallback.
GRAFTS.append((
    "# PN114 S",
    '''            state = self._state[seq_idx]
            if state.get("in_end", False):
                # logits processor in spec mode are called twice
''',
    '''            state = self._state[seq_idx]
            if state.get("in_end", False):
                # PN114 S: sound span forcing — mask every row this seq owns
                # this step with force_seq[end_count + k] (bonus row gets
                # end_count + n_spec), clamped at span end. end_count is the
                # landed-output position authority from site B; it never
                # moves inside a step, so the bonus/target calls are
                # consistent and cannot double-force a position.
                _pn114_sfs = state.get("force_seq")
                if _pn114_sfs:
                    _pn114_sdone = False
                    _pn114_srows = []
                    _pn114_stoks = []
                    try:
                        _pn114_semit = state.get("end_count", 0)
                        if self.in_spec_mode:
                            _pn114_sspec = (
                                spec_token_ids_for_layout[seq_idx]
                                if seq_idx < len(spec_token_ids_for_layout)
                                else []
                            )
                            if predict_bonus_token:
                                _pn114_spos = [
                                    (0, _pn114_semit + len(_pn114_sspec))
                                ]
                            else:
                                _pn114_spos = [
                                    (_pn114_sk, _pn114_semit + _pn114_sk)
                                    for _pn114_sk in range(len(_pn114_sspec))
                                ]
                        else:
                            _pn114_spos = [(0, _pn114_semit)]
                        for _pn114_sk, _pn114_sti in _pn114_spos:
                            if (_pn114_sti >= len(_pn114_sfs)
                                    or _pn114_sfs[_pn114_sti] is None):
                                # past span end, or a free HOLE inside the
                                # span (probe letter): row free-samples
                                continue
                            _pn114_smask = (
                                self.cu_num_tokens[seq_idx] + _pn114_sk
                            )
                            if (
                                _pn114_smask < self._mask_capacity
                                and _pn114_smask < logits.shape[0]
                            ):
                                _pn114_srows.append(_pn114_smask)
                                _pn114_stoks.append(_pn114_sfs[_pn114_sti])
                        _pn114_sdone = True
                    except Exception:
                        import logging as _pn114_slog
                        _pn114_slog.getLogger(
                            "vllm.genesis.pn114"
                        ).warning(
                            "PN114 span forcing raised — stock fallback",
                            exc_info=True,
                        )
                    if _pn114_sdone:
                        active_indices_cpu.extend(_pn114_srows)
                        force_tokens_cpu.extend(_pn114_stoks)
                        continue
                # logits processor in spec mode are called twice
''',
    "S span-sound forcing",
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
    '''                                if _pn114_fseq[end_count] is not None:
                                    active_indices_cpu.append(mask_idx)  # PN114 F2
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


# H: P-pen hesitation-marker logit penalty (GENESIS_PPEN_LAMBDA>0; ids from
# the boot ids file; thinking-phase rows only; spans excluded). FAIR-recipe
# λ subtract at the raw-logits seat (before temperature). Expectation at
# W4A16: −5-9% CoT, acc ±2 (headline 12-23% is 3-bit). Pre-registered λ=1.0.
GRAFTS.append((
    "# P-pen:",
    "        # Build the active index / forced-token lists entirely on CPU so we\n",
    '''        # P-pen: hesitation-marker logit penalty — runtime-gated, dark.
        try:
            import os as _ppen_os
            _ppen_l = float(
                _ppen_os.environ.get('GENESIS_PPEN_LAMBDA', '0') or 0
            )
            if _ppen_l > 0 and self._state:
                _ppen_ids = getattr(self, '_ppen_ids_t', None)
                if _ppen_ids is None:
                    import json as _ppen_json
                    import torch as _ppen_t
                    try:
                        with open('/tmp/genesis_pn114_ids.json') as _ppen_f:
                            _ppen_lst = _ppen_json.load(_ppen_f).get(
                                'ppen', []
                            )
                    except Exception:
                        _ppen_lst = []
                    _ppen_ids = (
                        _ppen_t.tensor(
                            _ppen_lst, dtype=_ppen_t.long,
                            device=logits.device,
                        )
                        if _ppen_lst else False
                    )
                    self._ppen_ids_t = _ppen_ids
                if _ppen_ids is not False and len(_ppen_ids) > 0:
                    for _ppen_si, _ppen_state in self._state.items():
                        if not _ppen_state.get('in_think'):
                            continue
                        if (_ppen_state.get('_pn114') or {}).get('phase'):
                            continue
                        _ppen_row = self.cu_num_tokens.get(_ppen_si)
                        if _ppen_row is None or _ppen_row >= logits.shape[0]:
                            continue
                        _ppen_end = min(
                            self.cu_num_tokens.get(
                                _ppen_si + 1, logits.shape[0]
                            ),
                            logits.shape[0],
                        )
                        logits[_ppen_row:_ppen_end, _ppen_ids] -= _ppen_l
        except Exception:
            import logging as _ppen_log
            _ppen_log.getLogger('vllm.genesis.pn114').warning(
                'P-pen raised', exc_info=True
            )
        # Build the active index / forced-token lists entirely on CPU so we
''',
    "H P-pen logit penalty",
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
