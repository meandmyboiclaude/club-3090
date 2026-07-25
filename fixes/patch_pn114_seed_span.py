#!/usr/bin/env python3
"""PN114-SEED — move the PN102 think-seed from the PROMPT into a FORCED SPAN.

Run LAST in the entrypoint (after patch_pn114_forced_span.py, after
patch_h119_lens_router.py and after patch_pn101_answer_rescue.py): every one
of this patch's anchors is text one of those three wrote. Verify the counts
the BOOT sees, never the pristine image:

    python3 fixes/verify_pn114_seed_anchors.py

WHY
---
H119 scores a request from its own prefill hidden states, so the only thing a
route can steer today is the thinking-token CAP. That ceiling is low — the
zero-risk cap-only oracle saving is ~11.5% and deep-side ``cap_hit`` is 0/31,
i.e. the deep cap never binds — because the original headline came from
TREATMENT SELECTION, not capping. The PN102 seed
(``Budget: ~N short steps.\\nStep 1:``) is rendered INSIDE ``<think>``
(chat_template_v2json.jinja:147-151), which makes it the one half of the
treatment a post-prefill decision can still deliver: forcing it as OUTPUT is
position- and voice-identical to rendering it in the prompt. The banner is a
system turn, out of the model's voice, and is NOT equivalent — it is out of
scope for this patch on purpose.

WHAT IS WIRED (all inert unless GENESIS_ENABLE_PN114_SEED_SPAN=1)
-----------------------------------------------------------------
  S1a/S1b  sync_batch: stash the seed the API server stripped, on every path
           that keeps a state entry (H119-provisional, relaxed, budgeted).
  S2       update_state: arm the span after H119 resolves the routed budget
           and before _update_think_state, on the step where no token has
           been produced yet — the only instant a span can start at output
           position 0.
  S3       the completion divert, ahead of pn114's own on_force_complete.
  S4       serving.py: pop ``pn_env_seed`` out of chat_template_kwargs and
           carry the exact seed text in ``vllm_xargs``.

ALL-OR-NOTHING ACROSS BOTH FILES. S4 without S1-S3 would strip the seed and
never force it (a silent quality regression); S1-S3 without S4 would emit the
seed twice. Any anchor that does not resolve leaves BOTH files untouched, and
if the operator actually asked for the feature the skip is shouted to stderr
and logged at ERROR — a soft-skip buried in INFO is indistinguishable from a
working no-op arm (the 2026-07-25 lesson: a GPQA-30 "with it on" came back
byte-identical to the run with it off).

Never raises into serving: a drifted anchor degrades to "seed stays in the
prompt", which is exactly today's behaviour.
"""
from __future__ import annotations

import logging
import os
import pathlib
import py_compile
import shutil
import sys

LOG = "[pn114-seed-span]"
VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
TBS = VLLM / "v1/sample/thinking_budget_state.py"
SRV = VLLM / "entrypoints/openai/chat_completion/serving.py"
SIDECAR_SRC = pathlib.Path("/fixes/pn114_seed.py")
SIDECAR_DST = VLLM / "_genesis_pn114_seed.py"
MARKER = "# PN114-SEED"


# ═══════════════════════════════════════════════════════════════════════════
# Site S1a — the H119 provisional path `continue`s before the loop tail, so
# routed requests (exactly the ones this feature exists for) never reach S1b.
# Present only when patch_h119_lens_router.py wired its F site.
# ═══════════════════════════════════════════════════════════════════════════
S1A_OLD = (
    "                if _h119_on_add(self, index, params, prompt_tok_ids,\n"
    "                                output_tok_ids) and index in self._state:\n"
    "                    continue\n"
)
S1A_NEW = (
    "                if _h119_on_add(self, index, params, prompt_tok_ids,\n"
    "                                output_tok_ids) and index in self._state:\n"
    "                    # PN114-SEED S1a: the H119 provisional entry skips the\n"
    "                    # loop tail, so the stash has to happen here too.\n"
    "                    try:\n"
    "                        from vllm import _genesis_pn114_seed as _pn114seed\n"
    "                        _pn114seed.note_params(self._state[index], params)\n"
    "                    except Exception:\n"
    "                        import logging as _pn114seed_log1\n"
    "                        _pn114seed_log1.getLogger(\n"
    "                            'vllm.genesis.pn114seed'\n"
    "                        ).warning('PN114-SEED note raised', exc_info=True)\n"
    "                    continue\n"
)

# Site S1b — the loop tail, reached by the relaxed and budgeted branches.
S1B_OLD = (
    '            self._state[index]["output_tok_ids"] = output_tok_ids\n'
    '            self._state[index]["spec_token_ids"] = []\n'
)
S1B_NEW = (
    '            self._state[index]["output_tok_ids"] = output_tok_ids\n'
    '            self._state[index]["spec_token_ids"] = []\n'
    "            # PN114-SEED S1b: stash the seed the API server stripped out\n"
    "            # of the prompt; the arm at S2 is the only reader.\n"
    "            try:\n"
    "                from vllm import _genesis_pn114_seed as _pn114seed\n"
    "                _pn114seed.note_params(self._state[index], params)\n"
    "            except Exception:\n"
    "                import logging as _pn114seed_log2\n"
    "                _pn114seed_log2.getLogger(\n"
    "                    'vllm.genesis.pn114seed'\n"
    "                ).warning('PN114-SEED note raised', exc_info=True)\n"
)

# Legacy sync_batch (pins before upstream added `relaxed_thinking` and
# restructured the add loop — e.g. dev1474cherry-1711-20260725). There is no
# `continue` in that loop, so ONE anchor at the loop exit covers every path,
# including H119's provisional entry.
S1L_OLD = (
    "            elif not _h119_on_add(self, index, params, prompt_tok_ids,\n"
    "                                  output_tok_ids):\n"
    "                # H119-BUDGET: unchanged stock path. H119 returns False\n"
    "                # whenever it is off, unavailable, or declined the row.\n"
    "                self._state.pop(index, None)\n"
    "\n"
    "        for i1, i2, direction in batch_update.moved:\n"
)
S1L_NEW = (
    "            elif not _h119_on_add(self, index, params, prompt_tok_ids,\n"
    "                                  output_tok_ids):\n"
    "                # H119-BUDGET: unchanged stock path. H119 returns False\n"
    "                # whenever it is off, unavailable, or declined the row.\n"
    "                self._state.pop(index, None)\n"
    "            # PN114-SEED S1 (legacy loop shape): no `continue` above, so\n"
    "            # one stash at the loop tail covers every surviving entry.\n"
    "            try:\n"
    "                _pn114seed_entry = self._state.get(index)\n"
    "                if _pn114seed_entry is not None:\n"
    "                    from vllm import _genesis_pn114_seed as _pn114seed\n"
    "                    _pn114seed.note_params(_pn114seed_entry, params)\n"
    "            except Exception:\n"
    "                import logging as _pn114seed_log0\n"
    "                _pn114seed_log0.getLogger(\n"
    "                    'vllm.genesis.pn114seed'\n"
    "                ).warning('PN114-SEED note raised', exc_info=True)\n"
    "\n"
    "        for i1, i2, direction in batch_update.moved:\n"
)

# S1 variants, counted, most-recent shape first. "h119" is the shipped shape
# on the boot pin; "legacy" is the older add loop; "stock" is a boot where the
# lens router soft-skipped its F site — there are no provisional entries then,
# so the loop tail alone covers every tracked row.
S1_VARIANTS = (
    ("S1-stash/h119", ((S1A_OLD, S1A_NEW), (S1B_OLD, S1B_NEW))),
    ("S1-stash/legacy", ((S1L_OLD, S1L_NEW),)),
    ("S1-stash/stock", ((S1B_OLD, S1B_NEW),)),
)


# ═══════════════════════════════════════════════════════════════════════════
# Site S2 — the arm seat. Anchored on PN114 graft G's tail so it lands after
# the pn108/pn112/pn114 observers and immediately before _update_think_state,
# which is what turns `in_end` into force_index=[0] on this very step.
# Requires patch_pn114_forced_span.py: without its force_seq machinery there
# is nothing to arm.
# ═══════════════════════════════════════════════════════════════════════════
S2_OLD = (
    "                ).warning('PN114 observe raised', exc_info=True)\n"
    "            self._update_think_state(state)\n"
)
S2_NEW = (
    "                ).warning('PN114 observe raised', exc_info=True)\n"
    "            # PN114-SEED S2: arm the forced seed span. Runs after H119\n"
    "            # resolved this request's routed budget (top of update_state)\n"
    "            # and before _update_think_state, which converts in_end into\n"
    "            # force_index=[0] while output is still empty. The module\n"
    "            # declines every case where position 0 is already gone.\n"
    "            try:\n"
    "                from vllm import _genesis_pn114_seed as _pn114seed\n"
    "                _pn114seed.maybe_arm(\n"
    "                    state, len(self.think_start_token_ids),\n"
    "                    req_id=getattr(\n"
    "                        self, '_genesis_req_id_by_index', {}\n"
    "                    ).get(seq_idx),\n"
    "                )\n"
    "            except Exception:\n"
    "                import logging as _pn114seed_log3\n"
    "                _pn114seed_log3.getLogger(\n"
    "                    'vllm.genesis.pn114seed'\n"
    "                ).warning('PN114-SEED arm raised', exc_info=True)\n"
    "            self._update_think_state(state)\n"
)


# ═══════════════════════════════════════════════════════════════════════════
# Site S3 — completion divert, AHEAD of pn114's. A seed span must return to
# think mode; pn114.on_force_complete does not know the phase and would fall
# through to the holder's answer-mode reset, which closes the think block.
# ═══════════════════════════════════════════════════════════════════════════
S3_OLD = (
    "                _pn114_handled = False\n"
    "                try:\n"
    "                    from vllm._genesis.plateau import pn114 as _pn114_mod\n"
    "                    _pn114_handled = _pn114_mod.on_force_complete(state)\n"
)
S3_NEW = (
    "                _pn114_handled = False\n"
    "                # PN114-SEED S3: our span resumes THINK mode; pn114 would\n"
    "                # not recognise the phase and the answer-mode reset below\n"
    "                # would close the block after 12 forced tokens.\n"
    "                try:\n"
    "                    from vllm import _genesis_pn114_seed as _pn114seed\n"
    "                    _pn114_handled = _pn114seed.on_force_complete(state)\n"
    "                except Exception:\n"
    "                    import logging as _pn114seed_log4\n"
    "                    _pn114seed_log4.getLogger(\n"
    "                        'vllm.genesis.pn114seed'\n"
    "                    ).warning('PN114-SEED complete raised', exc_info=True)\n"
    "                try:\n"
    "                    from vllm._genesis.plateau import pn114 as _pn114_mod\n"
    "                    if not _pn114_handled:\n"
    "                        _pn114_handled = _pn114_mod.on_force_complete(state)\n"
)


# ═══════════════════════════════════════════════════════════════════════════
# Site S4 — serving.py. Anchored on PN101's hint block, which is where PN102
# has just written pn_env_seed into chat_template_kwargs; the strip must sit
# after it and before the request is rendered / turned into SamplingParams.
# ═══════════════════════════════════════════════════════════════════════════
S4_OLD = (
    "            _pn101_hint(request)\n"
    "        except Exception:\n"
)
S4_NEW = (
    "            _pn101_hint(request)\n"
    "            # PN114-SEED S4: with the forced-span seed armed, PN102's seed\n"
    "            # must NOT be rendered into the prompt — the engine forces the\n"
    "            # same ids as OUTPUT at the same positions instead. Fail-closed:\n"
    "            # a seed the boot table has not proven splits exactly is left\n"
    "            # in the prompt, and the engine declines it symmetrically.\n"
    "            try:\n"
    "                from vllm import _genesis_pn114_seed as _pn114seed\n"
    "                _pn114seed.strip_prompt_seed(request)\n"
    "            except Exception:\n"
    "                import logging as _pn114seed_log5\n"
    "                _pn114seed_log5.getLogger(\n"
    "                    'vllm.genesis.pn114seed'\n"
    "                ).warning('PN114-SEED strip raised', exc_info=True)\n"
    "        except Exception:\n"
)


def resolve_tbs_sites(text: str):
    """(sites, problem) for thinking_budget_state.py against THIS text."""
    fixed = {"S2-arm": (S2_OLD, S2_NEW), "S3-complete": (S3_OLD, S3_NEW)}
    bad = {n: text.count(o) for n, (o, _) in fixed.items()
           if text.count(o) != 1}
    sites = []
    detail = []
    for vname, pairs in S1_VARIANTS:
        counts = [text.count(old) for old, _ in pairs]
        if all(c == 1 for c in counts):
            sites.extend((f"{vname}[{i}]", old, new)
                         for i, (old, new) in enumerate(pairs))
            break
        detail.append(f"{vname}={counts}")
    else:
        bad["S1-stash"] = "no variant matched (" + ", ".join(detail) + ")"
    for name, (old, new) in fixed.items():
        sites.append((name, old, new))
    if bad:
        return None, ", ".join(f"{n} count={c}" for n, c in bad.items())
    return sites, None


def resolve_srv_sites(text: str):
    """(sites, problem) for chat_completion/serving.py against THIS text."""
    n = text.count(S4_OLD)
    if n != 1:
        return None, (f"S4-strip count={n} (PN101's hint block is the anchor; "
                      f"patch_pn101_answer_rescue.py must run first)")
    return [("S4-strip", S4_OLD, S4_NEW)], None


def _requested() -> bool:
    v = os.environ.get("GENESIS_ENABLE_PN114_SEED_SPAN", "")
    return v.strip().lower() in ("1", "true", "yes", "on")


def _shout(what: str, detail: str) -> None:
    """A requested-but-not-installed site group must never read as INFO."""
    bar = "=" * 72
    msg = (f"{bar}\n"
           f"{LOG} ERROR: {what} was REQUESTED but is NOT INSTALLED.\n"
           f"{LOG} ERROR: {detail}\n"
           f"{LOG} ERROR: this boot will behave EXACTLY as if the flag were "
           f"off — any A/B against it measures nothing.\n"
           f"{LOG} ERROR: re-derive the anchors against POST-PATCH content: "
           f"python3 /fixes/verify_pn114_seed_anchors.py\n"
           f"{bar}")
    print(msg, file=sys.stderr, flush=True)
    try:
        logging.getLogger("genesis.pn114seed").error(msg.replace("\n", " | "))
    except Exception:  # noqa: BLE001 — logging must never break a boot
        pass


def main() -> int:
    if not TBS.exists() or not SRV.exists():
        missing = [str(p) for p in (TBS, SRV) if not p.exists()]
        print(f"{LOG} soft-skip: {missing} absent on this pin — seed span NOT "
              f"wired (prompt-rendered seed unchanged)")
        if _requested():
            _shout("GENESIS_ENABLE_PN114_SEED_SPAN=1 (sites S1-S4)",
                   f"target file(s) missing: {missing}")
        return 0

    tbs_text = TBS.read_text(encoding="utf-8")
    srv_text = SRV.read_text(encoding="utf-8")
    if MARKER in tbs_text and MARKER in srv_text:
        print(f"{LOG} already applied — skipping")
        return 0
    if MARKER in tbs_text or MARKER in srv_text:
        # Half-applied is the one state this patch must never leave behind.
        print(f"{LOG} soft-skip: marker present in exactly one of the two "
              f"targets — refusing to complete a half-applied install")
        if _requested():
            _shout("GENESIS_ENABLE_PN114_SEED_SPAN=1 (sites S1-S4)",
                   "one target already carries the marker and the other does "
                   "not; re-create the container rather than patching over it")
        return 0

    tbs_sites, tbs_problem = resolve_tbs_sites(tbs_text)
    srv_sites, srv_problem = resolve_srv_sites(srv_text)
    if tbs_problem or srv_problem:
        problem = "; ".join(p for p in (tbs_problem, srv_problem) if p)
        print(f"{LOG} soft-skip S1-S4: {problem} — anchors do not fit this "
              f"pin AS PATCHED BY THE EARLIER /fixes PATCHES; seed span NOT "
              f"wired (prompt-rendered seed unchanged)")
        if _requested():
            _shout("GENESIS_ENABLE_PN114_SEED_SPAN=1 (sites S1-S4)", problem)
        return 0

    # Build both files fully before writing either: all-or-nothing.
    new_tbs = tbs_text
    for _name, old, new in tbs_sites:
        new_tbs = new_tbs.replace(old, new, 1)
    new_srv = srv_text
    for _name, old, new in srv_sites:
        new_srv = new_srv.replace(old, new, 1)

    try:
        shutil.copy2(SIDECAR_SRC, SIDECAR_DST)
    except OSError as e:
        print(f"{LOG} soft-skip: sidecar install failed ({e}) — seed span NOT "
              f"wired")
        if _requested():
            _shout("GENESIS_ENABLE_PN114_SEED_SPAN=1 (sites S1-S4)",
                   f"could not install {SIDECAR_DST}: {e}")
        return 0

    try:
        TBS.write_text(new_tbs, encoding="utf-8")
        SRV.write_text(new_srv, encoding="utf-8")
        py_compile.compile(str(TBS), doraise=True)
        py_compile.compile(str(SRV), doraise=True)
        py_compile.compile(str(SIDECAR_DST), doraise=True)
    except Exception as e:  # noqa: BLE001
        # Restore both files rather than serve a half-written holder.
        try:
            TBS.write_text(tbs_text, encoding="utf-8")
            SRV.write_text(srv_text, encoding="utf-8")
            SIDECAR_DST.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        print(f"{LOG} soft-skip: write/compile failed ({e}) — both targets "
              f"restored, seed span NOT wired", file=sys.stderr)
        if _requested():
            _shout("GENESIS_ENABLE_PN114_SEED_SPAN=1 (sites S1-S4)", str(e))
        return 0

    _drop_stale_pyc()
    names = [n for n, _, _ in tbs_sites] + [n for n, _, _ in srv_sites]
    print(f"{LOG} applied {names}; inert unless "
          f"GENESIS_ENABLE_PN114_SEED_SPAN=1 "
          f"(mode: GENESIS_PN114_SEED_MODE=mirror|routed)")
    return 0


def _drop_stale_pyc() -> None:
    for target in (TBS, SRV, SIDECAR_DST):
        cache = target.parent / "__pycache__"
        if not cache.is_dir():
            continue
        for pyc in cache.glob(target.stem + ".*.pyc"):
            try:
                pyc.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
