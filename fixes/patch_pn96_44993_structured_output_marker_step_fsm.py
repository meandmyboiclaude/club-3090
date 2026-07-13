#!/usr/bin/env python3
"""PN96 — advance the grammar FSM across the reasoning boundary (marker step).

Backport of vllm#44993 (OPEN, stacked on merged #44297) to the dev1060 pin
9e57de71, 2 files:

  - vllm/v1/structured_output/__init__.py :: should_advance
  - vllm/v1/core/sched/scheduler.py       :: update_from_output call site

Bug (BUG-070, upstream vllm#48228): with response_format json_schema
(xgrammar) + MTP spec-decode + a reasoning parser, ~75% of responses start
with a DOUBLED first token (content = `{{"name": ...`). Mechanism: at the
decode step whose spec window contains `</think>`, the merged #44297 bitmask
code correctly constrains post-marker positions from the grammar ROOT state,
so the step emits `[..., </think>, {]`. But `should_advance()` flips
`reasoning_ended = True` and then RETURNS FALSE for every constraint type
except STRUCTURAL_TAG ("defer FSM advance until the next pass") — so the
scheduler never calls `grammar.accept_tokens` for that step and the emitted
`{` never enters the FSM, despite already being streamed to the client. The
next step builds its bitmask from the still-ROOT FSM, forcing the model to
emit `{` a second time. The grammar itself only ever sees one `{`, which is
why the remainder is grammar-perfect and finish_reason=stop.

Fix (upstream-faithful, #44993): should_advance grows an optional
`new_token_ids` parameter — the exact tokens appended this step — used as
the reasoning-end delta window (the placeholder-derived fallback also breaks
under async scheduling + spec decode when drafts are rejected, vllm#43388).
On boundary detection it records `reasoning_end_token_index` and returns
True for ALL constraint types (the STRUCTURAL_TAG-only carve-out is
removed); the scheduler's existing `trim_reasoning_for_advance` (#44297,
already in this pin) drops the reasoning prefix so the FSM advances through
exactly the post-marker tokens emitted in the marker step.

Supersedes Genesis P62 (vllm#36138 backport): P62 NEVER APPLIES on dev1060
(and did not apply on dev799 either) — boot log `[Genesis] DRIFT skipped:
P62 ... anchor drifted` — because upstream's #44297 rewrote both of its
anchor regions. Sander's sndr_core_engine P62 module names #44297+#44993 as
P62's retirement path via drift markers; this patcher is that path for our
pin. Keep GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING harmless (it skips); if
genesis ever re-anchors P62 for dev1060+, retire P62 — P62 and PN96 patch
the same function and MUST NOT both apply.

Retire when the pin advances past vllm#44993 (upstream merge): the patcher
self-retires per file when the upstream signature/call-site text is already
present (drift markers verbatim from `gh pr diff 44993`, matching Sander's
DRIFT_MARKER_44993_* convention).

Anchor drift vs PR: none — the pin (dev1060) matches the PR base text
byte-exact in all three hunks AFTER the full genesis + /fixes chain
(verified against the live container's as-patched files 2026-07-13; genesis
P58/P34 edits sit elsewhere in scheduler.py, P62/PN58 drift-skip, and no
other patcher touches these regions). Analysis + evidence:
diagnostics/tq-lane/BUG-070-ANALYSIS.md. Fix gate:
diagnostics/tq-lane/canary_grammar_mtp.py (pass = 0/12 `{{`).
"""
import pathlib
import sys

LOG = "[pn96-structured-output-marker-step-fsm]"
VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
SO_TARGET = VLLM / "v1/structured_output/__init__.py"
SCHED_TARGET = VLLM / "v1/core/sched/scheduler.py"
MARKER = "# PN96:"

# ── structured_output/__init__.py hunk 1: should_advance signature ──
SO_SIG_OLD = (
    '    def should_advance(self, request: "Request") -> bool:\n'
)
SO_SIG_NEW = (
    "    # PN96: vllm#44993 backport — optional new_token_ids (this step's\n"
    "    # appended tokens) as the reasoning-end delta window.\n"
    "    def should_advance(\n"
    "        self,\n"
    '        request: "Request",\n'
    "        new_token_ids: list[int] | None = None,\n"
    "    ) -> bool:\n"
)

# ── structured_output/__init__.py hunk 2: delta window + defer removal ──
# Byte-exact block from the pin (should_advance body, after the
# reasoning_ended short-circuit) — includes the STRUCTURAL_TAG-only defer
# carve-out that #44993 deletes.
SO_BODY_OLD = (
    "        # Check if reasoning ends in *this* step\n"
    "        delta_from = request.num_computed_tokens - request.num_output_placeholders\n"
    "        all_token_ids = request.all_token_ids\n"
    "        start = (\n"
    "            delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)\n"
    "        )\n"
    "        if reasoner.is_reasoning_end_streaming(\n"
    "            all_token_ids, itertools.islice(all_token_ids, start, None)\n"
    "        ):\n"
    "            structured_req.reasoning_ended = True\n"
    "\n"
    "            # Reasoning just ended this step. Defer FSM advance until the next\n"
    "            # pass (see reasoning_ended check above) for JSON/regex/choice/grammar:\n"
    "            # advancing on the closing boundary token can accept tokens that still\n"
    "            # belong to the reasoning stream. Structural tags are the only safe\n"
    "            # same-step exception: they model phased output (e.g. thinking tag ->\n"
    "            # answer tag), and speculative decoding must run grammar.validate_tokens\n"
    "            # on draft tokens produced immediately after that transition.\n"
    "            if (\n"
    "                self.vllm_config.speculative_config is not None\n"
    "                and structured_req.structured_output_key[0]\n"
    "                == StructuredOutputOptions.STRUCTURAL_TAG\n"
    "            ):\n"
    "                # The scheduler will advance the grammar with this step's\n"
    "                # tokens right away, but the step still contains reasoning\n"
    "                # content up to and including the end marker. Record where\n"
    "                # it ends so trim_reasoning_for_advance() can drop it.\n"
    "                structured_req.reasoning_end_token_index = (\n"
    "                    self._find_reasoning_end_index(reasoner, all_token_ids, start)\n"
    "                )\n"
    "                return True\n"
    "\n"
    "        return False\n"
)
SO_BODY_NEW = (
    "        # PN96: vllm#44993 backport (fixes vllm#48228 doubled first constrained\n"
    "        # token + vllm#43388 missed boundary under async+spec).\n"
    "        # Check if reasoning ends in *this* step.\n"
    "        # When the caller passes new_token_ids (the tokens that were just\n"
    "        # appended this step), use it directly as the delta window. The\n"
    "        # placeholder-derived fallback assumes num_output_placeholders ==\n"
    "        # len(new_token_ids), which breaks under async scheduling + spec\n"
    "        # decode when some drafts are rejected (#43388): the placeholder\n"
    "        # count remains > 0 after the step and the computed delta window\n"
    "        # starts past the reasoning-end marker.\n"
    "        all_token_ids = request.all_token_ids\n"
    "        if new_token_ids:\n"
    "            # The tokens were already appended this step, so the step window\n"
    "            # starts exactly len(new_token_ids) from the end.\n"
    "            start = len(all_token_ids) - len(new_token_ids)\n"
    "            delta_ids: Iterable[int] = new_token_ids\n"
    "        else:\n"
    "            delta_from = (\n"
    "                request.num_computed_tokens - request.num_output_placeholders\n"
    "            )\n"
    "            start = (\n"
    "                delta_from\n"
    "                if delta_from >= 0\n"
    "                else max(len(all_token_ids) + delta_from, 0)\n"
    "            )\n"
    "            delta_ids = itertools.islice(all_token_ids, start, None)\n"
    "        if reasoner.is_reasoning_end_streaming(all_token_ids, delta_ids):\n"
    "            structured_req.reasoning_ended = True\n"
    "\n"
    "            # PN96: no STRUCTURAL_TAG carve-out — EVERY constraint type records\n"
    "            # the boundary and returns True; the scheduler trims the reasoning\n"
    "            # prefix (trim_reasoning_for_advance, #44297) and advances the FSM\n"
    "            # through the post-marker tokens emitted in THIS step, so the next\n"
    "            # step's bitmask is built from the advanced state (no re-emission).\n"
    "            end_index = self._find_reasoning_end_index(\n"
    "                reasoner, all_token_ids, start\n"
    "            )\n"
    "\n"
    "            structured_req.reasoning_end_token_index = end_index\n"
    "            return True\n"
    "\n"
    "        return False\n"
)

# ── scheduler.py hunk 3: update_from_output call site passes new_token_ids ──
SCHED_CALL_OLD = (
    "            if new_token_ids and self.structured_output_manager.should_advance(request):\n"
)
SCHED_CALL_NEW = (
    "            # PN96: vllm#44993 — pass this step's tokens so the boundary is\n"
    "            # detected and the FSM advances in the marker step itself.\n"
    "            if new_token_ids and self.structured_output_manager.should_advance(\n"
    "                request, new_token_ids=new_token_ids\n"
    "            ):\n"
)

# Upstream-merged drift markers (verbatim from `gh pr diff 44993`; same
# strings sndr_core_engine uses as DRIFT_MARKER_44993_*). Checked only when
# MARKER is absent, so our own emitted text never self-collides.
SO_DRIFT = (
    "        new_token_ids: list[int] | None = None,\n"
    "    ) -> bool:"
)
SCHED_DRIFT = (
    "self.structured_output_manager.should_advance(\n"
    "                request, new_token_ids=new_token_ids\n"
    "            )"
)


def _apply(target: pathlib.Path, drift: str, drift_what: str,
           hunks: list[tuple[str, str, str]]) -> int:
    text = target.read_text()
    if MARKER in text:
        print(f"{LOG} {target.name}: already applied (idempotent)")
        return 0
    if drift in text:
        print(f"{LOG} {target.name}: upstream drift — {drift_what} already "
              f"present (vllm#44993 merged?) — self-retire (no-op)")
        return 0
    for name, old, _ in hunks:
        if old not in text:
            print(f"{LOG} FATAL: anchor-not-found ({target.name}/{name}) — "
                  f"upstream refactor past #44297; re-derive before boot "
                  f"(json_schema x MTP first-token duplication `{{{{` returns "
                  f"without this fix — see BUG-070-ANALYSIS.md)",
                  file=sys.stderr)
            return 1
        if text.count(old) != 1:
            print(f"{LOG} FATAL: ambiguous anchor ({target.name}/{name})",
                  file=sys.stderr)
            return 1
    for _, old, new in hunks:
        text = text.replace(old, new, 1)
    target.write_text(text)
    print(f"{LOG} {target.name}: applied {len(hunks)} hunk(s)")
    return 0


def main() -> int:
    for t in (SO_TARGET, SCHED_TARGET):
        if not t.exists():
            print(f"{LOG} FATAL: {t} not present", file=sys.stderr)
            return 1
    # Precondition sanity: this backport composes with the #44297 machinery
    # (trim_reasoning_for_advance) already native in the pin. FAIL-LOUD if a
    # future pin removes it — the fix depends on it.
    if "def trim_reasoning_for_advance" not in SO_TARGET.read_text():
        print(f"{LOG} FATAL: trim_reasoning_for_advance missing "
              f"(#44297 machinery gone) — PN96 depends on it; re-derive",
              file=sys.stderr)
        return 1
    rc = _apply(SO_TARGET, SO_DRIFT, "new should_advance signature", [
        ("should-advance-sig", SO_SIG_OLD, SO_SIG_NEW),
        ("delta-window-defer-removal", SO_BODY_OLD, SO_BODY_NEW),
    ])
    if rc:
        return rc
    rc = _apply(SCHED_TARGET, SCHED_DRIFT, "updated call site", [
        ("update-from-output-callsite", SCHED_CALL_OLD, SCHED_CALL_NEW),
    ])
    if rc:
        return rc
    print(f"{LOG} applied: grammar FSM now advances across the reasoning "
          f"boundary in the marker step (vllm#44993 backport; fixes BUG-070 "
          f"`{{{{` duplication, vllm#48228)")
    return 0


sys.exit(main())
