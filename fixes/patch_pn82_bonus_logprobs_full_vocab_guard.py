#!/usr/bin/env python3
"""PN82 — don't let logprob_token_ids override the full-vocab logprobs sentinel.

Upstream bug (present at least dev424..dev799+HEAD, vllm/v1/sample/sampler.py):
the generative_scoring API sets sampling_metadata.logprob_token_ids, and
Sampler.forward "prefers" the specific-token gather whenever num_logprobs is
not None. But RejectionSampler's bonus-token call uses max_num_logprobs=-1 as
an internal sentinel meaning "return FULL raw/processed logits — needed to
compute accepted-token logprobs" (rejection_sampler.py:131-141). -1 is not
None, so a generative_scoring request sharing a batch with ANY spec-decode
request replaces the [n, vocab] tensor with the gathered [n, k+1] values →
    final_logits[bonus_logits_indices] = bonus_logits
    RuntimeError: shape mismatch: [3, 2] vs [3, 248320]
→ EngineCore fatal. Live-reproduced 2026-07-07 02:30 on dev799 (MTP n=3 +
/generative_scoring probe during chat load).

Fix (hunk 1, sampler.py): only prefer the specific-token gather for real
top-k requests (num_logprobs >= 0), never for the -1 full-tensor sentinel.

Fix (hunks 2+3, rejection_sampler.py, v2): the spec-decode logprobs path has
NO specific-token gather at all — a generative_scoring request batched with
MTP requests gets top-k logprobs only, so the serving layer fails with
"Token IDs [...] not found in logprobs" (HTTP 400; live-reproduced 2026-07-07
02:41 with a /rerank call under streaming chat). v2 threads sampling_metadata
into _get_logprobs_tensors and, when any request in the batch carries
logprob_token_ids, gathers those specific ids for every accepted-token row
(rows map req-major, width = max_spec_len+1). Rows of requests without ids
get [sampled, -inf...] padding — harmless, those requests never asked for
logprobs. Known simplification: a batch mixing a top-k-logprobs chat request
AND a generative_scoring request returns specific-ids only (nothing on aibox
sends top-k logprobs chat traffic).
"""
import pathlib
import sys

LOG = "[pn82-bonus-logprobs-guard]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/sampler.py"
)
RS_TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py"
)
MARKER = "# PN82:"

OLD = (
    "        # If we have both num_logprobs and logprob_token_ids, prefer\n"
    "        # logprob_token_ids as it's more specific\n"
    "        if logprob_token_ids_tensors is not None and num_logprobs is not None:\n"
    "            logprobs_tensors = logprob_token_ids_tensors\n"
)
NEW = (
    "        # If we have both num_logprobs and logprob_token_ids, prefer\n"
    "        # logprob_token_ids as it's more specific.\n"
    "        # PN82: NEVER when num_logprobs == -1 — that is RejectionSampler's\n"
    "        # bonus-call sentinel demanding FULL raw logits ([n, vocab]); replacing\n"
    "        # them with the gathered [n, k+1] values crashes _get_logprobs_tensors\n"
    "        # (shape-mismatch EngineCore fatal) whenever generative_scoring shares\n"
    "        # a batch with spec-decode requests.\n"
    "        if (\n"
    "            logprob_token_ids_tensors is not None\n"
    "            and num_logprobs is not None\n"
    "            and num_logprobs >= 0\n"
    "        ):\n"
    "            logprobs_tensors = logprob_token_ids_tensors\n"
)


# ── rejection_sampler.py hunk 2: thread sampling_metadata into the call ──
RS_CALL_OLD = (
    "            logprobs_tensors = self._get_logprobs_tensors(\n"
    "                sampling_metadata.max_num_logprobs,\n"
    "                metadata,\n"
    "                logits,\n"
    "                target_logits if self.is_processed_logprobs_mode else raw_target_logits,\n"
    "                bonus_sampler_output.logprobs_tensors.logprobs,\n"
    "                output_token_ids,\n"
    "            )\n"
)
RS_CALL_NEW = (
    "            logprobs_tensors = self._get_logprobs_tensors(\n"
    "                sampling_metadata.max_num_logprobs,\n"
    "                metadata,\n"
    "                logits,\n"
    "                target_logits if self.is_processed_logprobs_mode else raw_target_logits,\n"
    "                bonus_sampler_output.logprobs_tensors.logprobs,\n"
    "                output_token_ids,\n"
    "                sampling_metadata=sampling_metadata,  # PN82: for logprob_token_ids\n"
    "            )\n"
)

# ── rejection_sampler.py hunk 3: signature + specific-token gather ──
RS_SIG_OLD = (
    "        bonus_logits: torch.Tensor,\n"
    "        sampled_token_ids: torch.Tensor,\n"
    "    ) -> LogprobsTensors:\n"
)
RS_SIG_NEW = (
    "        bonus_logits: torch.Tensor,\n"
    "        sampled_token_ids: torch.Tensor,\n"
    "        sampling_metadata=None,  # PN82: SamplingMetadata for logprob_token_ids\n"
    "    ) -> LogprobsTensors:\n"
)

RS_GATHER_OLD = (
    "        return self.sampler.gather_logprobs(\n"
    "            accepted_logprobs,\n"
    "            max_num_logprobs,\n"
    "            accepted_tokens.to(torch.int64),\n"
    "        )\n"
)
RS_GATHER_NEW = (
    "        # PN82: specific-token gather for generative_scoring under spec decode.\n"
    "        # Upstream only gathers top-k here, so label token ids are absent and\n"
    "        # the serving layer 400s with 'Token IDs not found in logprobs'.\n"
    "        # Rows are req-major with width = max_spec_len+1.\n"
    "        req_token_ids = getattr(sampling_metadata, \"logprob_token_ids\", None)\n"
    "        if req_token_ids:\n"
    "            _pn82_width = sampled_token_ids.shape[-1]\n"
    "            _pn82_expanded = {}\n"
    "            for _pn82_req, _pn82_tids in req_token_ids.items():\n"
    "                for _pn82_j in range(_pn82_width):\n"
    "                    _pn82_expanded[_pn82_req * _pn82_width + _pn82_j] = _pn82_tids\n"
    "            return self.sampler.gather_specific_token_logprobs(\n"
    "                accepted_logprobs,\n"
    "                _pn82_expanded,\n"
    "                accepted_tokens.to(torch.int64),\n"
    "            )\n"
    "        return self.sampler.gather_logprobs(\n"
    "            accepted_logprobs,\n"
    "            max_num_logprobs,\n"
    "            accepted_tokens.to(torch.int64),\n"
    "        )\n"
)


def _apply(target: pathlib.Path, hunks: list[tuple[str, str, str]]) -> int:
    text = target.read_text()
    if MARKER in text:
        print(f"{LOG} {target.name}: already applied (idempotent)")
        return 0
    for name, old, new in hunks:
        if old not in text:
            print(f"{LOG} FATAL: anchor-not-found ({target.name}/{name}) — re-derive "
                  f"(generative_scoring + spec-decode breaks without this)",
                  file=sys.stderr)
            return 1
        if text.count(old) != 1:
            print(f"{LOG} FATAL: ambiguous anchor ({target.name}/{name})", file=sys.stderr)
            return 1
    for name, old, new in hunks:
        text = text.replace(old, new, 1)
    target.write_text(text)
    print(f"{LOG} {target.name}: applied {len(hunks)} hunk(s)")
    return 0


def main() -> int:
    for t in (TARGET, RS_TARGET):
        if not t.exists():
            print(f"{LOG} FATAL: {t} not present", file=sys.stderr)
            return 1
    if "num_logprobs >= 0" in TARGET.read_text() and MARKER not in TARGET.read_text():
        print(f"{LOG} upstream drift: sampler guard already present — verify hunks 2-3 manually")
    rc = _apply(TARGET, [("sentinel-guard", OLD, NEW)])
    if rc:
        return rc
    rc = _apply(RS_TARGET, [
        ("call-site", RS_CALL_OLD, RS_CALL_NEW),
        ("signature", RS_SIG_OLD, RS_SIG_NEW),
        ("specific-gather", RS_GATHER_OLD, RS_GATHER_NEW),
    ])
    if rc:
        return rc
    print(f"{LOG} applied: full-vocab sentinel guard + spec-decode specific-token gather")
    return 0


sys.exit(main())
