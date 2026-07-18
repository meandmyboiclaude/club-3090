#!/usr/bin/env python3
"""PN106D — TEMPORARY ground-zero probe for BUG-076 (bench only).

When VLLM_COMPUTE_NANS_IN_LOGITS detects NaN logits rows, dump the batch's
align-mode mamba state-slot arithmetic: per request — req_id, num computed
tokens, seq len, the (seq_len-1)//mamba_block_size slot start index, and the
gathered block-table columns. One WARNING per NaN event names exactly which
state slot the victim read, so read-before-write vs seq-len skew is decided
by data instead of theory. Remove after PN106 (root fix) lands.

Anchor: the _get_nans_in_logits call in GPUModelRunner._sample (gated on the
same env, so zero cost when detection is off).
"""
import pathlib
import sys

LOG = "[pn106d-nan-slot-audit]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
)
MARKER = "# PN106D:"

SITE_OLD = (
    "        num_nans_in_logits = {}\n"
    "        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:\n"
    "            num_nans_in_logits = self._get_nans_in_logits(logits)\n"
)

SITE_NEW = (
    "        num_nans_in_logits = {}\n"
    "        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:\n"
    "            num_nans_in_logits = self._get_nans_in_logits(logits)\n"
    "            # PN106D: BUG-076 ground-zero probe — on any NaN event, dump\n"
    "            # the align-mode mamba state-slot arithmetic for the batch.\n"
    "            if any(num_nans_in_logits.values()):\n"
    "                try:\n"
    "                    _p6_bs = self.input_batch.num_reqs\n"
    "                    _p6_rows = []\n"
    "                    _p6_mbs = (\n"
    "                        self.vllm_config.cache_config.mamba_block_size or 0\n"
    "                    )\n"
    "                    for _p6_i in range(_p6_bs):\n"
    "                        _p6_rid = self.input_batch.req_ids[_p6_i]\n"
    "                        _p6_comp = int(\n"
    "                            self.input_batch.num_computed_tokens_cpu[_p6_i]\n"
    "                        )\n"
    "                        _p6_seql = int(self.seq_lens[_p6_i].item())\n"
    "                        _p6_slot = (\n"
    "                            max(_p6_seql - 1, 0) // _p6_mbs if _p6_mbs else -1\n"
    "                        )\n"
    "                        _p6_rows.append(\n"
    "                            (\n"
    "                                _p6_rid[-9:],\n"
    "                                _p6_comp,\n"
    "                                _p6_seql,\n"
    "                                _p6_slot,\n"
    "                                int(num_nans_in_logits.get(_p6_rid, 0)),\n"
    "                            )\n"
    "                        )\n"
    "                    logger.warning(\n"
    "                        \"PN106D nan-event: mamba_block=%s rows\"\n"
    "                        \"(id,computed,seq_len,slot,nans)=%s\",\n"
    "                        _p6_mbs,\n"
    "                        _p6_rows,\n"
    "                    )\n"
    "                except Exception as _p6_e:\n"
    "                    logger.warning(\"PN106D probe error: %r\", _p6_e)\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"{LOG} already applied — skip")
        return 0
    n = src.count(SITE_OLD)
    if n != 1:
        print(f"{LOG} FATAL: anchor hits={n} — upstream drifted", file=sys.stderr)
        return 1
    TARGET.write_text(src.replace(SITE_OLD, SITE_NEW, 1), encoding="utf-8")
    print(f"{LOG} applied: NaN events now dump state-slot arithmetic")
    return 0


sys.exit(main())
