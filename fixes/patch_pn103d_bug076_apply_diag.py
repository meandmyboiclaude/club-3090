#!/usr/bin/env python3
"""PN103D — TEMPORARY bench diagnostic for BUG-076 (not for prod compose).

Instruments vllm/v1/structured_output/utils.py::apply_grammar_bitmask to
discriminate the two candidate mechanisms behind token-0 "!" completions:
  (a) logits already garbage BEFORE the mask (all -inf / NaN row -> forward
      corruption, GDN state class), vs
  (b) an all-masked (popcount 0) bitmask row applied to a healthy logits row
      (CPU-side row accounting skew).
Logs a WARNING with the full (req, row, popcount) map only when an anomaly is
seen; silent otherwise. Remove after BUG-076 is closed.
"""
import pathlib
import sys

LOG = "[pn103d-apply-diag]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/utils.py"
)
MARKER = "# PN103D:"

SITE_OLD = (
    "    if not logits.is_cpu:\n"
    "        index_tensor = None\n"
    "        if not skip_out_indices:\n"
)

SITE_NEW = (
    "    # PN103D: BUG-076 anomaly probe — pre-mask logits health + applied-row\n"
    "    # popcounts. Bench-only diagnostic; WARN on (a) all-inf/NaN logits row\n"
    "    # (forward corruption) or (b) zero-popcount mask row (row skew).\n"
    "    try:\n"
    "        import numpy as _pn103d_np\n"
    "        _pn103d_rows = []\n"
    "        for _pn103d_rid in grammar_output.structured_output_request_ids:\n"
    "            _pn103d_li = struct_out_req_batch_indices.get(_pn103d_rid)\n"
    "            if _pn103d_li is None:\n"
    "                continue\n"
    "            _pn103d_ns = len(spec_tokens.get(_pn103d_rid, ()))\n"
    "            for _pn103d_i in range(1 + _pn103d_ns):\n"
    "                _pn103d_pc = int(\n"
    "                    _pn103d_np.unpackbits(\n"
    "                        sorted_bitmask[_pn103d_li + _pn103d_i].view(_pn103d_np.uint8)\n"
    "                    ).sum()\n"
    "                )\n"
    "                _pn103d_rows.append(\n"
    "                    (_pn103d_rid[-9:], _pn103d_li + _pn103d_i, _pn103d_pc)\n"
    "                )\n"
    "        _pn103d_zero = [_pn103d_r for _pn103d_r in _pn103d_rows if _pn103d_r[2] == 0]\n"
    "        _pn103d_prebad = []\n"
    "        for _pn103d_rid2, _pn103d_idx, _pn103d_pc in _pn103d_rows:\n"
    "            _pn103d_m = float(logits[_pn103d_idx].max().item())\n"
    "            if _pn103d_m != _pn103d_m or _pn103d_m == float(\"-inf\"):\n"
    "                _pn103d_prebad.append((_pn103d_rid2, _pn103d_idx, _pn103d_m))\n"
    "        if _pn103d_zero or _pn103d_prebad:\n"
    "            logger.warning(\n"
    '                "PN103D anomaly: zero_mask_rows=%s pre_mask_bad_logits=%s "\n'
    '                "rowmap=%s logits_shape=%s out_indices=%s skip=%s",\n'
    "                _pn103d_zero,\n"
    "                _pn103d_prebad,\n"
    "                _pn103d_rows,\n"
    "                tuple(logits.shape),\n"
    "                out_indices,\n"
    "                skip_out_indices,\n"
    "            )\n"
    "    except Exception as _pn103d_e:  # diagnostics must never break serving\n"
    '        logger.warning("PN103D probe error: %r", _pn103d_e)\n'
    "\n"
    "    if not logits.is_cpu:\n"
    "        index_tensor = None\n"
    "        if not skip_out_indices:\n"
)

IMPORT_OLD = "logger = init_logger(__name__)\n"


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"{LOG} already applied — skip")
        return 0
    if "logger" not in src.split("def apply_grammar_bitmask")[0]:
        # utils.py may not define a logger; add one after imports if missing.
        if IMPORT_OLD not in src:
            anchor = "import torch\n"
            if anchor not in src:
                print(f"{LOG} FATAL: no logger and no import anchor", file=sys.stderr)
                return 1
            src = src.replace(
                anchor,
                anchor
                + "from vllm.logger import init_logger  # PN103D: diag logger\n"
                + "logger = init_logger(__name__)  # PN103D: diag logger\n",
                1,
            )
    n = src.count(SITE_OLD)
    if n != 1:
        print(f"{LOG} FATAL: apply-site anchor hits={n}", file=sys.stderr)
        return 1
    TARGET.write_text(src.replace(SITE_OLD, SITE_NEW, 1), encoding="utf-8")
    print(f"{LOG} applied: anomaly probe live in apply_grammar_bitmask")
    return 0


sys.exit(main())
