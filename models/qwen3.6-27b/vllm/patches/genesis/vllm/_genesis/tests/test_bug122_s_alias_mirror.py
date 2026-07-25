#!/usr/bin/env python3
"""BUG-122 regression test: S-alias must reach the module-level gate.

The phantom-apply bug: sndr_lane.apply_policy installs GENESIS_ENABLE_S<BARE>
as an entry-level env_flag_alias. The dispatcher honors it and announces
"APPLY PN71", but each module's private _enabled() reads the CANONICAL flag,
finds it unset, and returns "skipped" without logging. Result: the record DB
says applied while the target file was never touched.

This test asserts the mirror: a truthy S-alias sets the canonical flag, which
is the only thing the module's own gate ever reads.

Run: python3 test_bug122_s_alias_mirror.py
"""
import os
import sys
import pathlib

# _genesis package root (…/genesis/vllm/_genesis) — tests/ sits directly under it.
_GEN = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_GEN.parent.parent))  # …/genesis  → import vllm._genesis

FLAGS = {
    "PN71": ("GENESIS_ENABLE_SPN71_THINKING_TAG_NORMALIZE",
             "GENESIS_ENABLE_PN71_THINKING_TAG_NORMALIZE"),
    "PN73": ("GENESIS_ENABLE_SPN73_TOOL_ARGS_SAFE_NORMALIZE",
             "GENESIS_ENABLE_PN73_TOOL_ARGS_SAFE_NORMALIZE"),
    "PN92": ("GENESIS_ENABLE_SPN92_NIXL_EP_TRIAL_IMPORT",
             "GENESIS_ENABLE_PN92_NIXL_EP_TRIAL_IMPORT"),
}

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main():
    # Arrange: exactly the compose's state — S-form set, canonical unset.
    for s_flag, bare in FLAGS.values():
        os.environ[s_flag] = "1"
        os.environ.pop(bare, None)

    try:
        from vllm._genesis.patches import sndr_lane
    except Exception as exc:  # pragma: no cover - import-env dependent
        print(f"SKIP: cannot import sndr_lane outside the container ({exc})")
        return 0

    print("BUG-122: S-alias -> canonical flag mirror")
    # apply_policy() reaches into the vendored sndr package; the runtime does
    # this via bootstrap_sndr_alias() before any sndr import.
    try:
        sndr_lane.bootstrap_sndr_alias()
    except Exception as exc:  # pragma: no cover - import-env dependent
        print(f"SKIP: vendored sndr not importable here ({exc})")
        return 0
    summary = sndr_lane.apply_policy()

    check("s_alias_mirrored" in summary,
          "apply_policy reports an s_alias_mirrored field")

    for pid, (s_flag, bare) in FLAGS.items():
        check(os.environ.get(bare) == "1",
              f"{pid}: canonical {bare} set from {s_flag}")

    # The mirror must never clobber an explicit operator value: if the canonical
    # flag is already present (even "0"), the operator's choice wins.
    pid = "PN71"
    s_flag, bare = FLAGS[pid]
    os.environ[bare] = "0"
    sndr_lane.apply_policy()
    check(os.environ.get(bare) == "0",
          f"{pid}: explicit operator {bare}=0 is NOT overwritten by the mirror")

    print(f"\n== RESULT: {len(FLAGS) + 2 - len(failures)} pass / "
          f"{len(failures)} fail ==")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
