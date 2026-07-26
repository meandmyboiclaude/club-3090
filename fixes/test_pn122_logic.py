#!/usr/bin/env python3
"""PN122 logic tests — no GPU, no container, no engine.

Copies the PINNED vllm sources into a temp tree, points the patch at that
tree, applies it for real, then exercises the grafted code with stub objects.
That is stronger than a hand reimplementation: the anchors, the emitted
markers and the resulting Python are the ones the boot will run.

    python3 fixes/test_pn122_logic.py
"""
import importlib.util
import pathlib
import shutil
import sys
import tempfile
import types

PIN = pathlib.Path("/var/tmp/led-vllm-pin/vllm")
PATCH = pathlib.Path(__file__).with_name("patch_pn122_structured_force_guard.py")
FILES = [
    "v1/structured_output/utils.py",
    "v1/worker/gpu_model_runner.py",
    "v1/sample/thinking_budget_state.py",
]
MARKERS = {
    "v1/structured_output/utils.py": ["# PN122 graft: row registry",
                                      "# PN122 graft: publish rows"],
    "v1/worker/gpu_model_runner.py": ["# PN122 graft: clear rows"],
    "v1/sample/thinking_budget_state.py": ["# PN122 graft: row helper",
                                           "# PN122 graft: skip grammar row"],
}

fails = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        fails.append(what)


def load_patch(base: pathlib.Path):
    spec = importlib.util.spec_from_file_location("pn122patch", PATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.BASE = base
    mod.SOUTIL = base / "v1/structured_output/utils.py"
    mod.RUNNER = base / "v1/worker/gpu_model_runner.py"
    mod.HOLDER = base / "v1/sample/thinking_budget_state.py"
    mod.GRAFTS = [
        (mod.SOUTIL, mod.MARK_A, mod.ANCH_A, mod.REPL_A, "A"),
        (mod.SOUTIL, mod.MARK_B, mod.ANCH_B, mod.REPL_B, "B"),
        (mod.RUNNER, mod.MARK_C, mod.ANCH_C, mod.REPL_C, "C"),
        (mod.HOLDER, mod.MARK_D2, mod.ANCH_D2, mod.REPL_D2, "D2"),
        (mod.HOLDER, mod.MARK_D, mod.ANCH_D, mod.REPL_D, "D"),
    ]
    return mod


def main() -> int:
    if not PIN.is_dir():
        print(f"SKIP: pinned tree missing at {PIN}")
        return 0

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pn122-"))
    base = tmp / "vllm"
    for rel in FILES:
        dst = base / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PIN / rel, dst)

    mod = load_patch(base)

    print("T1: patch applies cleanly against the pinned sources")
    check(mod.main() == 0, "main() returns 0")

    print("T2: every marker is present on disk, exactly once")
    for rel, marks in MARKERS.items():
        src = (base / rel).read_text(encoding="utf-8")
        for m in marks:
            check(src.count(m) == 1, f"{rel}: {m!r} x1")

    print("T3: idempotent — re-running is a no-op")
    before = {rel: (base / rel).read_text(encoding="utf-8") for rel in FILES}
    check(mod.main() == 0, "second main() returns 0")
    check(all((base / rel).read_text(encoding="utf-8") == before[rel]
              for rel in FILES), "no file changed on the second pass")

    print("T4: grafted files still parse")
    for rel in FILES:
        try:
            compile((base / rel).read_text(encoding="utf-8"), rel, "exec")
            check(True, f"{rel} compiles")
        except SyntaxError as exc:
            check(False, f"{rel} compiles ({exc})")

    print("T5: publish/clear/skip semantics, exercised on the grafted text")
    # Build a tiny module carrying ONLY the grafted registry + publish block,
    # so we can run them without importing torch.
    src = (base / "v1/structured_output/utils.py").read_text(encoding="utf-8")
    reg = src[src.index("PN122_STRUCTURED_ROWS"):src.index("def apply_grammar_bitmask(")]
    pub_start = src.index("    # PN122 graft: publish rows")
    pub_end = src.index("    out_indices = []", pub_start)
    pub = src[pub_start:pub_end]

    sou = types.ModuleType("vllm.v1.structured_output.utils")
    exec(reg, sou.__dict__)
    pkg = types.ModuleType("vllm.v1.structured_output")
    pkg.utils = sou
    for name, m in (("vllm", types.ModuleType("vllm")),
                    ("vllm.v1", types.ModuleType("vllm.v1")),
                    ("vllm.v1.structured_output", pkg),
                    ("vllm.v1.structured_output.utils", sou)):
        sys.modules[name] = m

    def run_publish(indices, spec, env):
        body = "def _pub(struct_out_req_batch_indices, spec_tokens):\n" + pub
        g = dict(sou.__dict__)
        g["os"] = __import__("os")
        old = __import__("os").environ.get(
            "GENESIS_ENABLE_PN122_STRUCTURED_FORCE_GUARD")
        if env is None:
            __import__("os").environ.pop(
                "GENESIS_ENABLE_PN122_STRUCTURED_FORCE_GUARD", None)
        else:
            __import__("os").environ[
                "GENESIS_ENABLE_PN122_STRUCTURED_FORCE_GUARD"] = env
        try:
            exec(body, g)
            g["_pub"](indices, spec)
        finally:
            if old is None:
                __import__("os").environ.pop(
                    "GENESIS_ENABLE_PN122_STRUCTURED_FORCE_GUARD", None)
            else:
                __import__("os").environ[
                    "GENESIS_ENABLE_PN122_STRUCTURED_FORCE_GUARD"] = old

    sou.PN122_STRUCTURED_ROWS.clear()
    run_publish({"r1": 3}, {}, None)
    check(sou.PN122_STRUCTURED_ROWS == set(),
          "flag unset -> nothing published (dark by default)")

    sou.PN122_STRUCTURED_ROWS.clear()
    run_publish({"r1": 3}, {}, "1")
    check(sou.PN122_STRUCTURED_ROWS == {3},
          "flag on, no spec tokens -> the single row")

    sou.PN122_STRUCTURED_ROWS.clear()
    run_publish({"r1": 3, "r2": 9}, {"r1": (1, 2)}, "1")
    check(sou.PN122_STRUCTURED_ROWS == {3, 4, 5, 9},
          "spec span covered for r1, r2 unaffected")

    # the holder helper, extracted from the grafted holder text
    hsrc = (base / "v1/sample/thinking_budget_state.py").read_text(
        encoding="utf-8")
    h_start = hsrc.index("    @staticmethod\n    def _pn122_row_masked")
    h_end = hsrc.index("    def _apply_forcing_to_logits(", h_start)
    helper = "\n".join(l[4:] if l.startswith("    ") else l
                       for l in hsrc[h_start:h_end].splitlines())
    g = {}
    exec(helper.replace("@staticmethod\n", "", 1), g)
    row_masked = g["_pn122_row_masked"]

    check(row_masked(3) is True, "helper: row 3 reported masked")
    check(row_masked(7) is False, "helper: row 7 not masked")
    sou.PN122_STRUCTURED_ROWS.clear()
    check(row_masked(3) is False,
          "helper: after clear, no row masked (no stale carry-over)")

    print("T6: the runner clear-graft precedes the grammar branch")
    rsrc = (base / "v1/worker/gpu_model_runner.py").read_text(encoding="utf-8")
    check(rsrc.index("PN122_STRUCTURED_ROWS.clear()")
          < rsrc.index("apply_grammar_bitmask(\n                scheduler_output"),
          "clear() runs before apply_grammar_bitmask")
    check(rsrc.index("PN122_STRUCTURED_ROWS.clear()")
          < rsrc.index("sampler_output = self._sample("),
          "clear() runs before sampling")

    shutil.rmtree(tmp, ignore_errors=True)
    print()
    if fails:
        print(f"FAILED {len(fails)}:")
        for f in fails:
            print("  -", f)
        return 1
    print("all PN122 logic tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
