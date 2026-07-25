#!/usr/bin/env python3
"""Count PN101's anchors as the BOOT will see them, on every live pin.

PN101's anchors do not live in the pristine image content alone. `apply_all`
rewrites the whole top of `_create_chat_completion` (genesis PN16 / PN40 /
edge-guard / P68-69 all re-emit the `# Streaming response` + tokenizer pair),
and two /fixes siblings — pn74 and pn100 — write serving.py before PN101's line
in the entrypoint. Counting against pristine image bytes is how a patch shipped
as a silent no-op here on 2026-07-25, so this replays the real thing: the boot
compose's environment plus its entrypoint script, then asks the patcher's own
resolver what it sees.

Three arms:

  boot-env      — the compose's environment verbatim, entrypoint truncated
                  immediately before the PN101 line. This is the live boot.
  comment-drift — the same, with upstream's `# Streaming response` comment
                  reworded. One comment, the most drift-prone thing a patch can
                  key on, and the shape that used to end the boot: the sole
                  anchor vanished, the patcher returned 1 and `set -e` killed
                  the entrypoint before PN71T ever ran (measured 2026-07-26).
                  PN101 must now resolve anyway, via a code-only variant.
  drift-matrix  — seven upstream-drift shapes applied to the REAL replayed boot
                  file (comment reworded / duplicated, tokenizer fetch renamed,
                  return arguments reflowed, kv-transfer wrapper dropped, every
                  hint landmark gone, anchor drifted ahead of PN100), each
                  asserting WHICH variant absorbs it and that PN114-SEED's S4
                  anchor survives. None of them may exit non-zero.
  chain         — the ENTIRE entrypoint replayed with `set -e` INTACT, through
                  PN71T and PN114-SEED, asserting rc == 0, that every marker
                  landed, that PN114-SEED's S4 site still finds PN101's hint
                  block, and that serving.py compiles after all of them. Their
                  insertion offsets interact; nothing else proves they compose.

    python3 fixes/verify_pn101_anchors.py [--pin TAG] [--arm NAME] [--keep DIR]

No GPU, no serving container touched — one throwaway container per arm. The
gpu-guard line is stripped from the replay (it is the one entrypoint step that
cannot run CPU-only) and so is `exec vllm serve`.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import patch_pn101_answer_rescue as P  # noqa: E402
import patch_pn114_seed_span as SEED  # noqa: E402
import patch_pn71t_truncation_signal as P71T  # noqa: E402

# The compose whose entrypoint + environment IS the boot.
COMPOSE = REPO / "models/qwen3.6-27b/vllm/compose/single/tcbench8021.yml"

# dev1060cherry-20260713 is deliberately absent: it will never be booted again.
PINS = (
    "localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725",  # the boot pin
    "localhost/vllm-qwen36-endgame:dev1474cherry-1711-20260725",
)
TARGET = ("/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/"
          "chat_completion/serving.py")
STOP_AT = "patch_pn101_answer_rescue"
DELIM = "@@@PN101-VERIFY-SERVING@@@"

MOUNTS = (
    (REPO / "fixes", "/fixes", "ro"),
    (REPO / "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis",
     "/usr/local/lib/python3.12/dist-packages/vllm/_genesis", "ro"),
    (REPO / "models/qwen3.6-27b/vllm/patches/genesis/vllm/_genesis/sndr",
     "/usr/local/lib/python3.12/dist-packages/sndr", "ro"),
    (REPO / "models/qwen3.6-27b/vllm/patches/local/"
            "qwen3coder_tool_parser_deferred_commit.py",
     "/patches/qwen3coder_tool_parser_deferred_commit.py", "ro"),
)

# The reword the "comment-drift" arm applies before PN101's line.
DRIFT = (
    "python3 - <<'PY'\n"
    "import pathlib\n"
    f"p = pathlib.Path({TARGET!r})\n"
    "s = p.read_text(encoding='utf-8')\n"
    "s = s.replace('        # Streaming response\\n', "
    "'        # Streaming / non-streaming dispatch\\n')\n"
    "p.write_text(s, encoding='utf-8')\n"
    "print('[arm] reworded upstream comment: # Streaming response')\n"
    "PY\n"
)


def _expand(value: str) -> str:
    """Resolve compose interpolation the way `docker compose` would with no .env."""
    prev = None
    while prev != value:
        prev = value
        value = re.sub(r"\$\{([A-Za-z_]\w*):-([^{}]*)\}", r"\2", value)
        value = re.sub(r"\$\{([A-Za-z_]\w*)\}", "", value)
    return value


def compose_environment() -> list[str]:
    txt = COMPOSE.read_text(encoding="utf-8")
    m = re.search(r"\n    environment:\n(.*?)\n    [a-z_]+:", txt, re.S)
    if not m:
        raise SystemExit(f"no environment: block found in {COMPOSE}")
    out = []
    for line in m.group(1).split("\n"):
        s = line.strip()
        if not s.startswith("- "):
            continue
        kv = s[2:]
        if kv[:1] in "\"'":
            kv = kv[1:-1]
        if "=" not in kv:
            continue
        key, val = kv.split("=", 1)
        out.append(f"{key}={_expand(val)}")
    return out


def entrypoint_lines(stop_at: str | None, keep_set_e: bool) -> list[str]:
    """The boot's entrypoint script, minus the two steps a CPU-only replay can't run."""
    txt = COMPOSE.read_text(encoding="utf-8")
    m = re.search(
        r"\n    entrypoint:\n      - /bin/bash\n      - -c\n      - \|\n(.*?)\n      - --\n",
        txt, re.S)
    if not m:
        raise SystemExit(f"no block-scalar entrypoint found in {COMPOSE}")
    lines = [ln[8:] if ln.startswith(" " * 8) else ln
             for ln in m.group(1).split("\n")]
    if stop_at and not any(stop_at in ln for ln in lines):
        raise SystemExit(f"{stop_at} not found in the entrypoint of {COMPOSE}")
    out = []
    for ln in lines:
        if stop_at and stop_at in ln:
            break
        # The gpu-guard cannot run CPU-only and would abort the replay.
        if "torch.cuda.is_available" in ln or "[gpu-guard]" in ln:
            continue
        # No engine in a verifier.
        if ln.strip().startswith("exec vllm serve"):
            continue
        if not keep_set_e and ln.strip() == "set -e":
            out.append("set +e")
            continue
        out.append(ln)
    return out


def run(pin: str, env: list[str], body: str, workdir: pathlib.Path):
    script = workdir / "replay.sh"
    script.write_text(body, encoding="utf-8")
    envfile = workdir / "env.list"
    envfile.write_text("\n".join(env) + "\n", encoding="utf-8")
    cmd = ["sudo", "podman", "run", "--rm", "--network", "none",
           "--env-file", str(envfile)]
    for src, dst, mode in MOUNTS:
        cmd += ["-v", f"{src}:{dst}:{mode}"]
    cmd += ["-v", f"{workdir}:/work:ro", "--entrypoint", "/bin/bash", pin,
            "/work/replay.sh"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=2400)


def _serving(out: str) -> str | None:
    if DELIM + "\n" not in out:
        return None
    return out.split(DELIM + "\n", 1)[1]


# ── arms ──────────────────────────────────────────────────────────────────

def arm_anchors(pin: str, env: list[str], workdir: pathlib.Path,
                drift: bool) -> bool:
    """Count + resolve PN101's anchors against the pre-PN101 boot file."""
    body = ("\n".join(entrypoint_lines(STOP_AT, keep_set_e=False)) + "\n"
            + (DRIFT if drift else "")
            + f"echo {DELIM}\ncat {TARGET}\n")
    r = run(pin, env, body, workdir)
    src = _serving(r.stdout)
    if src is None:
        print(f"    container rc={r.returncode}; "
              f"replay produced no serving.py: {r.stderr.strip()[-400:]}")
        return False
    pn16 = "applied" if "_genesis_pN16_apply_hook(self, request)" in src \
        else "NOT applied"
    pn100 = "applied" if P.PN100_CALL in src else "NOT applied"
    print(f"    pre-PN101 serving.py: {len(src.splitlines())} lines · "
          f"genesis PN16 {pn16} · PN100 {pn100}")

    if P.HINT_MARKER in src or P.REPAIR_MARKER in src:
        print("    FAIL a PN101 marker is already in the pre-PN101 file — "
              "another step writes it")
        return False

    ok = True
    hname, hoff, hcounts, hprobs = P.resolve_hint(src)
    for n, c in hcounts:
        print(f"      hint   {n:<16} count={c}")
    if hname is None:
        for p in hprobs:
            print(f"      {p}")
        print("    FAIL hint site unresolved — the PN102 banner would have no "
              "call site (loud skip, boot survives)")
        ok = False

    rname, rspan, rcounts, rprobs = P.resolve_repair(src)
    for n, c in rcounts:
        print(f"      repair {n:<16} count={c}")
    if rname is None:
        for p in rprobs:
            print(f"      {p}")
        print("    FAIL repair site unresolved — the answer-rescue post-pass "
              "would have no call site")
        ok = False
    if not ok:
        return False

    patched, applied, missing, warnings = P.build(src)
    if missing:
        for m in missing:
            print(f"    FAIL {m}")
        return False
    try:
        compile(patched, "serving.py", "exec")
    except SyntaxError as e:
        print(f"    FAIL patched serving.py does not compile: {e}")
        return False
    for w in warnings:
        print(f"    WARN {w}")
        ok = False
    # The downstream anchor PN114-SEED keys on must survive this insertion.
    n_s4 = patched.count(SEED.S4_OLD)
    print(f"      pn114-seed S4    count={n_s4} (in PN101's output)")
    if n_s4 != 1:
        print("    FAIL PN114-SEED's S4 site would not resolve against PN101's "
              "output")
        ok = False
    print(f"    {'OK  ' if ok else 'FAIL'} {'; '.join(applied)} · compiles")
    return ok


def arm_chain(pin: str, env: list[str], workdir: pathlib.Path) -> bool:
    """Replay the WHOLE entrypoint under `set -e` and prove the chain composes."""
    body = ("\n".join(entrypoint_lines(None, keep_set_e=True)) + "\n"
            + f"python3 -c \"import py_compile,sys; "
              f"py_compile.compile({TARGET!r}, doraise=True); "
              f"print('[chain] serving.py compiles')\"\n"
            + f"echo {DELIM}\ncat {TARGET}\n")
    r = run(pin, env, body, workdir)
    print(f"    entrypoint rc={r.returncode} (set -e intact, "
          f"gpu-guard + `exec vllm serve` stripped)")
    src = _serving(r.stdout)
    if r.returncode != 0 or src is None:
        tail = "\n".join(
            [ln for ln in r.stdout.splitlines()[-6:]]
            + [ln for ln in r.stderr.splitlines()[-10:]])
        print("    FAIL the entrypoint did not survive to the end:\n"
              + "\n".join(f"      {ln[:160]}" for ln in tail.splitlines()))
        return False
    checks = [
        ("PN100 hook", P.PN100_CALL, 1),
        ("genesis PN16 hook", "_genesis_pN16_apply_hook(self, request)", 1),
        ("PN101a hint", P.HINT_MARKER, 1),
        ("PN101b repair", P.REPAIR_MARKER, 1),
        ("PN71T", P71T.MARKER, 2),
    ]
    ok = "[chain] serving.py compiles" in r.stdout
    if not ok:
        print("    FAIL serving.py did not byte-compile in-container")
    for label, needle, want in checks:
        n = src.count(needle)
        good = n >= want
        ok &= good
        print(f"      {'OK  ' if good else 'FAIL'} {label:<18} "
              f"count={n} (want >= {want})")
    # PN114-SEED is default-OFF, so it soft-skips; what must hold is that its
    # anchor into PN101's block is still exactly resolvable.
    n_s4 = src.count(SEED.S4_OLD)
    seeded = src.count(SEED.MARKER)
    good = (n_s4 == 1) or (seeded > 0)
    ok &= good
    print(f"      {'OK  ' if good else 'FAIL'} {'PN114-SEED S4':<18} "
          f"anchor={n_s4} applied-marker={seeded}")
    print(f"    {'OK  ' if ok else 'FAIL'} PN100 -> PN16 -> PN101 -> PN71T "
          f"compose under `set -e`")
    return bool(ok)


# ── drift matrix ──────────────────────────────────────────────────────────
# Mutations applied to the REAL replayed boot file, each with the variant that
# must absorb it. `None` means "no site may resolve" — the loud-skip path.
_TOK_PAIR = P.V_TOKENIZER
_PN100_HEAD = "        # PN100: automatic reasoning-budget router"


def _move_tokenizer_above_pn100(s: str) -> str:
    """Put the hint's primary anchor ahead of PN100's hook — an illegal offset."""
    if _PN100_HEAD not in s or _TOK_PAIR not in s:
        return s
    s = s.replace(_TOK_PAIR, "", 1)
    return s.replace(_PN100_HEAD, _TOK_PAIR + _PN100_HEAD, 1)


DRIFT_MATRIX = (
    ("upstream reworded the comment",
     lambda s: s.replace("        # Streaming response\n",
                         "        # Streaming / non-streaming dispatch\n"),
     "tokenizer-fetch", "cleanup-wrapper", 0),
    ("a sibling emitted the comment twice",
     lambda s: s.replace("        # Streaming response\n",
                         "        # Streaming response\n"
                         "        # Streaming response\n", 1),
     "tokenizer-fetch", "cleanup-wrapper", 0),
    ("upstream renamed the tokenizer fetch",
     lambda s: s.replace("        tokenizer = self.renderer.tokenizer\n",
                         "        tokenizer = self.renderer.get_tokenizer()\n"),
     "ctk-read", "cleanup-wrapper", 0),
    ("upstream reflowed the return arguments",
     lambda s: s.replace(
         "            self._create_chat_completion(request, raw_request), "
         "request, raw_request\n",
         "            self._create_chat_completion(request, raw_request),\n"
         "            request,\n            raw_request,\n"),
     "tokenizer-fetch", "cleanup-wrapper", 0),
    ("upstream dropped the kv-transfer wrapper",
     lambda s: re.sub(
         r"        return await self\._with_kv_transfer_rejection_cleanup\(\n"
         r".*?\n        \)\n",
         "        return await self._create_chat_completion(request, raw_request)\n",
         s, count=1, flags=re.S),
     "tokenizer-fetch", "direct-call", 0),
    ("every hint landmark gone",
     lambda s: (s.replace("        # Streaming response\n", "        # X\n")
                 .replace("        tokenizer = self.renderer.tokenizer\n",
                          "        tokenizer = self.rndr.tok\n")
                 .replace("        chat_template_kwargs = "
                          "self._effective_chat_template_kwargs(request)\n",
                          "        ctk = self._eck(request)\n")),
     None, "cleanup-wrapper", 0),
    ("the anchor drifted ahead of PN100",
     _move_tokenizer_above_pn100, "tokenizer-fetch", "cleanup-wrapper", 1),
)


def arm_drift(pin: str, env: list[str], workdir: pathlib.Path) -> bool:
    """Mutate the REAL boot file and prove which variant absorbs each drift."""
    body = ("\n".join(entrypoint_lines(STOP_AT, keep_set_e=False)) + "\n"
            + f"echo {DELIM}\ncat {TARGET}\n")
    r = run(pin, env, body, workdir)
    src = _serving(r.stdout)
    if src is None:
        print(f"    container rc={r.returncode}; replay produced no serving.py")
        return False
    ok = True
    for label, mutate, want_hint, want_repair, want_warn in DRIFT_MATRIX:
        s = mutate(src)
        if s == src:
            print(f"    FAIL {label}: mutation did not change the file")
            ok = False
            continue
        patched, applied, missing, warnings = P.build(s)
        hname = P.resolve_hint(s)[0]
        rname = P.resolve_repair(s)[0]
        good = (hname == want_hint and rname == want_repair
                and len(warnings) == want_warn
                and len(missing) == (want_hint is None) + (want_repair is None))
        if patched != s:
            try:
                compile(patched, "serving.py", "exec")
            except SyntaxError as e:
                good = False
                print(f"      compile: {e}")
        if want_hint is not None and patched.count(SEED.S4_OLD) != 1:
            good = False
            print("      PN114-SEED S4 anchor did not survive")
        ok &= good
        print(f"      {'OK  ' if good else 'FAIL'} {label:<38} "
              f"hint={hname} repair={rname} warn={len(warnings)}")
    print(f"    {'OK  ' if ok else 'FAIL'} {len(DRIFT_MATRIX)} drift shapes, "
          f"none exits non-zero")
    return bool(ok)


ARMS = {
    "boot-env": lambda pin, env, wd: arm_anchors(pin, env, wd, drift=False),
    "comment-drift": lambda pin, env, wd: arm_anchors(pin, env, wd, drift=True),
    "drift-matrix": arm_drift,
    "chain": arm_chain,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", action="append", help="override the pin list")
    ap.add_argument("--arm", action="append", choices=sorted(ARMS),
                    help="run only these arms")
    ap.add_argument("--keep", help="write the replayed scripts to this dir")
    args = ap.parse_args()
    pins = args.pin or list(PINS)
    arms = args.arm or ["boot-env", "comment-drift", "drift-matrix", "chain"]

    env = compose_environment()
    print(f"compose: {COMPOSE}")
    print(f"prefix:  {len(entrypoint_lines(STOP_AT, False))} entrypoint lines "
          f"replayed (up to, not including, {STOP_AT})")
    print(f"full:    {len(entrypoint_lines(None, True))} entrypoint lines "
          f"in the chain arm")
    print(f"env:     {len(env)} variables from the compose")

    ctx = pathlib.Path(args.keep) if args.keep else None
    if ctx:
        ctx.mkdir(parents=True, exist_ok=True)
    bad = 0
    for pin in pins:
        for arm in arms:
            print(f"\n=== {pin}  [{arm}]")
            if ctx:
                d = ctx / f"{pin.rsplit(':', 1)[-1]}-{arm}"
                d.mkdir(exist_ok=True)
                if not ARMS[arm](pin, env, d):
                    bad += 1
            else:
                with tempfile.TemporaryDirectory() as td:
                    if not ARMS[arm](pin, env, pathlib.Path(td)):
                        bad += 1
    print()
    print("RESULT: every arm resolves both call sites and the chain composes"
          if not bad else f"RESULT: {bad} arm(s) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
