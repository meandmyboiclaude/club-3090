"""v0.8.2 CONTRACT-1.3 — the user-invoked, consented failure on-ramp submit.

`scripts/pull.sh --submit-last` / `scripts/pull.sh --submit <capture-dir>`.

This is the SEPARATE, explicit, user-invoked, consented step the locked
`[F]`-offline boundary mandates: the gate path (`pull.py run_pull`) stays
I/O-free and only PRINTS a pointer (CONTRACT-1.2); submission — the ONLY
place network ever happens, and only after an explicit `y` — lives HERE.

What this module does (and explicitly does NOT do):

  * It RESOLVES a capture bundle dir (`--submit-last` reads the V1-written
    `.pull-captures/.last` marker; `--submit <dir>` is the explicit form).
  * It re-reads `.last` and re-shows the resolved bundle's IDENTITY AND the
    exact already-redacted payload that will be sent, THEN requires an
    explicit `y` before ANY network (the race defense: a second pull may
    have overwritten `.last` between capture and submit — we always surface
    the CURRENT bundle, never a silent wrong-bundle submit).
  * After `y`, it reuses the SHIPPED F5 path (`dedup.submit` — the
    `effective_dedup_hash` -> `gh issue` +1-or-open with the bounded
    `loop:dedup-<hash>` label scheme + body tuple + collision-safe verify).
    It does NOT reimplement dedup. A bundle that classifies into
    `_SUPPRESSED_NEVER_FILED` is review-queued (local spool) by the shipped
    loop's existing behaviour — reused, not re-added here.
  * `gh`-less fallback runs AFTER F2 classification, gated on
    `should_file`: `should_file=True` -> a prefilled PUBLIC `issues/new`
    URL; review-queued (`unknown`) -> the LOCAL `_dedup-queue` spool path +
    "captured for maintainer triage; not a public issue", and NO public
    URL / no `issues/new` link (the §6.1 review-queue boundary).

It NEVER reimplements F1/F2/F5 — it imports them as a library and reuses
the shipped `loop_input.read_capture_bundle` / `read_gate_bundle`,
`classifier.classify`, `dedup.submit` / `dedup.build_issue_body` verbatim.
It NEVER raises for an expected outcome (house style: structured result).

Console-is-not-a-safe-source: the on-ramp ONLY ever emits the redacted
`.pull-captures/` artifact's payload — never the user's terminal
scrollback. The surfaced message never says "paste your terminal output".
The redacted artifacts are produced by `[E]`/the gate emitter via
`capture._redact_text`; this module does NOT re-redact (do not double
scrub) and asserts (via the V2 test) that nothing it tells the user to
share carries an unredacted absolute path.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path
from typing import Optional

from scripts.lib.profiles import dedup as _DEDUP
from scripts.lib.profiles.classifier import FailureClass, classify
from scripts.lib.profiles.loop_input import (
    CaptureBundleError,
    GATE_SCHEMA,
    read_capture_bundle,
    read_gate_bundle,
)

# The PUBLIC repo (CONTRACT-1.3 gh-less fallback). Hard-coded public
# constant — never derived from a local path (the leak-hygiene stack rule).
_PUBLIC_REPO = "noonghunna/club-3090"
_ISSUES_NEW = f"https://github.com/{_PUBLIC_REPO}/issues/new"

# The on-ramp keeps a URL well under GitHub's prefilled-issue limit; the
# body is hard-truncated so the encoded URL stays < 8 KB (CONTRACT-1.3).
_MAX_URL_BYTES = 8 * 1024

_MARKER_FILENAME = ".last"
_PULL_CAPTURES_DIRNAME = ".pull-captures"


# ---------------------------------------------------------------------------
# `.last` marker resolution (read-only at submit — V1 writes it atomically
# from BOTH emitters via the shared `capture.write_last_marker` helper; we
# RE-READ it here, never re-implement the write).
# ---------------------------------------------------------------------------
def _pull_captures_root(repo_root: Path) -> Path:
    return Path(repo_root) / _PULL_CAPTURES_DIRNAME


def resolve_last(repo_root: Path) -> Optional[Path]:
    """Resolve the most-recent capture dir from `.pull-captures/.last`.

    Read-only. Returns the absolute bundle dir, or None when the marker is
    absent / empty / points at a now-missing dir (stale). The CALLER turns
    None into the CONTRACT-1.3 error ("no recent capture; use
    `--submit <dir>`"). NEVER raises.
    """
    try:
        root = _pull_captures_root(repo_root)
        marker = root / _MARKER_FILENAME
        if not marker.is_file():
            return None
        rel = marker.read_text(encoding="utf-8").strip()
        if not rel:
            return None
        cand = (root / rel).resolve()
        if not cand.is_dir():
            return None
        return cand
    except Exception:  # pragma: no cover - marker read is best-effort.
        return None


# ---------------------------------------------------------------------------
# Bundle read — schema-aware (schema==1 [E] bundle OR schema==2 gate-only).
# Reuses the SHIPPED F1 readers verbatim; does NOT re-validate.
# ---------------------------------------------------------------------------
def _read_bundle(capture_dir: Path):
    """Return the F1 bundle object for `capture_dir`, dispatching on the
    manifest's `schema` (1 -> `read_capture_bundle`/`FInput`; 2 -> the
    gate-only `read_gate_bundle`/`FInputGate`). Raises `CaptureBundleError`
    on a bad/missing manifest (the caller turns it into a clean error).
    """
    cdir = Path(capture_dir)
    manifest_path = cdir / "manifest.json"
    if not cdir.is_dir() or not manifest_path.is_file():
        raise CaptureBundleError(
            f"not a capture bundle (no manifest.json): {cdir}"
        )
    try:
        schema = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "schema"
        )
    except (ValueError, OSError) as exc:
        raise CaptureBundleError(f"unreadable manifest.json: {exc}") from exc
    if schema == GATE_SCHEMA:
        return read_gate_bundle(cdir)
    return read_capture_bundle(cdir)


# ---------------------------------------------------------------------------
# CONTRACT-1.3 — re-show the resolved bundle IDENTITY + the exact redacted
# payload that will be sent, then require an explicit `y` before ANY
# network. `_input` is injectable so the V2 test drives consent with ZERO
# tty (and ZERO network).
# ---------------------------------------------------------------------------
def _bundle_identity(capture_dir: Path, finput) -> str:
    """A short, human IDENTITY line for the resolved bundle (so a racing
    second pull can't cause a silent wrong-bundle submit — the user sees
    WHICH bundle before consenting)."""
    m = getattr(finput, "manifest", {}) or {}
    return (
        f"slug={m.get('slug')!r} "
        f"utc_ts={m.get('utc_ts')!r} "
        f"outcome={m.get('outcome')!r} "
        f"schema={m.get('schema')} "
        f"dir={capture_dir.name}"
    )


def _redacted_payload_preview(finput, classification) -> str:
    """The EXACT already-redacted payload that will be sent — the shipped
    F5 `build_issue_body` rendering of the redacted manifest/tuple. We do
    NOT re-redact (the artifact is `[E]`/gate-emitter-redacted upstream);
    we only DISPLAY it so the user consents to exactly what is sent.
    """
    return _DEDUP.build_issue_body(finput, classification)


def _confirm(prompt: str, _input) -> bool:
    """Explicit `y` gate. Anything other than a leading 'y'/'Y' is a
    decline. EOF / no-tty -> decline (never assume consent)."""
    try:
        ans = _input(prompt)
    except (EOFError, KeyboardInterrupt):
        return False
    return str(ans).strip().lower().startswith("y")


# ---------------------------------------------------------------------------
# gh-less fallback (post-classification, `should_file`-gated). CONTRACT-1.3:
# PUBLIC URL ONLY for should_file=True; review-queued -> local spool path +
# the no-public-issue line, NEVER an `issues/new` link.
# ---------------------------------------------------------------------------
def _ghless_title(finput, classification) -> str:
    """The DETERMINISTIC gh-less title template (CONTRACT-1.3):
    `[<model>] <failure-class> on <topology> (dedup:<hash8>)`.

    Intentionally DIFFERENT from the shipped `dedup.build_issue_title`
    (`[loop][<cls>] <slug> (dedup <h>)`): the `gh`-path carries the dedup
    key as a `loop:dedup-<hash>` LABEL (the real dedup primitive) while the
    gh-less path has no label API so encodes the hash in the title. Dedup
    keys on the label/hash, NOT the title string — divergent titles do not
    split the bucket. Do NOT "unify" these (would byte-change the shipped
    `build_issue_title` for zero benefit).
    """
    h = _DEDUP.effective_dedup_hash(finput, classification)
    m = getattr(finput, "manifest", {}) or {}
    model = m.get("model") or m.get("model_id") or "<unknown-model>"
    cls = classification.failure_class
    cls_v = cls.value if isinstance(cls, FailureClass) else str(cls)
    topo = m.get("topology_class") or "unknown"
    return f"[{model}] {cls_v} on {topo} (dedup:{h[:8]})"


def _rel_capture(bundle_dir: Path) -> str:
    """Repo-relative capture pointer (`.pull-captures/<slug>/<ts>`), NEVER an
    absolute path. The gh-less body is pasted into a PUBLIC issue, so it must
    not carry the user's filesystem layout (CONTRACT-1.3 "console is not a
    safe source" + the acceptance: nothing the on-ramp tells a user to share
    contains an unredacted absolute path). Falls back to the leaf name.
    """
    parts = Path(bundle_dir).parts
    if _PULL_CAPTURES_DIRNAME in parts:
        i = parts.index(_PULL_CAPTURES_DIRNAME)
        return str(Path(*parts[i:]))
    return Path(bundle_dir).name


def _ghless_body(finput, classification, bundle_dir: Path) -> str:
    """URL-encoded-ready MARKDOWN body from the redacted manifest tuple
    (model/engine/arch/failure-class/topology + the dedup hash), hard-
    truncated so the final encoded URL stays < 8 KB, with a trailing
    "full redacted bundle at `<path>` — attach it". Markdown, not JSON.

    The body is built from ALREADY-redacted F1 fields; we do NOT re-redact.
    """
    h = _DEDUP.effective_dedup_hash(finput, classification)
    m = getattr(finput, "manifest", {}) or {}
    cls = classification.failure_class
    cls_v = cls.value if isinstance(cls, FailureClass) else str(cls)
    lines = [
        "## club-3090 failed-pull report (gh-less paste fallback)",
        "",
        "Auto-prefilled by `scripts/pull.sh --submit*` (the on-ramp could "
        "not reach `gh`; this is the manual paste path).",
        "",
        f"- **model:** `{m.get('model') or m.get('model_id')}`",
        f"- **failure-class:** `{cls_v}`",
        f"- **arch_family:** `{m.get('arch_family')}`",
        f"- **engine_version:** `{m.get('engine_version') or m.get('engine_pin')}`",
        f"- **topology_class:** `{m.get('topology_class')}`",
        f"- **abort_reason:** `{m.get('abort_reason')}`",
        f"- **dedup hash:** `{h}` (label `loop:dedup-{h}`)",
        "",
        "The redacted error substring (already `[E]`-scrubbed; not "
        "re-redacted):",
        "",
        "```",
        (classification.error_substring or "")[:1200],
        "```",
        "",
        f"redacted bundle dir (from your own run): "
        f"`{_rel_capture(bundle_dir)}` — attach it to the issue",
    ]
    body = "\n".join(lines)
    # Hard-truncate so the FINAL encoded URL stays < _MAX_URL_BYTES. We
    # budget against the encoded length (worst case ~3x for urlencode);
    # trim the body until the assembled URL fits, always preserving the
    # trailing "attach it" pointer.
    tail = (
        f"\n\nredacted bundle dir (from your own run): "
        f"`{_rel_capture(bundle_dir)}` — attach it to the issue"
    )
    while True:
        url = _build_issue_url(_ghless_title(finput, classification), body, h)
        if len(url.encode("utf-8")) < _MAX_URL_BYTES or len(body) <= len(tail):
            return body
        # Drop the middle, keep the header + the tail pointer.
        keep = max(len(tail), len(body) - 512)
        body = body[: keep - len(tail)].rstrip() + tail


def _build_issue_url(title: str, body: str, dedup_hash: str) -> str:
    """`https://github.com/<repo>/issues/new?title=&body=&labels=` —
    URL-encoded. The `labels` carries `loop:dedup-<hash>` so a gh-less
    paste lands on the SAME dedup bucket as a `gh`-path submit.
    """
    q = urllib.parse.urlencode(
        {
            "title": title,
            "body": body,
            "labels": f"{_DEDUP._DEDUP_LABEL_PREFIX}{dedup_hash}",
        }
    )
    return f"{_ISSUES_NEW}?{q}"


def ghless_fallback(
    finput,
    classification,
    *,
    repo_root: Path,
    bundle_dir: Path,
) -> dict:
    """The CONTRACT-1.3 gh-less fallback. Runs AFTER F2 classification;
    branches on `should_file`. NEVER raises; if URL-build fails, degrade to
    the existing local spool + printed paste-path.

    Returns a structured dict (`{kind, lines:[...], url|spool_path}`) so the
    caller (and the V2 test) can assert exactly what is emitted — in
    particular that a review-queued class emits NO public URL.
    """
    try:
        h = _DEDUP.effective_dedup_hash(finput, classification)
        if classification.should_file:
            # should_file=True (engine-support-unknown/no-arch-row ->
            # kernel-unsupported): print the redacted bundle path + a
            # prefilled PUBLIC issues/new URL.
            title = _ghless_title(finput, classification)
            body = _ghless_body(finput, classification, bundle_dir)
            url = _build_issue_url(title, body, h)
            return {
                "kind": "ghless-public-url",
                "should_file": True,
                "dedup_hash": h,
                "url": url,
                "lines": [
                    f"[submit] gh unavailable — paste this prefilled "
                    f"PUBLIC issue (it lands on the loop:dedup-{h} bucket):",
                    f"[submit] redacted bundle: {bundle_dir}",
                    f"[submit] {url}",
                ],
            }
        # review-queued (`unknown` ∈ _SUPPRESSED_NEVER_FILED): print the
        # redacted bundle path + the LOCAL review-queue spool path the
        # shipped F5 would write. NO public-issue URL / no issues/new link
        # (the §6.1 review-queue boundary — correct-refusals stay out of
        # the public tracker).
        spool = (
            _pull_captures_root(repo_root)
            / _DEDUP._REVIEW_QUEUE_DIRNAME
            / f"{h}.json"
        )
        return {
            "kind": "ghless-review-queued",
            "should_file": False,
            "dedup_hash": h,
            "spool_path": str(spool),
            "lines": [
                f"[submit] redacted bundle: {bundle_dir}",
                f"[submit] captured for maintainer triage; not a public "
                f"issue (correct-refusal / unactionable class)",
                f"[submit] local review-queue spool: {spool}",
            ],
        }
    except Exception as exc:  # pragma: no cover - degrade, never raise.
        # URL-build (or anything) failed -> degrade to the local spool +
        # printed paste-path; never raise out of the on-ramp.
        try:
            h = _DEDUP.effective_dedup_hash(finput, classification)
        except Exception:
            h = "unknown"
        spool = (
            _pull_captures_root(repo_root)
            / _DEDUP._DEDUP_QUEUE_DIRNAME
            / f"{h}.json"
        )
        return {
            "kind": "ghless-degraded",
            "should_file": bool(
                getattr(classification, "should_file", False)
            ),
            "dedup_hash": h,
            "spool_path": str(spool),
            "lines": [
                f"[submit] gh-less URL build failed ({exc!r}) — degraded",
                f"[submit] redacted bundle: {bundle_dir}",
                f"[submit] local spool: {spool}",
            ],
        }


# ---------------------------------------------------------------------------
# `gh` availability probe — reuses the shipped `dedup._real_gh_runner`
# (NEVER raises; a missing/unauthed `gh` reports non-ok).
# ---------------------------------------------------------------------------
def _gh_available(gh_runner) -> bool:
    """True iff `gh auth status` reports authenticated. Reuses the shipped
    injectable runner seam — the V2 test injects a mock so this is ZERO
    network. NEVER raises (a missing binary -> non-ok -> False)."""
    try:
        res = gh_runner(["auth", "status"])
        return bool(getattr(res, "ok", False))
    except Exception:  # pragma: no cover - runner never raises by contract.
        return False


# ---------------------------------------------------------------------------
# The top-level submit verb.
# ---------------------------------------------------------------------------
def submit_pull(
    *,
    capture_dir: Optional[str],
    submit_last: bool,
    repo_root: Path,
    repo: Optional[str] = None,
    gh_runner=None,
    _input=input,
    _print=print,
) -> int:
    """`scripts/pull.sh --submit-last` / `--submit <dir>`.

    Returns a process exit code: 0 on a clean submit / consented decline /
    review-queue spool; 2 on an unresolvable bundle ("no recent capture").
    NEVER raises (CONTRACT-1: the on-ramp must never blow up a user's tty).

    Flow:
      1. Resolve the bundle dir. `--submit-last` RE-READS `.pull-captures/
         .last` HERE (the race defense — a second pull may have overwritten
         it since capture; we surface the CURRENT bundle, never a silent
         wrong-bundle submit). Unresolvable -> the CONTRACT-1.3 error.
      2. Read (schema-aware) + classify (F2).
      3. Re-SHOW the resolved bundle IDENTITY AND the exact already-redacted
         payload, then require explicit `y` before ANY network.
      4. After `y`: `gh` present -> reuse the shipped F5 `dedup.submit`
         (effective_dedup_hash -> +1-or-open, bounded labels, body tuple,
         collision-safe verify, suppression/review-queue all reused).
         `gh` absent/unauthed -> the post-classification, should_file-gated
         gh-less fallback (public URL only for should_file=True).
    """
    gh_runner = gh_runner or _DEDUP._real_gh_runner
    root = Path(repo_root)

    # ---- 1. resolve the bundle dir (re-read `.last` HERE) --------------
    if submit_last:
        resolved = resolve_last(root)
        if resolved is None:
            _print(
                "[submit] no recent capture; use "
                "`scripts/pull.sh --submit <dir>`",
                file=sys.stderr,
            )
            return 2
        bundle_dir = resolved
    else:
        if not capture_dir:
            _print(
                "[submit] --submit needs a capture dir; or use "
                "`--submit-last`",
                file=sys.stderr,
            )
            return 2
        bundle_dir = Path(capture_dir).resolve()
        if not bundle_dir.is_dir():
            _print(
                f"[submit] capture dir does not exist: {bundle_dir}",
                file=sys.stderr,
            )
            return 2

    # ---- 2. read (schema-aware) + classify (F2) -----------------------
    try:
        finput = _read_bundle(bundle_dir)
    except CaptureBundleError as exc:
        _print(f"[submit] not a usable capture bundle: {exc}",
               file=sys.stderr)
        return 2
    classification = classify(finput)

    # ---- 3. re-show identity + the EXACT redacted payload, then `y` ---
    _print(f"[submit] resolved bundle: {bundle_dir}")
    _print(f"[submit] identity: {_bundle_identity(bundle_dir, finput)}")
    _print(
        "[submit] this is the EXACT already-redacted payload that will be "
        "sent (no terminal scrollback, no paths/tokens):"
    )
    _print("-" * 72)
    _print(_redacted_payload_preview(finput, classification))
    _print("-" * 72)
    cls = classification.failure_class
    cls_v = cls.value if isinstance(cls, FailureClass) else str(cls)
    _print(
        f"[submit] §6.1 class={cls_v} should_file={classification.should_file}"
    )
    if not _confirm("[submit] submit this redacted report? [y/N] ", _input):
        _print("[submit] declined — nothing sent (no network performed).")
        return 0

    # ---- 4. consented: gh path (reuse F5) OR gh-less fallback ----------
    if _gh_available(gh_runner):
        result = _DEDUP.submit(
            finput,
            classification,
            repo_root=root,
            repo=repo or _PUBLIC_REPO,
            gh_runner=gh_runner,
        )
        _print(
            f"[submit] F5 action={result.action.value} "
            f"dedup_hash={result.dedup_hash} "
            f"issue={result.issue_number} reason={result.reason}"
        )
        if result.spool_path:
            _print(f"[submit] spool: {result.spool_path}")
        return 0

    # gh absent/unauthed -> the post-F2, should_file-gated fallback.
    fb = ghless_fallback(
        finput,
        classification,
        repo_root=root,
        bundle_dir=bundle_dir,
    )
    for ln in fb["lines"]:
        _print(ln)
    return 0
