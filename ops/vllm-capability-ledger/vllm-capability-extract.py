#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the machine-readable vLLM capability inventory from the repo.

Walks the three patch lanes and mechanically derives, per capability:

    id · lane · source file · target file(s) · env flag(s) · kind ·
    verification handle (marker string / module path / symbol)

The point is that this is REGENERABLE.  A hand-written inventory rots the
moment someone lands a patch; this one is re-run as step 0 of every bump and
its diff against the committed copy is itself a signal.

Curated overlay
---------------
Anything the AST cannot see (one-line descriptions, upstream PR numbers,
retirement evidence, non-text verification probes) lives in
`inventory-overlay.json` and is merged on top, keyed by capability id.  The
overlay is hand-maintained; the extraction is not.

Usage:
    python3 vllm-capability-extract.py [-o vllm-capability-inventory.json]
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))  # realpath: these are symlinked into ~/shared/tools
from vllm_ledger_lib import (  # noqa: E402
    FIXES, GENESIS, LEDGER_DIR, MOUNTS, REPO, SNDR, VLLM_DIR, dump_json,
    eprint, load_json,
)

CONTAINER_VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"

FLAG_RE = re.compile(r"\b((?:GENESIS|SNDR|VLLM)_[A-Z0-9_]{3,})\b")
# ids: patch_101_x -> P101 ; patch_N40_x -> PN40 ; patch_pn100_x -> PN100 ;
#      patch_67b_x  -> P67B ; g4_75_x -> G4_75
ID_PATTERNS = [
    (re.compile(r"^patch_pn([0-9]+[a-zA-Z]*)_"), lambda m: "PN" + m.group(1).upper()),
    (re.compile(r"^patch_h([0-9]+[a-zA-Z]*)_"), lambda m: "H" + m.group(1).upper()),
    (re.compile(r"^patch_N([0-9]+[a-zA-Z]*)_"), lambda m: "PN" + m.group(1).upper()),
    (re.compile(r"^patch_([0-9]+[a-zA-Z]*)_"), lambda m: "P" + m.group(1).upper()),
    (re.compile(r"^pn([0-9]+[a-zA-Z]*)_"), lambda m: "PN" + m.group(1).upper()),
    (re.compile(r"^p([0-9]+[a-zA-Z]*)_"), lambda m: "P" + m.group(1).upper()),
    (re.compile(r"^(g4_[0-9]+[a-zA-Z]*)_"), lambda m: m.group(1).upper()),
    (re.compile(r"^(spn[0-9]+[a-zA-Z]*)_"), lambda m: m.group(1).upper()),
]


# ------------------------------------------------------------- ast helpers --

def _const_str(node) -> str | None:
    """Fold a string expression: literals, implicit concat, and `A + B`."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _const_str(node.left), _const_str(node.right)
        if a is not None and b is not None:
            return a + b
    if isinstance(node, ast.JoinedStr):
        return None
    return None


def _path_strs(node) -> list[str]:
    """Collect string literals out of a path expression.

    Handles the two shapes both lanes use:
        pathlib.Path("/usr/.../vllm/x/y.py")      -> the whole literal
        VLLM / "v1/core/sched/scheduler.py"       -> the relative literal
        resolve_vllm_file("v1/attention/....py")  -> the relative literal
    """
    out: list[str] = []
    for sub in ast.walk(node):
        s = _const_str(sub)
        if s and ("/" in s) and s.endswith((".py", ".pyi", ".so", ".json", ".jinja")):
            out.append(s)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                and sub.value.startswith("/usr/"):
            out.append(sub.value)
    # implicit-concat of a long absolute path arrives as several Constants
    if not out:
        joined = "".join(
            sub.value for sub in ast.walk(node)
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
        )
        if joined.endswith(".py") and "/" in joined:
            out.append(joined)
    return out


def _norm_target(raw: str) -> str:
    """Normalise a target to a CONTAINER-absolute path."""
    raw = raw.strip()
    if raw.startswith("/"):
        return raw
    raw = raw.lstrip("./")
    return f"{CONTAINER_VLLM}/{raw}"


def _derive_id(fname: str, body: str) -> str:
    stem = fname[:-3] if fname.endswith(".py") else fname
    for rx, fn in ID_PATTERNS:
        m = rx.match(stem)
        if m:
            return fn(m)
    m = re.search(r"\[(?:Genesis[- ]?(?:house )?)?((?:S?PN|P|H|G4_)[0-9][0-9a-zA-Z_]*)", body)
    if m:
        return m.group(1).upper()
    return stem.upper()


# --------------------------------------------------------------- extractor --

def scan_module(path: str, lane: str, mount_host_root: str) -> dict | None:
    try:
        src = open(path, "r", encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        eprint(f"  !! syntax error {path}: {e}")
        return None

    fname = os.path.basename(path)
    markers: list[str] = []
    targets: list[str] = []

    # module-level constant assignments
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        for name in names:
            up = name.upper()
            if "MARKER" in up and "DRIFT" not in up and "UPSTREAM" not in up:
                s = _const_str(node.value)
                if s:
                    markers.append(s)
                elif isinstance(node.value, (ast.Tuple, ast.List)):
                    markers += [x for x in (_const_str(e) for e in node.value.elts) if x]
            if up.startswith("TARGET") or up.endswith("_TARGET") or up in ("PATHS", "FILES"):
                targets += _path_strs(node.value)

    # TextPatcher(...) / resolve_vllm_file(...) anywhere in the module
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        fname_call = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if fname_call == "resolve_vllm_file":
            for a in node.args:
                targets += _path_strs(a)
        elif fname_call in ("TextPatcher", "TextPatchGroup"):
            for kw in node.keywords or []:
                if kw.arg == "marker":
                    s = _const_str(kw.value)
                    if s:
                        markers.append(s)
                elif kw.arg == "target_file":
                    targets += _path_strs(kw.value)

    # kind classification — decides HOW it can be verified at all
    has_text = "TextPatcher(" in src or "write_text(" in src or ".replace(" in src and "MARKER" in src
    has_selfinstall = "SELF_INSTALL" in src or "_install_at_import" in src
    has_setattr = bool(re.search(r"\bsetattr\(|\brebind\(", src))
    if has_text:
        kind = "text_patch_selfinstall" if has_selfinstall else "text_patch"
    elif has_setattr:
        kind = "monkey_patch"
    else:
        kind = "module"

    flags = sorted({f for f in FLAG_RE.findall(src)
                    if not f.startswith(("VLLM_USE", "VLLM_LOGGING"))})

    # container path of this SOURCE file, via the ro lane mounts
    cpath = path
    for host, cont in MOUNTS.items():
        if path.startswith(host + "/"):
            cpath = cont + path[len(host):]
            break

    doc = ast.get_docstring(tree) or ""
    summary = " ".join(doc.strip().splitlines()[:1])[:220]

    status = "live"
    low = (doc + src[:4000]).lower()
    if "/_archive/" in path or "/_retired/" in path:
        status = "retired"
    elif re.search(r"\bretired\b", low):
        status = "retired?"
    elif re.search(r"\bsuperseded\b", low):
        status = "superseded?"
    elif re.search(r"dark by default|default[- ]dark|shadow default|default off|DEFAULT OFF", doc):
        status = "default-dark"

    prs = sorted({("vllm#" + m) for m in re.findall(r"vllm#(\d{4,6})", src)})
    prs += sorted({("sglang#" + m) for m in re.findall(r"SGLang#(\d{4,6})", src, re.I)})

    return {
        "id": _derive_id(fname, src),
        "lane": lane,
        "source_host": os.path.relpath(path, REPO),
        "source_container": cpath,
        "kind": kind,
        "targets": sorted(set(_norm_target(t) for t in targets)),
        "markers": sorted(set(m for m in markers if len(m) >= 4)),
        "flags": flags,
        "upstream": prs,
        "status": status,
        "summary": summary,
    }


def load_registry(path: str) -> dict:
    """Both lanes ship a data-only PATCH_REGISTRY; runpy is the honest reader.

    lane-1: _genesis/dispatcher.py          (126 entries)
    lane-2: _genesis/sndr/dispatcher/registry.py (329 entries, richer schema
            incl. apply_module / lifecycle / implementation_status / superseded_by)
    """
    import runpy
    try:
        g = runpy.run_path(path)
    except Exception as e:
        eprint(f"  !! registry {path} unreadable: {type(e).__name__}: {e}")
        return {}
    reg = g.get("PATCH_REGISTRY")
    return reg if isinstance(reg, dict) else {}


def walk_lane(root: str, lane: str, pred) -> list[dict]:
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in sorted(fn):
            if not f.endswith(".py") or not pred(f):
                continue
            rec = scan_module(os.path.join(dp, f), lane, root)
            if rec:
                out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out",
                    default=os.path.join(LEDGER_DIR, "vllm-capability-inventory.json"))
    ap.add_argument("--overlay",
                    default=os.path.join(LEDGER_DIR, "inventory-overlay.json"))
    args = ap.parse_args()

    caps: list[dict] = []
    # lane-1: genesis wiring (house patches, applied by apply_all)
    caps += walk_lane(os.path.join(GENESIS, "wiring"), "lane1-genesis",
                      lambda f: f.startswith("patch_"))
    # lane-1 support modules that ARE the capability (kernels + middleware)
    for sub, lane in (("kernels", "lane1-kernel"), ("middleware", "lane1-middleware")):
        caps += walk_lane(os.path.join(GENESIS, sub), lane,
                          lambda f: not f.startswith("__"))
    # lane-2: vendored sndr engine patches
    caps += walk_lane(os.path.join(SNDR, "engines/vllm"), "lane2-sndr",
                      lambda f: not f.startswith("__"))
    # fixes lane: always-on entrypoint text patches
    caps += walk_lane(FIXES, "fixes",
                      lambda f: f.startswith(("patch_", "pn", "apply_")))

    # ---- fold in the two authoritative dispatcher registries -------------
    # These carry lifecycle / default_on / superseded_by / conflicts_with,
    # which no amount of module grepping can recover.  Join lane-2 by
    # apply_module (its registry names the file), lane-1 by id.
    reg1 = load_registry(os.path.join(GENESIS, "dispatcher.py"))
    reg2 = load_registry(os.path.join(SNDR, "dispatcher/registry.py"))
    print(f"  registries: lane1={len(reg1)} lane2={len(reg2)}")

    by_src = {}
    for c in caps:
        by_src[c["source_host"]] = c

    def _reg_fields(meta: dict) -> dict:
        out = {}
        for k in ("title", "env_flag", "env_flag_aliases", "default_on", "category",
                  "family", "tier", "lifecycle", "implementation_status",
                  "upstream_pr", "superseded_by", "retired_reason", "conflicts_with",
                  "requires_patches", "deprecated", "deprecation_note",
                  "vllm_version_range", "production_validated_pins",
                  "anchor_breaker_watch", "stable_since"):
            if k in meta and meta[k] not in (None, "", [], {}):
                out[k] = meta[k]
        return out

    matched2 = 0
    for pid, meta in reg2.items():
        # apply_module is a dotted path rooted at the top-level `sndr` package,
        # e.g. sndr.engines.vllm.patches.serving.p62_...  -> <SNDR>/engines/...py
        am = meta.get("apply_module")
        rel = None
        if am:
            dotted = am[len("sndr."):] if am.startswith("sndr.") else am
            abs_p = os.path.join(SNDR, dotted.replace(".", "/") + ".py")
            rel = os.path.relpath(abs_p, REPO)
        cap = by_src.get(rel) if rel else None
        if cap is not None:
            cap["registry"] = _reg_fields(meta)
            cap["registry_id"] = pid
            matched2 += 1
        else:
            # registry row with no reachable module — a real inventory landmine:
            # it announces an APPLY decision and there is nothing to verify.
            caps.append({
                "id": pid, "lane": "lane2-sndr", "source_host": am or "",
                "source_container": "", "kind": "registry_only", "targets": [],
                "markers": [], "flags": [meta["env_flag"]] if meta.get("env_flag") else [],
                "upstream": [], "status": meta.get("lifecycle", "unknown"),
                "summary": meta.get("title", ""), "registry": _reg_fields(meta),
                "registry_id": pid,
            })
    matched1 = 0
    l1_by_id = {}
    for c in caps:
        if c["lane"].startswith("lane1"):
            l1_by_id.setdefault(c["id"], c)
    for pid, meta in reg1.items():
        cap = l1_by_id.get(pid.upper()) or l1_by_id.get(pid)
        if cap is not None:
            cap["registry"] = _reg_fields(meta)
            cap["registry_id"] = pid
            matched1 += 1
        else:
            caps.append({
                "id": pid, "lane": "lane1-genesis", "source_host": "",
                "source_container": "", "kind": "registry_only", "targets": [],
                "markers": [], "flags": [meta["env_flag"]] if meta.get("env_flag") else [],
                "upstream": [], "status": "registry-only",
                "summary": meta.get("title", ""), "registry": _reg_fields(meta),
                "registry_id": pid,
            })
    print(f"  registry joined: lane1 {matched1}/{len(reg1)}  lane2 {matched2}/{len(reg2)}")

    # A lane-2 module with working apply()/flags/markers that NO registry row
    # points at cannot be reached by the dispatcher — real code, unreachable.
    # This is the "lost work" class (g4_upstream_tq_wip, spec_decode/probes,
    # de-duplicated standalones).  Mark it so the verifier can report it.
    orphans = 0
    for c in caps:
        if c["lane"] == "lane2-sndr" and c["kind"] != "registry_only" \
                and "registry" not in c and "/patches/" in c["source_host"]:
            src = os.path.join(REPO, c["source_host"])
            try:
                has_apply = "def apply(" in open(src, encoding="utf-8",
                                                 errors="replace").read()
            except OSError:
                has_apply = False
            if has_apply:
                c["unreachable_by_dispatcher"] = True
                c["status"] = "orphan-no-registry-row"
                orphans += 1
    print(f"  lane-2 modules with apply() but NO registry row: {orphans}")

    # de-dupe ids across lanes by suffixing the lane (ids DO collide — see
    # PATCH-ID-COLLISION-AUDIT-20260725.md; 3 lanes share one id space).
    seen: dict[str, int] = {}
    for c in caps:
        key = c["id"]
        seen[key] = seen.get(key, 0) + 1
    for c in caps:
        if seen[c["id"]] > 1:
            c["key"] = f"{c['id']}@{c['lane']}:{os.path.basename(c['source_host'])[:-3]}"
        else:
            c["key"] = c["id"]

    # ---- shared-id suppression: sndr_lane.py exports GENESIS_DISABLE_<id> for
    # every id present in BOTH registries, so lane-1 owns it and lane-2's copy
    # deliberately no-ops.  Its boot line reads "explicitly disabled by
    # operator", which looks like operator action but is self-inflicted by
    # design.  Without this flag the verifier calls ~120 of them lost.
    # sndr_lane.apply_policy() computes this as
    #     shared = set(lane2_registry) & set(lane1_registry)   (+ a rename map)
    # i.e. on REGISTRY KEYS.  Keying it on the extractor's own filename-derived
    # id instead marked 6 rows suppressed that the boot plainly applies: the
    # chunk_o consolidated module is registry id PN298 (no lane-1 twin) but its
    # filename `pn29_pn298_...` derives to PN29, which lane-1 does own.  Its
    # boot line reads `APPLY PN298`, and the ledger called it partially lost.
    # Cross-checked all 7 flips against the boot's own words: P18B_TEXT,
    # PN16_V6 and PN298 log APPLY; P23_WIRE, P29_HEAL and PN102 log a plain
    # "opt-in only"; only P18b logs "explicitly disabled by operator".
    # Rows with no registry join keep the filename-derived fallback.
    l1_ids = {c["id"] for c in caps if c["lane"].startswith("lane1")}
    l1_reg_ids = {str(k).upper() for k in reg1}
    renamed = {"PN40-CLASSIFIER": "PN40C"}  # sndr_lane._LANE1_RENAMED_SHARED
    shared = 0
    for c in caps:
        if c["lane"] != "lane2-sndr":
            continue
        rid = c.get("registry_id")
        if rid:
            rid_u = str(rid).upper()
            owned = rid_u in l1_reg_ids or renamed.get(rid_u, "") in l1_reg_ids
            if not (c.get("registry") or {}).get("env_flag"):
                owned = False   # apply_policy `continue`s on a flagless row
        else:
            owned = c["id"] in l1_ids
        if owned:
            c["lane2_suppressed_by_lane1"] = True
            shared += 1
    print(f"  lane-2 rows suppressed as lane-1-owned shared ids: {shared}")

    # lane-2's registry is the richer of the two: it carries lifecycle and
    # superseded_by that lane-1's does not.  For a shared id, lane-1 IS the
    # same patch, so inherit the retirement verdict instead of calling the
    # lane-1 copy lost.
    l2_life = {c["id"]: (c.get("registry") or {}) for c in caps
               if c["lane"] == "lane2-sndr" and c.get("registry")}
    for c in caps:
        if c["lane"].startswith("lane1") and c["id"] in l2_life:
            peer = l2_life[c["id"]]
            if peer.get("lifecycle"):
                c["peer_lifecycle"] = peer["lifecycle"]
            if peer.get("superseded_by"):
                c["peer_superseded_by"] = peer["superseded_by"]

    # Model gating: the rig serves Qwen3.6.  Every G4_* / gemma-family row is
    # inert by model, not lost.  Recorded explicitly so a future Gemma rig can
    # invert the filter rather than rediscover it.
    gated = 0
    for c in caps:
        reg = c.get("registry") or {}
        fam = str(reg.get("family", "")) + " " + str(reg.get("category", ""))
        if c["id"].upper().startswith("G4_") or "gemma" in fam.lower() \
                or "gemma4" in c["source_host"].lower():
            c["model_family"] = "gemma4"
            gated += 1
    print(f"  rows gated to the gemma4 model family (inert on a Qwen rig): {gated}")

    # ---- compose invocation: the /fixes lane only runs if the entrypoint
    # actually calls it.  A commented-out line (PN73/PN73T superseded by PN76,
    # PN79 retired) is NOT a missing capability — without this the verifier
    # reports every deliberately-parked sidecar as lost.
    composes = {}
    cdir = os.path.join(VLLM_DIR, "compose/single")
    for name in ("endgame8020.yml", "tcbench8021.yml"):
        p = os.path.join(cdir, name)
        if os.path.exists(p):
            composes[name[:-4]] = open(p, encoding="utf-8", errors="replace").read()
    for c in caps:
        if c["lane"] != "fixes":
            continue
        base = os.path.basename(c["source_host"])
        active, commented = [], []
        for cname, body in composes.items():
            for line in body.splitlines():
                s = line.strip()
                if f"/fixes/{base}" not in s and f"/fixes/cliff2b/{base}" not in s:
                    continue
                if s.lstrip("-").strip().startswith("#"):
                    commented.append(cname)
                else:
                    active.append(cname)
        c["compose_invoked"] = sorted(set(active))
        c["compose_commented"] = sorted(set(commented) - set(active))

    overlay = {}
    if os.path.exists(args.overlay):
        overlay = load_json(args.overlay)
    for c in caps:
        ov = overlay.get(c["key"]) or overlay.get(c["id"])
        if ov:
            c.update(ov)

    doc = {
        "schema": "vllm-capability-inventory/1",
        "generated_from": REPO,
        "container_vllm_root": CONTAINER_VLLM,
        "mounts": MOUNTS,
        "note": ("Verification is an EFFECT check (marker present in the installed "
                 "file / module present on disk), NEVER a boot-log line. "
                 "`exec vllm serve` discards every setattr apply_all made, so "
                 "kind=monkey_patch entries in lane1-genesis are INERT unless "
                 "they carry a self-install hook."),
        "capabilities": sorted(caps, key=lambda c: (c["lane"], c["key"])),
    }
    dump_json(args.out, doc)
    by_lane: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for c in caps:
        by_lane[c["lane"]] = by_lane.get(c["lane"], 0) + 1
        by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
    print(f"wrote {args.out}: {len(caps)} capabilities")
    print("  by lane:", dict(sorted(by_lane.items())))
    print("  by kind:", dict(sorted(by_kind.items())))
    print("  with marker:", sum(1 for c in caps if c["markers"]))
    print("  with target:", sum(1 for c in caps if c["targets"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
