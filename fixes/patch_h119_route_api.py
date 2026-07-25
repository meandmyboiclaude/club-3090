#!/usr/bin/env python3
"""H119 — expose the lens router's deep/lean route over the HTTP API.

WHY THIS EXISTS
---------------
The H119 route consumer that shipped earlier today can only choose the thinking
BUDGET. That is the smaller half of the deep/lean treatment: the other half is
the PN102 prompt banner (v5-class vs v3-class), which is rendered BEFORE prefill
while the route is derived FROM that prefill. Choosing the banner from the route
in one pass is circular, so an in-engine consumer cannot do it. The measured
ceiling of the budget half alone is small — the zero-risk cap-only oracle saving
is ~11.5% and on the deep side cap_hit is 0/31, so raising the deep cap does
nothing whatsoever. The headline result came from TREATMENT SELECTION, which is
a thing a CLIENT does, not a thing the sampler does.

So: hand the decision to the caller and let the caller apply the full treatment
out of band. Two surfaces, both implemented here:

  (a) `X-H119-Route` / `X-H119-Score` / `X-H119-Source` / `X-H119-Mode` /
      `X-H119-Req` response headers on NON-STREAMING chat completions, plus the
      same payload in the response body under `kv_transfer_params.h119`.
      Cost: nothing — it rides a field the response already carries.
  (b) `POST /v1/h119/score` — takes an ordinary chat-completion body, runs it at
      `max_tokens=1` (a prefill and one token; the score IS the prefill, so this
      is the floor) and returns the route WITHOUT generating an answer. This is
      the surface that unblocks banner selection: score first, then send the
      real request with the banner and the budget the route implies.

THE HARD PART — WHY A NAIVE VERSION CANNOT WORK
-----------------------------------------------
`vllm._genesis_pn119.ROUTES` lives in the EngineCore worker process.
`AsyncLLM` calls `EngineCoreClient.make_async_mp_client()` unconditionally, so
the HTTP layer is a DIFFERENT PROCESS and a read from the serving layer sees
`{}` forever (fixes/pn119_router.py docstring, "WHERE THE CONSUMER CANNOT LIVE",
blocker 3 — this mistake has already been made once here and cost a day).

The escape is an already-plumbed cross-process channel. VERIFIED AGAINST THE
ACTUAL IMAGE (localhost/vllm-qwen36-endgame:dev1474cherrymax-1757-20260725,
extracted read-only with `podman run --rm --entrypoint cat`), the chain is:

    v1/engine/__init__.py:196   EngineCoreOutput.kv_transfer_params
                                    dict[str, Any] | None = None      <- msgspec
    v1/engine/output_processor.py:637  kv_transfer_params =
                                    engine_core_output.kv_transfer_params
    v1/engine/output_processor.py:678  -> make_request_output(...)
    outputs.py:153              RequestOutput.kv_transfer_params = ...
    entrypoints/openai/chat_completion/serving.py:1085
                                kv_transfer_params=final_res.kv_transfer_params
    entrypoints/openai/chat_completion/protocol.py:455
                                ChatCompletionResponse.kv_transfer_params

Every one of those five line numbers is exact on the 1757 pin, and the same
five statements are present, byte-identical, on dev1474cherry-1711-20260725 and
dev1060cherry-20260713 (only the line numbers move). The field is declared and
threaded end-to-end and is `None` on every request on this stack: no KV
connector is configured, so `_free_request()` returns `(None, None)` and nothing
ever writes it. It is a free lane.

Consequence worth stating plainly: the RESPONSE BODY surface needs no serving
patch at all. The engine-side publish alone makes the route appear in
`kv_transfer_params` of every non-streaming chat completion. Sites I and J below
exist to add the headers and the score endpoint on top of that.

WHERE THE PUBLISH HAPPENS (site H)
----------------------------------
`Scheduler.update_from_output`, on the `if finished:` branch, immediately after
`self._free_request(request)`. That is:
  * in the EngineCore process, which under TP=1 (`distributed_executor_backend`
    resolves to "uni" -> UniProcExecutor) is the SAME process that runs
    gpu_model_runner and therefore owns the live `ROUTES` dict;
  * keyed by the engine's `req_id`, the same namespace the router writes;
  * executed exactly ONCE per request, on the final output — not per token. The
    route is read at the moment `RequestOutput.kv_transfer_params` is last
    assigned, which is what `final_res` carries;
  * before the runner's `on_finish()` pops the request from ROUTES/scored, which
    happens in the NEXT step's `_update_states`.

TP>1 CAVEAT, STATED NOT HIDDEN: with MultiprocExecutor the model runner is in a
child process and the scheduler's `ROUTES` is empty. The publish then simply
never fires — the field stays `None` and every surface reports "unavailable".
It is never WRONG, only absent, and after 20 consecutive misses the bridge
prints one boxed ERROR naming this as the likely cause instead of going quiet.

SHADOW MODE WORKS TOO (deliberate)
----------------------------------
`ROUTES` is only written when `PN119_MODE=enforce`. The bridge therefore falls
back to `ROUTER.scored[req_id]` + `ROUTER.tdeep`, which are populated in EVERY
mode, and to `ROUTER.unscored` + `ROUTER.fallback_route` for requests the probe
refused. So an operator can run the router in SHADOW — mutating no budget, no
sampling, nothing — and still get the route over HTTP for client-side treatment
selection. That is the zero-risk configuration and it is the recommended one.
`X-H119-Source` says which of the three answered: routes | score | fallback.

HOW A CLIENT ACTUALLY USES THIS
-------------------------------
The point is treatment selection, which is a two-call shape. Only the caller can
do it, because only the caller controls the prompt.

  Bench harness / anything that renders its own banner (the real product):
      1. POST /v1/h119/score with the messages it is ABOUT to send.
      2. Read `route`. On "deep" render the v5-class PN102 banner and send
         thinking_token_budget=H119_DEEP_BUDGET; on "lean" render the v3-class
         banner and send H119_LEAN_BUDGET. On `available: false` do exactly what
         it does today — the endpoint is additive, never a dependency.
      3. POST /v1/chat/completions with that banner and that budget.
     This is the same client-side path that produced the 25-to-v5 / 75-to-v3
     result, so it is the full treatment, not the budget half. Cost is one extra
     prefill; with prefix caching on, step 3 reuses whatever prefix step 2
     shares with it.
     Reasonable belt-and-braces: send `thinking_token_budget` explicitly, which
     the in-engine consumer treats as a caller budget and never overrides, so
     the two mechanisms cannot fight if both are ever on.

  hindsight / agent traffic (no banner control, one call):
      Send the request as usual and read `X-H119-Route` off the response. That
      is after the fact for THIS turn, but it is a free per-turn label: route a
      multi-turn session's next turn, pick a model, tag the trace, or just log
      `X-H119-Req` and join it to the router's v2 sink offline. Note it is
      absent on STREAMING responses (see the weakness note below).

  Offline analysis:
      `X-H119-Req` is the ENGINE request id — the same key the v2 sink's row and
      finish lines use — so a response can be joined to its feature row and its
      true rtok without instrumenting the engine any further.

  curl, to see all of it:
      curl -s localhost:8021/v1/h119/score -H 'content-type: application/json' \
        -d '{"model":"...","messages":[{"role":"user","content":"..."}]}' | jq .
      curl -sD- -o/dev/null localhost:8021/v1/chat/completions ... | grep -i x-h119

KNOWN WEAKNESSES, STATED UP FRONT
---------------------------------
  * STREAMING gets no header. Starlette sends a StreamingResponse's headers
    before the generator runs, and the route does not exist until prefill has
    happened, so there is nothing to put there. The body field is non-streaming
    only too (serving.py fills it on ChatCompletionResponse). Streaming clients
    use /v1/h119/score, which is non-streaming by construction.
  * TP>1 publishes nothing (MultiprocExecutor puts the runner in another
    process). Absent, never wrong, and it shouts once.
  * `n>1` parallel sampling: the children are separate engine requests with
    separate routes, and `final_res` carries whichever finished last. The header
    is then one of the n, not a summary. Do not use it for n>1.
  * The score endpoint costs a real prefill. It is the floor, not free.
  * It is a two-call protocol, so the client can render a banner that disagrees
    with the prompt it scored. Scoring the exact messages you will send is the
    caller's job and nothing here can enforce it.

Env flags (BOTH default OFF; the code is always installed, always inert):
  GENESIS_ENABLE_H119_ROUTE_API=1  — publish the route + attach the headers +
                                     enable POST /v1/h119/score.
  H119_API_BODY=0                  — strip `kv_transfer_params.h119` from the
                                     chat-completion response body (headers and
                                     the score endpoint still work). For callers
                                     that assert on the body shape.
The lens router itself must be installed and enabled (GENESIS_ENABLE_H119_LENS_
ROUTER=1) or there is nothing to publish; GENESIS_ENABLE_H119_ROUTE_BUDGET is
INDEPENDENT of this patch — the API surface neither needs nor implies it.

Sites (three groups, each soft-skipping on its own):
  SIDECAR  vllm/_genesis_h119_api.py — written from SIDECAR_SRC below. NOTE: a
           patch here must WRITE FILES. `apply_all` runs standalone and the
           entrypoint then does `exec vllm serve`, which replaces the process,
           so anything installed by setattr at patch time is gone before the
           server starts. Everything in this patch is bytes on disk.
  H        v1/core/sched/scheduler.py — H-shim (module-level lazy bridge import)
           and H-tag (the one-line publish).
  I/J      entrypoints/openai/chat_completion/api_router.py — I-header rewrites
           the ChatCompletionResponse JSONResponse to carry the headers; J is an
           APPEND (no anchor) adding the shims and POST /v1/h119/score.

DUAL-PIN: anchors are content-sniffed from counted variant sets, never matched
by image tag and never rewritten in place. Verify the counts THE BOOT WILL SEE
(i.e. after apply_all and the sibling /fixes patches rewrite the same files) on
every pin, without booting anything:

    python3 fixes/verify_h119_route_api_anchors.py

A SOFT-SKIP IS NOT SILENT: if GENESIS_ENABLE_H119_ROUTE_API=1 and a group fails
to install, this prints a boxed ERROR to stderr naming the exact anchor and logs
at ERROR — the same treatment patch_h119_lens_router.py gives its own groups,
because the 07-25 no-op arm cost a whole GPQA-30 and its only trace was an INFO.

Never raises into serving: every entry point on the request path is guarded and
returns its input unchanged on any failure.
"""
import logging
import os
import pathlib
import sys

LOG = "[h119-route-api]"
VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
SCHED = VLLM / "v1/core/sched/scheduler.py"
APIR = VLLM / "entrypoints/openai/chat_completion/api_router.py"
SIDECAR_DST = VLLM / "_genesis_h119_api.py"

SCHED_MARKER = "# H119-API:"
APIR_MARKER = "# H119-API:"


# ═══════════════════════════════════════════════════════════════════════════
# The sidecar. Installed as vllm/_genesis_h119_api.py; imported lazily from
# BOTH processes (EngineCore for tag(), API server for headers()/payload()).
# It is the only place that knows the shape of the router's state, so a router
# refactor breaks one file instead of two patched anchors.
# ═══════════════════════════════════════════════════════════════════════════
SIDECAR_SRC = r'''"""H119 route API bridge (installed by /fixes/patch_h119_route_api.py).

Two halves of one hop, deliberately in one file so the payload schema cannot
drift between the process that writes it and the process that reads it:

  ENGINE SIDE (EngineCore process, called from Scheduler.update_from_output)
      tag(req_id, kv_transfer_params) -> kv_transfer_params
    Reads the live lens-router state in THIS process and folds a small dict in
    under the "h119" key of the request's kv_transfer_params — the one field
    already threaded EngineCore -> API server -> ChatCompletionResponse and
    never populated on this stack (no KV connector is configured).

  SERVING SIDE (API server process, called from chat_completion/api_router.py)
      payload(obj)  -> the dict the engine put there, or None
      headers(obj)  -> the X-H119-* response headers
      strip_body(d) -> optionally remove the payload from the dumped body

Resolution order on the engine side, and why there are three sources:
  1. ROUTES[req_id]           — enforce mode. The decision of record; the exact
                                value the budget consumer acted on.
  2. ROUTER.scored[req_id]    — populated in EVERY mode, including shadow, so
     vs ROUTER.tdeep            the API surface works with the router mutating
                                nothing at all. This is the recommended config.
  3. ROUTER.unscored[req_id]  — the probe refused this request (partial prefill
     -> ROUTER.fallback_route   / prefix-cache hit). Reporting the fallback is
                                the truth; reporting nothing would let a
                                fallback storm read as "the router is off".
`source` in the payload says which one answered. A request none of the three
knows about publishes NOTHING — an absent field is honest, an invented route
is not.

Guarantees:
  * enabled() is False by default; every entry point returns its input
    unchanged when the flag is off.
  * Nothing here raises. tag() returns its argument on any exception.
  * An existing kv_transfer_params dict is MERGED, never replaced, so a KV
    connector's keys survive if one is ever configured.
"""
from __future__ import annotations

import collections
import logging
import os
import sys

logger = logging.getLogger("vllm.h119api")

SCHEMA = "h119.route/1"
# Single namespace key inside kv_transfer_params: one obviously-ours key is
# trivially ignorable by any consumer of that field, including a future KV
# connector that round-trips it back into a request.
NS = "h119"

ENV_FLAG = "GENESIS_ENABLE_H119_ROUTE_API"
ENV_BODY = "H119_API_BODY"

HDR_ROUTE = "X-H119-Route"
HDR_SCORE = "X-H119-Score"
HDR_SOURCE = "X-H119-Source"
HDR_MODE = "X-H119-Mode"
HDR_REQ = "X-H119-Req"

ROUTE_DEEP = "deep"
ROUTE_LEAN = "lean"

# The engine cannot tell "no route yet" from "the router is in another process"
# by looking at one request. It can tell by looking at many: N consecutive
# misses with not a single hit is the cross-process signature. 20 matches the
# router's own _ALARM_CONSUMER_MIN_N and is ~2 minutes of bench traffic, far
# beyond any legitimate warm-up (a request is decided inside its own prefill).
_MISS_SHOUT_N = 20

STATS: dict = collections.defaultdict(int)

_ENABLED = None
_BODY = None
_MOD = None
_SHOUTED = False


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    """Cached: this is read once per finished request on the engine side."""
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = _truthy(os.environ.get(ENV_FLAG, ""))
    return _ENABLED


def body_enabled() -> bool:
    global _BODY
    if _BODY is None:
        v = os.environ.get(ENV_BODY, "")
        _BODY = True if v.strip() == "" else _truthy(v)
    return _BODY


def reset_cache() -> None:
    """Tests only: drop the env/module caches."""
    global _ENABLED, _BODY, _MOD, _SHOUTED
    _ENABLED = _BODY = _MOD = None
    _SHOUTED = False
    STATS.clear()


def _router_mod():
    """vllm._genesis_pn119, or None. Resolved once; False caches the absence."""
    global _MOD
    if _MOD is None:
        try:
            from vllm import _genesis_pn119 as _m

            _MOD = _m
        except Exception:
            _MOD = False
    return _MOD or None


def _shout(detail: str) -> None:
    """A route API that publishes nothing must not do it quietly."""
    global _SHOUTED
    if _SHOUTED:
        return
    _SHOUTED = True
    bar = "=" * 72
    msg = (
        bar + "\n"
        "[h119-route-api] ERROR: " + ENV_FLAG + "=1 but the route API has "
        "published NOTHING.\n"
        "[h119-route-api] ERROR: " + detail + "\n"
        "[h119-route-api] ERROR: every X-H119-Route header and every "
        "/v1/h119/score call is returning 'unavailable'; any A/B against "
        "them measures nothing.\n"
        "[h119-route-api] ERROR: check, in order: GENESIS_ENABLE_H119_LENS_"
        "ROUTER=1 (is the router installed at all?), the boot log's "
        "[h119-lens-router] line (did sites A-D apply?), and the executor "
        "(TP>1 puts the model runner in a child process, where this bridge "
        "cannot see ROUTES).\n" + bar
    )
    try:
        print(msg, file=sys.stderr, flush=True)
    except Exception:
        pass
    try:
        logger.error(msg.replace("\n", " | "))
    except Exception:
        pass


def route_payload(req_id: str):
    """ENGINE SIDE. The route for `req_id` in THIS process, or None."""
    mod = _router_mod()
    if mod is None:
        return None
    router = getattr(mod, "ROUTER", None)
    route = source = reason = None
    score = None
    try:
        route = mod.ROUTES.get(req_id)
    except Exception:
        route = None
    if route is not None:
        source = "routes"
        try:
            score = mod.SCORES.get(req_id)
        except Exception:
            score = None
    elif router is not None:
        try:
            score = router.scored.get(req_id)
        except Exception:
            score = None
        if score is not None:
            source = "score"
            try:
                route = ROUTE_DEEP if float(score) >= float(router.tdeep) \
                    else ROUTE_LEAN
            except Exception:
                route = None
                score = None
        else:
            try:
                reason = router.unscored.get(req_id)
            except Exception:
                reason = None
            if reason is not None:
                source = "fallback"
                route = getattr(router, "fallback_route", ROUTE_DEEP)
    if route is None:
        return None
    out = {"v": SCHEMA, "route": str(route), "source": source,
           "req_id": str(req_id)}
    if score is not None:
        try:
            out["score"] = round(float(score), 6)
        except Exception:
            pass
    if reason:
        out["reason"] = str(reason)
    if router is not None:
        try:
            out["mode"] = str(getattr(router, "mode", ""))
            out["tdeep"] = float(getattr(router, "tdeep", 0.0))
        except Exception:
            pass
    try:
        if req_id in mod.EXPLORE:
            out["explore"] = True
    except Exception:
        pass
    return out


def tag(req_id, kv_transfer_params):
    """ENGINE SIDE. Fold the route into kv_transfer_params. Never raises."""
    try:
        if not enabled():
            return kv_transfer_params
        if kv_transfer_params is not None and \
                not isinstance(kv_transfer_params, dict):
            # Something else owns this field on this pin; leave it alone.
            STATS["tag_skip_nondict"] += 1
            return kv_transfer_params
        STATS["tag_calls"] += 1
        p = route_payload(req_id)
        if p is None:
            STATS["tag_miss"] += 1
            if STATS["tag_hit"] == 0 and STATS["tag_miss"] >= _MISS_SHOUT_N:
                mod = _router_mod()
                if mod is None:
                    detail = ("vllm._genesis_pn119 does not import in the "
                              "EngineCore process — the lens router is not "
                              "installed on this boot")
                elif getattr(mod, "ROUTER", None) is None:
                    detail = ("the lens router module imports but ROUTER is "
                              "None in this process — either the router is "
                              "disabled, or the model runner is in a DIFFERENT "
                              "process (TP>1 / MultiprocExecutor) and its "
                              "ROUTES can never be read from here")
                else:
                    detail = ("the router is live in this process but has no "
                              "decision for any of the last %d finished "
                              "requests" % STATS["tag_miss"])
                _shout(detail)
            return kv_transfer_params
        STATS["tag_hit"] += 1
        STATS["route_" + p["route"]] += 1
        STATS["source_" + str(p.get("source"))] += 1
        out = dict(kv_transfer_params) if kv_transfer_params else {}
        out[NS] = p
        return out
    except Exception:
        STATS["tag_error"] += 1
        return kv_transfer_params


def payload(obj):
    """SERVING SIDE. Pull the payload out of a response or a raw dict."""
    try:
        kv = obj
        if obj is not None and not isinstance(obj, dict):
            kv = getattr(obj, "kv_transfer_params", None)
        if not isinstance(kv, dict):
            return None
        p = kv.get(NS)
        return p if isinstance(p, dict) else None
    except Exception:
        return None


def headers(obj) -> dict:
    """SERVING SIDE. {} whenever there is nothing to say."""
    try:
        if not enabled():
            return {}
        p = payload(obj)
        if not p:
            return {}
        h = {HDR_ROUTE: str(p.get("route", ""))}
        if "score" in p:
            try:
                h[HDR_SCORE] = "%.6f" % float(p["score"])
            except Exception:
                pass
        if p.get("source"):
            h[HDR_SOURCE] = str(p["source"])
        if p.get("mode"):
            h[HDR_MODE] = str(p["mode"])
        if p.get("req_id"):
            h[HDR_REQ] = str(p["req_id"])
        # Header values must be latin-1 encodable or starlette raises while
        # building the response. Everything above is ASCII by construction;
        # this is the belt-and-braces that keeps a surprise out of serving.
        return {k: v.encode("ascii", "ignore").decode("ascii")
                for k, v in h.items()}
    except Exception:
        return {}


def strip_body(dumped):
    """SERVING SIDE. Honour H119_API_BODY=0 on the dumped response dict."""
    try:
        if body_enabled() or not isinstance(dumped, dict):
            return dumped
        kv = dumped.get("kv_transfer_params")
        if not isinstance(kv, dict) or NS not in kv:
            return dumped
        kv = {k: v for k, v in kv.items() if k != NS}
        dumped["kv_transfer_params"] = kv or None
        return dumped
    except Exception:
        return dumped


def snapshot() -> dict:
    """Operator view: `curl .../v1/h119/score` reports this on unavailable."""
    mod = _router_mod()
    return {
        "schema": SCHEMA,
        "enabled": enabled(),
        "body": body_enabled(),
        "router_importable": mod is not None,
        "router_live_here": bool(mod is not None
                                 and getattr(mod, "ROUTER", None) is not None),
        "stats": {k: int(v) for k, v in sorted(STATS.items())},
    }
'''


# ═══════════════════════════════════════════════════════════════════════════
# Site H — vllm/v1/core/sched/scheduler.py  (the EngineCore-side publish)
# ═══════════════════════════════════════════════════════════════════════════
# H-shim: the lazy bridge import, at module level. Anchored on the logger line,
# which is the most stable statement in any vLLM module. The leading/trailing
# newlines pin it to a whole line, so `sub_logger = init_logger(__name__)` on
# some future pin cannot be mistaken for it.
H_SHIM_OLD = "\nlogger = init_logger(__name__)\n"
H_SHIM_NEW = (
    "\nlogger = init_logger(__name__)\n"
    "\n"
    "\n"
    "# H119-API: bridge to vllm/_genesis_h119_api.py. Resolved once — False\n"
    "# caches the absence, so a boot without the bridge pays one ImportError\n"
    "# in total rather than one per finished request. This function is on the\n"
    "# engine step loop and MUST NOT raise: every failure returns the caller's\n"
    "# own kv_transfer_params, which is what upstream would have used.\n"
    "_H119_API_MOD = None\n"
    "\n"
    "\n"
    "def _h119_api_tag(req_id, kv_transfer_params):\n"
    "    global _H119_API_MOD\n"
    "    if _H119_API_MOD is None:\n"
    "        try:\n"
    "            from vllm import _genesis_h119_api as _m\n"
    "\n"
    "            _H119_API_MOD = _m\n"
    "        except Exception:\n"
    "            _H119_API_MOD = False\n"
    "    if not _H119_API_MOD:\n"
    "        return kv_transfer_params\n"
    "    try:\n"
    "        return _H119_API_MOD.tag(req_id, kv_transfer_params)\n"
    "    except Exception:\n"
    "        return kv_transfer_params\n"
)

# H-tag: the publish itself. `if finished:` runs exactly once per request, on
# the step that produces its FINAL EngineCoreOutput — which is the output
# `final_res.kv_transfer_params` comes from, and which still precedes the
# runner's on_finish() (next step's _update_states) that pops ROUTES/scored.
_H_TAG_INSERT = (
    "                    # H119-API: publish this request's deep/lean route on\n"
    "                    # the one field already threaded EngineCore -> API\n"
    "                    # server -> ChatCompletionResponse. Returns its input\n"
    "                    # unchanged when the flag is off, when no route exists,\n"
    "                    # or on ANY error — it never raises into the step loop.\n"
    "                    kv_transfer_params = _h119_api_tag(\n"
    "                        req_id, kv_transfer_params)\n"
)
H_TAG_A_OLD = (
    "                if finished:\n"
    "                    kv_transfer_params, ec_transfer_params = "
    "self._free_request(request)\n"
)
H_TAG_A_NEW = H_TAG_A_OLD + _H_TAG_INSERT
# Bare variant: same statement, no `if finished:` above it (an upstream that
# restructures the stop handling). Same indentation, so the insert still fits.
H_TAG_B_OLD = (
    "                    kv_transfer_params, ec_transfer_params = "
    "self._free_request(request)\n"
)
H_TAG_B_NEW = H_TAG_B_OLD + _H_TAG_INSERT
# Two-level-shallower variant, for a pin that flattens the branch.
H_TAG_C_OLD = (
    "            kv_transfer_params, ec_transfer_params = "
    "self._free_request(request)\n"
)
H_TAG_C_NEW = H_TAG_C_OLD + "\n".join(
    ln[8:] if ln.startswith(" " * 8) else ln
    for ln in _H_TAG_INSERT.rstrip("\n").split("\n")
) + "\n"

H_SHIM_VARIANTS = (
    ("H-shim/logger", ((H_SHIM_OLD, H_SHIM_NEW),)),
)
H_TAG_VARIANTS = (
    ("H-tag/finished", ((H_TAG_A_OLD, H_TAG_A_NEW),)),
    ("H-tag/bare20", ((H_TAG_B_OLD, H_TAG_B_NEW),)),
    ("H-tag/bare12", ((H_TAG_C_OLD, H_TAG_C_NEW),)),
)


# ═══════════════════════════════════════════════════════════════════════════
# Sites I/J — entrypoints/openai/chat_completion/api_router.py
# ═══════════════════════════════════════════════════════════════════════════
# I-header: the only change to stock behaviour is that the header mapping gains
# the X-H119-* entries and the dumped body passes through strip_body(). With
# the flag off both are identity, so the response is byte-identical to stock.
I_A_OLD = (
    "    elif isinstance(generator, ChatCompletionResponse):\n"
    "        return JSONResponse(\n"
    "            content=generator.model_dump(),\n"
    "            headers=metrics_header(metrics_header_format),\n"
    "        )\n"
)
I_A_NEW = (
    "    elif isinstance(generator, ChatCompletionResponse):\n"
    "        # H119-API: attach the route headers. `metrics_header` is typed\n"
    "        # Mapping|None, hence the `or {}`. _h119_api_headers() returns {}\n"
    "        # whenever the flag is off, the engine published nothing, or\n"
    "        # anything at all went wrong, so this is stock behaviour + 0..5\n"
    "        # extra headers and can never fail the response.\n"
    "        _h119_hdrs = dict(metrics_header(metrics_header_format) or {})\n"
    "        _h119_hdrs.update(_h119_api_headers(generator))\n"
    "        return JSONResponse(\n"
    "            content=_h119_api_body(generator.model_dump()),\n"
    "            headers=_h119_hdrs,\n"
    "        )\n"
)
# Variant for a pin that never grew the ORCA header argument.
I_B_OLD = (
    "    elif isinstance(generator, ChatCompletionResponse):\n"
    "        return JSONResponse(content=generator.model_dump())\n"
)
I_B_NEW = (
    "    elif isinstance(generator, ChatCompletionResponse):\n"
    "        # H119-API: attach the route headers (see patch_h119_route_api).\n"
    "        return JSONResponse(\n"
    "            content=_h119_api_body(generator.model_dump()),\n"
    "            headers=_h119_api_headers(generator),\n"
    "        )\n"
)
I_VARIANTS = (
    ("I-header/orca", ((I_A_OLD, I_A_NEW),)),
    ("I-header/plain", ((I_B_OLD, I_B_NEW),)),
)

# Site J — APPENDED, no anchor. Function definitions are resolved at call time,
# so appending after `attach_router` is fine: the decorator registers the route
# on `router` at import, and `attach_router(app)` runs later at startup.
# Everything it references (router, chat, Depends, JSONResponse, Request,
# ChatCompletionRequest, ChatCompletionResponse, ErrorResponse,
# validate_json_request) is already imported at the top of this module on all
# three pins; the import is re-checked by _resolve_api_sites() before we write.
J_APPEND = '''

# ═══════════════════════════════════════════════════════════════════════════
# H119-API: route exposure (installed by /fixes/patch_h119_route_api.py)
# ═══════════════════════════════════════════════════════════════════════════
# The lens router scores each request's prefill hidden states in the EngineCore
# worker process; the scheduler folds the resulting deep/lean route into
# EngineCoreOutput.kv_transfer_params, which is threaded to
# ChatCompletionResponse.kv_transfer_params. These two shims read it back out.
# Both are total: they return {} / their input on any failure, so a broken or
# absent bridge cannot affect a response.
_H119_API_MOD = None


def _h119_api():
    global _H119_API_MOD
    if _H119_API_MOD is None:
        try:
            from vllm import _genesis_h119_api as _m

            _H119_API_MOD = _m
        except Exception:
            _H119_API_MOD = False
    return _H119_API_MOD or None


def _h119_api_headers(obj) -> dict:
    mod = _h119_api()
    if mod is None:
        return {}
    try:
        return mod.headers(obj)
    except Exception:
        return {}


def _h119_api_body(dumped):
    mod = _h119_api()
    if mod is None:
        return dumped
    try:
        return mod.strip_body(dumped)
    except Exception:
        return dumped


# Fields the probe overrides, when the pin has them. Filtered against
# model_fields so a pin that renames one degrades to "not overridden" instead
# of raising: model_copy(update=...) bypasses validation and would happily
# attach an attribute nothing reads.
_H119_PROBE_OVERRIDES = {
    "stream": False,
    "stream_options": None,
    "n": 1,
    "max_tokens": 1,
    "max_completion_tokens": 1,
    "logprobs": False,
    "top_logprobs": 0,
    "prompt_logprobs": None,
    "echo": False,
    "kv_transfer_params": None,
}


@router.post(
    "/v1/h119/score",
    dependencies=[Depends(validate_json_request)],
)
async def h119_score(request: ChatCompletionRequest, raw_request: Request):
    """Score a prompt and return its deep/lean route WITHOUT answering it.

    Body: an ordinary /v1/chat/completions body. Send the SAME messages,
    model and chat_template_kwargs you intend to send for real — the route is a
    function of the rendered prompt, so a different rendering is a different
    question.

    What it costs: one prefill and one token. The score IS the prefill, so this
    is the floor; there is no cheaper way to obtain it. With prefix caching on,
    the real request that follows reuses the prefill this call warmed, for the
    prefix the two share.

    Response 200 always (a scoring failure must not break a caller's flow):
        {"available": true, "route": "deep"|"lean", "score": float,
         "source": "routes"|"score"|"fallback", "mode": "shadow"|"enforce",
         "tdeep": float, "req_id": str, "prompt_tokens": int, "id": str}
        {"available": false, "route": null, "reason": str, "bridge": {...}}
    The same X-H119-* headers are attached.
    """
    mod = _h119_api()
    if mod is None:
        return JSONResponse(
            status_code=503,
            content={"available": False, "route": None,
                     "reason": "H119 route API bridge is not installed"},
        )
    if not mod.enabled():
        return JSONResponse(
            status_code=404,
            content={"available": False, "route": None,
                     "reason": "H119 route API is disabled "
                               "(set GENESIS_ENABLE_H119_ROUTE_API=1)"},
        )
    handler = chat(raw_request)
    if handler is None:
        return JSONResponse(
            status_code=501,
            content={"available": False, "route": None,
                     "reason": "the model does not support Chat Completions"},
        )
    try:
        fields = set(type(request).model_fields)
        update = {k: v for k, v in _H119_PROBE_OVERRIDES.items()
                  if k in fields}
        probe = request.model_copy(update=update)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"available": False, "route": None,
                     "reason": "could not build the probe request: %r" % (e,)},
        )
    result = await handler.create_chat_completion(probe, raw_request)
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(),
                            status_code=result.error.code)
    if not isinstance(result, ChatCompletionResponse):
        return JSONResponse(
            status_code=500,
            content={"available": False, "route": None,
                     "reason": "probe did not return a ChatCompletionResponse"},
        )
    info = _h119_api_payload_of(result)
    body = {"available": info is not None, "route": None, "id": result.id}
    try:
        body["prompt_tokens"] = result.usage.prompt_tokens if result.usage \\
            else None
    except Exception:
        body["prompt_tokens"] = None
    if info is None:
        body["reason"] = ("the engine published no route for this request — "
                          "the lens router is off, not installed, or in a "
                          "different process (TP>1)")
        try:
            body["bridge"] = mod.snapshot()
        except Exception:
            pass
    else:
        for k in ("route", "score", "source", "mode", "tdeep", "reason",
                  "explore", "req_id"):
            if k in info:
                body[k] = info[k]
    return JSONResponse(content=body, headers=_h119_api_headers(result))


def _h119_api_payload_of(obj):
    mod = _h119_api()
    if mod is None:
        return None
    try:
        return mod.payload(obj)
    except Exception:
        return None
'''


# ═══════════════════════════════════════════════════════════════════════════
# Anchor resolution — content sniff, counted, never a tag match
# ═══════════════════════════════════════════════════════════════════════════
def _pick(text: str, variants, out_sites, problems, label):
    """First variant whose EVERY (old, new) pair counts exactly 1 wins."""
    detail = []
    for vname, pairs in variants:
        counts = [text.count(old) for old, _ in pairs]
        if all(c == 1 for c in counts):
            out_sites.extend((f"{vname}[{i}]" if len(pairs) > 1 else vname,
                              old, new) for i, (old, new) in enumerate(pairs))
            return True
        detail.append(f"{vname}={counts}")
    problems[label] = "no variant matched (" + ", ".join(detail) + ")"
    return False


def counts_report(text: str, variants) -> str:
    return "  ".join(
        f"{vname}={[text.count(o) for o, _ in pairs]}"
        for vname, pairs in variants)


def _resolve_sched_sites(text: str):
    """(sites, problem) for scheduler.py, counted against THIS text."""
    sites: list = []
    problems: dict = {}
    _pick(text, H_SHIM_VARIANTS, sites, problems, "H-shim")
    _pick(text, H_TAG_VARIANTS, sites, problems, "H-tag")
    if problems:
        return None, ", ".join(f"{k}: {v}" for k, v in problems.items())
    return sites, None


# Names site J's appended code needs to already exist in api_router.py. J is an
# append with no anchor, so this import check IS its anchor: a pin that stopped
# importing any of these would give a NameError at request time instead of a
# soft-skip at patch time.
J_REQUIRED = (
    "router = APIRouter()",
    "def chat(request: Request)",
    "JSONResponse",
    "Depends",
    "validate_json_request",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ErrorResponse",
)


def _resolve_api_sites(text: str):
    """(sites, problem) for chat_completion/api_router.py."""
    sites: list = []
    problems: dict = {}
    _pick(text, I_VARIANTS, sites, problems, "I-header")
    missing = [n for n in J_REQUIRED if n not in text]
    if missing:
        problems["J-endpoint"] = f"module lacks {missing}"
    if problems:
        return None, ", ".join(f"{k}: {v}" for k, v in problems.items())
    return sites, None


# ═══════════════════════════════════════════════════════════════════════════
# Application
# ═══════════════════════════════════════════════════════════════════════════
def _flag_requested() -> bool:
    return os.environ.get("GENESIS_ENABLE_H119_ROUTE_API", "").strip().lower() \
        in ("1", "true", "yes", "on")


def _shout(what: str, detail: str) -> None:
    bar = "=" * 72
    msg = (f"{bar}\n"
           f"{LOG} ERROR: {what} was REQUESTED but is NOT INSTALLED.\n"
           f"{LOG} ERROR: {detail}\n"
           f"{LOG} ERROR: this boot will behave EXACTLY as if the flag were "
           f"off — no X-H119-Route header, no /v1/h119/score, and any A/B "
           f"against them measures nothing.\n"
           f"{LOG} ERROR: re-derive the anchors against POST-PATCH content: "
           f"python3 /fixes/verify_h119_route_api_anchors.py\n"
           f"{bar}")
    print(msg, file=sys.stderr, flush=True)
    try:
        logging.getLogger("genesis.h119api").error(msg.replace("\n", " | "))
    except Exception:  # noqa: BLE001 — logging must never break a boot
        pass


def _install_sidecar() -> tuple[str, bool]:
    try:
        SIDECAR_DST.write_text(SIDECAR_SRC, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return (f"sidecar install FAILED ({e}) — route API cannot work", False)
    return (f"sidecar installed at {SIDECAR_DST.name} "
            f"({len(SIDECAR_SRC)} bytes)", True)


def _patch_file(path: pathlib.Path, marker: str, resolver, append: str,
                group: str) -> tuple[str, bool]:
    """Apply one site group. NEVER fails the boot; returns (status, installed)."""
    if not path.exists():
        return (f"soft-skip {group}: {path} absent on this pin", False)
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return (f"soft-skip {group}: unreadable ({e})", False)
    if marker in text:
        return (f"{group} already applied (idempotent)", True)
    sites, problem = resolver(text)
    if problem:
        return (f"soft-skip {group}: {problem} — anchors do not fit this pin's "
                f"{path.name} AS PATCHED BY THE EARLIER BOOT PATCHES", False)
    applied = []
    for name, old, new in sites:
        text = text.replace(old, new, 1)
        applied.append(name)
    if append:
        text = text + append
        applied.append(f"{group}-append")
    try:
        path.write_text(text, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return (f"soft-skip {group}: write failed ({e})", False)
    return (f"applied {applied}", True)


def main() -> int:
    requested = _flag_requested()
    status, side_ok = _install_sidecar()
    print(f"{LOG} {status}")
    if not side_ok and requested:
        _shout("GENESIS_ENABLE_H119_ROUTE_API=1 (sidecar)", status)

    # The two groups are INDEPENDENT on purpose. Group H alone already puts the
    # route in the response BODY (kv_transfer_params.h119), which is a complete,
    # usable surface; group I/J adds the headers and the score endpoint. A drift
    # in one must not cost the other.
    h_status, h_ok = _patch_file(SCHED, SCHED_MARKER, _resolve_sched_sites,
                                 "", "H (scheduler publish)")
    print(f"{LOG} {h_status}")
    if not h_ok and requested:
        _shout("GENESIS_ENABLE_H119_ROUTE_API=1 (site group H)", h_status)

    a_status, a_ok = _patch_file(APIR, APIR_MARKER, _resolve_api_sites,
                                 J_APPEND, "I/J (headers + /v1/h119/score)")
    print(f"{LOG} {a_status}")
    if not a_ok and requested:
        _shout("GENESIS_ENABLE_H119_ROUTE_API=1 (site group I/J)", a_status)
    elif a_ok and not h_ok and requested:
        _shout("GENESIS_ENABLE_H119_ROUTE_API=1 (site group H)",
               "the serving side installed but the ENGINE side did not, so "
               "there is nothing to read: every header and every score call "
               "will report unavailable")

    print(f"{LOG} inert unless GENESIS_ENABLE_H119_ROUTE_API=1 "
          f"(needs GENESIS_ENABLE_H119_LENS_ROUTER=1 to have a route to "
          f"publish; PN119_MODE=shadow is sufficient and is the zero-risk "
          f"configuration)")
    return 0


sys.exit(main())
