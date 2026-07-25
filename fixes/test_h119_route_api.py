#!/usr/bin/env python3
"""H119 — the route API contract (patch_h119_route_api.py).

Run: python3 fixes/test_h119_route_api.py
     (no boot, no GPU, no container, stdlib only — CPU exec + fakes)

WHAT IS UNDER TEST, AND WHY EACH PART IS HERE
---------------------------------------------
1. THE PATCHER, end to end, against a temp tree whose two files carry the real
   anchors. This exercises main(): sidecar write, both site groups, the
   idempotency marker, and — the part that has actually bitten this lane twice —
   the LOUD soft-skip when the operator asked for the flag and an anchor drifted.
   (Anchor counts against real boot-time content are a different question and a
   different tool: fixes/verify_h119_route_api_anchors.py.)

2. THE SIDECAR's resolution order. The route can come from three places and the
   distinction is the whole point of the design:
       ROUTES        enforce mode, the decision the budget consumer acted on
       scored/tdeep  EVERY mode including shadow -> the zero-risk configuration
       unscored      the probe refused; the fallback is the truth about it
   A request none of the three knows publishes NOTHING. An absent field is
   honest; an invented route would silently poison a client-side A/B.

3. THE ENGINE-SIDE GUARANTEES: tag() never raises, never drops a KV connector's
   keys, and is the identity function when the flag is off.

4. THE SCORE ENDPOINT's logic, executed for real against fakes: probe-override
   filtering by model_fields, error passthrough, available:false with a reason
   when the engine published nothing, and headers on every path.

5. THE CROSS-PROCESS ALARM. If the model runner is in another process (TP>1)
   the bridge can only ever miss. That must produce one boxed ERROR, not
   silence — the failure mode this whole lane keeps re-learning.
"""
from __future__ import annotations

import asyncio
import io
import pathlib
import sys
import types

HERE = pathlib.Path(__file__).resolve().parent
PATCH = HERE / "patch_h119_route_api.py"

_fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


# ───────────────────────────────────────────────────────────── patcher loading
def patcher_ns(dist: pathlib.Path) -> dict:
    """The patcher's namespace with its dist-packages path re-pointed."""
    src = PATCH.read_text(encoding="utf-8")
    src = src.replace("/usr/local/lib/python3.12/dist-packages", str(dist))
    src = src.replace("\nsys.exit(main())", "\n")
    g: dict = {"__name__": "h119_api_probe"}
    exec(compile(src, "patch_h119_route_api.py", "exec"), g)  # noqa: S102
    return g


# Faithful excerpts: the anchors verbatim as they appear on all three pinned
# images (checked by verify_h119_route_api_anchors.py), with just enough
# surrounding code to be a valid module.
SCHED_FIXTURE = '''from vllm.logger import init_logger

logger = init_logger(__name__)


class Scheduler:
    def update_from_output(self, scheduler_output, model_runner_output):
        for req_id, request in self.requests.items():
            kv_transfer_params = None
            ec_transfer_params = None
            if stopped:
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params, ec_transfer_params = self._free_request(request)
            outputs.append(
                EngineCoreOutput(
                    request_id=req_id,
                    kv_transfer_params=kv_transfer_params,
                    ec_transfer_params=ec_transfer_params,
                )
            )
'''

APIR_FIXTURE = '''from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.orca_metrics import metrics_header

router = APIRouter()


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get("endpoint-load-metrics-format", "")
    handler = chat(raw_request)
    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    return StreamingResponse(content=generator, media_type="text/event-stream")


def attach_router(app: FastAPI):
    app.include_router(router)
'''


def _tree(root: pathlib.Path, sched: str = SCHED_FIXTURE,
          apir: str = APIR_FIXTURE) -> pathlib.Path:
    dist = root / "dist"
    (dist / "vllm/v1/core/sched").mkdir(parents=True, exist_ok=True)
    (dist / "vllm/entrypoints/openai/chat_completion").mkdir(
        parents=True, exist_ok=True)
    (dist / "vllm/v1/core/sched/scheduler.py").write_text(sched, encoding="utf-8")
    (dist / "vllm/entrypoints/openai/chat_completion/api_router.py").write_text(
        apir, encoding="utf-8")
    return dist


class _capture:
    """Capture stderr so the boxed ERROR can be asserted on, not eyeballed."""

    def __enter__(self):
        self._old, sys.stderr = sys.stderr, io.StringIO()
        return self

    def __exit__(self, *a):
        self.text = sys.stderr.getvalue()
        sys.stderr = self._old
        return False


# ═════════════════════════════════════════════════════ 1. the patcher, for real
def test_patcher(tmp: pathlib.Path) -> None:
    print("\n[1] patcher end-to-end")
    dist = _tree(tmp / "ok")
    g = patcher_ns(dist)
    rc = g["main"]()
    check("main() returns 0", rc == 0)

    side = dist / "vllm/_genesis_h119_api.py"
    check("sidecar written as a FILE (exec replaces the process, so setattr "
          "would not survive)", side.exists())
    compile(side.read_text(encoding="utf-8"), str(side), "exec")
    check("sidecar byte-compiles", True)

    sched = (dist / "vllm/v1/core/sched/scheduler.py").read_text(encoding="utf-8")
    apir = (dist / "vllm/entrypoints/openai/chat_completion/api_router.py"
            ).read_text(encoding="utf-8")
    compile(sched, "scheduler.py", "exec")
    compile(apir, "api_router.py", "exec")
    check("both patched files byte-compile", True)
    check("H-shim installed", "_h119_api_tag(req_id, kv_transfer_params)" in sched
          or "def _h119_api_tag" in sched)
    check("H-tag publishes on the finished branch",
          "kv_transfer_params = _h119_api_tag(" in sched)
    check("H-tag sits AFTER _free_request (else the connector's value would "
          "overwrite ours)",
          sched.index("self._free_request(request)")
          < sched.index("kv_transfer_params = _h119_api_tag("))
    check("I-header wraps the JSONResponse", "_h119_api_headers(generator)" in apir)
    check("I-header keeps the ORCA headers",
          "metrics_header(metrics_header_format) or {}" in apir)
    check("J registers /v1/h119/score", '"/v1/h119/score"' in apir)
    check("stock streaming path untouched",
          "return StreamingResponse(content=generator, "
          "media_type=\"text/event-stream\")" in apir)

    # Idempotency: a second run must change nothing.
    before = (sched, apir)
    g2 = patcher_ns(dist)
    check("second main() returns 0", g2["main"]() == 0)
    after = ((dist / "vllm/v1/core/sched/scheduler.py").read_text(encoding="utf-8"),
             (dist / "vllm/entrypoints/openai/chat_completion/api_router.py"
              ).read_text(encoding="utf-8"))
    check("idempotent (marker gate, not anchor luck)", before == after)


def test_soft_skip_is_loud(tmp: pathlib.Path, monkeyflag: bool) -> None:
    print(f"\n[2] soft-skip loudness (flag {'ON' if monkeyflag else 'OFF'})")
    # Drift BOTH anchors: rename the free_request call and the JSONResponse.
    sched = SCHED_FIXTURE.replace("self._free_request(request)",
                                  "self._release_request(request)")
    apir = APIR_FIXTURE.replace("headers=metrics_header(metrics_header_format),",
                                "headers=None, extra=1,")
    dist = _tree(tmp / ("loud" if monkeyflag else "quiet"), sched, apir)
    import os
    old = os.environ.get("GENESIS_ENABLE_H119_ROUTE_API")
    os.environ["GENESIS_ENABLE_H119_ROUTE_API"] = "1" if monkeyflag else "0"
    try:
        g = patcher_ns(dist)
        with _capture() as cap:
            rc = g["main"]()
    finally:
        if old is None:
            os.environ.pop("GENESIS_ENABLE_H119_ROUTE_API", None)
        else:
            os.environ["GENESIS_ENABLE_H119_ROUTE_API"] = old
    check("boot is never failed by a drifted anchor", rc == 0)
    got = (dist / "vllm/v1/core/sched/scheduler.py").read_text(encoding="utf-8")
    check("drifted file left BYTE-IDENTICAL", got == sched)
    if monkeyflag:
        check("boxed ERROR on stderr", "ERROR:" in cap.text and "=" * 72 in cap.text)
        check("names the site group", "site group H" in cap.text
              and "site group I/J" in cap.text)
        check("names the anchor that failed", "H-tag" in cap.text
              and "I-header" in cap.text)
        check("points at the verifier",
              "verify_h119_route_api_anchors.py" in cap.text)
        check("says the A/B would measure nothing",
              "measures nothing" in cap.text)
    else:
        check("silent when the flag was never requested",
              "ERROR:" not in cap.text)


# ═════════════════════════════════════════════════════════ 2. the sidecar
class FakeRouter:
    def __init__(self, mode="shadow", tdeep=0.495, fallback="deep"):
        self.mode = mode
        self.tdeep = tdeep
        self.fallback_route = fallback
        self.scored: dict = {}
        self.unscored: dict = {}


def load_sidecar(g: dict, router=None, routes=None, scores=None, explore=None,
                 broken=False):
    """Exec the sidecar with a fake vllm._genesis_pn119 in place."""
    pn119 = types.ModuleType("vllm._genesis_pn119")
    if broken:
        class Boom:
            def get(self, *a):
                raise RuntimeError("router state is broken")

            def __contains__(self, k):
                raise RuntimeError("router state is broken")
        pn119.ROUTES = Boom()
        pn119.SCORES = Boom()
        pn119.EXPLORE = Boom()
        pn119.ROUTER = router
    else:
        pn119.ROUTES = dict(routes or {})
        pn119.SCORES = dict(scores or {})
        pn119.EXPLORE = set(explore or ())
        pn119.ROUTER = router
    vllm = sys.modules.get("vllm")
    if vllm is None or not isinstance(vllm, types.ModuleType):
        vllm = types.ModuleType("vllm")
        sys.modules["vllm"] = vllm
    vllm._genesis_pn119 = pn119
    sys.modules["vllm._genesis_pn119"] = pn119
    ns: dict = {"__name__": "vllm._genesis_h119_api"}
    exec(compile(g["SIDECAR_SRC"], "_genesis_h119_api.py", "exec"), ns)  # noqa: S102
    mod = types.ModuleType("vllm._genesis_h119_api")
    mod.__dict__.update(ns)
    sys.modules["vllm._genesis_h119_api"] = mod
    vllm._genesis_h119_api = mod
    return mod


def _on(mod) -> None:
    import os
    os.environ["GENESIS_ENABLE_H119_ROUTE_API"] = "1"
    os.environ.pop("H119_API_BODY", None)
    mod.reset_cache()


def test_sidecar(g: dict) -> None:
    print("\n[3] sidecar resolution order")
    import os

    # -- flag OFF is the identity function -------------------------------
    r = FakeRouter()
    r.scored["a"] = 9.0
    mod = load_sidecar(g, router=r)
    os.environ["GENESIS_ENABLE_H119_ROUTE_API"] = "0"
    mod.reset_cache()
    check("flag off: None stays None", mod.tag("a", None) is None)
    kv = {"remote_block_ids": [1]}
    check("flag off: dict passes through untouched", mod.tag("a", kv) is kv)
    check("flag off: no headers", mod.headers({"h119": {"route": "deep"}}) == {})

    # -- source 1: ROUTES (enforce) --------------------------------------
    r = FakeRouter(mode="enforce")
    mod = load_sidecar(g, router=r, routes={"a": "lean"}, scores={"a": 0.11})
    _on(mod)
    out = mod.tag("a", None)
    p = out["h119"]
    check("enforce: route from ROUTES", p["route"] == "lean", str(p))
    check("enforce: source=routes", p["source"] == "routes")
    check("enforce: score from SCORES", p["score"] == 0.11)
    check("enforce: mode reported", p["mode"] == "enforce")
    check("enforce: req_id carried for sink correlation", p["req_id"] == "a")

    # -- source 2: scored/tdeep (SHADOW — the zero-risk configuration) ----
    r = FakeRouter(mode="shadow", tdeep=0.5)
    r.scored.update({"hi": 0.5, "lo": 0.499999, "way": 3.0})
    mod = load_sidecar(g, router=r)
    _on(mod)
    check("shadow works at all (ROUTES is empty in shadow)",
          mod.tag("way", None)["h119"]["route"] == "deep")
    check("shadow: source=score", mod.tag("way", None)["h119"]["source"] == "score")
    check("shadow: score >= tdeep is DEEP (boundary inclusive, matches "
          "_finalize)", mod.tag("hi", None)["h119"]["route"] == "deep")
    check("shadow: just under tdeep is LEAN",
          mod.tag("lo", None)["h119"]["route"] == "lean")
    check("shadow: tdeep reported so a client can re-derive the call",
          mod.tag("hi", None)["h119"]["tdeep"] == 0.5)

    # -- source 3: unscoreable -> fallback --------------------------------
    r = FakeRouter(mode="enforce", fallback="deep")
    r.unscored["u"] = "partial_prefill"
    mod = load_sidecar(g, router=r)
    _on(mod)
    p = mod.tag("u", None)["h119"]
    check("unscoreable: fallback route published", p["route"] == "deep")
    check("unscoreable: source=fallback", p["source"] == "fallback")
    check("unscoreable: reason carried", p["reason"] == "partial_prefill")
    check("unscoreable: no invented score", "score" not in p)

    # -- unknown request publishes NOTHING --------------------------------
    mod = load_sidecar(g, router=FakeRouter())
    _on(mod)
    check("unknown req: None stays None (absent, not invented)",
          mod.tag("nope", None) is None)
    kv = {"remote_block_ids": [7]}
    check("unknown req: caller's dict returned unchanged",
          mod.tag("nope", kv) is kv)

    # -- merge semantics ---------------------------------------------------
    r = FakeRouter()
    r.scored["m"] = 1.0
    mod = load_sidecar(g, router=r, explore={"m"})
    _on(mod)
    out = mod.tag("m", {"remote_block_ids": [1, 2], "remote_engine_id": "x"})
    check("merge: connector keys survive",
          out["remote_block_ids"] == [1, 2] and out["remote_engine_id"] == "x")
    check("merge: single namespaced key added", set(out) - {
        "remote_block_ids", "remote_engine_id"} == {"h119"})
    check("explore flag propagates (labels must stay uncensored)",
          out["h119"].get("explore") is True)
    sentinel = object()
    check("non-dict kv_transfer_params is left entirely alone",
          mod.tag("m", sentinel) is sentinel)

    # -- nothing raises ----------------------------------------------------
    mod = load_sidecar(g, router=FakeRouter(), broken=True)
    _on(mod)
    kv = {"k": 1}
    try:
        got = mod.tag("x", kv)
        raised = False
    except Exception:
        got, raised = None, True
    check("a broken router cannot raise into the engine step loop", not raised)
    check("...and the caller's value is returned unchanged", got is kv)

    # -- no router at all ---------------------------------------------------
    sys.modules.pop("vllm._genesis_pn119", None)
    vllm = sys.modules["vllm"]
    if hasattr(vllm, "_genesis_pn119"):
        del vllm._genesis_pn119
    ns: dict = {"__name__": "vllm._genesis_h119_api"}
    exec(compile(g["SIDECAR_SRC"], "_genesis_h119_api.py", "exec"), ns)  # noqa: S102
    bare = types.ModuleType("m")
    bare.__dict__.update(ns)
    _on(bare)
    check("router not installed: tag is the identity", bare.tag("a", None) is None)
    check("router not installed: snapshot says so",
          bare.snapshot()["router_importable"] is False)


def test_headers_and_body(g: dict) -> None:
    print("\n[4] serving-side extraction")
    import os
    r = FakeRouter(mode="shadow", tdeep=0.5)
    r.scored["req-7"] = 0.6
    mod = load_sidecar(g, router=r)
    _on(mod)
    kv = mod.tag("req-7", None)

    class Resp:
        kv_transfer_params = kv

    h = mod.headers(Resp())
    check("X-H119-Route set", h.get("X-H119-Route") == "deep", str(h))
    check("X-H119-Score is a parseable float",
          abs(float(h["X-H119-Score"]) - 0.6) < 1e-9)
    check("X-H119-Source set", h.get("X-H119-Source") == "score")
    check("X-H119-Mode set", h.get("X-H119-Mode") == "shadow")
    check("X-H119-Req set (joins the response to the sink row)",
          h.get("X-H119-Req") == "req-7")
    check("header values are latin-1 safe",
          all(v.encode("latin-1") for v in h.values()))
    check("accepts a raw dict too", mod.headers(kv)["X-H119-Route"] == "deep")
    check("no payload -> no headers", mod.headers({"other": 1}) == {})
    check("None -> no headers", mod.headers(None) == {})

    dumped = {"id": "x", "kv_transfer_params": dict(kv)}
    check("H119_API_BODY default keeps the payload in the body",
          mod.strip_body(dict(dumped))["kv_transfer_params"] is not None
          and "h119" in mod.strip_body(dict(dumped))["kv_transfer_params"])
    os.environ["H119_API_BODY"] = "0"
    mod.reset_cache()
    os.environ["GENESIS_ENABLE_H119_ROUTE_API"] = "1"
    stripped = mod.strip_body({"id": "x", "kv_transfer_params": dict(kv)})
    check("H119_API_BODY=0 removes the payload",
          stripped["kv_transfer_params"] is None)
    stripped2 = mod.strip_body(
        {"kv_transfer_params": dict(kv, remote_block_ids=[3])})
    check("H119_API_BODY=0 keeps a real connector's keys",
          stripped2["kv_transfer_params"] == {"remote_block_ids": [3]})
    os.environ.pop("H119_API_BODY", None)
    mod.reset_cache()
    os.environ["GENESIS_ENABLE_H119_ROUTE_API"] = "1"


def test_cross_process_alarm(g: dict) -> None:
    print("\n[5] cross-process alarm (the TP>1 / wrong-process failure)")
    # A router that imports but has no ROUTER in THIS process is exactly what
    # MultiprocExecutor looks like from the scheduler.
    mod = load_sidecar(g, router=None)
    _on(mod)
    with _capture() as cap:
        for i in range(19):
            mod.tag(f"r{i}", None)
    check("quiet below the threshold (a fresh boot must not shout)",
          "ERROR" not in cap.text, f"{mod.STATS['tag_miss']} misses")
    with _capture() as cap:
        mod.tag("r19", None)
    check("boxed ERROR at exactly the 20th consecutive miss", "=" * 72 in cap.text)
    check("names the cross-process cause",
          "DIFFERENT process" in cap.text and "TP>1" in cap.text)
    # One shout reaches stderr twice: the deliberate print, plus logging's
    # lastResort handler in this bare test process (in the container the
    # "vllm.h119api" logger has a real handler and it lands in the boot log).
    with _capture() as cap2:
        for i in range(20, 40):
            mod.tag(f"r{i}", None)
    check("fires AT MOST once (never a per-request log storm)",
          cap2.text == "", f"{len(cap2.text)} bytes from 20 further misses")

    # A boot that IS working must never shout, however many later misses.
    r = FakeRouter()
    r.scored["good"] = 1.0
    mod = load_sidecar(g, router=r)
    _on(mod)
    mod.tag("good", None)
    with _capture() as cap:
        for i in range(40):
            mod.tag(f"m{i}", None)
    check("a working bridge never shouts (hits suppress the alarm)",
          "ERROR" not in cap.text)


# ═══════════════════════════════════════════════ 3. the /v1/h119/score endpoint
class FakeJSONResponse:
    def __init__(self, content=None, status_code=200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}


class FakeReq:
    """The slice of ChatCompletionRequest the endpoint touches."""

    model_fields = {"model": None, "messages": None, "stream": None,
                    "n": None, "max_tokens": None, "logprobs": None,
                    "top_logprobs": None, "echo": None,
                    "kv_transfer_params": None}

    def __init__(self, **kw):
        self.__dict__.update({"model": "m", "messages": [], "stream": True,
                              "n": 3, "max_tokens": 4096, "logprobs": True,
                              "top_logprobs": 5, "echo": True,
                              "kv_transfer_params": {"x": 1}})
        self.__dict__.update(kw)

    def model_copy(self, update=None):
        new = FakeReq(**self.__dict__)
        new.__dict__.update(update or {})
        return new


class FakeResp:
    def __init__(self, kv=None, prompt_tokens=123):
        self.id = "cmpl-1"
        self.kv_transfer_params = kv
        self.usage = types.SimpleNamespace(prompt_tokens=prompt_tokens)


class FakeErr:
    def __init__(self):
        self.error = types.SimpleNamespace(code=400)

    def model_dump(self):
        return {"error": "bad"}


def endpoint_ns(g: dict, mod, handler) -> dict:
    """Exec the appended J block with the module globals it expects."""
    class _Router:
        def post(self, *a, **kw):
            def deco(fn):
                return fn
            return deco

    ns = {
        "router": _Router(),
        "Depends": lambda x: x,
        "validate_json_request": object(),
        "Request": object,
        "JSONResponse": FakeJSONResponse,
        "ChatCompletionRequest": FakeReq,
        "ChatCompletionResponse": FakeResp,
        "ErrorResponse": FakeErr,
        "chat": lambda raw: handler,
        "__name__": "api_router_probe",
    }
    exec(compile(g["J_APPEND"], "api_router.py(J)", "exec"), ns)  # noqa: S102
    return ns


def test_endpoint(g: dict) -> None:
    print("\n[6] POST /v1/h119/score")
    import os
    r = FakeRouter(mode="shadow", tdeep=0.5)
    r.scored["engine-req-9"] = 0.77
    mod = load_sidecar(g, router=r)
    _on(mod)

    seen = {}

    class Handler:
        async def create_chat_completion(self, req, raw):
            seen["req"] = req
            return FakeResp(kv=mod.tag("engine-req-9", None))

    ns = endpoint_ns(g, mod, Handler())
    res = asyncio.run(ns["h119_score"](FakeReq(), object()))
    body = res.content
    check("200 on the happy path", res.status_code == 200)
    check("available", body["available"] is True, str(body))
    check("route returned", body["route"] == "deep")
    check("score returned", body["score"] == 0.77)
    check("source returned", body["source"] == "score")
    check("prompt_tokens returned", body["prompt_tokens"] == 123)
    check("headers attached to the endpoint too",
          res.headers.get("X-H119-Route") == "deep")

    probe = seen["req"]
    check("probe forces max_tokens=1 (a prefill and one token IS the score)",
          probe.max_tokens == 1)
    check("probe forces stream off", probe.stream is False)
    check("probe forces n=1", probe.n == 1)
    check("probe clears the request-side kv_transfer_params",
          probe.kv_transfer_params is None)
    check("probe overrides are FILTERED by model_fields (a renamed field "
          "degrades, never raises)",
          not hasattr(probe, "max_completion_tokens"))
    check("caller's messages/model are NOT touched — the route is a function "
          "of the rendered prompt", probe.model == "m")

    # engine published nothing -> available:false with a diagnosis, still 200
    class NoRoute:
        async def create_chat_completion(self, req, raw):
            return FakeResp(kv=None)

    ns = endpoint_ns(g, mod, NoRoute())
    res = asyncio.run(ns["h119_score"](FakeReq(), object()))
    check("no route: still 200 (never break the caller's flow)",
          res.status_code == 200)
    check("no route: available false", res.content["available"] is False)
    check("no route: route is null, not guessed",
          res.content["route"] is None)
    check("no route: carries a reason", "reason" in res.content)
    check("no route: carries the bridge snapshot for triage",
          "bridge" in res.content)

    # upstream error passthrough
    class Err:
        async def create_chat_completion(self, req, raw):
            return FakeErr()

    ns = endpoint_ns(g, mod, Err())
    res = asyncio.run(ns["h119_score"](FakeReq(), object()))
    check("upstream ErrorResponse is passed through with its code",
          res.status_code == 400 and res.content == {"error": "bad"})

    # flag off -> 404, and no generation is attempted at all
    called = {"n": 0}

    class Counting:
        async def create_chat_completion(self, req, raw):
            called["n"] += 1
            return FakeResp(kv=None)

    os.environ["GENESIS_ENABLE_H119_ROUTE_API"] = "0"
    mod.reset_cache()
    ns = endpoint_ns(g, mod, Counting())
    res = asyncio.run(ns["h119_score"](FakeReq(), object()))
    check("flag off: 404", res.status_code == 404)
    check("flag off: no prefill is spent", called["n"] == 0)
    os.environ["GENESIS_ENABLE_H119_ROUTE_API"] = "1"
    mod.reset_cache()

    # handler missing (model does not support chat)
    ns = endpoint_ns(g, mod, None)
    res = asyncio.run(ns["h119_score"](FakeReq(), object()))
    check("no chat handler: 501, no crash", res.status_code == 501)


def main() -> int:
    import os
    import tempfile
    print("H119 route API contract\n")
    old = dict(os.environ)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="h119-api-test-"))
    try:
        g = patcher_ns(tmp / "unused")
        test_patcher(tmp)
        test_soft_skip_is_loud(tmp, True)
        test_soft_skip_is_loud(tmp, False)
        test_sidecar(g)
        test_headers_and_body(g)
        test_cross_process_alarm(g)
        test_endpoint(g)
    finally:
        os.environ.clear()
        os.environ.update(old)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if _fails:
        print(f"FAILED: {len(_fails)} — {', '.join(_fails)}")
        return 1
    print("ALL PASS")
    print("VERDICT: the route reaches the HTTP layer through "
          "EngineCoreOutput.kv_transfer_params, works in SHADOW mode, never "
          "invents a route it does not have, cannot raise into the engine or "
          "into serving, and shouts once when it can only ever miss.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
