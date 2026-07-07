#!/usr/bin/env python3
"""PN81 — /rerank + /v1/rerank on the generate engine (:8020), inside vLLM.

USER decision 2026-07-07: the reranker lives IN vLLM, not in anubis or an
external shim. Technique = Qwen3-Reranker yes/no scoring, implemented by
adapting onto upstream's own /generative_scoring feature (dev799+:
vllm/entrypoints/generate/generative_scoring/) which computes label-token
probabilities on the generate runner.

This patch:
  1. Vendors vllm/entrypoints/genesis_rerank.py — a router exposing
     POST /rerank and /v1/rerank in BOTH wire dialects:
       TEI/Infinity  (Hindsight RerankClient):  {query, texts[]}    -> [{index, score}]
       Cohere/vLLM:  {query, documents[], top_n} -> {results:[{index, relevance_score}]}
     Prompting follows Qwen3-Reranker ChatML (system yes/no judge, thinking
     hard-off via an explicit empty <think> block — bench-proven rule).
     Items are scored in bounded sequential batches (GENESIS_PN81_BATCH,
     default 16) so a 300-doc rerank cannot head-of-line-block chat traffic.
  2. Text-patches vllm/entrypoints/generate/api_router.py to attach the
     router (fail-soft: attach errors log and never break boot).

REQUIRES PN82 (rejection-sampler logprobs guard) — without it any
generative_scoring call sharing a batch with MTP spec-decode requests kills
the EngineCore. Kill-switch: GENESIS_PN81_RERANK=0 (routes 404).
"""
import pathlib
import sys

LOG = "[pn81-rerank-endpoint]"
MODULE_TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/genesis_rerank.py"
)
ROUTER_TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/generate/api_router.py"
)
MARKER = "# PN81:"

MODULE_SRC = '''# SPDX-License-Identifier: Apache-2.0
# PN81: Genesis rerank endpoint — yes/no-logit scoring on the generate engine.
# Vendored by club-3090 /fixes/patch_pn81_rerank_endpoint.py (NOT upstream).
import os

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
# Qwen3 ChatML, thinking hard-off (explicit empty think block).
PREFIX_TMPL = (
    "<|im_start|>system\\n{system}<|im_end|>\\n"
    "<|im_start|>user\\n<Instruct>: {instruction}\\n"
    "<Query>: {query}\\n<Document>: "
)
SUFFIX = "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"

_label_ids_cache: dict[int, list[int]] = {}


def _resolve_label_ids(tokenizer) -> list[int]:
    key = id(tokenizer)
    ids = _label_ids_cache.get(key)
    if ids is None:
        yes = tokenizer.encode("yes", add_special_tokens=False)
        no = tokenizer.encode("no", add_special_tokens=False)
        if len(yes) != 1 or len(no) != 1:
            raise ValueError(
                f"PN81 requires single-token yes/no labels; got yes={yes} no={no}"
            )
        ids = [yes[0], no[0]]
        _label_ids_cache[key] = ids
    return ids


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        content={"error": {"message": message, "type": "PN81RerankError"}},
        status_code=status,
    )


@router.post("/rerank")
@router.post("/v1/rerank")
@router.post("/v2/rerank")
async def genesis_rerank(raw_request: Request):
    handler = getattr(raw_request.app.state, "serving_generative_scoring", None)
    if handler is None:
        return _error(501, "generative scoring not available on this server")

    try:
        body = await raw_request.json()
    except Exception:
        return _error(400, "invalid JSON body")

    query = body.get("query")
    tei_dialect = "texts" in body
    docs = body.get("documents", body.get("texts"))
    if isinstance(docs, str):
        docs = [docs]
    if not query or not isinstance(query, str) or not docs:
        return _error(400, "both 'query' (str) and 'documents'/'texts' (list) required")
    if not all(isinstance(d, str) for d in docs):
        return _error(400, "documents/texts must be strings for PN81 rerank")
    top_n = body.get("top_n") or 0
    instruction = body.get("instruction") or DEFAULT_INSTRUCTION

    tokenizer = handler.renderer.tokenizer
    if tokenizer is None:
        return _error(500, "tokenizer unavailable")
    try:
        label_ids = _resolve_label_ids(tokenizer)
    except ValueError as e:
        return _error(500, str(e))

    prefix = PREFIX_TMPL.format(system=SYSTEM_PROMPT, instruction=instruction, query=query)
    items = [d + SUFFIX for d in docs]

    # Score in bounded batches so a big rerank cannot monopolize the scheduler.
    from vllm.entrypoints.generate.generative_scoring.serving import (
        GenerativeScoringRequest,
    )

    # Batch size bounds how many docs sit in the engine's FCFS waiting queue at
    # once (chat TTFT protection). Request-overridable: latency-sensitive quiet
    # lanes pass {"batch": 999} for one full wave (5 slots stay saturated,
    # no straggler dead-time between waves).
    try:
        batch_size = int(body.get("batch") or os.environ.get("GENESIS_PN81_BATCH", "32"))
    except (TypeError, ValueError):
        batch_size = 32
    batch_size = max(1, batch_size)
    model_name = body.get("model") or handler.models.model_name()
    scores: list[float] = [0.0] * len(docs)
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        gs_request = GenerativeScoringRequest(
            model=model_name,
            query=prefix,
            items=chunk,
            label_token_ids=label_ids,
            apply_softmax=True,
            item_first=False,
            add_special_tokens=False,
        )
        result = await handler.create_generative_scoring(gs_request, raw_request)
        if not hasattr(result, "data"):  # ErrorResponse
            code = getattr(getattr(result, "error", None), "code", 500) or 500
            msg = getattr(getattr(result, "error", None), "message", str(result))
            return _error(code, f"generative scoring failed: {msg}")
        for item in result.data:
            scores[start + item.index] = item.score

    order = sorted(range(len(docs)), key=lambda i: -scores[i])
    if top_n and top_n > 0:
        order = order[:top_n]

    if tei_dialect:
        # TEI/Infinity shape (Hindsight RerankClient): bare sorted array.
        return JSONResponse(content=[{"index": i, "score": scores[i]} for i in order])
    results = [{"index": i, "relevance_score": scores[i]} for i in order]
    if body.get("return_documents"):
        for r in results:
            r["document"] = {"text": docs[r["index"]]}
    return JSONResponse(
        content={
            "id": f"genesis-rerank-{id(raw_request):x}",
            "model": model_name,
            "results": results,
            "usage": {"total_tokens": 0},
        }
    )


def attach_router(app: FastAPI) -> None:
    if os.environ.get("GENESIS_PN81_RERANK", "1") != "1":
        logger.info("[PN81] rerank endpoint disabled via GENESIS_PN81_RERANK")
        return
    app.include_router(router)
    logger.info("[PN81] /rerank + /v1/rerank + /v2/rerank attached (generate engine)")
'''

ROUTER_OLD = (
    "    from .generative_scoring.api_router import register_generative_scoring_api_router\n"
    "\n"
    "    register_generative_scoring_api_router(app)\n"
)
ROUTER_NEW = (
    "    from .generative_scoring.api_router import register_generative_scoring_api_router\n"
    "\n"
    "    register_generative_scoring_api_router(app)\n"
    "\n"
    "    # PN81: Genesis rerank endpoint (club-3090). Fail-soft — never break boot.\n"
    "    try:\n"
    "        from vllm.entrypoints.genesis_rerank import attach_router as _pn81_attach\n"
    "\n"
    "        _pn81_attach(app)\n"
    "    except Exception as _pn81_exc:  # noqa: BLE001\n"
    "        import logging\n"
    "\n"
    "        logging.getLogger(__name__).warning(\"[PN81] attach failed: %s\", _pn81_exc)\n"
)


def main() -> int:
    if not ROUTER_TARGET.exists():
        print(f"{LOG} FATAL: {ROUTER_TARGET} not present", file=sys.stderr)
        return 1
    MODULE_TARGET.write_text(MODULE_SRC)
    print(f"{LOG} vendored module written: {MODULE_TARGET}")
    text = ROUTER_TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} router already patched (idempotent)")
        return 0
    if ROUTER_OLD not in text:
        print(f"{LOG} FATAL: anchor-not-found in generate/api_router.py — "
              f"re-derive attach point (rerank routes will be absent)", file=sys.stderr)
        return 1
    if text.count(ROUTER_OLD) != 1:
        print(f"{LOG} FATAL: ambiguous anchor", file=sys.stderr)
        return 1
    ROUTER_TARGET.write_text(text.replace(ROUTER_OLD, ROUTER_NEW, 1))
    print(f"{LOG} applied: rerank router attached after generative_scoring")
    return 0


sys.exit(main())
