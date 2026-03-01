"""FastAPI server for cross-encoder reranking via vLLM.

Uses the yes/no logit trick: prompt vLLM with "(query, document) -> relevant?"
and extract P(yes) / (P(yes) + P(no)) as the relevance score.

Run with: uvicorn services.reranker.baseline.server:app --host 0.0.0.0 --port 42983
"""

from __future__ import annotations

import logging
import math

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.models import RerankResult, ServiceHealth, ServiceStatus

logger = logging.getLogger(__name__)

app = FastAPI(title="reranker", version="1.0.0")

# Server state
_vllm_url: str = "http://localhost:42984"
_model_name: str = "Qwen/Qwen3.5-27B"
_port: int = 42983

_DEFAULT_INSTRUCTION = "Given the query, determine if the document is relevant."


class RankRequest(BaseModel):
    query: str
    documents: list[str]
    instruction: str = _DEFAULT_INSTRUCTION
    top_k: int = Field(default=0, description="Return only top-K results (0 = all)")


def _build_prompt(instruction: str, query: str, document: str) -> str:
    """Build the yes/no relevance prompt using ChatML format."""
    return (
        "<|im_start|>system\n"
        "Judge whether the Document is relevant to the Query and Instruction.\n"
        'Answer only "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _extract_score(logprobs_data: dict) -> float:
    """Extract P(yes) / (P(yes) + P(no)) from vLLM logprobs response.

    The logprobs_data is a single token's top_logprobs dict mapping
    token string -> logprob float.
    """
    # Look for yes/no tokens in the top logprobs
    logprob_yes = None
    logprob_no = None

    for token, logprob in logprobs_data.items():
        token_lower = token.strip().lower()
        if token_lower == "yes" and logprob_yes is None:
            logprob_yes = logprob
        elif token_lower == "no" and logprob_no is None:
            logprob_no = logprob

    # If either token is missing, fall back to the generated token
    if logprob_yes is None and logprob_no is None:
        return 0.5
    if logprob_yes is None:
        return 0.0
    if logprob_no is None:
        return 1.0

    # Normalize: P(yes) / (P(yes) + P(no))
    p_yes = math.exp(logprob_yes)
    p_no = math.exp(logprob_no)
    return p_yes / (p_yes + p_no)


@app.post("/rank", response_model=RerankResult)
async def rank(req: RankRequest) -> RerankResult:
    """Rerank documents by relevance to query using yes/no logit scoring."""
    if not req.documents:
        return RerankResult(scores=[], model=_model_name)

    # Build prompts for all documents
    prompts = [
        _build_prompt(req.instruction, req.query, doc) for doc in req.documents
    ]

    # Call vLLM completions API with logprobs
    payload = {
        "model": _model_name,
        "prompt": prompts,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 5,
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=30.0)
    ) as client:
        resp = await client.post(f"{_vllm_url}/v1/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

    # Extract scores from logprobs
    scores: list[float] = []
    for choice in data["choices"]:
        top_logprobs = choice.get("logprobs", {}).get("top_logprobs", [{}])
        if top_logprobs:
            score = _extract_score(top_logprobs[0])
        else:
            score = 0.5
        scores.append(score)

    return RerankResult(scores=scores, model=_model_name)


@app.get("/health", response_model=ServiceHealth)
async def health() -> ServiceHealth:
    """Health check endpoint."""
    # Also check if vLLM backend is reachable
    devices: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{_vllm_url}/health")
            if resp.status_code == 200:
                devices.append("vllm:ok")
            else:
                devices.append("vllm:error")
    except Exception:
        devices.append("vllm:unreachable")

    return ServiceHealth(
        name="reranker",
        status=ServiceStatus.RUNNING,
        endpoint=f"http://0.0.0.0:{_port}",
        devices=devices,
    )


def configure(
    vllm_url: str = "http://localhost:42984",
    model_name: str = "Qwen/Qwen3.5-27B",
    port: int = 42983,
) -> None:
    """Configure server settings (call before starting uvicorn)."""
    global _vllm_url, _model_name, _port
    _vllm_url = vllm_url
    _model_name = model_name
    _port = port
