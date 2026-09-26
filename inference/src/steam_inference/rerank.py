"""Stage 3: LLM reranking of the retrieved candidates (Amazon Bedrock, Converse API).

For each selected user, the prompt lists the games they liked (`games_reviewed_positive` of
their latest `user_features` row, the same input as the user tower), then the K candidates in
retrieval order. The model must
answer by calling the `submit_ranking` tool (structured output): every candidate number once,
best first, with a short explanation for the first N.

The answer is repaired rather than trusted: unknown / repeated numbers are dropped and missing
candidates are appended in retrieval order. A failed call (throttling after retries, invalid
output) leaves that user with the retrieval order and no explanations.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import boto3
from botocore.config import Config
from pydantic import ValidationError

from steam_inference.contracts import LlmRanking

log = logging.getLogger(__name__)

TOOL_NAME = "submit_ranking"
MAX_EXPLANATION_CHARS = 600
SYSTEM_PROMPT = (
    "You are a Steam game recommendation assistant. A recommender model retrieved candidate "
    "games for one user; you rerank them using the games the user liked, and you "
    "explain the best picks to the user. Use only the information given. Write every "
    'explanation in Spanish (neutral Latin American, informal "tú"). Always answer by '
    f"calling the {TOOL_NAME} tool."
)
TOOL_SPEC = {
    "toolSpec": {
        "name": TOOL_NAME,
        "description": "Submit the reranked candidates, best first.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "ranking": {
                        "type": "array",
                        "description": "Every candidate exactly once, best fit first.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate": {
                                    "type": "integer",
                                    "description": "Candidate number from the list.",
                                },
                                "explanation": {
                                    "type": "string",
                                    "description": "Why it fits this user (top entries only).",
                                },
                            },
                            "required": ["candidate"],
                        },
                    }
                },
                "required": ["ranking"],
            }
        },
    }
}


class RerankError(Exception):
    """The LLM answer is unusable (no tool call, invalid JSON, no known candidate)."""


@dataclass(frozen=True)
class RerankRequest:
    liked: list[str]  # game descriptions, most recent first
    candidates: list[str]  # retrieval order, best first


@dataclass(frozen=True)
class Reranked:
    order: list[int]  # 0-based candidate positions, best first (a permutation)
    explanations: dict[int, str]  # candidate position -> explanation (top N only)


# request -> raw LLM ranking (raises on failure)
RankFn = Callable[[RerankRequest], LlmRanking]


def build_prompt(request: RerankRequest, explain_top_n: int) -> str:
    lines = ["Games this user liked (most recent first):"]
    lines += [f"- {game}" for game in request.liked] or ["- (none)"]
    lines += ["", "Candidate games the user has NOT played yet (recommender order, best first):"]
    lines += [f"{i}. {game}" for i, game in enumerate(request.candidates, start=1)]
    n = len(request.candidates)
    lines += [
        "",
        f"Rerank all {n} candidates from best to worst fit for this user and call {TOOL_NAME} "
        f"with every candidate number (1-{n}) exactly once.",
    ]
    if explain_top_n:
        lines.append(
            f"For the first {explain_top_n} entries only, add an explanation: one or two "
            'sentences in Spanish, addressed to the user as "tú", on why the game fits their '
            "taste. Keep game names as given. Only games from the liked list may be named as games "
            "the user liked or played; never claim the user played a candidate. Leave the "
            "explanation out for the other entries."
        )
    return "\n".join(lines)


def merge_ranking(ranking: LlmRanking, num_candidates: int, explain_top_n: int) -> Reranked:
    order: list[int] = []
    explanations: dict[int, str] = {}
    seen: set[int] = set()
    for entry in ranking.ranking:
        position = entry.candidate - 1
        if not 0 <= position < num_candidates or position in seen:
            continue
        seen.add(position)
        order.append(position)
        text = " ".join((entry.explanation or "").split())
        if text:
            explanations[position] = text[:MAX_EXPLANATION_CHARS]
    if not order:
        raise RerankError("the ranking has no known candidate")
    order += [p for p in range(num_candidates) if p not in seen]
    top = set(order[:explain_top_n])
    return Reranked(order, {p: e for p, e in explanations.items() if p in top})


def parse_response(response: dict) -> LlmRanking:
    """The tool call's input, or a JSON object in the text as a fallback."""
    content = response.get("output", {}).get("message", {}).get("content", [])
    try:
        for block in content:
            if "toolUse" in block and block["toolUse"].get("name") == TOOL_NAME:
                return LlmRanking.model_validate(block["toolUse"]["input"])
        text = "".join(block.get("text", "") for block in content)
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return LlmRanking.model_validate(json.loads(text[start : end + 1]))
    except (ValidationError, json.JSONDecodeError, TypeError) as err:
        raise RerankError(f"invalid ranking: {err}") from err
    raise RerankError(f"no {TOOL_NAME} call (stop reason {response.get('stopReason')})")


class BedrockReranker:
    """Thread-safe `RankFn` over the Bedrock Converse API; counts tokens for cost logging.

    Unusable answers (e.g. stop reason `malformed_tool_use`: the model emitted a tool call
    Bedrock could not parse, ~0.3% of users with Nova 2 Lite) are retried up to
    `invalid_answer_retries` times; throttling / network errors are retried by botocore."""

    def __init__(
        self,
        model_id: str,
        *,
        explain_top_n: int,
        max_tokens: int,
        temperature: float,
        region: str,
        concurrency: int = 10,
        invalid_answer_retries: int = 2,
        client=None,
    ) -> None:
        self.model_id = model_id
        self.invalid_answer_retries = invalid_answer_retries
        self.explain_top_n = explain_top_n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = client or boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                retries={"max_attempts": 10, "mode": "adaptive"},
                read_timeout=120,
                # one pooled connection per calling thread (botocore's default is 10)
                max_pool_connections=max(concurrency, 10),
            ),
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def __call__(self, request: RerankRequest) -> LlmRanking:
        for attempt in range(self.invalid_answer_retries + 1):
            try:
                return parse_response(self._converse(request))
            except RerankError as err:
                if attempt == self.invalid_answer_retries:
                    raise
                log.info("retrying an unusable answer (%s)", err)
        raise AssertionError("unreachable")

    def _converse(self, request: RerankRequest) -> dict:
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": build_prompt(request, self.explain_top_n)}],
                }
            ],
            toolConfig={"tools": [TOOL_SPEC], "toolChoice": {"tool": {"name": TOOL_NAME}}},
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature},
        )
        usage = response.get("usage", {})
        with self._lock:
            self.input_tokens += usage.get("inputTokens", 0)
            self.output_tokens += usage.get("outputTokens", 0)
        return response


def rerank_all(
    rank_fn: RankFn,
    requests: dict[int, RerankRequest],
    *,
    explain_top_n: int,
    concurrency: int,
) -> tuple[dict[int, Reranked], int]:
    """Rerank every request (keyed by user position) concurrently -> (results, failures).
    Failed users are left out of the results."""
    results: dict[int, Reranked] = {}
    failures = 0

    def run(request: RerankRequest) -> Reranked:
        return merge_ranking(rank_fn(request), len(request.candidates), explain_top_n)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run, request): user for user, request in requests.items()}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                results[futures[future]] = future.result()
            except Exception as err:  # one user's failure never stops the run
                failures += 1
                if failures <= 5:
                    log.warning("rerank failed for user #%d: %s", futures[future], err)
            if done % 100 == 0 or done == len(futures):
                log.info("reranked %d/%d users (%d failed)", done, len(futures), failures)
    return results, failures
