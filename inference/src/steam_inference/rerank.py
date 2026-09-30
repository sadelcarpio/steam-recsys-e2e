"""Stage 3: two-stage LLM reranking of the retrieved candidates (spec 7; Bedrock Converse API).

For each selected user, the prompts list the games they liked (`games_reviewed_positive` of their
latest `user_features` row, the same input as the user tower).

1. **Ranking** (`submit_ranking`, temperature 0): the K candidates, shuffled with a seed derived
   from the user id (position bias), answered as candidate numbers only. The answer must be an
   exact permutation of 1..K; one retry states what was wrong, a second failure keeps the
   retrieval order. The final order blends both ranks (`RERANK_BLEND_WEIGHT`).
2. **Explanations** (`submit_explanations`): only the final top N, exactly one text each. Every
   text is checked in code (`explanation_problem`); failed ones are re-asked once, then replaced
   by a template built from the data (`template_explanation`), so every top-N entry of a
   reranked user has an explanation.

LLM output is untrusted: the final order is always a permutation of the candidates. A user whose
LLM calls all fail keeps the retrieval order without explanations; one user's failure never
fails the run.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal, Protocol

import boto3
import numpy as np
from botocore.config import Config
from pydantic import BaseModel, ValidationError

from steam_inference.contracts import LlmExplanations, LlmRanking
from steam_inference.features import GameInfo

log = logging.getLogger(__name__)

RANK_TOOL = "submit_ranking"
EXPLAIN_TOOL = "submit_explanations"
MAX_EXPLANATION_CHARS = 600
Stages = Literal["rank+explain", "explain"]
SYSTEM_PROMPT = (
    "You are a Steam game recommendation assistant. A recommender model retrieved candidate "
    "games for one user; you help order them using the games the user liked, and you explain "
    "the best picks to the user. Use only the information given. Write every explanation in "
    'Spanish (neutral Latin American, informal "tú"). Always answer by calling the tool you '
    "are given."
)

# Steam genre names (English in the lookups) as a Spanish explanation may write them.
GENRES_ES = {
    "action": "acción",
    "adventure": "aventura",
    "strategy": "estrategia",
    "simulation": "simulación",
    "racing": "carreras",
    "sports": "deportes",
    "rpg": "rol",
    "massively multiplayer": "multijugador masivo",
    "early access": "acceso anticipado",
}


class RerankError(Exception):
    """The LLM answer is unusable (no tool call, invalid JSON)."""


@dataclass(frozen=True)
class RerankRequest:
    liked: list[GameInfo]  # most recent first
    candidates: list[GameInfo]  # retrieval order, best first
    seed: int = 0  # shuffle seed (the user id): reproducible prompts


@dataclass(frozen=True)
class Reranked:
    order: list[int]  # 0-based candidate positions, best first (a permutation)
    explanations: dict[int, str]  # candidate position -> explanation (final top N only)
    rank_retried: bool = False
    rank_fallback: bool = False  # the LLM ranking failed: retrieval order
    explain_retries: int = 0  # explanations re-asked
    templates: int = 0  # explanations replaced by the template


@dataclass(frozen=True)
class RerankOptions:
    explain_top_n: int
    stages: Stages = "rank+explain"
    blend_weight: float = 1.0  # 1 = LLM order, 0 = retrieval order
    shuffle: bool = True


class RerankLlm(Protocol):
    """The two LLM calls; both raise on an unusable answer or a failed call."""

    def rank(self, liked: list[str], candidates: list[str], error: str | None = None) -> list[int]:
        """Candidate numbers (1-based), best first."""
        ...

    def explain(
        self, liked: list[str], candidates: list[str], note: str | None = None
    ) -> dict[int, str]:
        """Candidate number (1-based) -> explanation."""
        ...


# ---- prompts ---------------------------------------------------------------------------------


def _liked_lines(liked: list[str]) -> list[str]:
    return [
        "Games this user liked (most recent first):",
        *([f"- {g}" for g in liked] or ["- (none)"]),
    ]


def rank_prompt(liked: list[str], candidates: list[str], error: str | None = None) -> str:
    n = len(candidates)
    lines = _liked_lines(liked)
    lines += ["", "Candidate games the user has NOT played yet (in no particular order):"]
    lines += [f"{i}. {game}" for i, game in enumerate(candidates, start=1)]
    lines += [
        "",
        f"Order all {n} candidates from best to worst fit for this user and call {RANK_TOOL} "
        f"with every candidate number from 1 to {n} exactly once. Numbers only, no text.",
    ]
    if error:
        lines.append(f"Your previous answer was invalid ({error}). Answer again.")
    return "\n".join(lines)


def explain_prompt(liked: list[str], candidates: list[str], note: str | None = None) -> str:
    n = len(candidates)
    lines = _liked_lines(liked)
    lines += ["", "Games recommended to this user (they have NOT played them yet):"]
    lines += [f"{i}. {game}" for i, game in enumerate(candidates, start=1)]
    lines += [
        "",
        f"Call {EXPLAIN_TOOL} with exactly {n} explanations, one per recommended game number "
        f'(1-{n}): one or two sentences in Spanish, addressed to the user as "tú", on why the '
        "game fits their taste. Name at least one game from the liked list by its exact name, "
        "or a genre or developer they share. Keep game names as given. Do not name the "
        "recommended game itself (your text is shown next to it), and never say the user "
        "liked, played or enjoyed it.",
    ]
    if note:
        lines.append(f"Some previous explanations broke these rules: {note}. Write them again.")
    return "\n".join(lines)


# ---- validation --------------------------------------------------------------------------------


def permutation_error(answer: list[int], n: int) -> str | None:
    """What keeps `answer` from being a permutation of 1..n (None when it is one)."""
    seen: set[int] = set()
    unknown, duplicated = [], []
    for number in answer:
        if not 1 <= number <= n:
            unknown.append(number)
        elif number in seen:
            duplicated.append(number)
        seen.add(number)
    missing = [i for i in range(1, n + 1) if i not in seen]
    problems = [
        f"{label} {', '.join(map(str, values[:10]))}"
        for label, values in (
            ("missing", missing),
            ("duplicated", duplicated),
            ("unknown", unknown),
        )
        if values
    ]
    return "; ".join(problems) or None


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


def _contains(text: str, name: str) -> bool:
    """Whole-word match of a normalized name in normalized text."""
    return bool(name) and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text) is not None


def names_candidate(text: str, candidate: GameInfo, liked: list[GameInfo]) -> bool:
    """The explanation names the recommended game itself. It is shown next to the game, so a
    mention adds nothing, and it is how "you played <candidate>" claims start. Liked games are
    removed first: naming "Dark Souls III" is fine when the candidate is "Dark Souls"."""
    normalized = _normalize(text)
    for game in sorted(liked, key=lambda g: -len(g.name)):
        liked_name = _normalize(game.name)
        if liked_name:
            normalized = re.sub(rf"(?<!\w){re.escape(liked_name)}(?!\w)", " ", normalized)
    return _contains(normalized, _normalize(candidate.name))


def _grounding_terms(candidate: GameInfo, liked: list[GameInfo]) -> set[str]:
    terms = {g.name for g in liked}
    for game in (candidate, *liked):
        terms.update(game.developers)
        terms.update(game.tags)
        for genre in game.genres:
            terms.add(genre)
            terms.add(GENRES_ES.get(genre.casefold(), genre))
    return {term for term in map(_normalize, terms) if len(term) >= 3}


def explanation_problem(text: str, candidate: GameInfo, liked: list[GameInfo]) -> str | None:
    """Why an explanation is rejected (None when it passes)."""
    if len(text) > MAX_EXPLANATION_CHARS:
        return f"longer than {MAX_EXPLANATION_CHARS} characters"
    if names_candidate(text, candidate, liked):
        return "it names the recommended game itself"
    normalized = _normalize(text)
    if not any(_contains(normalized, term) for term in _grounding_terms(candidate, liked)):
        return "it names no liked game, genre or developer from the prompt"
    return None


def _genre_es(genre: str) -> str:
    return GENRES_ES.get(genre.casefold(), genre)


def template_explanation(candidate: GameInfo, liked: list[GameInfo]) -> str:
    """A grounded explanation built from the data (the last resort of stage 2)."""
    for game in liked:
        shared = [d for d in game.developers if d in candidate.developers]
        if shared:
            return f"Porque te gustó {game.name}, del mismo estudio ({shared[0]})."
        shared = [g for g in game.genres if g in candidate.genres]
        if shared:
            return f"Porque te gustó {game.name}, que también es de {_genre_es(shared[0])}."
    if liked:
        return f"Porque te gustó {liked[0].name}."
    if candidate.genres:
        return f"Encaja con tu gusto por los juegos de {_genre_es(candidate.genres[0])}."
    return "Recomendado a partir de tus reseñas."


# ---- ordering ---------------------------------------------------------------------------------


def shuffled_positions(n: int, seed: int, shuffle: bool) -> list[int]:
    """Retrieval position shown as number i + 1 of the ranking prompt."""
    if not shuffle:
        return list(range(n))
    return np.random.default_rng(seed).permutation(n).tolist()


def blend(llm_order: list[int], weight: float) -> list[int]:
    """Final order: ascending `weight * llm_rank + (1 - weight) * retrieval_rank`, ties by
    retrieval rank."""
    n = len(llm_order)
    llm_rank = np.empty(n)
    llm_rank[np.asarray(llm_order)] = np.arange(n)
    score = weight * llm_rank + (1 - weight) * np.arange(n)
    return np.lexsort((np.arange(n), score)).tolist()


# ---- one user ---------------------------------------------------------------------------------


def llm_ranking(
    llm: RerankLlm, request: RerankRequest, shuffle: bool
) -> tuple[list[int] | None, bool, Exception | None]:
    """Stage 1 -> (LLM order as retrieval positions or None, retried, call error)."""
    n = len(request.candidates)
    shown = shuffled_positions(n, request.seed, shuffle)
    liked = [g.text for g in request.liked]
    texts = [request.candidates[p].text for p in shown]
    error: str | None = None
    for attempt in range(2):
        try:
            answer = llm.rank(liked, texts, error)
        except Exception as err:  # the call itself failed (throttling after retries, ...)
            return None, attempt > 0, err
        error = permutation_error(answer, n)
        if error is None:
            return [shown[number - 1] for number in answer], attempt > 0, None
    return None, True, None


def rerank_user(llm: RerankLlm, request: RerankRequest, options: RerankOptions) -> Reranked:
    n = len(request.candidates)
    llm_order, rank_retried, rank_error = None, False, None
    ranked = options.stages == "rank+explain" and n > 1
    if ranked:
        llm_order, rank_retried, rank_error = llm_ranking(llm, request, options.shuffle)
    order = blend(llm_order, options.blend_weight) if llm_order else list(range(n))

    top = order[: options.explain_top_n]
    liked = [g.text for g in request.liked]
    explanations: dict[int, str] = {}
    pending, note = list(top), None
    retries = explained_by_llm = 0
    explain_error: Exception | None = None
    for attempt in range(2):
        if not pending:
            break
        try:
            answer = llm.explain(liked, [request.candidates[p].text for p in pending], note)
        except Exception as err:
            explain_error = err
            break
        failed, problems = [], set()
        for number, position in enumerate(pending, start=1):
            text = " ".join(answer.get(number, "").split())
            problem = (
                explanation_problem(text, request.candidates[position], request.liked)
                if text
                else "an explanation is missing"
            )
            if problem:
                failed.append(position)
                problems.add(problem)
            else:
                explanations[position] = text
                explained_by_llm += 1
        if attempt == 0:
            retries = len(failed)
        pending, note = failed, "; ".join(sorted(problems))

    if llm_order is None and not explained_by_llm and (rank_error or explain_error):
        # nothing came from the LLM (e.g. Bedrock unavailable): the user keeps retrieval order
        raise rank_error or explain_error  # type: ignore[misc]
    for position in pending:
        explanations[position] = template_explanation(request.candidates[position], request.liked)
    return Reranked(
        order=order,
        explanations=explanations,
        rank_retried=rank_retried,
        rank_fallback=ranked and llm_order is None,
        explain_retries=retries,
        templates=len(pending),
    )


@dataclass
class RerankStats:
    failures: int = 0  # users left with the retrieval order and no explanations
    rank_retries: int = 0
    rank_fallbacks: int = 0
    explain_retries: int = 0
    templates: int = 0

    def add(self, result: Reranked) -> None:
        self.rank_retries += result.rank_retried
        self.rank_fallbacks += result.rank_fallback
        self.explain_retries += result.explain_retries
        self.templates += result.templates


def rerank_all(
    llm: RerankLlm,
    requests: dict[int, RerankRequest],
    options: RerankOptions,
    *,
    concurrency: int,
) -> tuple[dict[int, Reranked], RerankStats]:
    """Rerank every request (keyed by user position) concurrently. Failed users are left out."""
    results: dict[int, Reranked] = {}
    stats = RerankStats()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(rerank_user, llm, request, options): user
            for user, request in requests.items()
        }
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as err:  # one user's failure never stops the run
                stats.failures += 1
                if stats.failures <= 5:
                    log.warning("rerank failed for user #%d: %s", futures[future], err)
            else:
                results[futures[future]] = result
                stats.add(result)
            if done % 100 == 0 or done == len(futures):
                log.info("reranked %d/%d users (%d failed)", done, len(futures), stats.failures)
    return results, stats


# ---- Bedrock ----------------------------------------------------------------------------------


def rank_tool(n: int) -> dict:
    return {
        "toolSpec": {
            "name": RANK_TOOL,
            "description": "Submit the candidate numbers, best fit first.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "ranking": {
                            "type": "array",
                            "description": f"Every candidate number from 1 to {n} exactly once.",
                            "items": {"type": "integer"},
                            "minItems": n,
                            "maxItems": n,
                        }
                    },
                    "required": ["ranking"],
                }
            },
        }
    }


def explain_tool(n: int) -> dict:
    return {
        "toolSpec": {
            "name": EXPLAIN_TOOL,
            "description": "Submit one explanation per recommended game.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "explanations": {
                            "type": "array",
                            "minItems": n,
                            "maxItems": n,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "candidate": {
                                        "type": "integer",
                                        "description": "Recommended game number from the list.",
                                    },
                                    "text": {
                                        "type": "string",
                                        "description": "Why it fits this user (Spanish).",
                                    },
                                },
                                "required": ["candidate", "text"],
                            },
                        }
                    },
                    "required": ["explanations"],
                }
            },
        }
    }


def parse_tool_input[M: BaseModel](response: dict, tool: str, model: type[M]) -> M:
    """The tool call's input, or a JSON object in the text as a fallback."""
    content = response.get("output", {}).get("message", {}).get("content", [])
    try:
        for block in content:
            if "toolUse" in block and block["toolUse"].get("name") == tool:
                return model.model_validate(block["toolUse"]["input"])
        text = "".join(block.get("text", "") for block in content)
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return model.model_validate(json.loads(text[start : end + 1]))
    except (ValidationError, json.JSONDecodeError, TypeError) as err:
        raise RerankError(f"invalid {tool} input: {err}") from err
    raise RerankError(f"no {tool} call (stop reason {response.get('stopReason')})")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


class BedrockRerankLlm:
    """Thread-safe `RerankLlm` over the Bedrock Converse API; counts tokens per stage.

    Unusable answers (e.g. stop reason `malformed_tool_use`: a tool call Bedrock could not
    parse) are retried up to `invalid_answer_retries` times; throttling / network errors are
    retried by botocore."""

    def __init__(
        self,
        model_id: str,
        *,
        max_tokens: int,
        temperature: float,
        region: str,
        concurrency: int = 10,
        invalid_answer_retries: int = 2,
        client=None,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.invalid_answer_retries = invalid_answer_retries
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
        self.usage: dict[str, Usage] = {}  # stage (rank / explain) -> tokens
        self._lock = threading.Lock()

    def rank(self, liked: list[str], candidates: list[str], error: str | None = None) -> list[int]:
        answer = self._call(
            "rank",
            rank_prompt(liked, candidates, error),
            rank_tool(len(candidates)),
            temperature=0.0,
            parse=lambda r: parse_tool_input(r, RANK_TOOL, LlmRanking),
        )
        return list(answer.ranking)

    def explain(
        self, liked: list[str], candidates: list[str], note: str | None = None
    ) -> dict[int, str]:
        answer = self._call(
            "explain",
            explain_prompt(liked, candidates, note),
            explain_tool(len(candidates)),
            temperature=self.temperature,
            parse=lambda r: parse_tool_input(r, EXPLAIN_TOOL, LlmExplanations),
        )
        return {e.candidate: e.text for e in answer.explanations}

    def _call(self, stage: str, prompt: str, tool: dict, *, temperature: float, parse: Callable):
        for attempt in range(self.invalid_answer_retries + 1):
            try:
                return parse(self._converse(stage, prompt, tool, temperature))
            except RerankError as err:
                if attempt == self.invalid_answer_retries:
                    raise
                log.info("retrying an unusable %s answer (%s)", stage, err)
        raise AssertionError("unreachable")

    def _converse(self, stage: str, prompt: str, tool: dict, temperature: float) -> dict:
        name = tool["toolSpec"]["name"]
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            toolConfig={"tools": [tool], "toolChoice": {"tool": {"name": name}}},
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": temperature},
        )
        tokens = response.get("usage", {})
        with self._lock:
            usage = self.usage.setdefault(stage, Usage())
            usage.input_tokens += tokens.get("inputTokens", 0)
            usage.output_tokens += tokens.get("outputTokens", 0)
        return response
