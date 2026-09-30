import threading

import pytest

from steam_inference.contracts import LlmRanking
from steam_inference.features import GameInfo
from steam_inference.rerank import (
    EXPLAIN_TOOL,
    MAX_EXPLANATION_CHARS,
    RANK_TOOL,
    SYSTEM_PROMPT,
    BedrockRerankLlm,
    RerankError,
    RerankOptions,
    RerankRequest,
    blend,
    explain_prompt,
    explanation_problem,
    names_candidate,
    parse_tool_input,
    permutation_error,
    rank_prompt,
    rerank_all,
    rerank_user,
    shuffled_positions,
    template_explanation,
)

HADES = GameInfo("Hades", "Hades | Action, Indie", ("Action", "Indie"), ("Supergiant Games",))
CELESTE = GameInfo("Celeste", "Celeste | Indie", ("Indie",), ("Maddy Makes Games",))
CANDIDATES = [
    GameInfo("Dead Cells", "Dead Cells | Action", ("Action",), ("Motion Twin",)),
    GameInfo("Bastion", "Bastion | Action, RPG", ("Action", "RPG"), ("Supergiant Games",)),
    GameInfo("Stardew Valley", "Stardew Valley | Simulation", ("Simulation",), ("ConcernedApe",)),
    GameInfo("Portal 2", "Portal 2 | Puzzle", ("Puzzle",), ("Valve",)),
]
REQUEST = RerankRequest(liked=[HADES, CELESTE], candidates=CANDIDATES, seed=76561198000000001)
OPTIONS = RerankOptions(explain_top_n=2, shuffle=False)
GOOD = "Te va a gustar si disfrutaste de Hades."


class ScriptedLlm:
    """Fake LLM answering from queues; records every prompt it gets."""

    def __init__(self, ranks=(), explains=()):
        self.ranks = list(ranks)
        self.explains = list(explains)
        self.rank_calls: list[tuple[list[str], list[str], str | None]] = []
        self.explain_calls: list[tuple[list[str], list[str], str | None]] = []
        self.lock = threading.Lock()

    def rank(self, liked, candidates, error=None):
        with self.lock:
            self.rank_calls.append((liked, candidates, error))
            answer = self.ranks.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def explain(self, liked, candidates, note=None):
        with self.lock:
            self.explain_calls.append((liked, candidates, note))
            answer = self.explains.pop(0) if self.explains else None
        if isinstance(answer, Exception):
            raise answer
        if answer is None:  # default: a grounded text per game
            return {i: GOOD for i in range(1, len(candidates) + 1)}
        return answer


# ---- stage 1 --------------------------------------------------------------------------------


def test_permutation_error_names_every_problem():
    assert permutation_error([3, 1, 2], 3) is None
    assert permutation_error([1, 1, 5], 3) == "missing 2, 3; duplicated 1; unknown 5"
    assert permutation_error([], 2) == "missing 1, 2"


def test_llm_order_maps_back_through_the_shuffle():
    shown = shuffled_positions(4, REQUEST.seed, shuffle=True)
    assert sorted(shown) == [0, 1, 2, 3]
    assert shown == shuffled_positions(4, REQUEST.seed, shuffle=True)  # reproducible
    llm = ScriptedLlm(ranks=[[1, 2, 3, 4]])  # the LLM keeps the order it was shown
    result = rerank_user(llm, REQUEST, RerankOptions(explain_top_n=0, shuffle=True))
    assert result.order == shown
    assert llm.rank_calls[0][1] == [CANDIDATES[p].text for p in shown]


def test_invalid_ranking_is_retried_with_the_error():
    llm = ScriptedLlm(ranks=[[2, 2, 1], [4, 3, 2, 1]])
    result = rerank_user(llm, REQUEST, OPTIONS)
    assert result.order == [3, 2, 1, 0]
    assert result.rank_retried and not result.rank_fallback
    assert llm.rank_calls[1][2] == "missing 3, 4; duplicated 2"


def test_second_invalid_ranking_keeps_the_retrieval_order():
    llm = ScriptedLlm(ranks=[[1], [9, 1, 2, 3]])
    result = rerank_user(llm, REQUEST, OPTIONS)
    assert result.order == [0, 1, 2, 3]
    assert result.rank_fallback and result.explanations == {0: GOOD, 1: GOOD}


def test_blend_weights_the_llm_and_retrieval_ranks():
    llm_order = [3, 2, 1, 0]
    assert blend(llm_order, 1.0) == [3, 2, 1, 0]
    assert blend(llm_order, 0.0) == [0, 1, 2, 3]
    # w = 0.5: every score ties at 1.5, retrieval rank breaks the tie
    assert blend(llm_order, 0.5) == [0, 1, 2, 3]
    assert blend([1, 0, 3, 2], 0.25) == [0, 1, 2, 3]
    assert blend([2, 0, 1, 3], 0.75) == [2, 0, 1, 3]


def test_explain_only_skips_the_ranking():
    llm = ScriptedLlm()
    result = rerank_user(llm, REQUEST, RerankOptions(explain_top_n=2, stages="explain"))
    assert llm.rank_calls == [] and result.order == [0, 1, 2, 3]
    assert not result.rank_fallback and set(result.explanations) == {0, 1}


# ---- stage 2 --------------------------------------------------------------------------------


def test_only_the_final_top_n_is_explained():
    llm = ScriptedLlm(ranks=[[3, 1, 4, 2]])
    result = rerank_user(llm, REQUEST, OPTIONS)
    assert result.order == [2, 0, 3, 1]
    assert llm.explain_calls[0][1] == [CANDIDATES[2].text, CANDIDATES[0].text]
    assert set(result.explanations) == {2, 0}


def test_failed_explanations_are_reasked_then_templated():
    llm = ScriptedLlm(
        ranks=[[1, 2, 3, 4]],
        explains=[
            {1: GOOD, 2: "Bastion es genial."},  # names the candidate, and nothing grounded
            {1: "Nada que ver."},  # still ungrounded
        ],
    )
    result = rerank_user(llm, REQUEST, OPTIONS)
    assert result.explanations[0] == GOOD
    assert result.explanations[1] == "Porque te gustó Hades, del mismo estudio (Supergiant Games)."
    assert (result.explain_retries, result.templates) == (1, 1)
    retried = llm.explain_calls[1]
    assert retried[1] == [CANDIDATES[1].text]  # only the failed one, renumbered
    assert "names the recommended game itself" in retried[2]


def test_missing_explanations_are_reasked():
    llm = ScriptedLlm(ranks=[[1, 2, 3, 4]], explains=[{1: GOOD}, {1: GOOD}])
    result = rerank_user(llm, REQUEST, OPTIONS)
    assert result.explanations == {0: GOOD, 1: GOOD}
    assert result.explain_retries == 1 and result.templates == 0
    assert "missing" in llm.explain_calls[1][2]


def test_failed_explain_call_uses_templates():
    llm = ScriptedLlm(ranks=[[2, 1, 3, 4]], explains=[RuntimeError("throttled")])
    result = rerank_user(llm, REQUEST, OPTIONS)
    assert result.order == [1, 0, 2, 3] and result.templates == 2
    assert all(text.startswith("Porque te gustó") for text in result.explanations.values())


def test_everything_failing_raises():
    llm = ScriptedLlm(ranks=[RuntimeError("down")], explains=[RuntimeError("down")])
    with pytest.raises(RuntimeError, match="down"):
        rerank_user(llm, REQUEST, OPTIONS)


def test_explanation_checks():
    assert explanation_problem(GOOD, CANDIDATES[0], [HADES]) is None
    assert explanation_problem("Como FROM Motion Twin, acción pura.", CANDIDATES[0], []) is None
    assert explanation_problem("Mucha acción, como te gusta.", CANDIDATES[0], []) is None  # ES
    assert "itself" in explanation_problem("Te gustó Dead Cells.", CANDIDATES[0], [HADES])
    assert "no liked game" in explanation_problem("Muy bueno.", CANDIDATES[0], [HADES])
    long = "Hades " + "x" * MAX_EXPLANATION_CHARS
    assert "longer" in explanation_problem(long, CANDIDATES[0], [HADES])


def test_candidate_name_is_matched_as_whole_words():
    rust = GameInfo("Rust", "Rust")
    assert not names_candidate("Es frustrante como Hades.", rust, [HADES])
    assert names_candidate("RUST es como Hades.", rust, [HADES])
    souls = GameInfo("Dark Souls", "Dark Souls")
    souls3 = GameInfo("Dark Souls III", "Dark Souls III")
    assert not names_candidate("Si te gustó Dark Souls III, este es el origen.", souls, [souls3])
    assert names_candidate("Dark Souls es el origen de Dark Souls III.", souls, [souls3])


def test_template_prefers_a_shared_developer_then_genre():
    assert template_explanation(CANDIDATES[1], [CELESTE, HADES]) == (
        "Porque te gustó Hades, del mismo estudio (Supergiant Games)."
    )
    assert template_explanation(CANDIDATES[0], [HADES]) == (
        "Porque te gustó Hades, que también es de acción."
    )
    assert template_explanation(CANDIDATES[2], [HADES]) == "Porque te gustó Hades."
    assert template_explanation(CANDIDATES[2], []) == (
        "Encaja con tu gusto por los juegos de simulación."
    )


# ---- prompts / Bedrock --------------------------------------------------------------------------


def test_prompts():
    ranking = rank_prompt(["Hades"], ["Dead Cells", "Bastion"], error="missing 2")
    assert "- Hades" in ranking and "2. Bastion" in ranking
    assert "from 1 to 2 exactly once" in ranking and "(missing 2)" in ranking
    explaining = explain_prompt(["Hades"], ["Dead Cells"], note="too long")
    assert "exactly 1 explanations" in explaining and "Spanish" in explaining
    assert "Do not name the recommended game" in explaining and "too long" in explaining
    assert "Spanish" in SYSTEM_PROMPT


def test_parse_tool_input_and_text_fallback():
    tool = {"output": {"message": {"content": [
        {"text": "thinking"},
        {"toolUse": {"name": RANK_TOOL, "input": {"ranking": [2, 1]}}},
    ]}}}  # fmt: skip
    assert parse_tool_input(tool, RANK_TOOL, LlmRanking).ranking == [2, 1]
    text = {"output": {"message": {"content": [{"text": 'ok {"ranking": [3]}'}]}}}
    assert parse_tool_input(text, RANK_TOOL, LlmRanking).ranking == [3]
    for bad in (
        {"output": {"message": {"content": [{"text": "no json"}]}}, "stopReason": "end_turn"},
        {"output": {"message": {"content": [{"text": '{"ranking": "nope"}'}]}}},
    ):
        with pytest.raises(RerankError):
            parse_tool_input(bad, RANK_TOOL, LlmRanking)


class StubBedrock:
    """Answers the forced tool: ranking 1..n, or one explanation per game."""

    def __init__(self, malformed: int = 0):
        self.calls = []
        self.malformed = malformed

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        usage = {"inputTokens": 100, "outputTokens": 20}
        if len(self.calls) <= self.malformed:
            return {
                "output": {"message": {"content": [{"text": "{broken"}]}},
                "stopReason": "malformed_tool_use",
                "usage": usage,
            }
        spec = kwargs["toolConfig"]["tools"][0]["toolSpec"]
        n = spec["inputSchema"]["json"]["properties"][
            "ranking" if spec["name"] == RANK_TOOL else "explanations"
        ]["maxItems"]
        answer = (
            {"ranking": list(range(1, n + 1))}
            if spec["name"] == RANK_TOOL
            else {"explanations": [{"candidate": i, "text": f"t{i}"} for i in range(1, n + 1)]}
        )
        return {
            "output": {
                "message": {"content": [{"toolUse": {"name": spec["name"], "input": answer}}]}
            },
            "usage": usage,
        }


def _llm(client, retries=2):
    return BedrockRerankLlm(
        "model-x",
        max_tokens=500,
        temperature=0.3,
        region="us-east-1",
        invalid_answer_retries=retries,
        client=client,
    )


def test_bedrock_forces_each_tool_and_counts_tokens_per_stage():
    client = StubBedrock()
    llm = _llm(client)
    assert llm.rank(["Hades"], ["a", "b", "c"]) == [1, 2, 3]
    assert llm.explain(["Hades"], ["a", "b"]) == {1: "t1", 2: "t2"}
    rank_call, explain_call = client.calls
    assert rank_call["toolConfig"]["toolChoice"] == {"tool": {"name": RANK_TOOL}}
    assert rank_call["inferenceConfig"] == {"maxTokens": 500, "temperature": 0.0}
    items = rank_call["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]
    assert items["properties"]["ranking"]["minItems"] == 3
    assert explain_call["toolConfig"]["toolChoice"] == {"tool": {"name": EXPLAIN_TOOL}}
    assert explain_call["inferenceConfig"]["temperature"] == 0.3
    assert {s: (u.input_tokens, u.output_tokens) for s, u in llm.usage.items()} == {
        "rank": (100, 20),
        "explain": (100, 20),
    }


def test_unusable_answers_are_retried_and_bounded():
    client = StubBedrock(malformed=2)
    llm = _llm(client)
    assert llm.rank([], ["a", "b"]) == [1, 2]
    assert len(client.calls) == 3 and llm.usage["rank"].input_tokens == 300  # all paid for
    with pytest.raises(RerankError, match="malformed_tool_use"):
        _llm(StubBedrock(malformed=5), retries=1).rank([], ["a"])


def test_connection_pool_matches_concurrency(aws):
    llm = BedrockRerankLlm(
        "model-x", max_tokens=500, temperature=0.1, region="us-east-1", concurrency=32
    )
    assert llm.client.meta.config.max_pool_connections == 32


# ---- many users -----------------------------------------------------------------------------


def test_rerank_all_isolates_failures_and_counts():
    class PerUser(ScriptedLlm):
        def rank(self, liked, candidates, error=None):
            if liked[0] == "fail":
                raise RuntimeError("throttled")
            return [2, 1]

        def explain(self, liked, candidates, note=None):
            if liked[0] == "fail":
                raise RuntimeError("throttled")
            return {i: "" for i in range(1, len(candidates) + 1)}  # all missing -> templates

    def request(i: int) -> RerankRequest:
        liked = GameInfo("fail" if i == 1 else "Hades", "fail" if i == 1 else "Hades")
        return RerankRequest(liked=[liked], candidates=CANDIDATES[:2])

    results, stats = rerank_all(
        PerUser(), {i: request(i) for i in range(4)}, OPTIONS, concurrency=3
    )
    assert sorted(results) == [0, 2, 3] and stats.failures == 1
    assert results[0].order == [1, 0]
    assert (stats.explain_retries, stats.templates) == (6, 6)
