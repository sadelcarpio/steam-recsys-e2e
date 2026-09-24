import threading

import pytest

from steam_inference.contracts import LlmRanking, RankedCandidate
from steam_inference.rerank import (
    TOOL_NAME,
    BedrockReranker,
    RerankError,
    RerankRequest,
    build_prompt,
    merge_ranking,
    parse_response,
    rerank_all,
)

REQUEST = RerankRequest(
    liked=["Liked A | RPG", "Liked B"],
    candidates=["Cand 1", "Cand 2", "Cand 3", "Cand 4"],
)


def ranking(*entries) -> LlmRanking:
    return LlmRanking(ranking=[RankedCandidate(candidate=c, explanation=e) for c, e in entries])


def test_merge_repairs_the_answer():
    merged = merge_ranking(
        ranking((3, "  great\n fit "), (3, "dup"), (9, "unknown"), (1, None), (0, "zero")),
        num_candidates=4,
        explain_top_n=2,
    )
    assert merged.order == [2, 0, 1, 3]  # missing candidates appended in retrieval order
    assert merged.explanations == {2: "great fit"}


def test_merge_keeps_explanations_of_the_top_n_only():
    merged = merge_ranking(
        ranking((2, "a"), (1, "b"), (4, "c"), (3, "d")), num_candidates=4, explain_top_n=2
    )
    assert merged.order == [1, 0, 3, 2]
    assert merged.explanations == {1: "a", 0: "b"}


def test_merge_without_known_candidates_fails():
    with pytest.raises(RerankError):
        merge_ranking(ranking((7, "x")), num_candidates=4, explain_top_n=2)


def test_prompt_lists_taste_and_candidates():
    prompt = build_prompt(REQUEST, explain_top_n=3)
    assert "- Liked A | RPG" in prompt and "- Liked B" in prompt
    assert "1. Cand 1" in prompt and "4. Cand 4" in prompt
    assert "(1-4)" in prompt and "first 3 entries" in prompt
    assert "explanation" not in build_prompt(REQUEST, explain_top_n=0)


def test_parse_tool_call_and_text_fallback():
    tool = {
        "output": {
            "message": {
                "content": [
                    {"text": "thinking"},
                    {"toolUse": {"name": TOOL_NAME, "input": {"ranking": [{"candidate": 2}]}}},
                ]
            }
        }
    }
    assert parse_response(tool).ranking[0].candidate == 2
    text = {"output": {"message": {"content": [{"text": 'ok {"ranking": [{"candidate": 3}]}'}]}}}
    assert parse_response(text).ranking[0].candidate == 3
    for bad in (
        {"output": {"message": {"content": [{"text": "no json"}]}}, "stopReason": "end_turn"},
        {"output": {"message": {"content": [{"text": '{"ranking": "nope"}'}]}}},
    ):
        with pytest.raises(RerankError):
            parse_response(bad)


class StubBedrock:
    def __init__(self):
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": TOOL_NAME,
                                "input": {"ranking": [{"candidate": 1, "explanation": "x"}]},
                            }
                        }
                    ]
                }
            },
            "usage": {"inputTokens": 100, "outputTokens": 20},
        }


def test_bedrock_reranker_forces_the_tool_and_counts_tokens():
    client = StubBedrock()
    reranker = BedrockReranker(
        "model-x",
        explain_top_n=2,
        max_tokens=500,
        temperature=0.1,
        region="us-east-1",
        client=client,
    )
    assert reranker(REQUEST).ranking[0].candidate == 1
    reranker(REQUEST)
    call = client.calls[0]
    assert call["modelId"] == "model-x"
    assert call["toolConfig"]["toolChoice"] == {"tool": {"name": TOOL_NAME}}
    assert call["inferenceConfig"] == {"maxTokens": 500, "temperature": 0.1}
    assert "Cand 4" in call["messages"][0]["content"][0]["text"]
    assert (reranker.input_tokens, reranker.output_tokens) == (200, 40)


def test_rerank_all_isolates_failures():
    lock = threading.Lock()
    seen = []

    def rank(request: RerankRequest) -> LlmRanking:
        with lock:
            seen.append(request.liked[0])
        if request.liked[0] == "fail":
            raise RuntimeError("throttled")
        return ranking((2, "why"))

    requests = {
        i: RerankRequest(liked=["fail" if i == 1 else "ok"], candidates=["a", "b"])
        for i in range(4)
    }
    results, failures = rerank_all(rank, requests, explain_top_n=1, concurrency=3)
    assert failures == 1 and sorted(results) == [0, 2, 3]
    assert results[0].order == [1, 0] and results[0].explanations == {1: "why"}
    assert len(seen) == 4
