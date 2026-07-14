from pathlib import Path

import pytest

from src.evaluation import load_agent_evaluation_cases
from src.evaluation.scenarios import build_dog_knowledge_scenario_runtime


DATASET_PATH = Path(
    "evaluation/datasets/dog_knowledge_behavior_cases.json"
)


@pytest.mark.asyncio
async def test_dog_knowledge_runtime_should_run_real_exact_lookup_graph() -> None:
    """
    测试确定性场景会执行真实 DogKnowledgeAgent 精确查询子图。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_case = load_agent_evaluation_cases(DATASET_PATH)[0]
    runtime = build_dog_knowledge_scenario_runtime(eval_case)

    result_state = await runtime.invoke()

    assert "evaluation_rag_context" not in runtime.initial_state
    assert runtime.parser.inputs
    assert runtime.retriever.queries
    assert runtime.reranker.calls
    assert runtime.llm_provider.prompts
    assert result_state["dog_knowledge_answer"]["query_type"] == (
        "exact_lookup"
    )
    assert result_state["dog_knowledge_answer"]["status"] == "success"
    assert "10 到 12 年" in result_state["final_answer"]
