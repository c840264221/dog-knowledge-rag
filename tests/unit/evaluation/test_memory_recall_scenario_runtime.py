from pathlib import Path

import pytest

from src.evaluation import load_agent_evaluation_cases
from src.evaluation.scenarios import build_memory_recall_scenario_runtime


DATASET_PATH = Path(
    "evaluation/datasets/memory_recall_behavior_cases.json"
)


@pytest.mark.asyncio
async def test_memory_runtime_should_apply_related_memory() -> None:
    """
    测试真实记忆召回链路会采用通过语义门槛的当前用户记忆。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_case = load_agent_evaluation_cases(DATASET_PATH)[0]
    runtime = build_memory_recall_scenario_runtime(eval_case)

    result_state = await runtime.invoke()

    assert "evaluation_memories" not in runtime.initial_state
    assert result_state["memory_recall_result"]["status"] == "applied"
    assert result_state["memory_recall_result"]["selected_memory_ids"] == [1]
    assert "金毛寻回犬" in result_state["memory_context"]


@pytest.mark.asyncio
async def test_memory_runtime_should_isolate_other_user_memory() -> None:
    """
    测试真实记忆召回链路不会返回其他用户的记忆。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_case = load_agent_evaluation_cases(DATASET_PATH)[3]
    runtime = build_memory_recall_scenario_runtime(eval_case)

    result_state = await runtime.invoke()

    assert result_state["memory_recall_result"]["status"] == "empty"
    assert result_state["memory_recall_result"]["selected_memory_ids"] == []
    assert "拉布拉多寻回犬" not in result_state["memory_context"]
