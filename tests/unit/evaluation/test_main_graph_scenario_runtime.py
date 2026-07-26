import pytest

from src.evaluation import AgentEvaluationCase
from src.evaluation.scenarios.main_graph_scenario_runtime import (
    EvaluationMainGraphLLMProvider,
    EvaluationMainGraphPlanningProvider,
    build_main_graph_scenario_runtime,
)


@pytest.mark.asyncio
async def test_main_graph_llm_provider_should_classify_real_prompt_roles() -> None:
    """
    测试主图评估 LLM Provider 能按真实 Prompt 职责返回固定结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = EvaluationMainGraphLLMProvider(
        general_answer="通用回答",
        dog_answer="狗狗回答",
        tool_answer="工具回答",
    )

    memory_response = await provider.safe_ainvoke(
        llm=provider.chinese_llm,
        prompt="你是一个长期记忆提取器。",
    )
    dog_response = await provider.safe_ainvoke(
        llm=provider.main_llm,
        prompt="你是 Dog Agent Framework 的犬种知识助手。",
    )

    assert '"should_save": false' in str(memory_response.content)
    assert dog_response.content == "狗狗回答"
    assert provider.count_calls("memory_extract") == 1
    assert provider.count_calls("dog_answer") == 1


@pytest.mark.asyncio
async def test_build_main_graph_runtime_should_strip_evaluation_fields() -> None:
    """
    测试主图评估场景使用真实编译图且不会污染业务 state。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_case = AgentEvaluationCase(
        case_id="main_graph_runtime_001",
        category="main_graph_behavior",
        question="法国的首都是什么？",
        input_state={
            "user_id": "evaluation_user",
            "evaluation_general_answer": "法国的首都是巴黎。",
            "evaluation_tool_parser_result": {
                "need_tool": False,
                "tool_calls": [],
            },
        },
        expected={
            "route": "general_agent",
        },
    )

    runtime = await build_main_graph_scenario_runtime(eval_case)

    assert hasattr(runtime.graph, "ainvoke")
    assert runtime.initial_state["question"] == "法国的首都是什么？"
    assert runtime.initial_state["user_id"] == "evaluation_user"
    assert all(
        not key.startswith("evaluation_")
        for key in runtime.initial_state
    )
    assert runtime.tool_parser.result == {
        "need_tool": False,
        "tool_calls": [],
    }


@pytest.mark.asyncio
async def test_main_graph_planning_provider_should_use_runtime_ids() -> None:
    """
    测试主图计划 Provider 会回填真实 Planner 本轮编号和目标。

    参数含义：
        无。

    返回值含义：
        None。
    """

    provider = EvaluationMainGraphPlanningProvider(
        {
            "plan_id": "golden_plan",
            "objective": "黄金目标",
            "steps": [
                {
                    "step_id": "step_one",
                    "title": "执行步骤",
                    "assigned_agent": "worker_agent",
                }
            ],
        }
    )

    message = await provider.safe_ainvoke(
        llm=provider.main_llm,
        prompt="""
8. plan_id 必须原样返回为 'runtime_plan_001'。
用户目标开始：
运行时真实目标
用户目标结束。
""".strip(),
    )

    assert '"plan_id": "runtime_plan_001"' in message.content
    assert '"objective": "运行时真实目标"' in message.content
    assert len(provider.prompts) == 1
