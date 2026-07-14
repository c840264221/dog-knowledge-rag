import pytest

from src.evaluation import AgentEvaluationCase
from src.evaluation.scenarios import build_tool_agent_scenario_runtime


@pytest.mark.asyncio
async def test_tool_agent_scenario_runtime_should_run_real_date_graph() -> None:
    """
    测试确定性场景运行环境会执行真实 ToolAgent 日期链路。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_case = AgentEvaluationCase(
        case_id="tool_date_runtime_001",
        category="tool_behavior",
        question="今天几号？",
        input_state={
            "evaluation_parser_result": {
                "need_tool": True,
                "tool_calls": [
                    {
                        "name": "date",
                        "args": {},
                    }
                ],
            }
        },
        expected={
            "response_status": "completed",
        },
    )

    runtime = build_tool_agent_scenario_runtime(eval_case)
    result_state = await runtime.graph.ainvoke(runtime.initial_state)

    assert "evaluation_parser_result" not in runtime.initial_state
    assert runtime.parser.inputs
    assert runtime.executor.calls == [
        {
            "tool_name": "date",
            "args": {},
        }
    ]
    assert result_state["tool_agent_response"]["status"] == "completed"
    assert "2026-07-08" in result_state["final_answer"]


@pytest.mark.asyncio
async def test_tool_agent_scenario_runtime_should_capture_confirmation() -> None:
    """
    测试天气场景会记录确认提示并使用预设确认回答。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_case = AgentEvaluationCase(
        case_id="tool_weather_runtime_001",
        category="tool_behavior",
        question="今天成都天气怎么样？",
        input_state={
            "evaluation_parser_result": {
                "need_tool": True,
                "tool_calls": [
                    {
                        "name": "weather",
                        "args": {
                            "city": "成都",
                        },
                    }
                ],
            },
            "evaluation_confirmation_response": "y",
        },
        expected={
            "response_status": "completed",
        },
    )

    runtime = build_tool_agent_scenario_runtime(eval_case)
    result_state = await runtime.graph.ainvoke(runtime.initial_state)

    assert runtime.confirmation_prompts
    assert "查询指定城市天气" in runtime.confirmation_prompts[0]
    assert runtime.executor.calls[0]["tool_name"] == "weather"
    assert result_state["tool_agent_response"]["status"] == "completed"
