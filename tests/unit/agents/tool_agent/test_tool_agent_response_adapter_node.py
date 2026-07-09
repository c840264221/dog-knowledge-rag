"""
ToolAgent 响应适配节点测试。

功能：
    测试 ToolAgent response adapter node 是否能把旧工具 state 转成 tool_agent_response。

测试重点：
    1. 节点能返回 tool_agent_response。
    2. 工具结果存在时 status=completed。
    3. 工具调用存在但还没有结果时 status=pending_confirmation。
    4. 用户取消时 status=cancelled。
    5. 节点输出是普通 dict，不是 Pydantic 对象。
"""

from __future__ import annotations

from src.agents.tool_agent.nodes.response_adapter_node import (
    build_tool_agent_response_adapter_node,
)
from src.agents.tool_agent.adapters.state_adapter import (
    TOOL_AGENT_RESPONSE_STATE_KEY,
)


def test_response_adapter_node_should_return_tool_agent_response() -> None:
    """
    测试节点会返回 tool_agent_response。

    功能：
        验证 build_tool_agent_response_adapter_node 返回的节点可以被直接调用。

    参数：
        无。

    返回值：
        None。
    """

    node = build_tool_agent_response_adapter_node()

    result = node(
        {
            "question": "你好",
        }
    )

    assert TOOL_AGENT_RESPONSE_STATE_KEY in result
    assert result[TOOL_AGENT_RESPONSE_STATE_KEY]["status"] == "no_tool"


def test_response_adapter_node_should_mark_completed_when_tool_results_exist() -> None:
    """
    测试存在工具结果时状态为 completed。

    功能：
        验证旧 tool_results 会通过节点转换成 execution_records。

    参数：
        无。

    返回值：
        None。
    """

    node = build_tool_agent_response_adapter_node()

    result = node(
        {
            "tool_results": [
                {
                    "success": True,
                    "tool_name": "weather",
                    "content": "晴天",
                }
            ],
            "final_answer": "今天是晴天。",
        }
    )

    response = result[TOOL_AGENT_RESPONSE_STATE_KEY]

    assert response["status"] == "completed"
    assert response["execution_records"][0]["tool_result"]["tool_name"] == "weather"
    assert response["execution_records"][0]["tool_result"]["content"] == "晴天"
    assert response["final_answer"] == "今天是晴天。"


def test_response_adapter_node_should_mark_pending_when_tool_calls_exist() -> None:
    """
    测试存在待执行工具调用时状态为 pending_confirmation。

    功能：
        验证旧 tool_calls 会通过节点转换成 planned_calls。

    参数：
        无。

    返回值：
        None。
    """

    node = build_tool_agent_response_adapter_node()

    result = node(
        {
            "need_tool": True,
            "tool_calls": [
                {
                    "name": "weather",
                    "args": {
                        "city": "成都",
                    },
                }
            ],
            "tool_results": [],
        }
    )

    response = result[TOOL_AGENT_RESPONSE_STATE_KEY]

    assert response["status"] == "pending_confirmation"
    assert response["planned_calls"][0]["tool_call"]["name"] == "weather"
    assert response["planned_calls"][0]["tool_call"]["args"] == {
        "city": "成都",
    }
    assert response["permission"]["status"] == "pending"


def test_response_adapter_node_should_mark_cancelled_when_user_cancelled() -> None:
    """
    测试用户取消时状态为 cancelled。

    功能：
        验证旧工具确认链路的取消结果可以被新响应契约表达。

    参数：
        无。

    返回值：
        None。
    """

    node = build_tool_agent_response_adapter_node()

    result = node(
        {
            "tool_confirmed": "n",
            "tool_results": [
                "用户取消了工具调用。",
            ],
        }
    )

    response = result[TOOL_AGENT_RESPONSE_STATE_KEY]

    assert response["status"] == "cancelled"
    assert response["permission"]["status"] == "rejected"


def test_response_adapter_node_should_return_plain_dict() -> None:
    """
    测试节点输出是普通 dict。

    功能：
        确认节点不会把 Pydantic 对象直接写入 state，避免 checkpoint 序列化风险。

    参数：
        无。

    返回值：
        None。
    """

    node = build_tool_agent_response_adapter_node()

    result = node(
        {
            "tool_results": [
                {
                    "success": True,
                    "tool_name": "weather",
                    "content": "晴天",
                }
            ],
        }
    )

    response = result[TOOL_AGENT_RESPONSE_STATE_KEY]

    assert isinstance(
        result,
        dict,
    )
    assert isinstance(
        response,
        dict,
    )
    assert isinstance(
        response["execution_records"][0],
        dict,
    )

