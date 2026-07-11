"""ToolAgent 参数澄清节点测试。"""

from src.agents.tool_agent.nodes.tool_clarification_node import (
    build_tool_agent_tool_clarification_node,
)
from src.agents.tool_agent.adapters.state_adapter import (
    build_tool_agent_response_from_state,
)


def test_tool_clarification_node_should_create_user_facing_question() -> None:
    """
    测试澄清节点生成用户可读问题。

    功能：
        验证节点会把结构化 question 写入最终回答并标记等待用户输入。

    参数：
        无。

    返回值：
        None。
    """

    node = build_tool_agent_tool_clarification_node(
        runtime_context_getter=lambda: None,
    )
    result = node(
        {
            "tool_agent_clarification_request": {
                "question": "请选择数据库：memory 或 rag。",
                "missing_fields": ["database_name"],
            }
        }
    )

    assert result["final_answer"] == "请选择数据库：memory 或 rag。"
    assert result["waiting_user_input"] is True
    assert result["has_asked_user"] is True
    assert result["tool_agent_answer_source"] == "tool_clarification"

    response = build_tool_agent_response_from_state(
        {
            "tool_agent_clarification_request": {
                "status": "pending",
            },
            **result,
        }
    )
    assert response.status == "awaiting_clarification"
