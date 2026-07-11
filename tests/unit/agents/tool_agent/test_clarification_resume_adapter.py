"""ToolAgent 参数澄清恢复适配器测试。"""

from src.agents.tool_agent.adapters.clarification_resume_adapter import (
    resolve_tool_clarification_input,
)


def build_pending_state(question: str) -> dict:
    """
    构建等待 database_name 参数的测试 state。

    参数：
        question:
            本轮用户输入。

    返回值：
        dict:
            包含澄清请求和待处理工具调用的测试状态。
    """

    return {
        "question": question,
        "tool_agent_clarification_request": {
            "status": "pending",
            "tool_name": "sqlite_list_tables",
            "missing_fields": ["database_name"],
            "options": {
                "database_name": ["memory", "rag"],
            },
        },
        "tool_agent_pending_tool_call": {
            "name": "sqlite_list_tables",
            "args": {},
        },
    }


def test_candidate_input_should_resume_pending_tool_call() -> None:
    """测试候选值会补全参数并恢复待处理工具调用。"""

    result = resolve_tool_clarification_input(
        state=build_pending_state(
            question="memory",
        )
    )
    update = result["state_update"]

    assert result["action"] == "resumed"
    assert update["tool_calls"] == [
        {
            "name": "sqlite_list_tables",
            "args": {
                "database_name": "memory",
            },
        }
    ]
    assert update["need_tool"] is True
    assert update["tool_agent_clarification_resume_ready"] is True
    assert update["tool_agent_clarification_request"] is None


def test_new_question_should_clear_pending_clarification() -> None:
    """测试不匹配候选值的完整问题会作为新问题处理。"""

    result = resolve_tool_clarification_input(
        state=build_pending_state(
            question="金毛的性格如何？",
        )
    )
    update = result["state_update"]

    assert result["action"] == "new_question"
    assert update["tool_agent_clarification_request"] is None
    assert update["tool_agent_pending_tool_call"] is None
    assert update["tool_agent_clarification_resume_ready"] is False


def test_cancel_input_should_cancel_pending_clarification() -> None:
    """测试取消词会清理待处理工具调用。"""

    result = resolve_tool_clarification_input(
        state=build_pending_state(
            question="取消",
        )
    )

    assert result["action"] == "cancelled"
    assert result["state_update"]["tool_agent_pending_tool_call"] is None


def test_candidate_should_partially_resume_multiple_missing_fields() -> None:
    """测试多个缺失字段可以先补数据库别名，再继续询问表名。"""

    result = resolve_tool_clarification_input(
        state={
            "question": "memory",
            "tool_agent_clarification_request": {
                "status": "pending",
                "tool_name": "sqlite_describe_table",
                "missing_fields": [
                    "database_name",
                    "table_name",
                ],
                "options": {
                    "database_name": ["memory", "rag"],
                    "table_name": [],
                },
                "question": "请补充数据库别名和表名。",
            },
            "tool_agent_pending_tool_call": {
                "name": "sqlite_describe_table",
                "args": {},
            },
        }
    )

    update = result["state_update"]
    assert result["action"] == "partial"
    assert update["tool_agent_pending_tool_call"]["args"] == {
        "database_name": "memory",
    }
    assert update["tool_agent_clarification_request"]["missing_fields"] == [
        "table_name",
    ]
    assert "请继续补充：表名" in update[
        "tool_agent_clarification_request"
    ]["question"]


def test_free_text_should_resume_last_missing_table_name() -> None:
    """测试最后一个无候选字段可以使用安全短文本完成补参。"""

    result = resolve_tool_clarification_input(
        state={
            "question": "collections",
            "tool_agent_clarification_request": {
                "status": "pending",
                "tool_name": "sqlite_describe_table",
                "missing_fields": ["table_name"],
                "options": {
                    "table_name": [],
                },
                "question": "请补充表名。",
            },
            "tool_agent_pending_tool_call": {
                "name": "sqlite_describe_table",
                "args": {
                    "database_name": "memory",
                },
            },
        }
    )

    update = result["state_update"]
    assert result["action"] == "resumed"
    assert update["tool_calls"][0]["args"] == {
        "database_name": "memory",
        "table_name": "collections",
    }
    assert update["tool_agent_clarification_resume_ready"] is True
