"""ToolAgent LLM 答案格式化节点测试。"""

from src.agents.tool_agent.nodes.tool_answer_llm_formatter_node import (
    build_tool_agent_tool_answer_llm_formatter_node,
)


class FakeLlmProvider:
    """可配置成功或失败的测试 LLM Provider。"""

    def __init__(
        self,
        answer: str = "自然语言工具回答",
        error: Exception | None = None,
    ) -> None:
        self.backup_llm = object()
        self.answer = answer
        self.error = error
        self.call_count = 0

    async def safe_ainvoke(
        self,
        llm,
        prompt,
        fallback_response="",
    ) -> str:
        """返回固定答案或抛出测试异常。"""

        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.answer


async def test_llm_formatter_node_should_replace_rule_answer() -> None:
    """测试成功工具结果会通过 LLM 覆盖规则答案。"""

    provider = FakeLlmProvider()
    node = build_tool_agent_tool_answer_llm_formatter_node(
        llm_provider=provider,
        runtime_context_getter=lambda: None,
    )
    update = await node(
        {
            "question": "数据库有哪些表？",
            "final_answer": "sqlite_list_tables 工具返回：{'tables': ['dogs']}",
            "tool_results": [
                {
                    "success": True,
                    "tool_name": "sqlite_list_tables",
                    "content": {
                        "tables": ["dogs"],
                    },
                }
            ],
        }
    )

    assert update["final_answer"] == "自然语言工具回答"
    assert update["tool_agent_answer_source"] == "llm_tool_result_formatter"
    assert update["tool_agent_llm_answer_used"] is True
    assert update["tool_agent_response"]["final_answer"] == "自然语言工具回答"


async def test_llm_formatter_node_should_keep_rule_answer_on_error() -> None:
    """测试 LLM 异常时节点返回空 update，保留上游规则答案。"""

    provider = FakeLlmProvider(
        error=RuntimeError("LLM unavailable"),
    )
    node = build_tool_agent_tool_answer_llm_formatter_node(
        llm_provider=provider,
        runtime_context_getter=lambda: None,
    )
    update = await node(
        {
            "final_answer": "规则答案",
            "tool_results": [
                {
                    "success": True,
                    "tool_name": "date",
                    "content": "2026-07-11",
                }
            ],
        }
    )

    assert update == {}


async def test_llm_formatter_node_should_skip_failed_results() -> None:
    """测试只有失败工具结果时不调用 LLM。"""

    provider = FakeLlmProvider()
    node = build_tool_agent_tool_answer_llm_formatter_node(
        llm_provider=provider,
        runtime_context_getter=lambda: None,
    )
    update = await node(
        {
            "tool_results": [
                {
                    "success": False,
                    "tool_name": "weather",
                    "error": "网络错误",
                }
            ],
        }
    )

    assert update == {}
    assert provider.call_count == 0
