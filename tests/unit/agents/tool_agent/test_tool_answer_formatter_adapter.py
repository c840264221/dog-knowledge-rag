"""ToolAgent LLM 答案格式化适配器测试。"""

from src.agents.tool_agent.adapters.tool_answer_formatter_adapter import (
    build_llm_tool_result_payload,
    build_tool_answer_prompt,
    format_tool_results_with_llm,
)


class FakeLlmProvider:
    """返回固定自然语言答案的测试 LLM Provider。"""

    def __init__(self) -> None:
        self.backup_llm = object()
        self.calls: list[dict] = []

    async def safe_ainvoke(
        self,
        llm,
        prompt,
        fallback_response="",
    ) -> str:
        """记录 LLM 调用并返回固定文本。"""

        self.calls.append(
            {
                "llm": llm,
                "prompt": prompt,
                "fallback_response": fallback_response,
            }
        )
        return "memory 数据库中包含 memories 和 users 两张表。"


def test_build_payload_should_remove_internal_metadata() -> None:
    """测试发送给 LLM 的载荷不会包含内部 metadata。"""

    payload = build_llm_tool_result_payload(
        tool_results=[
            {
                "tool_name": "sqlite_list_tables",
                "content": {
                    "tables": ["memories", "users"],
                },
                "error": "",
                "metadata": {
                    "database_path": "secret.sqlite3",
                },
            }
        ]
    )

    assert payload[0]["tool_name"] == "sqlite_list_tables"
    assert payload[0]["content"]["tables"] == ["memories", "users"]
    assert "metadata" not in payload[0]


def test_build_tool_answer_prompt_should_require_readable_long_lists() -> None:
    """测试答案 Prompt 会要求长列表逐行展示并给出总数。"""

    prompt = build_tool_answer_prompt(
        question="memory 数据库有哪些表？",
        payload_text='[{"content": {"tables": ["a", "b", "c"]}}]',
    )

    assert "先用一句话说明对象和总数" in prompt
    assert "Markdown 无序列表" in prompt
    assert "每项单独一行" in prompt
    assert "memory 数据库中共有 3 张表" in prompt


async def test_format_tool_results_with_llm_should_return_natural_answer() -> None:
    """测试适配器使用注入 Provider 生成自然语言回答。"""

    provider = FakeLlmProvider()
    answer = await format_tool_results_with_llm(
        question="memory 数据库有哪些表？",
        tool_results=[
            {
                "tool_name": "sqlite_list_tables",
                "content": {
                    "tables": ["memories", "users"],
                },
                "error": "",
            }
        ],
        llm_provider=provider,
    )

    assert answer == "memory 数据库中包含 memories 和 users 两张表。"
    assert "sqlite_list_tables" in provider.calls[0]["prompt"]
    assert "secret.sqlite3" not in provider.calls[0]["prompt"]
