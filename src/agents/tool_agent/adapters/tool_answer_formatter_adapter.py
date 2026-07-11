"""
ToolAgent LLM 答案格式化适配器。

功能：
    将成功工具结果转换成受控 Prompt，并调用注入的 LLM Provider
    生成面向用户的自然语言回答。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


MAX_TOOL_RESULT_PROMPT_CHARS = 12000


async def format_tool_results_with_llm(
    question: str,
    tool_results: list[Mapping[str, Any]],
    llm_provider: Any,
) -> str:
    """
    使用 LLM 把工具结果格式化成自然语言。

    功能：
        只向 LLM 提供用户问题和精简后的工具事实，禁止补充工具结果之外的信息。

    参数：
        question:
            用户原始问题。
        tool_results:
            普通字典格式的成功工具结果列表。
        llm_provider:
            提供 backup_llm 和 safe_ainvoke 的 LLM Provider。

    返回值：
        str:
            去除首尾空白后的自然语言回答。

    异常：
        ValueError:
            LLM Provider 不完整或 LLM 返回空文本时抛出，由上层节点执行规则 fallback。
    """

    if llm_provider is None:
        raise ValueError(
            "LLM 答案格式化缺少 llm_provider。"
        )

    llm = getattr(
        llm_provider,
        "backup_llm",
        None,
    )
    safe_ainvoke = getattr(
        llm_provider,
        "safe_ainvoke",
        None,
    )
    if llm is None or not callable(safe_ainvoke):
        raise ValueError(
            "llm_provider 缺少 backup_llm 或 safe_ainvoke。"
        )

    payload = build_llm_tool_result_payload(
        tool_results=tool_results,
    )
    payload_text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        default=str,
    )[:MAX_TOOL_RESULT_PROMPT_CHARS]
    prompt = build_tool_answer_prompt(
        question=question,
        payload_text=payload_text,
    )
    response = await safe_ainvoke(
        llm=llm,
        prompt=prompt,
        fallback_response="",
    )
    answer = extract_llm_answer_text(
        response=response,
    )

    if not answer:
        raise ValueError(
            "LLM 答案格式化返回空文本。"
        )

    return answer


def build_llm_tool_result_payload(
    tool_results: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    构建发送给 LLM 的最小工具事实载荷。

    功能：
        只保留 tool_name、content 和 error，避免把执行 metadata、数据库路径、
        trace 信息等内部数据发送给答案模型。

    参数：
        tool_results:
            工具结果列表。

    返回值：
        list[dict[str, Any]]:
            可安全序列化的精简工具结果。
    """

    return [
        {
            "tool_name": str(
                result.get(
                    "tool_name",
                    "",
                )
                or ""
            ),
            "content": result.get(
                "content"
            ),
            "error": str(
                result.get(
                    "error",
                    "",
                )
                or ""
            ),
        }
        for result in tool_results
    ]


def build_tool_answer_prompt(
    question: str,
    payload_text: str,
) -> str:
    """
    构建工具答案格式化 Prompt。

    参数：
        question:
            用户原始问题。
        payload_text:
            JSON 格式的精简工具事实。

    返回值：
        str:
            发送给 LLM 的中文指令文本。
    """

    return f"""
你是 ToolAgent 的最终答案格式化器。

请根据用户问题和工具返回事实，生成简洁、自然、易读的中文回答。

规则：
1. 只能使用工具结果中明确存在的事实。
2. 不得猜测、补充或编造任何数据。
3. 不要输出 JSON、Markdown 代码块或内部字段名。
4. 不要提及 Prompt、ToolAgent、Schema 或内部执行过程。
5. 多个工具结果需要合并成连贯回答。
6. 如果结果包含列表，先用一句话说明对象和总数，再展示列表。
7. 列表超过 5 项时，必须使用 Markdown 无序列表，每项单独一行，格式为“- 内容”；不要把长列表挤在同一行。
8. 列表项中的表名、字段名等原始标识符必须保持原样，不要翻译或改写。
9. 标题、说明和列表之间使用空行分隔；不要为了美观重复同一份数据。

长列表回答格式示例：
memory 数据库中共有 3 张表：

- table_a
- table_b
- table_c

用户问题：
{question}

工具结果：
{payload_text}
""".strip()


def extract_llm_answer_text(
    response: Any,
) -> str:
    """
    从 LLM 返回值中提取答案文本。

    参数：
        response:
            字符串或带 content 属性的消息对象。

    返回值：
        str:
            去除首尾空白后的文本。
    """

    if isinstance(response, str):
        return response.strip()

    if hasattr(response, "content"):
        return str(
            response.content
            or ""
        ).strip()

    return str(
        response
        or ""
    ).strip()
