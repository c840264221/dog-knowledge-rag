"""
ToolAgent 参数澄清契约。

功能：
    定义工具调用缺少必填参数时的结构化澄清请求。

设计说明：
    Schema 用于创建和校验数据；写入 LangGraph state 时应调用 model_dump，
    只保存普通 dict，避免 checkpoint 持久化自定义对象。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolClarificationRequest(BaseModel):
    """
    ToolAgent 工具参数澄清请求。

    功能：
        记录待补全工具、缺失字段、可选值和面向用户的澄清问题。

    参数：
        status:
            澄清状态。当前 MVP 固定为 pending。
        tool_name:
            等待补全参数的工具名称。
        missing_fields:
            当前工具调用缺少的必填参数名称。
        question:
            展示给用户的自然语言问题。
        options:
            按字段保存的候选值；没有候选值时对应列表为空。
        reason:
            为什么需要澄清的内部说明。

    返回值：
        ToolClarificationRequest:
            可通过 model_dump 转换成普通 dict 的澄清契约对象。
    """

    status: Literal["pending"] = "pending"
    tool_name: str
    missing_fields: list[str] = Field(default_factory=list)
    question: str
    options: dict[str, list[Any]] = Field(default_factory=dict)
    reason: str = "工具调用缺少必填参数。"
