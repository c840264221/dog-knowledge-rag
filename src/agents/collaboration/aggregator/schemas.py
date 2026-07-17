"""
结果聚合器使用的数据结构。

功能：
    规定 LLM 整理多个 Worker 结果后必须返回哪些字段，避免只得到一段
    无法检查是否漏掉步骤的普通文本。
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResultAggregationDraft(BaseModel):
    """
    保存 LLM 生成但尚未写入最终任务结果的回答草稿。

    功能：
        同时保存最终回答、实际使用的成功步骤编号和任务限制。程序会用
        used_step_ids 检查 LLM 是否漏掉某个 Worker 的成功结果。

    参数含义：
        final_answer:
            根据各 Worker 结果整理出的用户可读回答。
        used_step_ids:
            本次回答实际使用的成功步骤编号。
        limitations:
            因步骤失败、跳过或数据不足而需要告诉用户的限制。

    返回值含义：
        ResultAggregationDraft:
            字段完整且步骤编号没有重复的聚合回答草稿。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    final_answer: str = Field(
        ...,
        min_length=1,
        description="根据所有可用步骤结果整理出的最终回答。",
    )
    used_step_ids: list[str] = Field(
        default_factory=list,
        description="最终回答实际使用的成功步骤编号。",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="失败步骤或数据不足带来的回答限制。",
    )

    @model_validator(mode="after")
    def validate_unique_items(self) -> Self:
        """
        检查步骤编号和限制说明是否存在重复内容。

        功能：
            避免 LLM 重复声明使用同一步骤，或者重复输出相同限制。

        参数含义：
            self:
                已完成基础字段校验的聚合草稿。

        返回值含义：
            ResultAggregationDraft:
                内容没有重复时返回当前草稿，否则抛出 ValueError。
        """

        if len(self.used_step_ids) != len(set(self.used_step_ids)):
            raise ValueError("used_step_ids 不能包含重复步骤编号")
        if len(self.limitations) != len(set(self.limitations)):
            raise ValueError("limitations 不能包含重复内容")
        return self
