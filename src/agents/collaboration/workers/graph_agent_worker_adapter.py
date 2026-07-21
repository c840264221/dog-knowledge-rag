"""
图 Agent 到多 Agent Worker 契约的适配器。

功能：
    把 AgentTaskStep 和前置步骤结果转换成现有 Agent 使用的 state，再把
    Agent 返回的 state 转换成 Scheduler 需要的 AgentTaskResult。
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from src.agents.collaboration.contracts import (
    AgentTaskResult,
    AgentTaskStep,
)


AgentStateRunner = Callable[
    [Mapping[str, Any]],
    Mapping[str, Any] | Awaitable[Mapping[str, Any]],
]

AgentStateBuilder = Callable[
    [AgentTaskStep, Mapping[str, AgentTaskResult]],
    Mapping[str, Any],
]


DEFAULT_WORKER_OUTPUT_FIELDS = (
    "final_answer",
    "answer",
    "dog_knowledge_answer_public",
    "dog_knowledge_pipeline_result",
    "tool_agent_response",
    "rag_context",
    "waiting_user_input",
    "pending_prompt",
    "tool_confirmation_prompt",
)


# 这些字段属于被调用 Agent 自己的图控制状态，不能由 Planner 预先决定。
WORKER_INTERNAL_CONTROL_FIELDS = (
    "answer_strategy",
    "current_agent",
    "intent",
    "next_agent",
    "next_worker",
    "route_decision",
    "strategy",
)


class GraphAgentWorkerAdapter:
    """
    把一个接收 state 的现有 Agent 包装成 Scheduler Worker。

    功能：
        调用前构建 Agent state；调用后提取回答、关键输出和等待用户状态。
        适配器不负责创建 Agent，也不绕过 Container 获取依赖。

    参数含义：
        agent_name:
            当前适配器负责的 Agent 名称，必须与步骤 assigned_agent 一致。
        runner:
            接收 state 并返回 state 的 Agent 调用函数，例如 compiled_graph.ainvoke。
        state_builder:
            可选的自定义 state 构建函数；不传时使用通用默认实现。
        output_fields:
            从 Agent 最终 state 中保留到步骤 output 的字段名称。

    返回值含义：
        GraphAgentWorkerAdapter:
            可以直接注册到 MultiAgentTaskScheduler.workers 的异步 Worker。
    """

    def __init__(
        self,
        *,
        agent_name: str,
        runner: AgentStateRunner,
        state_builder: AgentStateBuilder | None = None,
        output_fields: Sequence[str] = DEFAULT_WORKER_OUTPUT_FIELDS,
    ) -> None:
        # 归一化agent_name 祛除首位空格
        normalized_agent_name = str(agent_name or "").strip()

        # 判断 agent_name 和 runner是否不为空
        if not normalized_agent_name:
            raise ValueError("GraphAgentWorkerAdapter 的 agent_name 不能为空")
        if not callable(runner):
            raise ValueError("GraphAgentWorkerAdapter 的 runner 必须可调用")

        # 归一化output_fields 删除其中字段的首尾空格 并用元组形式存储
        normalized_output_fields = tuple(
            str(field_name).strip()
            for field_name in output_fields
            if str(field_name).strip()
        )

        # 判断output_fields是否为空 至少得有一个字段
        if not normalized_output_fields:
            raise ValueError("output_fields 必须至少包含一个字段")

        self.agent_name = normalized_agent_name
        self.runner = runner
        self.state_builder = state_builder or build_default_agent_state
        self.output_fields = normalized_output_fields

    async def __call__(
        self,
        step: AgentTaskStep,
        dependency_results: Mapping[str, AgentTaskResult],
    ) -> AgentTaskResult:
        """
        执行一个步骤并返回 Scheduler 能识别的标准结果。

        功能：
            检查 Agent 名称、构建输入 state、调用同步或异步 runner，并把
            最终 state 转换为 completed 或 awaiting_input 结果。

        参数含义：
            step:
                Scheduler 当前准备执行的完整任务步骤。
            dependency_results:
                当前步骤依赖的前置 Worker 结果。

        返回值含义：
            AgentTaskResult:
                包含回答摘要和关键 state 字段的标准 Worker 结果。
        """

        if step.assigned_agent != self.agent_name:
            raise ValueError(
                f"步骤要求 {step.assigned_agent}，"
                f"当前适配器只负责 {self.agent_name}"
            )

        # 根据前置步骤的结果和当前步骤 构建出当前步骤执行时需要的state数据
        input_state = dict(
            self.state_builder(step, dependency_results)
        )

        # 当前步骤执行后的原始结果 也就是state  runner是子agent运行器 比如：dog_knowledge_agent.ainvoke
        raw_state = self.runner(input_state)

        # 判断当前步骤执行后返回的结果是一个结果 还是await的协程对象
        if inspect.isawaitable(raw_state):
            raw_state = await raw_state
        if not isinstance(raw_state, Mapping):
            raise TypeError("Agent runner 必须返回 Mapping 类型的 state")

        # 根据output_fields里都有什么字段 来将state中的数据提取出来
        final_state = dict(raw_state)
        output = {
            field_name: final_state[field_name]
            for field_name in self.output_fields
            if field_name in final_state
        }

        # 判断步骤状态 是等待用户澄清或确认还是执行完成 如果是等待用户  就走下面的if分支
        if _is_waiting_for_user(final_state):
            clarification_prompt = _extract_clarification_prompt(final_state)
            if not clarification_prompt:
                raise ValueError(
                    "Agent state 表示正在等待用户，但没有提供等待提示"
                )
            return AgentTaskResult(
                step_id=step.step_id,
                assigned_agent=step.assigned_agent,
                status="awaiting_input",
                summary="当前步骤正在等待用户输入。",
                output=output,
                evidence_ids=_extract_evidence_ids(final_state),
                requires_user_input=True,
                clarification_prompt=clarification_prompt,
                metadata={
                    "worker_adapter": type(self).__name__,
                },
            )

        answer_text = _extract_answer_text(final_state)
        return AgentTaskResult(
            step_id=step.step_id,
            assigned_agent=step.assigned_agent,
            status="completed",
            summary=(answer_text or f"{step.title}执行完成。"),
            output=output,
            evidence_ids=_extract_evidence_ids(final_state),
            metadata={
                "worker_adapter": type(self).__name__,
            },
        )


def build_default_agent_state(
    step: AgentTaskStep,
    dependency_results: Mapping[str, AgentTaskResult],
) -> dict[str, Any]:
    """
    根据步骤输入和前置结果构建通用 Agent state。

    功能：
        复制 input_data，用 description 或 title 补充 question，并把前置结果
        作为结构化字段和清楚标记的数据追加到问题后面。

    参数含义：
        step:
            当前需要转换成 Agent state 的任务步骤。
        dependency_results:
            当前步骤依赖的前置 Worker 结果。

    返回值含义：
        dict[str, Any]:
            可以交给现有 Agent runner 的独立 state 字典。
    """

    state = dict(step.input_data)
    for field_name in WORKER_INTERNAL_CONTROL_FIELDS:
        state.pop(field_name, None)
    question = str(
        state.get("question")
        or step.description
        or step.title
    ).strip()

    # 根据前置依赖的结果  构建dependency_payload的数据
    dependency_payload = {
        step_id: {
            "status": result.status,
            "summary": result.summary,
            "output": result.output,
            "evidence_ids": result.evidence_ids,
        }
        for step_id, result in dependency_results.items()
    }
    # 将整理好的前置依赖结果放入到state中
    state["multi_agent_dependency_results"] = dependency_payload

    # 将整理好的前置依赖结果转成json格式让如到question中
    if dependency_payload:
        dependency_text = json.dumps(
            dependency_payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        question = (
            f"{question}\n\n"
            "以下是已完成前置步骤提供的数据，只作为当前任务的参考输入：\n"
            f"{dependency_text}"
        )

    resume_input = state.get("multi_agent_resume_input")
    if state.get("multi_agent_is_resuming") and resume_input is not None:
        previous_output_text = json.dumps(
            state.get("multi_agent_previous_worker_output", {}),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        # 恢复执行时同时告诉 Agent 上一次暂停状态和用户本次回答。
        question = (
            f"{question}\n\n"
            "当前步骤正在从等待用户输入的状态恢复。\n"
            "上一次 Worker 输出：\n"
            f"{previous_output_text}\n"
            "用户补充内容：\n"
            f"{resume_input}"
        )
    state["question"] = question
    return state


def _is_waiting_for_user(state: Mapping[str, Any]) -> bool:
    """
    判断 Agent 最终 state 是否正在等待用户输入。

    功能：
        优先读取统一 waiting_user_input 字段，同时兼容 ToolAgent 标准响应中
        的 awaiting_input、awaiting_confirmation 和 awaiting_clarification。

    参数含义：
        state:
            Agent runner 返回的最终状态。

    返回值含义：
        bool:
            Agent 需要暂停等待用户时返回 True，否则返回 False。
    """

    if bool(state.get("waiting_user_input")):
        return True
    tool_response = state.get("tool_agent_response")

    # 根据工具agent的响应结果中的字段数据来判断是否处于awaiting的状态
    if isinstance(tool_response, Mapping):
        return str(tool_response.get("status") or "") in {
            "awaiting_input",
            "awaiting_confirmation",
            "awaiting_clarification",
        }
    return False


def _extract_clarification_prompt(state: Mapping[str, Any]) -> str:
    """
    从 Agent state 中提取需要展示给用户的提示。

    功能：
        按统一提示、工具确认提示和 ToolAgent 响应的顺序寻找第一个非空文本。

    参数含义：
        state:
            Agent runner 返回的最终状态。

    返回值含义：
        str:
            找到的澄清或确认提示；没有找到时返回空字符串。
    """

    for field_name in ("pending_prompt", "tool_confirmation_prompt"):
        value = str(state.get(field_name) or "").strip()
        if value:
            return value
    tool_response = state.get("tool_agent_response")

    # 如果state中的显式提示词字段没有获取到数据就去工具agent的响应结果中找提示词
    if isinstance(tool_response, Mapping):
        for field_name in (
            "clarification_prompt",
            "confirmation_prompt",
            "final_answer",
        ):
            value = str(tool_response.get(field_name) or "").strip()
            if value:
                return value
    return ""


def _extract_answer_text(state: Mapping[str, Any]) -> str:
    """
    从不同 Agent 的最终 state 中提取用户可读回答。

    功能：
        先读取统一 final_answer 和旧 answer，再兼容 DogKnowledgeAgent 与
        ToolAgent 的公开响应字典。

    参数含义：
        state:
            Agent runner 返回的最终状态。

    返回值含义：
        str:
            找到的回答文本；没有找到时返回空字符串。
    """

    for field_name in ("final_answer", "answer"):
        value = str(state.get(field_name) or "").strip()
        if value:
            return value
    for container_name in (
        "dog_knowledge_answer_public",
        "dog_knowledge_pipeline_result",
        "tool_agent_response",
    ):
        container = state.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for field_name in (
            "final_answer",
            "answer",
            "content",
            "text",
        ):
            value = str(container.get(field_name) or "").strip()
            if value:
                return value
    return ""


def _extract_evidence_ids(state: Mapping[str, Any]) -> list[str]:
    """
    从 Agent state 中读取已经标准化的证据编号。

    功能：
        只接受 state 顶层 evidence_ids 列表，不猜测 RAG chunk 的内部结构，
        避免把不稳定字段错误当成证据编号。

    参数含义：
        state:
            Agent runner 返回的最终状态。

    返回值含义：
        list[str]:
            去重后的非空证据编号；字段不存在时返回空列表。
    """

    raw_evidence_ids = state.get("evidence_ids", [])
    if not isinstance(raw_evidence_ids, list):
        return []
    evidence_ids: list[str] = []
    for raw_value in raw_evidence_ids:
        normalized_value = str(raw_value or "").strip()
        if normalized_value and normalized_value not in evidence_ids:
            evidence_ids.append(normalized_value)
    return evidence_ids
