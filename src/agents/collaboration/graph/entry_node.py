"""
多 Agent 协作主图入口节点。

功能：
    根据 DogState 中的恢复动作选择新建、重新规划或恢复多 Agent 任务，
    再把标准任务结果转换成可以写入主图和 Checkpoint 的普通字典字段。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Awaitable

from src.agents.collaboration.contracts import MultiAgentTaskResult
from src.graph.states.dog_state import DogState


MultiAgentEntryNode = Callable[
    [DogState],
    Awaitable[dict[str, Any]],
]


def build_multi_agent_entry_node(
    *,
    orchestrator: Any,
) -> MultiAgentEntryNode:
    """
    构建注入 MultiAgentOrchestrator 的主图异步节点。

    功能：
        新复杂目标调用 run；Worker 等待后的回答调用 resume；Planner 等待
        后的回答携带补充上下文重新规划；澄清未完成时只返回等待提示。

    参数含义：
        orchestrator:
            提供 run 和 resume 方法的多 Agent 总编排器。

    返回值含义：
        MultiAgentEntryNode:
            可以注册到 StateGraph 的异步多 Agent 入口节点。
    """

    if orchestrator is None:
        raise ValueError("多 Agent 主图入口必须提供 orchestrator")
    if not callable(getattr(orchestrator, "run", None)):
        raise ValueError("orchestrator 缺少 run 方法")
    if not callable(getattr(orchestrator, "resume", None)):
        raise ValueError("orchestrator 缺少 resume 方法")

    async def multi_agent_entry_node(
        state: DogState,
    ) -> dict[str, Any]:
        """
        执行或恢复一次多 Agent 协作任务。

        参数含义：
            state:
                RootAgent 已完成路由和恢复输入整理的当前 DogState。

        返回值含义：
            dict[str, Any]:
                多 Agent 结果、最终回答和下一轮恢复字段组成的局部状态。
        """

        action = str(
            state.get("multi_agent_resume_action") or "none"
        )
        if action == "needs_clarification":
            prompt = str(
                state.get("multi_agent_pending_prompt")
                or "请补充多 Agent 任务所需信息。"
            )
            return {
                "current_agent": "multi_agent",
                "final_answer": prompt,
                "pending_prompt": prompt,
                "waiting_user_input": True,
            }

        if action == "resume":
            task_result = _load_pending_task_result(state)
            user_inputs = _load_resume_inputs(state)
            result = await orchestrator.resume(
                task_result,
                user_inputs=user_inputs,
            )
        elif action == "replan":
            task_result = _load_pending_task_result(state)
            resume_inputs = _load_resume_inputs(state)
            result = await orchestrator.run(
                task_result.plan.objective,
                context={
                    "user_clarification": resume_inputs.get(
                        "planner_clarification",
                        "",
                    ),
                    "previous_clarification_prompt": (
                        task_result.plan.clarification_prompt
                    ),
                    "memory_context": state.get("memory_context", ""),
                    "user_id": state.get("user_id", ""),
                    "session_id": state.get("session_id", ""),
                    "trace_id": state.get("trace_id", ""),
                },
            )
        else:
            result = await orchestrator.run(
                str(state.get("question") or "").strip(),
                context={
                    "memory_context": state.get("memory_context", ""),
                    "user_id": state.get("user_id", ""),
                    "session_id": state.get("session_id", ""),
                    "trace_id": state.get("trace_id", ""),
                },
            )

        return build_multi_agent_state_update(result)

    return multi_agent_entry_node


def _load_pending_task_result(
    state: Mapping[str, Any],
) -> MultiAgentTaskResult:
    """
    从主图状态读取并校验暂停任务结果。

    参数含义：
        state:
            包含 multi_agent_task_result 的当前主图状态。

    返回值含义：
        MultiAgentTaskResult:
            通过 Schema 校验的暂停任务结果。
    """

    raw_result = state.get("multi_agent_task_result")
    if not isinstance(raw_result, Mapping):
        raise ValueError("主图状态缺少可恢复的 multi_agent_task_result")
    return MultiAgentTaskResult.model_validate(raw_result)


def _load_resume_inputs(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从主图状态读取已经整理好的恢复输入。

    参数含义：
        state:
            包含 multi_agent_resume_inputs 的当前主图状态。

    返回值含义：
        dict[str, Any]:
            等待步骤编号到用户回答的独立字典。
    """

    raw_inputs = state.get("multi_agent_resume_inputs")
    if not isinstance(raw_inputs, Mapping) or not raw_inputs:
        raise ValueError("主图状态缺少 multi_agent_resume_inputs")
    return dict(raw_inputs)


def build_multi_agent_state_update(
    task_result: MultiAgentTaskResult,
) -> dict[str, Any]:
    """
    把多 Agent 标准结果转换成主图局部状态。

    功能：
        始终保存可序列化任务结果；等待输入时写入提示和等待标记，任务结束
        时写入最终回答并清空恢复字段。

    参数含义：
        task_result:
            Orchestrator 返回的最新多 Agent 任务结果。

    返回值含义：
        dict[str, Any]:
            可以由 LangGraph 合并并自动写入 Checkpoint 的普通字典。
    """

    result_data = task_result.model_dump(mode="python")
    if task_result.status == "awaiting_input":
        prompt = _extract_waiting_prompt(task_result)
        return {
            "multi_agent_task_result": result_data,
            "multi_agent_resume_action": "none",
            "multi_agent_resume_inputs": {},
            "multi_agent_resume_ready": False,
            "multi_agent_pending_prompt": prompt,
            "pending_prompt": prompt,
            "waiting_user_input": True,
            "current_agent": "multi_agent",
            "final_answer": prompt,
        }

    final_answer = str(
        task_result.final_answer
        or task_result.error_message
        or "多 Agent 任务已经结束，但没有生成可展示的回答。"
    )
    return {
        "multi_agent_task_result": result_data,
        "multi_agent_resume_action": "none",
        "multi_agent_resume_inputs": {},
        "multi_agent_resume_ready": False,
        "multi_agent_pending_prompt": "",
        "pending_prompt": "",
        "waiting_user_input": False,
        "current_agent": "multi_agent",
        "final_answer": final_answer,
    }


def _extract_waiting_prompt(
    task_result: MultiAgentTaskResult,
) -> str:
    """
    从暂停任务中提取优先展示给用户的问题。

    参数含义：
        task_result:
            状态为 awaiting_input 的多 Agent 任务结果。

    返回值含义：
        str:
            Worker、metadata 或 Planner 提供的第一个非空等待提示。
    """

    for result in task_result.task_results:
        if result.status == "awaiting_input":
            prompt = str(result.clarification_prompt or "").strip()
            if prompt:
                return prompt
    metadata_prompt = str(
        task_result.metadata.get("clarification_prompt") or ""
    ).strip()
    if metadata_prompt:
        return metadata_prompt
    plan_prompt = str(task_result.plan.clarification_prompt or "").strip()
    if plan_prompt:
        return plan_prompt
    raise ValueError("awaiting_input 多 Agent 任务缺少等待提示")
