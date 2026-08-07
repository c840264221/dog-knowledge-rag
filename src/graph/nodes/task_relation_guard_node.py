"""主图任务关系门卫节点。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.graph.states.dog_state import DogState
from src.logger import logger
from src.runtime.resume import resolve_pending_task_relation


def _find_pending_prompt(
    state: Mapping[str, Any],
    pending_kind: str,
) -> str:
    """
    取得旧等待任务正在向用户询问的问题。

    功能：
        根据等待模块读取 ToolAgent、多智能体或 Skill 已保存的补充提示，
        用于给简短恢复输入补上业务上下文。

    参数含义：
        state:
            已恢复等待字段的当前主图状态。
        pending_kind:
            当前等待模块名称，例如 tool、multi_agent 或 skill。

    返回值含义：
        str:
            旧任务的补充提示；没有找到时返回空字符串。
    """

    if pending_kind == "tool":
        clarification = state.get("tool_agent_clarification_request")
        if isinstance(clarification, Mapping):
            return str(clarification.get("question") or "").strip()
    if pending_kind == "multi_agent":
        return str(state.get("multi_agent_pending_prompt") or "").strip()
    if pending_kind == "skill":
        return str(state.get("skill_pending_prompt") or "").strip()
    return ""


def _build_memory_source_text(
    *,
    state: Mapping[str, Any],
    action: str,
    normalized_input: str,
    pending_kind: str,
) -> str:
    """
    构建本轮长期记忆抽取器真正需要读取的文本。

    功能：
        普通问题和新任务使用清理后的业务输入；恢复任务时把旧补充问题与
        用户本轮回答组合起来；取消和模糊控制输入返回空字符串，从而跳过
        没有业务价值的记忆 LLM 调用。

    参数含义：
        state:
            当前主图状态。
        action:
            任务关系门卫返回的动作。
        normalized_input:
            去掉“新问题：”“继续任务：”等控制前缀后的输入。
        pending_kind:
            当前等待任务所属模块。

    返回值含义：
        str:
            可直接交给 Memory Extract 的业务文本；空字符串表示跳过抽取。
    """

    if action in {"cancel", "ambiguous"}:
        return ""
    if action != "resume":
        return normalized_input

    pending_prompt = _find_pending_prompt(state, pending_kind)
    if not pending_prompt:
        return normalized_input
    return (
        f"旧任务正在询问：{pending_prompt}\n"
        f"用户本轮补充：{normalized_input}"
    )


async def task_relation_guard_node(
    state: DogState,
) -> dict[str, Any]:
    """
    在 Memory 和 RootAgent 之前判断本轮输入与旧等待任务的关系。

    功能：
        调用统一任务关系适配器，生成规范化业务问题、任务关系状态和专用
        记忆抽取文本。该节点不执行具体业务恢复，也不选择目标 Agent。

    参数含义：
        state:
            包含原始输入和 Checkpoint 等待字段的当前 DogState。

    返回值含义：
        dict[str, Any]:
            可合并回 DogState 的局部更新，包含任务关系结果、规范化问题和
            memory_source_text。
    """

    resolution = resolve_pending_task_relation(state)
    state_update = dict(resolution.get("state_update") or {})
    action = str(resolution.get("action") or "none").strip()
    normalized_input = str(
        state_update.get("question")
        or state.get("question")
        or ""
    ).strip()
    pending_kind = str(
        state_update.get("task_relation_pending_kind")
        or ""
    ).strip()

    terminal_update: dict[str, Any] = {}
    if action in {"cancel", "ambiguous"}:
        terminal_update = {
            "route_decision": {
                "route": "FINISH",
                "query_type": "finish",
                "confidence": 1.0,
                "reason": (
                    "当前输入无法安全区分新旧任务，需要用户明确选择。"
                    if action == "ambiguous"
                    else "用户已明确取消上一条等待任务。"
                ),
                "requires_rag": False,
                "requires_tool": False,
                "requires_memory": False,
                "source": "task_relation_guard_v2",
                "hints": {
                    "task_relation": action,
                    "pending_kind": pending_kind,
                },
            },
            "next_agent": "FINISH",
            "current_agent": "root_agent",
        }

    memory_source_text = _build_memory_source_text(
        state=state,
        action=action,
        normalized_input=normalized_input,
        pending_kind=pending_kind,
    )
    logger.info(
        "[task_relation_guard_node] "
        f"action={action}, "
        f"pending_kind={pending_kind or 'none'}, "
        f"terminal={bool(terminal_update)}, "
        f"memory_source_prepared={bool(memory_source_text)}"
    )

    return {
        **state_update,
        **terminal_update,
        "task_relation_guard_processed": True,
        "raw_user_input": str(
            state.get("raw_user_input")
            or state.get("question")
            or ""
        ).strip(),
        "memory_source_text": memory_source_text,
    }
