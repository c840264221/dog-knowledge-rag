"""主图 Skill（技能）选择和输入准备节点。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.skills.default_catalog import build_default_skill_runtime
from src.skills.runtime import SkillRuntime
from src.skills.state_adapter import (
    build_skill_enhanced_question,
    build_skill_state_update,
)
from src.agents.root_agent.routes import (
    get_root_route_from_state,
    normalize_root_route,
)


SkillPrepareNode = Callable[[Mapping[str, Any]], dict[str, Any]]


DIRECT_SKILL_AGENT_ROUTES = {
    "dog_knowledge_agent",
    "general_agent",
}


def _resolve_skill_target_agent(state: Mapping[str, Any]) -> str:
    """
    确定 Skill 准备完成后应该继续进入的目标 Agent。

    功能：
        恢复执行时优先使用检查点保存的目标 Agent；首轮执行时读取
        RootAgent 刚写入的路由结果。只允许普通知识和通用 Agent 使用
        顶层 Skill，其他路由兜底到 general_agent。

    参数含义：
        state:
            当前主图状态，包含可选的 skill_target_agent 和路由决策。

    返回值含义：
        str:
            dog_knowledge_agent 或 general_agent。
    """

    # 检查点中保存的是首轮已经确定的目标，恢复时不能用简短回答重新猜测。
    saved_target = str(state.get("skill_target_agent") or "").strip()
    if saved_target in DIRECT_SKILL_AGENT_ROUTES:
        return saved_target

    routed_target = normalize_root_route(
        get_root_route_from_state(state),
    )
    if routed_target in DIRECT_SKILL_AGENT_ROUTES:
        return routed_target
    return "general_agent"


def _build_skill_task_question(
    *,
    original_question: str,
    current_user_text: str,
) -> str:
    """
    组装交给下游 Agent 和 RAG 的业务问题。

    功能：
        保留首轮完整问题，并在恢复时追加用户本轮补充内容。Skill 操作说明
        继续保存在独立的 skill_context 字段中，避免污染 RAG 检索语义。

    参数含义：
        original_question:
            首轮触发 Skill 的完整用户问题。
        current_user_text:
            当前轮用户输入；恢复时通常是简短补充回答。
    返回值含义：
        str:
            只包含原始任务和可选补充信息的业务问题。
    """

    parts = [f"用户原始任务：\n{original_question}"]
    if current_user_text and current_user_text != original_question:
        parts.append(f"用户本轮补充信息：\n{current_user_text}")
    return "\n\n".join(parts)


def build_skill_prepare_node(
    skill_runtime: SkillRuntime | None = None,
) -> SkillPrepareNode:
    """
    构建主图使用的 Skill 准备节点。

    功能：
        创建一个读取 DogState、调用 SkillRuntime 并返回局部状态更新的同步节点。
        允许测试或未来 Container 从外部注入运行器。

    参数含义：
        skill_runtime:
            可选的技能运行器；为空时构建项目默认运行器。

    返回值含义：
        SkillPrepareNode:
            可以注册到 LangGraph StateGraph 的同步节点函数。
    """

    # 外部传入的运行器优先，测试可以借此隔离默认目录和提取规则。
    resolved_runtime = skill_runtime or build_default_skill_runtime()

    def skill_prepare_node(
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        为当前主图状态准备 Skill。

        功能：
            首轮根据 question 选择技能；如果 checkpoint 表示某个技能仍在等待
            输入，则继续使用已保存的技能编号和输入处理本轮补充内容。

        参数含义：
            state:
                当前 LangGraph 主图状态。

        返回值含义：
            dict[str, Any]:
                Skill 选择、输入、状态、提示和上下文组成的局部状态更新。
        """

        # 本轮用户自然语言，首轮是完整问题，恢复时是用户补充的简短回答。
        user_text = str(state.get("question") or "").strip()

        # 只有上一轮确实在等待 Skill 输入时，才沿用旧技能编号和历史输入。
        is_resuming_skill = (
            str(state.get("skill_status") or "").strip()
            == "awaiting_input"
        )
        selected_skill_id = (
            str(state.get("skill_selected_id") or "").strip()
            if is_resuming_skill
            else ""
        )
        raw_existing_inputs = state.get("skill_inputs")
        existing_inputs = (
            dict(raw_existing_inputs)
            if is_resuming_skill
            and isinstance(raw_existing_inputs, Mapping)
            else {}
        )

        # 首轮保存完整问题；恢复轮沿用检查点中的原始任务，避免只执行“6岁”。
        saved_original_question = str(
            state.get("skill_original_question") or ""
        ).strip()
        original_question = (
            saved_original_question
            if is_resuming_skill and saved_original_question
            else user_text
        )

        # 记住 RootAgent 首轮选择的目标，下一轮恢复时仍回到同一个 Agent。
        target_agent = _resolve_skill_target_agent(state)

        result = resolved_runtime.prepare(
            user_text=user_text,
            existing_inputs=existing_inputs,
            selected_skill_id=selected_skill_id or None,
        )
        update = build_skill_state_update(result)
        update["skill_original_question"] = original_question
        update["skill_target_agent"] = target_agent
        if result.status == "ready":
            # 先保存干净检索问题，再恢复原有的 Skill 增强 question。
            task_question = _build_skill_task_question(
                original_question=original_question,
                current_user_text=user_text,
            )
            update["retrieval_question"] = task_question
            update["memory_retrieval_text"] = task_question
            update["question"] = build_skill_enhanced_question(
                question=task_question,
                skill_inputs=result.extraction.merged_inputs,
                skill_context=result.skill_context,
            )
        return update

    return skill_prepare_node


skill_prepare_node = build_skill_prepare_node()
