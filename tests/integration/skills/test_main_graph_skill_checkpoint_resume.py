"""顶层 Skill（技能）主图 Checkpoint 跨轮恢复集成测试。"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.graph.graph_run import run_main_graph_with_result
from src.graph.nodes.router_node import semantic_router_node
from src.graph.nodes.skill_prepare_node import build_skill_prepare_node
from src.graph.routes.route_after_semantic import route_after_semantic
from src.graph.routes.route_after_skill_prepare import (
    build_skill_prepare_route_map,
    route_after_skill_prepare,
)
from src.graph.states.dog_state import DogState
from src.runtime.resume.contracts import GraphFinalResult, GraphInterruptResult


class FakeRuntimeContext:
    """保存主图入口在测试中写入的最小运行时身份字段。"""

    def __init__(self) -> None:
        self.trace_id: str | None = None
        self.user_id: str | None = None
        self.session_id: str | None = None


def build_skill_checkpoint_test_graph():
    """
    构建带真实 Skill、RootAgent 和内存 Checkpoint 的最小主图。

    功能：
        保留真实语义路由、Skill 准备和 Skill 后置路由，只用假 Agent 收敛
        LLM、RAG 等外部依赖，专门验证跨轮恢复链路。

    参数含义：
        无。

    返回值含义：
        CompiledStateGraph:
            使用 InMemorySaver 保存跨轮状态的已编译测试图。
    """

    graph = StateGraph(DogState)
    graph.add_node("semantic_router", semantic_router_node)
    graph.add_node("skill_prepare", build_skill_prepare_node())
    graph.add_node("dog_knowledge_agent", fake_dog_agent_node)
    graph.add_node("general", fake_other_agent_node)
    graph.add_node("tool_agent", fake_other_agent_node)
    graph.add_node("multi_agent", fake_other_agent_node)
    graph.set_entry_point("semantic_router")
    graph.add_conditional_edges(
        "semantic_router",
        route_after_semantic,
        {
            "dog_knowledge_agent": "skill_prepare",
            "general_agent": "skill_prepare",
            "tool_agent": "tool_agent",
            "multi_agent": "multi_agent",
            "FINISH": END,
        },
    )
    graph.add_conditional_edges(
        "skill_prepare",
        route_after_skill_prepare,
        build_skill_prepare_route_map(END),
    )
    for node_name in (
        "dog_knowledge_agent",
        "general",
        "tool_agent",
        "multi_agent",
    ):
        graph.add_edge(node_name, END)
    return graph.compile(checkpointer=InMemorySaver())


async def fake_dog_agent_node(state: DogState) -> dict[str, Any]:
    """
    验证恢复后的完整问题并返回测试答案。

    功能：
        检查下游业务问题保留原始任务和用户补充，同时 Skill 说明使用独立
        字段传递，不再污染交给 RAG 的检索文本。

    参数含义：
        state:
            Skill 准备完成后的主图状态。

    返回值含义：
        dict[str, Any]:
            包含最终测试答案的局部状态更新。
    """

    question = str(state.get("question") or "")
    retrieval_question = str(state.get("retrieval_question") or "")
    assert "帮我为6岁的金毛制定训练计划" in question
    assert "它目前会坐下，希望学习等待和召回" in question
    assert "已经校验通过的 Skill 输入" in question
    assert "技能：狗狗训练计划" in question
    assert "帮我为6岁的金毛制定训练计划" in retrieval_question
    assert "它目前会坐下，希望学习等待和召回" in retrieval_question
    assert "技能：狗狗训练计划" not in retrieval_question
    assert "技能：狗狗训练计划" in str(state.get("skill_context") or "")
    return {"final_answer": "已根据补充资料和 Skill 生成训练计划。"}


async def fake_other_agent_node(_state: DogState) -> dict[str, Any]:
    """
    标记请求错误进入了非目标 Agent。

    参数含义：
        _state:
            当前主图状态，本测试无需读取。

    返回值含义：
        dict[str, Any]:
            可用于识别错误路由的答案。
    """

    return {"final_answer": "错误路由到其他 Agent。"}


@pytest.mark.asyncio
async def test_top_level_skill_should_resume_from_same_thread_checkpoint() -> None:
    """
    测试顶层 Skill 能按同一 thread_id 跨两轮补全输入并恢复执行。

    功能：
        第一轮缺少当前行为和训练目标时返回统一中断；第二轮通过 resume_value
        补充后，从 Checkpoint 恢复原问题和目标 Agent，最终执行狗狗知识节点。

    参数含义：
        无。

    返回值含义：
        None。
    """

    app = build_skill_checkpoint_test_graph()
    runtime_context = FakeRuntimeContext()

    first_result = await run_main_graph_with_result(
        question="帮我为6岁的金毛制定训练计划。",
        thread_id="top-level-skill-thread",
        trace_id="top-level-skill-trace-1",
        graph_app=app,
        runtime_context=runtime_context,
    )

    assert isinstance(first_result, GraphInterruptResult)
    assert first_result.metadata["skill_selected_id"] == "dog-training-plan"

    second_result = await run_main_graph_with_result(
        question="它目前会坐下，希望学习等待和召回。",
        thread_id="top-level-skill-thread",
        trace_id="top-level-skill-trace-2",
        resume_value="它目前会坐下，希望学习等待和召回。",
        graph_app=app,
        runtime_context=runtime_context,
    )

    assert isinstance(second_result, GraphFinalResult)
    assert second_result.answer == "已根据补充资料和 Skill 生成训练计划。"
