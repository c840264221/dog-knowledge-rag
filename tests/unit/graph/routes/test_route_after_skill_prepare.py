"""主图 Skill（技能）准备节点后置路由测试。"""

from src.graph.routes.route_after_skill_prepare import (
    build_skill_prepare_route_map,
    route_after_skill_prepare,
)


def test_skill_route_should_stop_when_waiting_for_input() -> None:
    """测试 Skill 缺少输入时结束本轮并等待用户补充。"""

    route = route_after_skill_prepare(
        {
            "skill_status": "awaiting_input",
            "skill_target_agent": "dog_knowledge_agent",
        }
    )

    assert route == "awaiting_input"


def test_skill_route_should_continue_to_saved_target() -> None:
    """测试 Skill 已准备完成时继续进入首轮目标 Agent。"""

    route = route_after_skill_prepare(
        {
            "skill_status": "ready",
            "skill_target_agent": "dog_knowledge_agent",
        }
    )

    assert route == "dog_knowledge_agent"


def test_skill_route_map_should_map_waiting_to_end() -> None:
    """测试逻辑等待路由会映射到 LangGraph END。"""

    route_map = build_skill_prepare_route_map("__END__")

    assert route_map == {
        "dog_knowledge_agent": "dog_knowledge_agent",
        "general_agent": "general",
        "awaiting_input": "__END__",
    }
