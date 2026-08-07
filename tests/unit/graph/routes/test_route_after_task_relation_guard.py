"""任务关系门卫后置路由单元测试。"""

from src.graph.routes.route_after_task_relation_guard import (
    build_task_relation_guard_route_map,
    route_after_task_relation_guard,
)


def test_guard_route_should_finish_cancel_and_ambiguous() -> None:
    """
    验证取消和模糊关系直接结束本轮主图。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    for relation in ("cancel", "ambiguous"):
        assert route_after_task_relation_guard(
            {
                "task_relation_decision": {
                    "relation": relation,
                }
            }
        ) == "finish"


def test_guard_route_should_continue_business_input() -> None:
    """
    验证普通、恢复和新任务继续进入记忆抽取节点。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    for relation in ("", "resume", "new_task"):
        assert route_after_task_relation_guard(
            {
                "task_relation_decision": {
                    "relation": relation,
                }
            }
        ) == "continue"


def test_guard_route_map_should_point_to_memory_and_end() -> None:
    """
    验证逻辑路由正确映射到 Memory 节点和结束节点。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    end_node = object()
    assert build_task_relation_guard_route_map(end_node) == {
        "continue": "memory_extract",
        "finish": end_node,
    }
