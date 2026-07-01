"""
RootAgent 路由测试脚本。

功能：
    用最小状态测试 V1.7.1 RootAgent 是否能正确完成主路由判断。

测试内容：
    1. 狗狗推荐问题 -> dog_knowledge_agent
    2. 狗狗知识问题 -> dog_knowledge_agent
    3. 工具问题 -> tool_agent
    4. 普通聊天 -> general_agent
    5. 结束类问题 -> FINISH

运行方式：
    python scripts/debug_root_agent_router.py
"""

import asyncio
from pprint import pprint

from src.agents.root_agent.supervisor import root_supervisor_node
from src.graph.nodes.router_node import semantic_router_node
from src.graph.routes.route_after_semantic import route_after_semantic


TEST_CASES = [
    {
        "name": "狗狗推荐问题",
        "question": "推荐几种适合公寓养的狗",
        "expected_route": "dog_knowledge_agent",
    },
    {
        "name": "狗狗知识问题",
        "question": "金毛寿命多久？",
        "expected_route": "dog_knowledge_agent",
    },
    {
        "name": "工具类问题",
        "question": "现在几点？",
        "expected_route": "tool_agent",
    },
    {
        "name": "普通聊天",
        "question": "你好，你是谁？",
        "expected_route": "general_agent",
    },
    {
        "name": "结束类问题",
        "question": "先这样，结束",
        "expected_route": "FINISH",
    },
]


def build_test_state(question: str) -> dict:
    """
    构建测试用 DogState。

    参数：
        question:
            用户输入问题。

    返回值：
        dict:
            模拟 LangGraph 运行时传入节点的 state。
    """

    return {
        "question": question,
        "user_id": "debug_user",
        "session_id": "debug_session",
        "trace_id": "debug_trace",
    }


async def test_root_supervisor_directly() -> None:
    """
    直接测试 root_supervisor_node。

    功能：
        不经过旧 semantic_router_node，直接调用新版 RootAgent。
    """

    print("\n========== 1. 直接测试 root_supervisor_node ==========")

    for case in TEST_CASES:
        state = build_test_state(
            question=case["question"],
        )

        result = await root_supervisor_node(
            state,
        )

        route_decision = result.get(
            "route_decision",
            {},
        )

        actual_route = route_decision.get(
            "route",
        )

        print(f"\n测试用例：{case['name']}")
        print(f"问题：{case['question']}")
        print(f"预期 route：{case['expected_route']}")
        print(f"实际 route：{actual_route}")

        pprint(
            result,
        )

        assert actual_route == case["expected_route"], (
            f"路由不符合预期："
            f"expected={case['expected_route']}, "
            f"actual={actual_route}"
        )


async def test_semantic_router_adapter() -> None:
    """
    测试 semantic_router_node 适配器。

    功能：
        确认旧主图节点 semantic_router_node 已经转调新版 RootAgent。
    """

    print("\n========== 2. 测试 semantic_router_node Adapter ==========")

    for case in TEST_CASES:
        state = build_test_state(
            question=case["question"],
        )

        result = await semantic_router_node(
            state,
        )

        route_decision = result.get(
            "route_decision",
            {},
        )

        actual_route = route_decision.get(
            "route",
        )

        print(f"\n测试用例：{case['name']}")
        print(f"问题：{case['question']}")
        print(f"预期 route：{case['expected_route']}")
        print(f"实际 route：{actual_route}")

        assert actual_route == case["expected_route"], (
            f"Adapter 路由不符合预期："
            f"expected={case['expected_route']}, "
            f"actual={actual_route}"
        )


def test_route_after_semantic() -> None:
    """
    测试 route_after_semantic。

    功能：
        确认主图条件边读取 route_decision 后，
        可以返回正确的路由 key。
    """

    print("\n========== 3. 测试 route_after_semantic ==========")

    for case in TEST_CASES:
        state = {
            "route_decision": {
                "route": case["expected_route"],
            }
        }

        actual_route = route_after_semantic(
            state,
        )

        print(f"\n测试用例：{case['name']}")
        print(f"预期 route：{case['expected_route']}")
        print(f"实际 route：{actual_route}")

        assert actual_route == case["expected_route"], (
            f"route_after_semantic 不符合预期："
            f"expected={case['expected_route']}, "
            f"actual={actual_route}"
        )


async def main() -> None:
    """
    测试入口。

    功能：
        依次运行 RootAgent 直接测试、semantic_router 适配器测试、
        route_after_semantic 路由测试。
    """

    await test_root_supervisor_directly()

    await test_semantic_router_adapter()

    test_route_after_semantic()

    print("\n✅ RootAgent V1.7.1 最小路由测试全部通过")


if __name__ == "__main__":
    asyncio.run(
        main(),
    )