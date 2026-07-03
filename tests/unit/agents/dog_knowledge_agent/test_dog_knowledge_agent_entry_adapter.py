"""
DogKnowledgeAgent Entry Adapter 单元测试。

功能：
    测试 V1.7.2 Step 4 新增的 DogKnowledgeAgent 入口节点适配器。

测试目标：
    1. Adapter 可以只返回 pipeline skeleton metadata。
    2. Adapter 可以包装同步 delegate node。
    3. Adapter 可以包装异步 delegate node。
    4. delegate_update 优先级高于 adapter_update。
    5. Adapter 不修改原始 state。
    6. delegate 返回 None 时可以正常处理。
    7. delegate 返回非法类型时会抛出 TypeError。
"""

from __future__ import annotations

import pytest

from src.agents.dog_knowledge_agent.entry_adapter import (
    build_dog_knowledge_entry_adapter,
    build_dog_knowledge_entry_skeleton_node,
    call_delegate_node,
    merge_entry_adapter_updates,
    run_dog_knowledge_entry_adapter,
)


EXPECTED_LAYERS = (
    "entry",
    "query_builder",
    "retrieval",
    "rerank",
    "quality",
    "context_builder",
    "memory_context",
    "strategy",
    "generation",
    "debug_report",
)


@pytest.mark.asyncio
async def test_skeleton_node_returns_pipeline_metadata() -> None:
    """
    测试 skeleton node 返回 pipeline metadata。

    功能：
        验证 build_dog_knowledge_entry_skeleton_node
        可以构建一个只返回 pipeline skeleton 的节点。

    参数：
        无。

    返回值：
        None。
    """

    node = build_dog_knowledge_entry_skeleton_node()

    result = await node(
        {
            "question": "推荐几种适合公寓养的狗",
            "user_id": "test_user",
        }
    )

    assert result[
        "current_agent"
    ] == "dog_knowledge_agent"

    assert result[
        "dog_knowledge_pipeline_status"
    ] == "skeleton_ready"

    assert tuple(
        step[
            "layer"
        ]
        for step in result[
            "dog_knowledge_pipeline_steps"
        ]
    ) == EXPECTED_LAYERS


@pytest.mark.asyncio
async def test_entry_adapter_wraps_sync_delegate_node() -> None:
    """
    测试 Entry Adapter 包装同步 delegate node。

    功能：
        验证 delegate_node 是普通同步函数时，
        adapter 可以正常调用并合并结果。

    参数：
        无。

    返回值：
        None。
    """

    def sync_delegate(
            state: dict,
    ) -> dict:
        """
        同步测试 delegate 节点。

        功能：
            模拟旧 DogKnowledgeAgent 同步入口节点。

        参数：
            state:
                当前测试状态。

        返回值：
            dict:
                模拟业务节点返回结果。
        """

        return {
            "final_answer": f"回答：{state['question']}",
            "delegate_called": True,
        }

    node = build_dog_knowledge_entry_adapter(
        delegate_node=sync_delegate,
    )

    result = await node(
        {
            "question": "金毛适合新手养吗？",
            "user_id": "test_user",
        }
    )

    assert result[
        "delegate_called"
    ] is True

    assert result[
        "final_answer"
    ] == "回答：金毛适合新手养吗？"

    assert result[
        "dog_knowledge_pipeline_status"
    ] == "skeleton_ready"


@pytest.mark.asyncio
async def test_entry_adapter_wraps_async_delegate_node() -> None:
    """
    测试 Entry Adapter 包装异步 delegate node。

    功能：
        验证 delegate_node 是 async 函数时，
        adapter 可以 await 并合并结果。

    参数：
        无。

    返回值：
        None。
    """

    async def async_delegate(
            state: dict,
    ) -> dict:
        """
        异步测试 delegate 节点。

        功能：
            模拟旧 DogKnowledgeAgent 异步入口节点。

        参数：
            state:
                当前测试状态。

        返回值：
            dict:
                模拟业务节点返回结果。
        """

        return {
            "final_answer": f"异步回答：{state['question']}",
            "delegate_called": True,
        }

    node = build_dog_knowledge_entry_adapter(
        delegate_node=async_delegate,
    )

    result = await node(
        {
            "question": "边牧聪明吗？",
            "user_id": "test_user",
        }
    )

    assert result[
        "delegate_called"
    ] is True

    assert result[
        "final_answer"
    ] == "异步回答：边牧聪明吗？"

    assert result[
        "dog_knowledge_pipeline_status"
    ] == "skeleton_ready"


def test_merge_entry_adapter_updates_delegate_has_priority() -> None:
    """
    测试 delegate_update 优先级更高。

    功能：
        当 adapter_update 和 delegate_update 有相同字段时，
        delegate_update 应该覆盖 adapter_update。

    参数：
        无。

    返回值：
        None。
    """

    result = merge_entry_adapter_updates(
        adapter_update={
            "current_agent": "dog_knowledge_agent",
            "final_answer": "adapter answer",
            "dog_knowledge_pipeline_status": "skeleton_ready",
        },
        delegate_update={
            "final_answer": "delegate answer",
            "business_field": "value",
        },
    )

    assert result[
        "current_agent"
    ] == "dog_knowledge_agent"

    assert result[
        "final_answer"
    ] == "delegate answer"

    assert result[
        "business_field"
    ] == "value"

    assert result[
        "dog_knowledge_pipeline_status"
    ] == "skeleton_ready"


@pytest.mark.asyncio
async def test_entry_adapter_does_not_mutate_original_state() -> None:
    """
    测试 Entry Adapter 不修改原始 state。

    功能：
        确认 adapter 只返回 state update，
        不会直接修改传入的 state dict。

    参数：
        无。

    返回值：
        None。
    """

    state = {
        "question": "柯基掉毛严重吗？",
        "user_id": "test_user",
    }

    original_state = dict(
        state,
    )

    node = build_dog_knowledge_entry_adapter()

    _ = await node(
        state,
    )

    assert state == original_state


@pytest.mark.asyncio
async def test_call_delegate_node_with_none_delegate() -> None:
    """
    测试 delegate_node 为 None。

    功能：
        如果没有传入真实业务节点，
        call_delegate_node 应该返回空 dict。

    参数：
        无。

    返回值：
        None。
    """

    result = await call_delegate_node(
        state={
            "question": "测试问题",
        },
        delegate_node=None,
    )

    assert result == {}


@pytest.mark.asyncio
async def test_call_delegate_node_with_delegate_returning_none() -> None:
    """
    测试 delegate 返回 None。

    功能：
        如果真实业务节点返回 None，
        adapter 应该将其视为空 dict。

    参数：
        无。

    返回值：
        None。
    """

    def none_delegate(
            state: dict,
    ) -> None:
        """
        返回 None 的测试 delegate。

        功能：
            模拟某些旧节点没有返回值的情况。

        参数：
            state:
                当前测试状态。

        返回值：
            None。
        """

        return None

    result = await call_delegate_node(
        state={
            "question": "测试问题",
        },
        delegate_node=none_delegate,
    )

    assert result == {}


@pytest.mark.asyncio
async def test_delegate_returning_invalid_type_raises_type_error() -> None:
    """
    测试 delegate 返回非法类型时抛出 TypeError。

    功能：
        delegate_node 必须返回 Mapping 或 None。
        如果返回字符串、列表等非法类型，应该抛出 TypeError。

    参数：
        无。

    返回值：
        None。
    """

    def invalid_delegate(
            state: dict,
    ) -> str:
        """
        返回非法类型的测试 delegate。

        功能：
            模拟错误实现的业务节点。

        参数：
            state:
                当前测试状态。

        返回值：
            str:
                非法返回值。
        """

        return "invalid"

    with pytest.raises(
            TypeError,
    ):
        await call_delegate_node(
            state={
                "question": "测试问题",
            },
            delegate_node=invalid_delegate,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_run_entry_adapter_without_pipeline_skeleton() -> None:
    """
    测试关闭 pipeline skeleton metadata。

    功能：
        当 include_pipeline_skeleton=False 时，
        adapter 不返回 pipeline skeleton metadata，
        但仍然会返回 DogKnowledgeAgent debug report。

    参数：
        无。

    返回值：
        None。
    """

    def sync_delegate(
            state: dict,
    ) -> dict:
        """
        同步测试 delegate 节点。

        功能：
            返回简单业务结果。

        参数：
            state:
                当前测试状态。

        返回值：
            dict:
                模拟业务结果。
        """

        return {
            "final_answer": "只返回业务结果",
        }

    result = await run_dog_knowledge_entry_adapter(
        state={
            "question": "测试问题",
        },
        delegate_node=sync_delegate,
        include_pipeline_skeleton=False,
    )

    assert result[
        "final_answer"
    ] == "只返回业务结果"

    assert "dog_knowledge_pipeline_status" not in result

    assert "dog_knowledge_pipeline_steps" not in result

    assert "dog_knowledge_debug_report" in result

    report = result[
        "dog_knowledge_debug_report"
    ]

    assert report[
        "status"
    ] == "missing_pipeline"

    assert report[
        "answer"
    ][
        "has_final_answer"
    ] is True

@pytest.mark.asyncio
async def test_entry_adapter_returns_dog_knowledge_debug_report() -> None:
    """
    测试 Entry Adapter 返回 DogKnowledgeAgent Debug Report。

    功能：
        验证 adapter 合并 pipeline metadata 和 delegate 业务字段后，
        会自动生成 dog_knowledge_debug_report。

    参数：
        无。

    返回值：
        None。
    """

    def sync_delegate(
            state: dict,
    ) -> dict:
        """
        同步测试 delegate 节点。

        功能：
            模拟真实 DogKnowledgeAgent 业务入口返回 final_answer。

        参数：
            state:
                当前测试状态。

        返回值：
            dict:
                模拟业务结果。
        """

        return {
            "final_answer": "测试答案",
            "rag_query": {
                "question": state[
                    "question"
                ],
            },
        }

    node = build_dog_knowledge_entry_adapter(
        delegate_node=sync_delegate,
    )

    result = await node(
        {
            "question": "金毛适合新手吗？",
            "user_id": "test_user",
        }
    )

    assert "dog_knowledge_debug_report" in result

    report = result[
        "dog_knowledge_debug_report"
    ]

    assert report[
        "section"
    ] == "dog_knowledge_agent"

    assert report[
        "status"
    ] == "ready"

    assert report[
        "pipeline"
    ][
        "status"
    ] == "skeleton_ready"

    assert report[
        "rag"
    ][
        "has_rag_query"
    ] is True

    assert report[
        "answer"
    ][
        "has_final_answer"
    ] is True
