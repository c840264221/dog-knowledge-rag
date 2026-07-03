"""
DogKnowledgeAgent Entry Integration 单元测试。

功能：
    验证 DogKnowledgeAgent 集成层可以同时兼容普通节点函数
    和 LangGraph 已编译子图对象。
"""

from __future__ import annotations

import pytest

from src.agents.dog_knowledge_agent.entry_integration import (
    build_integrated_dog_knowledge_entry_node,
    call_integrated_delegate_node,
)


class FakeAsyncCompiledGraph:
    """
    模拟支持 ainvoke 的已编译 LangGraph 子图。

    功能：
        用最小对象模拟 CompiledStateGraph 的异步调用协议。

    参数含义：
        无。

    返回值含义：
        该测试类本身不直接返回值，返回值由 ainvoke 方法提供。
    """

    async def ainvoke(
            self,
            state: dict,
    ) -> dict:
        """
        模拟异步调用已编译子图。

        功能：
            接收 state 并返回业务节点状态更新。

        参数含义：
            state:
                当前测试状态字典。

        返回值含义：
            dict:
                模拟 DogKnowledgeAgent 子图返回的状态更新。
        """

        return {
            "final_answer": f"async graph: {state['question']}",
            "delegate_called_by": "ainvoke",
        }


class FakeSyncCompiledGraph:
    """
    模拟只支持 invoke 的已编译 LangGraph 子图。

    功能：
        用最小对象模拟 CompiledStateGraph 的同步调用协议。

    参数含义：
        无。

    返回值含义：
        该测试类本身不直接返回值，返回值由 invoke 方法提供。
    """

    def invoke(
            self,
            state: dict,
    ) -> dict:
        """
        模拟同步调用已编译子图。

        功能：
            接收 state 并返回业务节点状态更新。

        参数含义：
            state:
                当前测试状态字典。

        返回值含义：
            dict:
                模拟 DogKnowledgeAgent 子图返回的状态更新。
        """

        return {
            "final_answer": f"sync graph: {state['question']}",
            "delegate_called_by": "invoke",
        }


@pytest.mark.asyncio
async def test_integrated_node_wraps_async_compiled_graph() -> None:
    """
    测试集成节点可以包装支持 ainvoke 的已编译子图。

    功能：
        验证 build_integrated_dog_knowledge_entry_node 可以把
        异步已编译子图接入 Entry Adapter。

    参数含义：
        无。

    返回值含义：
        None。
    """

    node = build_integrated_dog_knowledge_entry_node(
        delegate_node=FakeAsyncCompiledGraph(),
    )

    result = await node(
        {
            "question": "金毛适合新手养吗？",
        },
    )

    assert result["delegate_called_by"] == "ainvoke"
    assert result["final_answer"] == "async graph: 金毛适合新手养吗？"
    assert result["dog_knowledge_pipeline_status"] == "skeleton_ready"


@pytest.mark.asyncio
async def test_integrated_node_wraps_sync_compiled_graph() -> None:
    """
    测试集成节点可以包装只支持 invoke 的已编译子图。

    功能：
        验证 build_integrated_dog_knowledge_entry_node 可以兼容
        同步调用协议的已编译子图。

    参数含义：
        无。

    返回值含义：
        None。
    """

    node = build_integrated_dog_knowledge_entry_node(
        delegate_node=FakeSyncCompiledGraph(),
    )

    result = await node(
        {
            "question": "边牧聪明吗？",
        },
    )

    assert result["delegate_called_by"] == "invoke"
    assert result["final_answer"] == "sync graph: 边牧聪明吗？"
    assert result["dog_knowledge_pipeline_status"] == "skeleton_ready"


@pytest.mark.asyncio
async def test_integrated_delegate_still_supports_plain_node_function() -> None:
    """
    测试集成层仍然支持普通节点函数。

    功能：
        验证新增 compiled graph 兼容逻辑不会破坏原有函数节点接入方式。

    参数含义：
        无。

    返回值含义：
        None。
    """

    def plain_node(
            state: dict,
    ) -> dict:
        """
        模拟普通 DogKnowledgeAgent 节点函数。

        功能：
            接收 state 并返回状态更新。

        参数含义：
            state:
                当前测试状态字典。

        返回值含义：
            dict:
                模拟业务节点返回的状态更新。
        """

        return {
            "final_answer": f"plain node: {state['question']}",
            "delegate_called_by": "callable",
        }

    result = await call_integrated_delegate_node(
        state={
            "question": "柯基掉毛严重吗？",
        },
        delegate_node=plain_node,
    )

    assert result == {
        "final_answer": "plain node: 柯基掉毛严重吗？",
        "delegate_called_by": "callable",
    }


@pytest.mark.asyncio
async def test_integrated_delegate_rejects_invalid_result_type() -> None:
    """
    测试集成层拒绝非法返回值类型。

    功能：
        验证 delegate 返回非 mapping 类型时会抛出 TypeError。

    参数含义：
        无。

    返回值含义：
        None。
    """

    class InvalidCompiledGraph:
        """
        模拟返回非法类型的已编译子图。

        功能：
            用于测试集成层的返回值校验。

        参数含义：
            无。

        返回值含义：
            该测试类本身不直接返回值，返回值由 ainvoke 方法提供。
        """

        async def ainvoke(
                self,
                state: dict,
        ) -> str:
            """
            返回非法字符串结果。

            功能：
                模拟错误的子图返回值。

            参数含义：
                state:
                    当前测试状态字典。

            返回值含义：
                str:
                    非法返回值类型。
            """

            return "invalid result"

    with pytest.raises(
            TypeError,
    ):
        await call_integrated_delegate_node(
            state={
                "question": "测试问题",
            },
            delegate_node=InvalidCompiledGraph(),
        )
