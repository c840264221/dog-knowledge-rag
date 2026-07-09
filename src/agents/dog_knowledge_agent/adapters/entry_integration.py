"""
DogKnowledgeAgent 真实入口接入模块。

功能：
    将已有 DogKnowledgeAgent 真实入口节点接入 V1.7.2 Entry Adapter。

    当前模块主要负责：
    1. 包装已有 DogKnowledgeAgent 入口函数。
    2. 保留旧业务逻辑。
    3. 增加 dog_knowledge_pipeline_* metadata。
    4. 为 GraphRuntimeService 或主图构建代码提供统一入口。

当前不负责：
    1. 不构建真实 RagQuery。
    2. 不执行真实 RAG 检索。
    3. 不执行真实 rerank。
    4. 不执行真实质量检测。
    5. 不生成最终答案。

专业名词：
    Integration：集成，把已有模块接入新结构。
    Adapter：适配器，在不修改旧实现的情况下增加新能力。
    Delegate：被包装的真实业务节点。
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from src.agents.dog_knowledge_agent.adapters.entry_adapter import (
    build_dog_knowledge_entry_adapter,
)


DogKnowledgeState = dict[str, Any]

DogKnowledgeStateUpdate = dict[str, Any]

DogKnowledgeEntryNode = Callable[
    [DogKnowledgeState],
    DogKnowledgeStateUpdate | Awaitable[DogKnowledgeStateUpdate] | None,
]

DogKnowledgeCompiledDelegate = Any


async def call_integrated_delegate_node(
        state: DogKnowledgeState,
        delegate_node: DogKnowledgeEntryNode | DogKnowledgeCompiledDelegate,
) -> DogKnowledgeStateUpdate:
    """
    调用 DogKnowledgeAgent 集成层 delegate。

    功能：
        兼容普通节点函数和 LangGraph 已编译子图。
        如果 delegate_node 提供 ainvoke 方法，优先按异步子图调用。
        如果 delegate_node 只提供 invoke 方法，则按同步子图调用。
        如果 delegate_node 是普通 callable，则按普通节点函数调用。

    参数含义：
        state:
            当前 LangGraph 状态字典。
        delegate_node:
            被集成的真实 DogKnowledgeAgent 节点。
            可以是普通函数、异步函数，也可以是已编译 LangGraph 子图。

    返回值含义：
        dict[str, Any]:
            delegate_node 返回的状态更新字典。
            如果 delegate_node 返回 None，则统一转换为空字典。
    """

    if hasattr(delegate_node, "ainvoke"):
        result = await delegate_node.ainvoke(
            state,
        )
    elif hasattr(delegate_node, "invoke"):
        result = delegate_node.invoke(
            state,
        )
    elif callable(
            delegate_node,
    ):
        result = delegate_node(
            state,
        )
    else:
        raise TypeError(
            "DogKnowledgeAgent delegate must be a callable node "
            "or a compiled graph with invoke/ainvoke.",
        )

    if inspect.isawaitable(
            result,
    ):
        result = await result

    if result is None:
        return {}

    if not isinstance(
            result,
            Mapping,
    ):
        raise TypeError(
            "DogKnowledgeAgent delegate result must be a mapping or None.",
        )

    return dict(
        result,
    )


def build_integrated_dog_knowledge_entry_node(
        delegate_node: DogKnowledgeEntryNode | DogKnowledgeCompiledDelegate,
) -> Callable[
    [DogKnowledgeState],
    Awaitable[DogKnowledgeStateUpdate],
]:
    """
    构建接入 pipeline metadata 的 DogKnowledgeAgent 入口节点。

    功能：
        将已有 DogKnowledgeAgent 真实入口节点包装为新入口节点。

        包装后：
        1. 原有业务逻辑继续执行。
        2. 原有业务返回字段继续保留。
        3. 新增 dog_knowledge_pipeline_* metadata。
        4. delegate_node 返回字段优先级高于 adapter metadata。

    参数：
        delegate_node:
            当前项目里已有的 DogKnowledgeAgent 真实入口节点。
            可以是同步函数，也可以是异步函数。

    返回值：
        Callable[[DogKnowledgeState], Awaitable[DogKnowledgeStateUpdate]]:
            可以注册到 LangGraph 的异步节点函数。
    """

    async def integrated_delegate_node(
            state: DogKnowledgeState,
    ) -> DogKnowledgeStateUpdate:
        """
        调用已适配的 DogKnowledgeAgent delegate。

        功能：
            将普通节点函数或已编译 LangGraph 子图统一转换成
            Entry Adapter 可以调用的异步节点函数。

        参数含义：
            state:
                当前 LangGraph 状态字典。

        返回值含义：
            dict[str, Any]:
                DogKnowledgeAgent 真实业务节点返回的状态更新字典。
        """

        return await call_integrated_delegate_node(
            state=state,
            delegate_node=delegate_node,
        )

    return build_dog_knowledge_entry_adapter(
        delegate_node=integrated_delegate_node,
        include_pipeline_skeleton=True,
    )

