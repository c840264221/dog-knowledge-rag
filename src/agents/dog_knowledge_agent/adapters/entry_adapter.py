"""
DogKnowledgeAgent 入口节点适配器。

功能：
    为 DogKnowledgeAgent 的真实入口节点提供 Adapter（适配器）能力。

    当前模块用于 V1.7.2 Step 4：
    1. 包装已有的 DogKnowledgeAgent 入口节点。
    2. 在不破坏旧业务逻辑的前提下，写入 pipeline skeleton metadata。
    3. 支持同步入口函数和异步入口函数。
    4. 保证 delegate 节点返回的业务字段优先保留。
    5. 为后续逐步接入 query_builder、retrieval、rerank、quality 等真实模块做准备。

当前不负责：
    1. 不构建真实 RagQuery。
    2. 不执行真实 RAG 检索。
    3. 不执行真实 rerank。
    4. 不判断真实检索质量。
    5. 不生成最终答案。
    6. 不修改旧业务节点内部逻辑。

专业名词：
    Adapter：适配器，在不改变旧接口的情况下接入新结构。
    Delegate：委托对象，表示被 Adapter 包装的真实业务函数。
    Metadata：元数据，表示辅助调试和观测的结构化信息。
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from src.agents.dog_knowledge_agent.skeleton.pipeline_skeleton import (
    build_dog_knowledge_pipeline_skeleton_state_update,
)

from src.agents.dog_knowledge_agent.debug.debug_report import (
    build_dog_knowledge_debug_report,
)

from src.logger import logger


DogKnowledgeState = dict[str, Any]

DogKnowledgeStateUpdate = dict[str, Any]

DogKnowledgeDelegateResult = (
    Mapping[str, Any]
    | Awaitable[Mapping[str, Any]]
    | None
)

DogKnowledgeDelegateNode = Callable[
    [DogKnowledgeState],
    DogKnowledgeDelegateResult,
]


def build_dog_knowledge_entry_adapter(
        delegate_node: DogKnowledgeDelegateNode | None = None,
        include_pipeline_skeleton: bool = True,
) -> Callable[
    [DogKnowledgeState],
    Awaitable[DogKnowledgeStateUpdate],
]:
    """
    构建 DogKnowledgeAgent 入口节点适配器。

    功能：
        返回一个可被 LangGraph 调用的异步节点函数。

        该节点会：
        1. 构建 DogKnowledgeAgent pipeline skeleton metadata。
        2. 如果传入 delegate_node，则继续调用真实业务入口节点。
        3. 将 skeleton metadata 和 delegate_node 返回值合并。
        4. 保证 delegate_node 的业务结果优先级更高。
        5. 不直接修改原始 state。

    参数：
        delegate_node:
            被适配器包装的真实 DogKnowledgeAgent 入口节点。
            可以是同步函数，也可以是异步函数。
            如果为 None，则只返回 pipeline skeleton metadata。

        include_pipeline_skeleton:
            是否写入 pipeline skeleton metadata。
            默认 True。
            后续如果某些测试或特殊场景不需要 metadata，可以设置为 False。

    返回值：
        Callable[[DogKnowledgeState], Awaitable[DogKnowledgeStateUpdate]]:
            一个异步 LangGraph 节点函数。
    """

    async def dog_knowledge_entry_adapter_node(
            state: DogKnowledgeState,
    ) -> DogKnowledgeStateUpdate:
        """
        DogKnowledgeAgent 入口适配节点。

        功能：
            在真实 DogKnowledgeAgent 入口节点外层增加 pipeline metadata。

        参数：
            state:
                当前 LangGraph 状态。

        返回值：
            DogKnowledgeStateUpdate:
                DogKnowledgeAgent 节点状态更新。
        """

        return await run_dog_knowledge_entry_adapter(
            state=state,
            delegate_node=delegate_node,
            include_pipeline_skeleton=include_pipeline_skeleton,
        )

    return dog_knowledge_entry_adapter_node


async def run_dog_knowledge_entry_adapter(
        state: DogKnowledgeState,
        delegate_node: DogKnowledgeDelegateNode | None = None,
        include_pipeline_skeleton: bool = True,
) -> DogKnowledgeStateUpdate:
    """
    执行 DogKnowledgeAgent 入口适配逻辑。

    功能：
        该函数是 Entry Adapter 的核心执行逻辑。

        执行顺序：
        1. 根据 state 构建 pipeline skeleton metadata。
        2. 调用 delegate_node 执行真实业务逻辑。
        3. 合并两个结果。
        4. 返回合并后的 state update。

    参数：
        state:
            当前 LangGraph 状态。

        delegate_node:
            被包装的真实业务节点。
            可以为空。

        include_pipeline_skeleton:
            是否包含 pipeline skeleton metadata。

    返回值：
        DogKnowledgeStateUpdate:
            合并后的 DogKnowledgeAgent 状态更新。
    """

    adapter_update: DogKnowledgeStateUpdate = {}

    if include_pipeline_skeleton:
        adapter_update = build_dog_knowledge_pipeline_skeleton_state_update(
            state=state,
        )

    delegate_update = await call_delegate_node(
        state=state,
        delegate_node=delegate_node,
    )

    merged_update = merge_entry_adapter_updates(
        adapter_update=adapter_update,
        delegate_update=delegate_update,
    )

    merged_state_preview = {
        **state,
        **merged_update,
    }

    debug_report = build_dog_knowledge_debug_report(
        state=merged_state_preview
    )


    state_snapshot = {
        "input_state": state,
        "adapter_update": adapter_update,
        "delegate_update": delegate_update,
        "merged_state_preview": merged_state_preview,
    }

    # state_snapshot_json = json.dumps(
    #     state_snapshot,
    #     indent=4,
    #     ensure_ascii=False,
    #     default=json_safe_default,
    # )
    #
    # logger.info(
    #     "DogKnowledgeAgent Entry Adapter 状态快照：\n"
    #     f"{state_snapshot_json}"
    # )

    return {
        **merged_update,
        "dog_knowledge_debug_report": debug_report,
    }


def json_safe_default(
        value: Any,
) -> Any:
    """
    为 Entry Adapter 日志提供 JSON 安全兜底转换。

    功能：
        当 json.dumps 遇到 AIMessage 等不可直接序列化对象时，
        将对象转换成可展示的 dict 或字符串，避免日志影响主流程。

    参数：
        value:
            json.dumps 无法直接序列化的对象。

    返回值：
        Any:
            可被 JSON 序列化的对象。
    """

    if hasattr(
            value,
            "model_dump",
    ):
        return value.model_dump()

    if hasattr(
            value,
            "dict",
    ) and callable(
            value.dict,
    ):
        return value.dict()

    if hasattr(
            value,
            "__dict__",
    ):
        return {
            key: item
            for key, item in vars(
                value,
            ).items()
            if not key.startswith(
                "_",
            )
        }

    return str(
        value,
    )


async def call_delegate_node(
        state: DogKnowledgeState,
        delegate_node: DogKnowledgeDelegateNode | None,
) -> DogKnowledgeStateUpdate:
    """
    调用被适配的真实 DogKnowledgeAgent 入口节点。

    功能：
        兼容三种情况：
        1. delegate_node 为 None。
        2. delegate_node 是同步函数。
        3. delegate_node 是异步函数。

        如果 delegate_node 返回 None，则视为空 dict。

    参数：
        state:
            当前 LangGraph 状态。

        delegate_node:
            被包装的真实业务节点。

    返回值：
        DogKnowledgeStateUpdate:
            delegate_node 返回的状态更新。
    """

    if delegate_node is None:
        return {}

    result = delegate_node(
        state,
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
            "DogKnowledgeAgent delegate_node 必须返回 dict / Mapping / None，"
            f"实际返回类型为: {type(result)!r}"
        )

    return dict(
        result,
    )


def merge_entry_adapter_updates(
        adapter_update: DogKnowledgeStateUpdate,
        delegate_update: DogKnowledgeStateUpdate,
) -> DogKnowledgeStateUpdate:
    """
    合并 Entry Adapter 结果和真实业务节点结果。

    功能：
        将 adapter_update 和 delegate_update 合并成一个 state update。

        合并规则：
        1. adapter_update 先写入。
        2. delegate_update 后写入。
        3. 如果字段冲突，delegate_update 优先。

    为什么 delegate_update 优先：
        因为真实业务节点可能已经写入 final_answer、rag_context、
        current_agent 等关键业务字段。
        Adapter 只负责补充 metadata，不应该覆盖真实业务结果。

    参数：
        adapter_update:
            适配器生成的 metadata 更新。

        delegate_update:
            真实业务节点返回的业务更新。

    返回值：
        DogKnowledgeStateUpdate:
            合并后的状态更新。
    """

    merged: DogKnowledgeStateUpdate = {
        **adapter_update,
        **delegate_update,
    }

    return merged


def build_dog_knowledge_entry_skeleton_node() -> Callable[
    [DogKnowledgeState],
    Awaitable[DogKnowledgeStateUpdate],
]:
    """
    构建只返回 pipeline skeleton 的 DogKnowledgeAgent 节点。

    功能：
        用于测试、调试或早期迁移阶段。

        这个节点不调用真实业务逻辑，
        只返回 DogKnowledgeAgent pipeline skeleton metadata。

    参数：
        无。

    返回值：
        Callable[[DogKnowledgeState], Awaitable[DogKnowledgeStateUpdate]]:
            一个异步 LangGraph 节点函数。
    """

    return build_dog_knowledge_entry_adapter(
        delegate_node=None,
        include_pipeline_skeleton=True,
    )

