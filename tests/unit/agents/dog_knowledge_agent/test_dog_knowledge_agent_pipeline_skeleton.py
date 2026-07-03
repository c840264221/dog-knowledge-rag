"""
DogKnowledgeAgent Pipeline Skeleton 单元测试。

功能：
    测试 V1.7.2 Step 3 新增的 DogKnowledgeAgent 入口编排骨架。

测试目标：
    1. pipeline steps 可以正确构建。
    2. pipeline 顺序与 module_contracts 保持一致。
    3. quality 位于 rerank 之后、context_builder 之前。
    4. memory_context 与 RagContextBuilder 分离。
    5. skeleton state update 不修改原始 state。
    6. Markdown 渲染结果包含关键职责层。
"""

from __future__ import annotations

from src.agents.dog_knowledge_agent.pipeline_skeleton import (
    build_dog_knowledge_pipeline_skeleton_state_update,
    build_dog_knowledge_pipeline_steps,
    build_dog_knowledge_pipeline_trace,
    get_dog_knowledge_pipeline_layer_order,
    render_dog_knowledge_pipeline_skeleton_markdown,
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


def test_build_dog_knowledge_pipeline_steps() -> None:
    """
    测试构建 DogKnowledgeAgent pipeline steps。

    功能：
        验证 pipeline steps 可以根据 module_contracts 正确生成。

    参数：
        无。

    返回值：
        None。
    """

    steps = build_dog_knowledge_pipeline_steps()

    assert steps

    assert tuple(
        step.layer
        for step in steps
    ) == EXPECTED_LAYERS

    assert all(
        step.status == "planned"
        for step in steps
    )


def test_pipeline_layer_order_matches_expected_layers() -> None:
    """
    测试 pipeline 层级顺序。

    功能：
        验证 get_dog_knowledge_pipeline_layer_order 返回标准执行顺序。

    参数：
        无。

    返回值：
        None。
    """

    layers = get_dog_knowledge_pipeline_layer_order()

    assert layers == EXPECTED_LAYERS


def test_quality_layer_is_after_rerank_and_before_context_builder() -> None:
    """
    测试 quality 层位置。

    功能：
        确保质量检测位于 rerank 之后、context_builder 之前。

    参数：
        无。

    返回值：
        None。
    """

    layers = get_dog_knowledge_pipeline_layer_order()

    rerank_index = layers.index(
        "rerank",
    )

    quality_index = layers.index(
        "quality",
    )

    context_builder_index = layers.index(
        "context_builder",
    )

    assert rerank_index < quality_index < context_builder_index


def test_memory_context_is_after_context_builder() -> None:
    """
    测试 memory_context 层位置。

    功能：
        确保 memory_context 与 RagContextBuilder 分离，
        且位于 context_builder 之后。

    参数：
        无。

    返回值：
        None。
    """

    layers = get_dog_knowledge_pipeline_layer_order()

    context_builder_index = layers.index(
        "context_builder",
    )

    memory_context_index = layers.index(
        "memory_context",
    )

    assert context_builder_index < memory_context_index


def test_build_dog_knowledge_pipeline_trace() -> None:
    """
    测试构建 DogKnowledgeAgent pipeline trace。

    功能：
        验证 trace 数量与 steps 一致，
        并且每个 trace item 都包含 layer、status、message。

    参数：
        无。

    返回值：
        None。
    """

    steps = build_dog_knowledge_pipeline_steps()

    trace = build_dog_knowledge_pipeline_trace(
        steps=steps,
    )

    assert len(
        trace,
    ) == len(
        steps,
    )

    assert tuple(
        item.layer
        for item in trace
    ) == EXPECTED_LAYERS

    assert all(
        item.status == "planned"
        for item in trace
    )

    assert all(
        item.message
        for item in trace
    )


def test_pipeline_skeleton_state_update() -> None:
    """
    测试 pipeline skeleton state update。

    功能：
        验证 build_dog_knowledge_pipeline_skeleton_state_update
        会返回标准 state 更新字段。

    参数：
        无。

    返回值：
        None。
    """

    state = {
        "question": "推荐几种适合公寓养的狗",
        "user_id": "test_user",
        "session_id": "test_session",
        "trace_id": "test_trace",
    }

    result = build_dog_knowledge_pipeline_skeleton_state_update(
        state=state,
    )

    assert result[
        "current_agent"
    ] == "dog_knowledge_agent"

    assert result[
        "dog_knowledge_pipeline_status"
    ] == "skeleton_ready"

    assert result[
        "dog_knowledge_pipeline_version"
    ] == "v1.7.2-step3"

    assert result[
        "dog_knowledge_pipeline_question"
    ] == "推荐几种适合公寓养的狗"

    assert tuple(
        step[
            "layer"
        ]
        for step in result[
            "dog_knowledge_pipeline_steps"
        ]
    ) == EXPECTED_LAYERS

    assert tuple(
        item[
            "layer"
        ]
        for item in result[
            "dog_knowledge_pipeline_trace"
        ]
    ) == EXPECTED_LAYERS


def test_pipeline_skeleton_does_not_mutate_original_state() -> None:
    """
    测试 pipeline skeleton 不修改原始 state。

    功能：
        确认 skeleton 函数只返回新的 state update，
        不会直接修改传入的 state。

    参数：
        无。

    返回值：
        None。
    """

    state = {
        "question": "金毛适合新手养吗？",
        "user_id": "test_user",
    }

    original_state = dict(
        state,
    )

    _ = build_dog_knowledge_pipeline_skeleton_state_update(
        state=state,
    )

    assert state == original_state


def test_render_dog_knowledge_pipeline_skeleton_markdown() -> None:
    """
    测试 Markdown 渲染。

    功能：
        验证 pipeline skeleton 可以被渲染成 Markdown 文档片段。

    参数：
        无。

    返回值：
        None。
    """

    markdown = render_dog_knowledge_pipeline_skeleton_markdown()

    assert "# DogKnowledgeAgent Pipeline Skeleton" in markdown
    assert "entry -> query_builder -> retrieval" in markdown
    assert "rerank" in markdown
    assert "quality" in markdown
    assert "context_builder" in markdown
    assert "memory_context" in markdown
    assert "debug_report" in markdown