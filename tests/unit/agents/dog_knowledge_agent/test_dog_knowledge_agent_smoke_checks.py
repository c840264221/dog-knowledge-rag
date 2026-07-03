"""
DogKnowledgeAgent Smoke Check 单元测试。

功能：
    测试 V1.7.2 Step 6 新增的 DogKnowledgeAgent smoke check 工具。

测试目标：
    1. 可以从 state 中提取 pipeline layers。
    2. 可以验证完整 pipeline metadata。
    3. 可以发现缺失 metadata。
    4. 可以发现错误 pipeline 顺序。
    5. 可以渲染 smoke report markdown。
"""

from __future__ import annotations

from src.agents.dog_knowledge_agent.smoke_checks import (
    EXPECTED_DOG_KNOWLEDGE_PIPELINE_LAYERS,
    extract_pipeline_layers_from_state,
    render_dog_knowledge_smoke_report_markdown,
    validate_dog_knowledge_pipeline_metadata,
    validate_dog_knowledge_pipeline_order,
    validate_dog_knowledge_smoke_state,
)


def build_valid_smoke_state() -> dict:
    """
    构建有效的 smoke test state。

    功能：
        生成一个包含 dog_knowledge_pipeline_* metadata 的测试 state。

    参数：
        无。

    返回值：
        dict:
            有效测试状态。
    """

    return {
        "current_agent": "dog_knowledge_agent",
        "final_answer": "测试答案",
        "dog_knowledge_pipeline_status": "skeleton_ready",
        "dog_knowledge_pipeline_version": "v1.7.2-step3",
        "dog_knowledge_pipeline_question": "金毛适合新手养吗？",
        "dog_knowledge_pipeline_steps": [
            {
                "index": index,
                "layer": layer,
                "status": "planned",
            }
            for index, layer in enumerate(
                EXPECTED_DOG_KNOWLEDGE_PIPELINE_LAYERS,
                start=1,
            )
        ],
        "dog_knowledge_pipeline_trace": [
            {
                "index": index,
                "layer": layer,
                "status": "planned",
                "message": "测试 trace",
            }
            for index, layer in enumerate(
                EXPECTED_DOG_KNOWLEDGE_PIPELINE_LAYERS,
                start=1,
            )
        ],
    }


def test_extract_pipeline_layers_from_state() -> None:
    """
    测试从 state 中提取 pipeline layers。

    功能：
        验证 extract_pipeline_layers_from_state 可以正确读取 layer 顺序。

    参数：
        无。

    返回值：
        None。
    """

    state = build_valid_smoke_state()

    layers = extract_pipeline_layers_from_state(
        state=state,
    )

    assert layers == EXPECTED_DOG_KNOWLEDGE_PIPELINE_LAYERS


def test_validate_dog_knowledge_pipeline_metadata_passes() -> None:
    """
    测试有效 pipeline metadata 通过检查。

    功能：
        验证完整的 dog_knowledge_pipeline_* 字段可以通过检查。

    参数：
        无。

    返回值：
        None。
    """

    state = build_valid_smoke_state()

    result = validate_dog_knowledge_pipeline_metadata(
        state=state,
    )

    assert result.passed is True
    assert result.errors == ()
    assert result.observed_layers == EXPECTED_DOG_KNOWLEDGE_PIPELINE_LAYERS


def test_validate_dog_knowledge_smoke_state_passes() -> None:
    """
    测试综合 smoke state 检查通过。

    功能：
        验证完整 state 可以通过综合 smoke check。

    参数：
        无。

    返回值：
        None。
    """

    state = build_valid_smoke_state()

    result = validate_dog_knowledge_smoke_state(
        state=state,
    )

    assert result.passed is True
    assert result.errors == ()


def test_validate_pipeline_metadata_fails_when_missing_fields() -> None:
    """
    测试缺少 metadata 时检查失败。

    功能：
        当 state 不包含 dog_knowledge_pipeline_* 字段时，
        检查结果应该失败。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_dog_knowledge_pipeline_metadata(
        state={
            "current_agent": "dog_knowledge_agent",
        },
    )

    assert result.passed is False
    assert result.errors


def test_validate_pipeline_order_fails_when_quality_is_wrong_position() -> None:
    """
    测试错误 pipeline 顺序会失败。

    功能：
        如果 quality 没有位于 rerank 之后、context_builder 之前，
        应该返回错误。

    参数：
        无。

    返回值：
        None。
    """

    layers = (
        "entry",
        "query_builder",
        "retrieval",
        "quality",
        "rerank",
        "context_builder",
        "memory_context",
        "strategy",
        "generation",
        "debug_report",
    )

    errors = validate_dog_knowledge_pipeline_order(
        layers=layers,
    )

    assert errors
    assert any(
        "quality 必须位于 rerank 之后" in error
        for error in errors
    )


def test_validate_pipeline_order_fails_when_memory_context_is_wrong_position() -> None:
    """
    测试 memory_context 顺序错误会失败。

    功能：
        如果 memory_context 出现在 context_builder 之前，
        应该返回错误。

    参数：
        无。

    返回值：
        None。
    """

    layers = (
        "entry",
        "query_builder",
        "retrieval",
        "rerank",
        "quality",
        "memory_context",
        "context_builder",
        "strategy",
        "generation",
        "debug_report",
    )

    errors = validate_dog_knowledge_pipeline_order(
        layers=layers,
    )

    assert errors
    assert any(
        "memory_context 必须位于 context_builder 之后" in error
        for error in errors
    )


def test_render_dog_knowledge_smoke_report_markdown() -> None:
    """
    测试渲染 Smoke Check Markdown 报告。

    功能：
        验证 smoke check 结果可以渲染为 Markdown。

    参数：
        无。

    返回值：
        None。
    """

    state = build_valid_smoke_state()

    result = validate_dog_knowledge_smoke_state(
        state=state,
    )

    markdown = render_dog_knowledge_smoke_report_markdown(
        result=result,
    )

    assert "# DogKnowledgeAgent Smoke Check Report" in markdown
    assert "passed" in markdown
    assert "Observed Pipeline Layers" in markdown
    assert "entry -> query_builder -> retrieval" in markdown
    assert "Errors" in markdown
    assert "Warnings" in markdown