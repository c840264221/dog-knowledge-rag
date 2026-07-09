"""
DogKnowledgeAgent Debug Report Smoke Check 单元测试。

功能：
    测试 V1.7.2 Step 8 新增的 DogKnowledgeAgent Debug Report smoke check 工具。

测试目标：
    1. 可以从 state 中提取 dog_knowledge_debug_report。
    2. 可以提取 report sections。
    3. 可以提取 pipeline layers。
    4. 完整 report 可以通过 smoke check。
    5. 缺少 report 时 smoke check 失败。
    6. 缺少核心 section 时 smoke check 失败。
    7. 可以渲染 Markdown smoke report。
"""

from __future__ import annotations

from src.agents.dog_knowledge_agent.debug.debug_report_smoke_checks import (
    EXPECTED_DOG_KNOWLEDGE_DEBUG_SECTIONS,
    EXPECTED_DOG_KNOWLEDGE_PIPELINE_LAYERS,
    extract_debug_report_pipeline_layers,
    extract_debug_report_sections,
    extract_dog_knowledge_debug_report,
    render_dog_knowledge_debug_report_smoke_markdown,
    validate_dog_knowledge_debug_report_smoke_state,
)


def build_valid_debug_report_state() -> dict:
    """
    构建有效 Debug Report smoke state。

    功能：
        生成一个包含完整 dog_knowledge_debug_report 的测试 state。

    参数：
        无。

    返回值：
        dict:
            有效测试状态。
    """

    return {
        "question": "金毛适合新手养吗？",
        "dog_knowledge_debug_report": {
            "section": "dog_knowledge_agent",
            "section_title": "DogKnowledgeAgent 调试报告",
            "status": "ready",
            "summary": "测试 summary",
            "created_at": "2026-01-01T00:00:00+00:00",
            "pipeline": {
                "status": "skeleton_ready",
                "version": "v1.7.2-step3",
                "question": "金毛适合新手养吗？",
                "step_count": 10,
                "trace_count": 10,
                "layers": list(
                    EXPECTED_DOG_KNOWLEDGE_PIPELINE_LAYERS,
                ),
                "steps": [],
                "trace": [],
            },
            "rag": {
                "has_rag_query": True,
                "rag_query": {
                    "question": "金毛适合新手养吗？",
                },
                "retrieved_chunk_count": 2,
                "reranked_chunk_count": 1,
                "retrieval_quality": {
                    "status": "good",
                },
                "rag_context": {
                    "has_rag_context": True,
                    "status": "ready",
                    "source_count": 1,
                    "context_preview": "测试 RAG 上下文",
                },
            },
            "memory": {
                "has_memory_context": True,
                "memory_context_type": "str",
                "memory_context_preview": "用户喜欢新手友好犬种。",
            },
            "strategy": {
                "has_answer_strategy": True,
                "answer_strategy": {
                    "name": "grounded_answer",
                },
            },
            "answer": {
                "has_final_answer": True,
                "answer_preview": "金毛通常适合新手。",
            },
        },
    }


def test_extract_dog_knowledge_debug_report() -> None:
    """
    测试提取 dog_knowledge_debug_report。

    功能：
        验证 extract_dog_knowledge_debug_report 可以从 state 中读取 report。

    参数：
        无。

    返回值：
        None。
    """

    state = build_valid_debug_report_state()

    report = extract_dog_knowledge_debug_report(
        state=state,
    )

    assert report[
        "section"
    ] == "dog_knowledge_agent"


def test_extract_dog_knowledge_debug_report_returns_empty_dict_when_missing() -> None:
    """
    测试缺少 report 时返回空 dict。

    功能：
        state 中没有 dog_knowledge_debug_report 时，
        应该安全返回空 dict。

    参数：
        无。

    返回值：
        None。
    """

    report = extract_dog_knowledge_debug_report(
        state={},
    )

    assert report == {}


def test_extract_debug_report_sections() -> None:
    """
    测试提取 Debug Report sections。

    功能：
        验证可以正确识别 pipeline、rag、memory、strategy、answer。

    参数：
        无。

    返回值：
        None。
    """

    report = extract_dog_knowledge_debug_report(
        state=build_valid_debug_report_state(),
    )

    sections = extract_debug_report_sections(
        debug_report=report,
    )

    assert sections == EXPECTED_DOG_KNOWLEDGE_DEBUG_SECTIONS


def test_extract_debug_report_pipeline_layers() -> None:
    """
    测试提取 pipeline layers。

    功能：
        验证可以从 report["pipeline"]["layers"] 中提取标准顺序。

    参数：
        无。

    返回值：
        None。
    """

    report = extract_dog_knowledge_debug_report(
        state=build_valid_debug_report_state(),
    )

    layers = extract_debug_report_pipeline_layers(
        debug_report=report,
    )

    assert layers == EXPECTED_DOG_KNOWLEDGE_PIPELINE_LAYERS


def test_validate_debug_report_smoke_state_passes() -> None:
    """
    测试完整 Debug Report smoke state 通过。

    功能：
        完整 dog_knowledge_debug_report 应该通过 smoke check。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_dog_knowledge_debug_report_smoke_state(
        state=build_valid_debug_report_state(),
    )

    assert result.passed is True
    assert result.errors == ()
    assert result.report_status == "ready"
    assert result.observed_sections == EXPECTED_DOG_KNOWLEDGE_DEBUG_SECTIONS
    assert result.observed_layers == EXPECTED_DOG_KNOWLEDGE_PIPELINE_LAYERS


def test_validate_debug_report_smoke_state_fails_when_report_missing() -> None:
    """
    测试缺少 dog_knowledge_debug_report 时失败。

    功能：
        state 中没有 dog_knowledge_debug_report 时，
        smoke check 应该失败。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_dog_knowledge_debug_report_smoke_state(
        state={
            "question": "测试问题",
        },
    )

    assert result.passed is False
    assert result.errors
    assert any(
        "缺少 dog_knowledge_debug_report" in error
        for error in result.errors
    )


def test_validate_debug_report_smoke_state_fails_when_section_missing() -> None:
    """
    测试缺少核心 section 时失败。

    功能：
        删除 memory section 后，smoke check 应该失败。

    参数：
        无。

    返回值：
        None。
    """

    state = build_valid_debug_report_state()

    del state[
        "dog_knowledge_debug_report"
    ][
        "memory"
    ]

    result = validate_dog_knowledge_debug_report_smoke_state(
        state=state,
    )

    assert result.passed is False
    assert any(
        "缺少核心 section" in error
        for error in result.errors
    )


def test_validate_debug_report_smoke_state_fails_when_layers_wrong() -> None:
    """
    测试 pipeline layers 错误时失败。

    功能：
        如果 report 中的 pipeline layers 顺序错误，
        smoke check 应该失败。

    参数：
        无。

    返回值：
        None。
    """

    state = build_valid_debug_report_state()

    state[
        "dog_knowledge_debug_report"
    ][
        "pipeline"
    ][
        "layers"
    ] = [
        "entry",
        "query_builder",
        "quality",
    ]

    result = validate_dog_knowledge_debug_report_smoke_state(
        state=state,
    )

    assert result.passed is False
    assert any(
        "pipeline.layers 不符合" in error
        for error in result.errors
    )


def test_render_debug_report_smoke_markdown() -> None:
    """
    测试渲染 Debug Report Smoke Markdown。

    功能：
        验证 smoke result 可以渲染为 Markdown 文本。

    参数：
        无。

    返回值：
        None。
    """

    state = build_valid_debug_report_state()

    result = validate_dog_knowledge_debug_report_smoke_state(
        state=state,
    )

    markdown = render_dog_knowledge_debug_report_smoke_markdown(
        result=result,
        state=state,
    )

    assert "# DogKnowledgeAgent Debug Report Smoke Check" in markdown
    assert "Observed Sections" in markdown
    assert "Observed Pipeline Layers" in markdown
    assert "Full DogKnowledgeAgent Debug Report" in markdown
    assert "DogKnowledgeAgent 调试报告" in markdown
