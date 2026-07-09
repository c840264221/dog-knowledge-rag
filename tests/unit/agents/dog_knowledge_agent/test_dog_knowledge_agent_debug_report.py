"""
DogKnowledgeAgent Debug Report 单元测试。

功能：
    测试 V1.7.2 Step 7 新增的 DogKnowledgeAgent 调试报告构建能力。

测试目标：
    1. 可以构建完整 dog_knowledge_debug_report。
    2. 可以汇总 pipeline 信息。
    3. 可以汇总 RAG 信息。
    4. 可以汇总 memory_context 信息。
    5. 可以汇总 answer_strategy 信息。
    6. 可以汇总 final_answer 信息。
    7. 可以渲染 Markdown。
"""

from __future__ import annotations

from src.agents.dog_knowledge_agent.debug.debug_report import (
    build_dog_knowledge_debug_report,
    build_text_preview,
    extract_layers_from_steps,
    render_dog_knowledge_debug_report_markdown,
    summarize_rag_context,
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


def build_debug_test_state() -> dict:
    """
    构建 Debug Report 测试状态。

    功能：
        生成一个包含 pipeline、RAG、Memory、Strategy、Answer 字段的测试 state。

    参数：
        无。

    返回值：
        dict:
            测试状态。
    """

    return {
        "question": "金毛适合新手养吗？",
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
                EXPECTED_LAYERS,
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
                EXPECTED_LAYERS,
                start=1,
            )
        ],
        "rag_query": {
            "question": "金毛适合新手养吗？",
            "user_id": "test_user",
            "top_k": 5,
        },
        "retrieved_chunks": [
            {
                "chunk_id": "chunk_1",
                "score": 0.8,
            },
            {
                "chunk_id": "chunk_2",
                "score": 0.7,
            },
        ],
        "reranked_chunks": [
            {
                "chunk_id": "chunk_1",
                "score": 0.92,
            },
        ],
        "retrieval_quality": {
            "status": "good",
            "reason": "召回内容足够回答问题。",
        },
        "rag_context": {
            "status": "ready",
            "source_count": 1,
            "context_text": "金毛性格友好，适合新手，但需要较多运动。",
            "chunks": [
                {
                    "chunk_id": "chunk_1",
                }
            ],
        },
        "memory_context": "用户喜欢性格稳定、适合新手的狗。",
        "answer_strategy": {
            "name": "grounded_answer",
            "reason": "RAG 召回质量良好。",
        },
        "final_answer": "金毛通常适合新手，但需要注意运动量和掉毛。",
    }


def test_build_dog_knowledge_debug_report() -> None:
    """
    测试构建完整 DogKnowledgeAgent Debug Report。

    功能：
        验证 build_dog_knowledge_debug_report 可以生成完整结构。

    参数：
        无。

    返回值：
        None。
    """

    report = build_dog_knowledge_debug_report(
        state=build_debug_test_state(),
    )

    assert report[
        "section"
    ] == "dog_knowledge_agent"

    assert report[
        "section_title"
    ] == "DogKnowledgeAgent 调试报告"

    assert report[
        "status"
    ] == "ready"

    assert report[
        "pipeline"
    ][
        "step_count"
    ] == 10

    assert report[
        "pipeline"
    ][
        "layers"
    ] == list(
        EXPECTED_LAYERS,
    )

    assert report[
        "rag"
    ][
        "has_rag_query"
    ] is True

    assert report[
        "rag"
    ][
        "retrieved_chunk_count"
    ] == 2

    assert report[
        "rag"
    ][
        "reranked_chunk_count"
    ] == 1

    assert report[
        "memory"
    ][
        "has_memory_context"
    ] is True

    assert report[
        "strategy"
    ][
        "has_answer_strategy"
    ] is True

    assert report[
        "answer"
    ][
        "has_final_answer"
    ] is True


def test_build_debug_report_pipeline_only() -> None:
    """
    测试只包含 pipeline metadata 的 Debug Report。

    功能：
        如果当前只有 pipeline skeleton，
        report 状态应该是 pipeline_only。

    参数：
        无。

    返回值：
        None。
    """

    state = {
        "question": "测试问题",
        "dog_knowledge_pipeline_status": "skeleton_ready",
        "dog_knowledge_pipeline_version": "v1.7.2-step3",
        "dog_knowledge_pipeline_steps": [],
        "dog_knowledge_pipeline_trace": [],
    }

    report = build_dog_knowledge_debug_report(
        state=state,
    )

    assert report[
        "status"
    ] == "pipeline_only"


def test_summarize_rag_context_with_dict() -> None:
    """
    测试 RagContext dict 摘要。

    功能：
        验证 summarize_rag_context 可以处理 dict 类型 RagContext。

    参数：
        无。

    返回值：
        None。
    """

    summary = summarize_rag_context(
        rag_context={
            "status": "ready",
            "source_count": 2,
            "context_text": "这是一段 RAG 上下文。",
            "chunks": [
                {
                    "id": 1,
                },
                {
                    "id": 2,
                },
            ],
        }
    )

    assert summary[
        "has_rag_context"
    ] is True

    assert summary[
        "status"
    ] == "ready"

    assert summary[
        "source_count"
    ] == 2

    assert "RAG 上下文" in summary[
        "context_preview"
    ]


def test_summarize_rag_context_with_none() -> None:
    """
    测试 RagContext 为空时的摘要。

    功能：
        验证 summarize_rag_context 可以安全处理 None。

    参数：
        无。

    返回值：
        None。
    """

    summary = summarize_rag_context(
        rag_context=None,
    )

    assert summary[
        "has_rag_context"
    ] is False

    assert summary[
        "status"
    ] == "missing"


def test_extract_layers_from_steps() -> None:
    """
    测试从 steps 中提取 layers。

    功能：
        验证 extract_layers_from_steps 可以正确提取 layer 字段。

    参数：
        无。

    返回值：
        None。
    """

    layers = extract_layers_from_steps(
        steps=[
            {
                "layer": "entry",
            },
            {
                "layer": "query_builder",
            },
            {
                "not_layer": "ignored",
            },
        ]
    )

    assert layers == [
        "entry",
        "query_builder",
    ]


def test_build_text_preview_truncates_long_text() -> None:
    """
    测试文本预览会截断长文本。

    功能：
        当文本长度超过 max_length 时，应该被截断并添加省略号。

    参数：
        无。

    返回值：
        None。
    """

    preview = build_text_preview(
        value="a" * 20,
        max_length=10,
    )

    assert preview == "aaaaaaaaaa..."


def test_render_dog_knowledge_debug_report_markdown() -> None:
    """
    测试渲染 DogKnowledgeAgent Debug Report Markdown。

    功能：
        验证 report 可以渲染成 Markdown 文本。

    参数：
        无。

    返回值：
        None。
    """

    report = build_dog_knowledge_debug_report(
        state=build_debug_test_state(),
    )

    markdown = render_dog_knowledge_debug_report_markdown(
        debug_report=report,
    )

    assert "# DogKnowledgeAgent 调试报告" in markdown
    assert "Pipeline / 管线" in markdown
    assert "RAG / 检索增强生成" in markdown
    assert "Memory / 长期记忆" in markdown
    assert "Strategy / 回答策略" in markdown
    assert "Answer / 答案" in markdown
    assert "entry -> query_builder -> retrieval" in markdown
