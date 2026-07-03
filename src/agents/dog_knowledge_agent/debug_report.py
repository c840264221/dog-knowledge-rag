"""
DogKnowledgeAgent 调试报告构建模块。

功能：
    将 DogKnowledgeAgent 执行过程中产生的 pipeline、RAG、Memory、
    AnswerStrategy、Answer 等字段整理成统一的 dog_knowledge_debug_report。

当前模块主要负责：
    1. 构建结构化 dog_knowledge_debug_report。
    2. 汇总 pipeline skeleton 信息。
    3. 汇总 RAG 查询、召回、重排、质量检测、RagContext 信息。
    4. 汇总 memory_context 信息。
    5. 汇总 answer_strategy 和 final_answer 信息。
    6. 渲染 Markdown 调试报告。

当前不负责：
    1. 不执行真实 RAG 检索。
    2. 不执行 rerank。
    3. 不执行质量检测。
    4. 不检索用户长期记忆。
    5. 不生成最终回答。
    6. 不改变业务 state。

专业名词：
    Debug Report：调试报告，用于复盘 Agent 执行过程。
    Pipeline：管线，表示按顺序执行的一组处理步骤。
    RAG：Retrieval-Augmented Generation，检索增强生成。
    Memory Context：记忆上下文，表示用户长期记忆召回结果。
    Answer Strategy：回答策略，表示最终回答采用的生成方式。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_dog_knowledge_debug_report(
        state: dict[str, Any],
) -> dict[str, Any]:
    """
    构建 DogKnowledgeAgent 调试报告。

    功能：
        从当前 state 中读取 DogKnowledgeAgent 相关字段，
        并整理成 dog_knowledge_debug_report。

    参数：
        state:
            当前 LangGraph 状态或 DogKnowledgeAgent 节点返回状态。
            可以包含 dog_knowledge_pipeline_*、rag_query、rag_context、
            memory_context、answer_strategy、final_answer 等字段。

    返回值：
        dict[str, Any]:
            结构化 DogKnowledgeAgent 调试报告。
    """

    pipeline_section = build_pipeline_debug_section(
        state=state,
    )

    rag_section = build_rag_debug_section(
        state=state,
    )

    memory_section = build_memory_debug_section(
        state=state,
    )

    strategy_section = build_strategy_debug_section(
        state=state,
    )

    answer_section = build_answer_debug_section(
        state=state,
    )

    status = infer_dog_knowledge_debug_status(
        pipeline_section=pipeline_section,
        rag_section=rag_section,
        answer_section=answer_section,
    )

    summary = build_dog_knowledge_debug_summary(
        status=status,
        pipeline_section=pipeline_section,
        rag_section=rag_section,
        memory_section=memory_section,
        strategy_section=strategy_section,
        answer_section=answer_section,
    )

    return {
        "section": "dog_knowledge_agent",
        "section_title": "DogKnowledgeAgent 调试报告",
        "status": status,
        "summary": summary,
        "created_at": datetime.now(
            timezone.utc,
        ).isoformat(),
        "pipeline": pipeline_section,
        "rag": rag_section,
        "memory": memory_section,
        "strategy": strategy_section,
        "answer": answer_section,
    }


def build_pipeline_debug_section(
        state: dict[str, Any],
) -> dict[str, Any]:
    """
    构建 pipeline 调试片段。

    功能：
        汇总 dog_knowledge_pipeline_* 字段。

    参数：
        state:
            当前状态。

    返回值：
        dict[str, Any]:
            pipeline 调试信息。
    """

    steps = safe_list(
        state.get(
            "dog_knowledge_pipeline_steps",
        )
    )

    trace = safe_list(
        state.get(
            "dog_knowledge_pipeline_trace",
        )
    )

    layers = extract_layers_from_steps(
        steps=steps,
    )

    return {
        "status": state.get(
            "dog_knowledge_pipeline_status",
            "missing",
        ),
        "version": state.get(
            "dog_knowledge_pipeline_version",
            "",
        ),
        "question": state.get(
            "dog_knowledge_pipeline_question",
            state.get(
                "question",
                "",
            ),
        ),
        "step_count": len(
            steps,
        ),
        "trace_count": len(
            trace,
        ),
        "layers": layers,
        "steps": steps,
        "trace": trace,
    }


def build_rag_debug_section(
        state: dict[str, Any],
) -> dict[str, Any]:
    """
    构建 RAG 调试片段。

    功能：
        汇总 rag_query、retrieved_chunks、reranked_chunks、
        retrieval_quality、rag_context 等字段。

    参数：
        state:
            当前状态。

    返回值：
        dict[str, Any]:
            RAG 调试信息。
    """

    rag_query = state.get(
        "rag_query",
    )

    retrieved_chunks = safe_list(
        state.get(
            "retrieved_chunks",
        )
    )

    reranked_chunks = safe_list(
        state.get(
            "reranked_chunks",
        )
    )

    retrieval_quality = safe_dict(
        state.get(
            "retrieval_quality",
        )
    )

    rag_context = state.get(
        "rag_context",
    )

    rag_context_summary = summarize_rag_context(
        rag_context=rag_context,
    )

    return {
        "has_rag_query": rag_query is not None,
        "rag_query": normalize_debug_value(
            rag_query,
        ),
        "retrieved_chunk_count": len(
            retrieved_chunks,
        ),
        "reranked_chunk_count": len(
            reranked_chunks,
        ),
        "retrieval_quality": retrieval_quality,
        "rag_context": rag_context_summary,
    }


def build_memory_debug_section(
        state: dict[str, Any],
) -> dict[str, Any]:
    """
    构建 Memory 调试片段。

    功能：
        汇总 memory_context 字段。

    参数：
        state:
            当前状态。

    返回值：
        dict[str, Any]:
            Memory 调试信息。
    """

    memory_context = state.get(
        "memory_context",
        "",
    )

    has_memory_context = bool(
        memory_context,
    )

    return {
        "has_memory_context": has_memory_context,
        "memory_context_type": type(
            memory_context,
        ).__name__,
        "memory_context_preview": build_text_preview(
            value=memory_context,
            max_length=300,
        ),
    }


def build_strategy_debug_section(
        state: dict[str, Any],
) -> dict[str, Any]:
    """
    构建 AnswerStrategy 调试片段。

    功能：
        汇总 answer_strategy 字段。

    参数：
        state:
            当前状态。

    返回值：
        dict[str, Any]:
            回答策略调试信息。
    """

    answer_strategy = state.get(
        "answer_strategy",
    )

    return {
        "has_answer_strategy": answer_strategy is not None,
        "answer_strategy": normalize_debug_value(
            answer_strategy,
        ),
    }


def build_answer_debug_section(
        state: dict[str, Any],
) -> dict[str, Any]:
    """
    构建 Answer 调试片段。

    功能：
        汇总 final_answer / answer 字段。

    参数：
        state:
            当前状态。

    返回值：
        dict[str, Any]:
            答案调试信息。
    """

    final_answer = state.get(
        "final_answer",
        state.get(
            "answer",
            "",
        ),
    )

    return {
        "has_final_answer": bool(
            final_answer,
        ),
        "answer_preview": build_text_preview(
            value=final_answer,
            max_length=300,
        ),
    }


def infer_dog_knowledge_debug_status(
        pipeline_section: dict[str, Any],
        rag_section: dict[str, Any],
        answer_section: dict[str, Any],
) -> str:
    """
    推断 DogKnowledgeAgent Debug Report 状态。

    功能：
        根据 pipeline、RAG、answer 字段判断当前报告状态。

    参数：
        pipeline_section:
            pipeline 调试片段。

        rag_section:
            RAG 调试片段。

        answer_section:
            answer 调试片段。

    返回值：
        str:
            报告状态。
            可能值：
            - ready
            - pipeline_only
            - missing_pipeline
            - incomplete
    """

    pipeline_status = pipeline_section.get(
        "status",
    )

    has_rag_query = rag_section.get(
        "has_rag_query",
        False,
    )

    has_final_answer = answer_section.get(
        "has_final_answer",
        False,
    )

    if pipeline_status == "missing":
        return "missing_pipeline"

    if pipeline_status == "skeleton_ready" and not has_rag_query and not has_final_answer:
        return "pipeline_only"

    if pipeline_status == "skeleton_ready" and has_final_answer:
        return "ready"

    return "incomplete"


def build_dog_knowledge_debug_summary(
        status: str,
        pipeline_section: dict[str, Any],
        rag_section: dict[str, Any],
        memory_section: dict[str, Any],
        strategy_section: dict[str, Any],
        answer_section: dict[str, Any],
) -> str:
    """
    构建 DogKnowledgeAgent Debug Report 中文摘要。

    功能：
        根据各个调试片段生成一段简短中文摘要。

    参数：
        status:
            调试报告状态。

        pipeline_section:
            pipeline 调试片段。

        rag_section:
            RAG 调试片段。

        memory_section:
            Memory 调试片段。

        strategy_section:
            AnswerStrategy 调试片段。

        answer_section:
            Answer 调试片段。

    返回值：
        str:
            中文摘要。
    """

    step_count = pipeline_section.get(
        "step_count",
        0,
    )

    retrieved_chunk_count = rag_section.get(
        "retrieved_chunk_count",
        0,
    )

    reranked_chunk_count = rag_section.get(
        "reranked_chunk_count",
        0,
    )

    has_memory = memory_section.get(
        "has_memory_context",
        False,
    )

    has_strategy = strategy_section.get(
        "has_answer_strategy",
        False,
    )

    has_answer = answer_section.get(
        "has_final_answer",
        False,
    )

    return (
        "DogKnowledgeAgent 调试报告状态为 "
        f"{status}。"
        f"pipeline 步骤数: {step_count}；"
        f"初始召回 chunks: {retrieved_chunk_count}；"
        f"重排 chunks: {reranked_chunk_count}；"
        f"是否包含记忆上下文: {has_memory}；"
        f"是否包含回答策略: {has_strategy}；"
        f"是否包含最终答案: {has_answer}。"
    )


def summarize_rag_context(
        rag_context: Any,
) -> dict[str, Any]:
    """
    汇总 RagContext 信息。

    功能：
        将 RagContext 对象、dict 或 None 转换为轻量摘要。
        避免 Debug Report 中塞入过大的 context_text。

    参数：
        rag_context:
            RagContext 对象、dict 或 None。

    返回值：
        dict[str, Any]:
            RagContext 摘要信息。
    """

    if rag_context is None:
        return {
            "has_rag_context": False,
            "status": "missing",
            "source_count": 0,
            "context_preview": "",
        }

    normalized = normalize_debug_value(
        rag_context,
    )

    if isinstance(
            normalized,
            dict,
    ):
        context_text = normalized.get(
            "context_text",
            "",
        )

        return {
            "has_rag_context": True,
            "status": normalized.get(
                "status",
                "unknown",
            ),
            "source_count": normalized.get(
                "source_count",
                len(
                    safe_list(
                        normalized.get(
                            "chunks",
                        )
                    )
                ),
            ),
            "context_preview": build_text_preview(
                value=context_text,
                max_length=300,
            ),
        }

    return {
        "has_rag_context": True,
        "status": "unknown",
        "source_count": 0,
        "context_preview": build_text_preview(
            value=normalized,
            max_length=300,
        ),
    }


def extract_layers_from_steps(
        steps: list[Any],
) -> list[str]:
    """
    从 pipeline steps 中提取 layer 列表。

    功能：
        读取每个 step 的 layer 字段。

    参数：
        steps:
            pipeline steps 列表。

    返回值：
        list[str]:
            layer 字符串列表。
    """

    layers: list[str] = []

    for step in steps:
        if not isinstance(
                step,
                dict,
        ):
            continue

        layer = step.get(
            "layer",
        )

        if isinstance(
                layer,
                str,
        ):
            layers.append(
                layer,
            )

    return layers


def normalize_debug_value(
        value: Any,
) -> Any:
    """
    将调试值转换为适合写入 report 的结构。

    功能：
        兼容 Pydantic model、dataclass、普通 dict、list 和基础类型。

    参数：
        value:
            任意待转换的值。

    返回值：
        Any:
            可序列化或更容易展示的调试值。
    """

    if value is None:
        return None

    if hasattr(
            value,
            "model_dump",
    ):
        return value.model_dump()

    if hasattr(
            value,
            "__dict__",
    ) and not isinstance(
            value,
            type,
    ):
        return {
            key: normalize_debug_value(
                item,
            )
            for key, item in vars(
                value,
            ).items()
            if not key.startswith(
                "_",
            )
        }

    if isinstance(
            value,
            dict,
    ):
        return {
            key: normalize_debug_value(
                item,
            )
            for key, item in value.items()
        }

    if isinstance(
            value,
            list,
    ):
        return [
            normalize_debug_value(
                item,
            )
            for item in value
        ]

    if isinstance(
            value,
            tuple,
    ):
        return [
            normalize_debug_value(
                item,
            )
            for item in value
        ]

    return value


def safe_list(
        value: Any,
) -> list[Any]:
    """
    安全转换为 list。

    功能：
        如果 value 是 list，则原样返回。
        如果 value 是 tuple，则转换为 list。
        其他情况返回空 list。

    参数：
        value:
            任意输入值。

    返回值：
        list[Any]:
            安全列表。
    """

    if isinstance(
            value,
            list,
    ):
        return value

    if isinstance(
            value,
            tuple,
    ):
        return list(
            value,
        )

    return []


def safe_dict(
        value: Any,
) -> dict[str, Any]:
    """
    安全转换为 dict。

    功能：
        如果 value 是 dict，则原样返回。
        如果 value 是 Pydantic model，则使用 model_dump 转换。
        其他情况返回空 dict。

    参数：
        value:
            任意输入值。

    返回值：
        dict[str, Any]:
            安全字典。
    """

    if isinstance(
            value,
            dict,
    ):
        return value

    if hasattr(
            value,
            "model_dump",
    ):
        dumped = value.model_dump()

        if isinstance(
                dumped,
                dict,
        ):
            return dumped

    return {}


def build_text_preview(
        value: Any,
        max_length: int = 300,
) -> str:
    """
    构建文本预览。

    功能：
        将任意值转换为字符串，并截断到指定长度。

    参数：
        value:
            任意输入值。

        max_length:
            最大长度。

    返回值：
        str:
            文本预览。
    """

    if value is None:
        return ""

    text = str(
        value,
    )

    if len(
            text,
    ) <= max_length:
        return text

    return (
        text[
            :max_length
        ]
        + "..."
    )


def render_dog_knowledge_debug_report_markdown(
        debug_report: dict[str, Any],
) -> str:
    """
    渲染 DogKnowledgeAgent Debug Report Markdown。

    功能：
        将 dog_knowledge_debug_report 渲染成 Markdown 文本，
        方便控制台查看、写入文档或放入整体 Debug Report。

    参数：
        debug_report:
            DogKnowledgeAgent 调试报告。

    返回值：
        str:
            Markdown 格式调试报告。
    """

    pipeline = safe_dict(
        debug_report.get(
            "pipeline",
        )
    )

    rag = safe_dict(
        debug_report.get(
            "rag",
        )
    )

    memory = safe_dict(
        debug_report.get(
            "memory",
        )
    )

    strategy = safe_dict(
        debug_report.get(
            "strategy",
        )
    )

    answer = safe_dict(
        debug_report.get(
            "answer",
        )
    )

    layers = pipeline.get(
        "layers",
        [],
    )

    lines = [
        "# DogKnowledgeAgent 调试报告",
        "",
        f"- section: `{debug_report.get('section', '')}`",
        f"- status: `{debug_report.get('status', '')}`",
        f"- summary: {debug_report.get('summary', '')}",
        f"- created_at: `{debug_report.get('created_at', '')}`",
        "",
        "## Pipeline / 管线",
        "",
        f"- version: `{pipeline.get('version', '')}`",
        f"- status: `{pipeline.get('status', '')}`",
        f"- step_count: `{pipeline.get('step_count', 0)}`",
        f"- trace_count: `{pipeline.get('trace_count', 0)}`",
        f"- layers: {' -> '.join(layers) if isinstance(layers, list) else ''}",
        "",
        "## RAG / 检索增强生成",
        "",
        f"- has_rag_query: `{rag.get('has_rag_query', False)}`",
        f"- retrieved_chunk_count: `{rag.get('retrieved_chunk_count', 0)}`",
        f"- reranked_chunk_count: `{rag.get('reranked_chunk_count', 0)}`",
        f"- retrieval_quality: `{rag.get('retrieval_quality', {})}`",
        f"- rag_context: `{rag.get('rag_context', {})}`",
        "",
        "## Memory / 长期记忆",
        "",
        f"- has_memory_context: `{memory.get('has_memory_context', False)}`",
        f"- memory_context_type: `{memory.get('memory_context_type', '')}`",
        f"- memory_context_preview: {memory.get('memory_context_preview', '')}",
        "",
        "## Strategy / 回答策略",
        "",
        f"- has_answer_strategy: `{strategy.get('has_answer_strategy', False)}`",
        f"- answer_strategy: `{strategy.get('answer_strategy', None)}`",
        "",
        "## Answer / 答案",
        "",
        f"- has_final_answer: `{answer.get('has_final_answer', False)}`",
        f"- answer_preview: {answer.get('answer_preview', '')}",
    ]

    return "\n".join(
        lines,
    )