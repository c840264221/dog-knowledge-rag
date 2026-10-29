"""意图示例检索实践：真实向量、独立内存索引，由用户手动运行。"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from functools import partial
from typing import Any
from uuid import uuid4

from chromadb.proto.convert import to_proto_knn
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from sympy.codegen.ast import continue_

from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.embedding_provider import EmbeddingProvider
from src.runtime.container.providers.llm_provider import LLMProvider
from src.runtime.context import RuntimeContext, runtime_ctx
from src.runtime.scopes.metrics_scope import MetricsScope
from tests.learning.test_plus_intent_recognition import recognize_intent_hybrid


# 示例与本次验证问题分开；这是人工整理的学习数据，不是生产完整数据集。
INTENT_POLICY_VERSION = "intent_boundary_v1"
INTENT_EXAMPLES = (
    ("rec_01", "dog_recommendation", "第一次养狗，请推荐比较容易照顾的犬种。",
     "用户希望系统帮助选择犬种。"),
    ("rec_02", "dog_recommendation", "不考虑大型犬，推荐适合小户型的狗。",
     "排除一个范围后仍要求选择犬种，不是查询被排除对象的知识。"),
    ("rec_03", "dog_recommendation", "除了拉布拉多，还有哪些犬种适合新手？",
     "提到了具体犬种，但目标是获取其他犬种选择。"),
    ("know_01", "dog_knowledge", "金毛适合没有养狗经验的人吗？",
     "判断指定犬种是否适合，属于知识查询。"),
    ("know_02", "dog_knowledge", "拉布拉多住小户型会不会不适应？",
     "了解指定犬种的适合程度，没有要求推荐其他犬种。"),
    ("know_03", "dog_knowledge", "不是查询气温，我想了解哈士奇冬天如何护理。",
     "真实目标是狗狗护理，不能因气温关键词选择工具。"),
    ("tool_01", "tool_request", "查询一下杭州明天是否下雨。",
     "需要获取具体地点和日期的天气信息。"),
    ("chat_01", "general_chat", "这门编程课程适合刚入门的人吗？",
     "询问课程而非犬种，属于普通问答。"),
)


def build_intent_example_store(embedding: Any) -> InMemoryVectorStore:
    """
    功能：将人工示例正文向量化，建立独立内存索引，不写入正式数据库。
    参数含义：embedding 是从项目向量提供者获取的真实向量客户端。
    返回值含义：保存示例正文、编号、类别、依据和边界版本的内存向量库。
    """
    documents = [
        Document(
            page_content=text,
            metadata={
                "example_id": example_id,
                "query_type": query_type,
                "boundary_reason": reason,
                "policy_version": INTENT_POLICY_VERSION,
            },
        )
        for example_id, query_type, text, reason in INTENT_EXAMPLES
    ]
    store = InMemoryVectorStore(embedding)
    store.add_documents(
        documents=documents,
        ids=[document.metadata["example_id"] for document in documents],
    )
    return store


def retrieve_intent_examples(
    *,
    store: InMemoryVectorStore,
    question: str,
    context_summary: str = "",
    minimum_similarity: float = 0.45,
    candidate_count: int = 6,
    maximum_examples: int = 3,
    maximum_characters: int = 900,
) -> dict[str, Any]:
    """
    功能：检索候选，按阈值、正文去重、同类数量和字符预算选择少量示例。
    参数含义：store 是独立示例索引；question 和 context_summary 组成查询。
        minimum_similarity 是可调的余弦相似度门槛，演示默认值尚未校准；
        candidate_count 是召回数量；maximum_examples 是最多采用数量；
        maximum_characters 是示例 JSON 的字符预算，不是精确 Token 数。
    返回值含义：返回查询、候选及排除原因、selected_examples 和回退状态。
        相似度越大越相关；没有合适示例时返回空列表，不决定本次意图。
    """
    query_text = question
    if context_summary:
        query_text += f"\n必要背景：{context_summary}"
    # 当前本地实现按余弦相似度降序返回，不是 Chroma 的距离值。
    hits = store.similarity_search_with_score(query_text, k=candidate_count)
    candidates = []
    selected_examples = []
    seen_texts: set[str] = set()
    category_counts: dict[str, int] = {}
    for document, similarity in hits:
        example = {
            "example_id": document.metadata["example_id"],
            "text": document.page_content,
            "query_type": document.metadata["query_type"],
            "boundary_reason": document.metadata["boundary_reason"],
            "policy_version": document.metadata["policy_version"],
        }
        normalized_text = " ".join(document.page_content.split())
        query_type = example["query_type"]
        proposed = [*selected_examples, example]
        if example["policy_version"] != INTENT_POLICY_VERSION:
            selection_reason = "policy_version_mismatch"
        elif float(similarity) < minimum_similarity:
            selection_reason = "below_similarity_threshold"
        elif normalized_text in seen_texts:
            selection_reason = "duplicate_text"
        elif len(selected_examples) >= maximum_examples:
            selection_reason = "example_count_limit"
        elif category_counts.get(query_type, 0) >= 2:
            selection_reason = "same_category_limit"
        elif len(json.dumps(proposed, ensure_ascii=False)) > maximum_characters:
            selection_reason = "character_budget_limit"
        else:
            selection_reason = "selected"
            selected_examples.append(example)
            seen_texts.add(normalized_text)
            category_counts[query_type] = category_counts.get(query_type, 0) + 1
        candidates.append({
            **example,
            "similarity": round(float(similarity), 4),
            "selection_reason": selection_reason,
        })
    return {
        "status": "selected" if selected_examples else "fallback_no_examples",
        "policy_version": INTENT_POLICY_VERSION,
        "query_text": query_text,
        "minimum_similarity": minimum_similarity,
        "candidates": candidates,
        "selected_examples": selected_examples,
    }


async def run_intent_example_retrieval_demo(
    minimum_similarity: float = 0.45,
) -> dict[str, Any]:
    """
    功能：手动运行三条学习样本，对照基础识别与示例增强，保留真实调用诊断。
    参数含义：minimum_similarity 是用户可调的示例相似度门槛。
    返回值含义：返回索引构建耗时、各模式通过数、错误数和逐条结果。
        预期标签仅在识别结束后检查，不传给检索器或模型；不保存报告文件。
        正常情况下两个非规则样本各调用两次模型，共四次，不执行业务工具。
    """
    if not -1.0 <= minimum_similarity <= 1.0:
        raise ValueError("余弦相似度门槛必须在 -1 到 1 之间")
    container = RuntimeContainer()
    container.register("llm", LLMProvider())
    container.register("embedding", EmbeddingProvider())
    previous_context = runtime_ctx.get()
    results = []
    cases = (
        ("recommendation_boundary", "新手一个人住公寓，不想养金毛，帮我选几种容易照顾的狗。",
         "", "dog_recommendation"),
        ("knowledge_control", "我住公寓，想了解柯基是否容易掉毛。",
         "", "dog_knowledge"),
        ("rule_fast_path", "今天几号", "", "tool_request"),
    )
    try:
        startup_started_at = time.perf_counter()
        await container.startup()
        startup_ms = round((time.perf_counter() - startup_started_at) * 1000, 2)
        index_started_at = time.perf_counter()
        store = await asyncio.to_thread(
            build_intent_example_store, container.get("embedding").embedding,
        )
        index_build_ms = round((time.perf_counter() - index_started_at) * 1000, 2)
        retriever = partial(
            retrieve_intent_examples,
            store=store,
            minimum_similarity=minimum_similarity,
        )
        for case_id, question, background, expected_type in cases:
            for mode in ("baseline", "with_examples"):
                print(f"正在识别：{mode} | {question}")
                context = RuntimeContext(
                    trace_id=str(uuid4()),
                    user_id="plus_learning_user",
                    session_id="plus_intent_example_demo",
                    component="plus_intent_example_retrieval",
                )
                runtime_ctx.set(context)
                row: dict[str, Any] = {
                    "case_id": case_id,
                    "mode": mode,
                    "question": question,
                    "context_summary": background,
                    "expected_query_type": expected_type,
                    "trace_id": context.trace_id,
                }
                try:
                    await context.startup()
                    started_at = time.perf_counter()
                    try:
                        actual = await recognize_intent_hybrid(
                            llm_provider=container.get("llm"),
                            question=question,
                            context_summary=background,
                            example_retriever=retriever if mode == "with_examples" else None,
                        )
                        recognition = actual["recognition_result"]
                        passed = (
                            recognition["recognition_action"] == "route"
                            and recognition["query_type"] == expected_type
                        )
                        row.update(
                            status="passed" if passed else "mismatch",
                            actual=actual,
                        )
                    except Exception as exc:
                        cause = exc.__cause__
                        row.update(status="error", error={
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "cause_type": type(cause).__name__ if cause else None,
                            "cause_message": str(cause) if cause else None,
                        })
                    row["latency_ms"] = round(
                        (time.perf_counter() - started_at) * 1000, 2,
                    )
                    row["metrics"] = dict(context.service(MetricsScope).get_metrics())
                    results.append(row)
                finally:
                    try:
                        await context.shutdown()
                    finally:
                        runtime_ctx.set(previous_context)
        return {
            "mode": "intent_example_retrieval_comparison",
            "minimum_similarity": minimum_similarity,
            "provider_startup_ms": startup_ms,
            "index_build_ms": index_build_ms,
            "cases_per_mode": len(cases),
            "passed_cases_by_mode": {
                mode: sum(row["mode"] == mode and row["status"] == "passed" for row in results)
                for mode in ("baseline", "with_examples")
            },
            "error_cases": sum(row["status"] == "error" for row in results),
            "results": results,
        }
    finally:
        await container.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对照基础意图识别与真实示例向量检索")
    parser.add_argument("--minimum-similarity", type=float, default=0.45)
    arguments = parser.parse_args()
    report = asyncio.run(run_intent_example_retrieval_demo(arguments.minimum_similarity))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # 退出码仅表示运行是否出错；识别是否正确要看 passed_cases_by_mode。
    raise SystemExit(1 if report["error_cases"] else 0)
