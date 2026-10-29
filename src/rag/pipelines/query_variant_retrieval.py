"""同一检索请求的中英文召回；只变更问题文本，完整保留过滤及身份。"""

import asyncio

from src.rag.retrievers.multi_collection_retriever import fuse_collection_results
from src.rag.schemas import RagContext, RagQuery, RagRetrievedChunk


async def retrieve_query_variants(query: RagQuery, english_question: str, *, retriever) -> tuple[RagContext, dict]:
    """并发查询并融合完整候选，留给下游原有节点重排。

    参数：query为中文标准请求，english_question为查询改写阶段的英文文本；
        retriever为注入的检索器，有过滤时必须明确禁止过滤放宽。
    返回值：标准RagContext及通道审计记录；某路失败则整体报错，不冒充完整成功。
        保留各路距离，不把RRF分数写成向量距离；完整池不在重排前截断。
    """
    if not query.question.strip() or not english_question.strip():
        raise ValueError("双语查询需要非空中英文检索文本。")
    if query.filters and getattr(retriever, "allow_filter_relaxation", None) is not False:
        raise ValueError("有过滤条件的双语查询必须关闭过滤放宽。")
    requests = {"original_zh": query.model_copy(deep=True),
                "translated_en": query.model_copy(update={"question": english_question}, deep=True)}
    contexts = await asyncio.gather(*(retriever.async_retrieve(query=q) for q in requests.values()),
                                    return_exceptions=True)
    for result in contexts:
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException):
            raise RuntimeError("中英文召回某一路失败，未产生完整双语结果。") from result
    channels = {name: context.chunks for name, context in zip(requests, contexts)}
    pool = fuse_collection_results(channels, rrf_k=60)
    hits = [RagRetrievedChunk(chunk=item.chunk.model_copy(deep=True),
                retrieval_score=item.contributions[0].retrieval_score,
                reason="中英文同条件召回，经RRF融合；原始距离见通道记录。") for item in pool]
    context = RagContext(question=query.question, chunks=hits,
        context_text="\n\n".join(f"[Chunk {i}]\n{h.chunk.content}" for i, h in enumerate(hits, 1)),
        source_count=len({h.chunk.doc_id for h in hits}), status="success" if hits else "empty")
    audit = {"mode": "bilingual", "queries": {k: q.model_dump(mode="json") for k, q in requests.items()},
             "channels": {k: [h.model_dump(mode="json") for h in v] for k, v in channels.items()},
             "fused_candidates": [h.model_dump(mode="json") for h in pool]}
    return context, audit
