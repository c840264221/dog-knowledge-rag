"""受控双语召回与重排管线：依赖外部注入，返回真实候选和完整排序。"""

import asyncio
import math
from time import perf_counter

from src.rag.retrievers.multi_collection_retriever import fuse_collection_results
from src.rag.schemas import RagQuery, RagRetrievedChunk
from src.rag.query_builders.english_query_rewriter import translate_retrieval_question


async def retrieve_auto_bilingual(query: RagQuery, *, llm, retriever, reranker) -> dict:
    """自动翻译后运行现有双语检索，不依赖规划器提供英文。

    参数：query为宿主构造的标准请求；llm、retriever、reranker为注入服务。
    返回值：原管线结果，新增翻译记录和耗时；保留原问题用于重排，不变更身份与过滤条件。
        当前沿用公共资料管线无过滤限制；翻译失败向上传播，不静默降级。
    """
    if not query.question.strip() or query.filters:
        raise ValueError("自动双语公共检索需要非空问题且不接受过滤条件。")
    started = perf_counter()
    translation = await translate_retrieval_question(llm, query.question)
    seconds = round(perf_counter() - started, 4)
    result = await retrieve_bilingual(query, translation["english_question"],
                                      retriever=retriever, reranker=reranker)
    return {**result, "translation": translation, "translation_seconds": seconds}


def rerank_pool(question: str, pool: list, reranker) -> list[RagRetrievedChunk]:
    """对完整去重池用原问题评分。

    参数：question为原中文问题；pool为融合候选；reranker为注入的模型。
    返回值：按模型分值降序的标准检索块；无候选返回空，不生成新证据。
        retrieval_score保留首个来源的原值，仅供追踪；final_score使用重排输出。
    """
    if not pool:
        return []
    scores = reranker.predict([(question, item.chunk.content) for item in pool],
                              batch_size=8, show_progress_bar=False)
    if len(scores) != len(pool):
        raise ValueError("重排分数数量与候选不一致。")
    hits = []
    for item, value in zip(pool, scores):
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("重排评分无效。")
        hits.append(RagRetrievedChunk(
            chunk=item.chunk.model_copy(deep=True), retrieval_score=item.contributions[0].retrieval_score,
            rerank_score=score, final_score=score, reason="双语真实召回去重后，按原中文问题重排。",
        ))
    return sorted(hits, key=lambda hit: -hit.rerank_score)


async def retrieve_bilingual(query: RagQuery, english_question: str, *, retriever, reranker) -> dict:
    """实际查询中文与已审核英文、融合去重，再对完整候选重排。

    参数：query为标准中文请求；english_question为已审核的英文检索文本；
        retriever与reranker为项目服务注入，不自行创建模型。
    返回值：含两路真实结果、完整融合池、完整重排列表与耗时的字典。
        当前仅支持无过滤公共资料；不处理自动翻译、集合路由、权限或最终上下文选择。
    """
    if not query.question.strip() or not english_question.strip() or query.filters:
        raise ValueError("需要非空中英文问题，当前公共资料管线不接受过滤条件。")
    started = perf_counter()
    queries = {"original_zh": query.model_copy(deep=True),
               "rewritten_en": query.model_copy(update={"question": english_question}, deep=True)}
    contexts = await asyncio.gather(*(
        retriever.async_retrieve(item) for item in queries.values()
    ), return_exceptions=True)
    for result in contexts:
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException):
            raise RuntimeError("双语召回某一路失败，不冒充完整成功。") from result
    channels = {name: context.chunks for name, context in zip(queries, contexts)}
    pool = fuse_collection_results(channels, rrf_k=60)
    retrieval_seconds = round(perf_counter() - started, 4)
    started = perf_counter()
    ranked = await asyncio.to_thread(rerank_pool, query.question, pool, reranker)
    return {"channels": channels, "fused_candidates": pool, "reranked_candidates": ranked,
            "retrieval_and_fusion_seconds": retrieval_seconds,
            "rerank_seconds": round(perf_counter() - started, 4)}
