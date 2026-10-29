"""统一知识检索：子问题内多集合双语召回，跨问题保留覆盖与引用。"""

import asyncio
import math
from time import perf_counter

from src.rag.context_builders.cited_context_builder import build_cited_context
from src.rag.retrievers.multi_collection_retriever import fuse_collection_results
from src.rag.schemas import RagQuery, RagContext, RagRetrievedChunk


class InjectedKnowledgeCatalog:
    """将已有检索器适配为统一目录，供独立案例与测试复用，不创建模型或向量库。"""

    def __init__(self, retrievers: dict, descriptions: dict[str, str]):
        """保存注入对象；参数为标识到检索器及内容说明；无返回值，拒绝目录不一致。"""
        if not retrievers or set(retrievers) != set(descriptions):
            raise ValueError("注入检索器与目录说明必须一致且非空。")
        self.retrievers, self.descriptions = dict(retrievers), dict(descriptions)

    def knowledge_catalog(self) -> dict[str, str]:
        """列出可查询集合；无参数；返回说明副本，不提供路径。"""
        return dict(self.descriptions)

    def get_collection_retriever(self, collection_id: str):
        """取得已注入检索器；参数为目录标识；返回原对象，未知标识拒绝。"""
        if collection_id not in self.retrievers:
            raise ValueError("未知集合标识。")
        return self.retrievers[collection_id]


def combine_group_contexts(question: str, groups: list[dict]) -> RagContext:
    """合并候选供既有质量节点检查；参数为整体问题与分组结果；返回完整去重池，不做最终选取。"""
    hits = {}
    for group in groups:
        for raw in group.get("candidates", []):
            hit = RagRetrievedChunk.model_validate(raw)
            key = (hit.chunk.doc_id, hit.chunk.chunk_id)
            if key in hits and hits[key].chunk.content != hit.chunk.content:
                raise ValueError("跨组相同块标识的正文不一致。")
            hits.setdefault(key, hit)
    selected = list(hits.values())
    return RagContext(question=question, chunks=selected,
        context_text="\n\n".join(h.chunk.content for h in selected),
        source_count=len({h.chunk.doc_id for h in selected}), status="success" if selected else "empty")


async def retrieve_groups(groups: list[dict], *, provider, bilingual: bool,
                          concurrency: int, max_groups: int) -> list[dict]:
    """执行可信目录中的分组查询。

    参数：groups含query_id、标准query、英文及集合标识；provider提供目录及检索器；
        bilingual为宿主开关，concurrency限制实际底层调用，max_groups限制子问题数。
    返回值：各组通道、完整候选及耗时；底层失败则抛出，不以空资料伪装服务失败。
        全部路由在首个请求前检查；相同身份和每组过滤条件用于该组所有通道。
    """
    if not groups or len(groups) > max_groups or concurrency < 1:
        raise ValueError("分组数量或并发配置无效。")
    catalog = provider.knowledge_catalog()
    prepared, ids = [], set()
    for group in groups:
        query_id = group["query_id"]
        names = group["collection_ids"]
        if query_id in ids or not names or len(names) > 4 or len(set(names)) != len(names) or not set(names).issubset(catalog):
            raise ValueError("子查询编号重复、集合为空或包含未知集合。")
        ids.add(query_id)
        query = RagQuery.model_validate(group["query"])
        requests = {}
        english = str(group.get("english_question") or "").strip()
        for name in names:
            retriever = provider.get_collection_retriever(name)
            if getattr(retriever, "allow_filter_relaxation", None) is not False:
                raise ValueError("统一知识检索要求禁止放宽过滤条件。")
            requests[f"{name}/zh"] = (retriever, query.model_copy(deep=True))
            if bilingual and english:
                requests[f"{name}/en"] = (retriever, query.model_copy(update={"question": english}, deep=True))
        prepared.append((group, requests))
    semaphore = asyncio.Semaphore(concurrency)

    async def one_call(retriever, query):
        """限制一次底层调用；参数为检索器和标准请求；返回单通道上下文。"""
        async with semaphore:
            return await retriever.async_retrieve(query=query)

    async def one_group(group, requests):
        """查询一组并融合；参数为子问题及已验证通道；返回分组结果，不修改共享状态。"""
        started = perf_counter()
        results = await asyncio.gather(*(one_call(r, q) for r, q in requests.values()), return_exceptions=True)
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                raise RuntimeError(f"子查询{group['query_id']}召回失败。") from result
        channels = {name: result.chunks for name, result in zip(requests, results)}
        pool = fuse_collection_results(channels)
        hits = [RagRetrievedChunk(chunk=item.chunk, retrieval_score=item.contributions[0].retrieval_score,
                                  reason="本子问题多集合/语言通道RRF融合。") for item in pool]
        return {**group, "status": "retrieved" if hits else "empty",
            "channels": {name: [hit.model_dump(mode="json") for hit in values] for name, values in channels.items()},
            "candidates": [hit.model_dump(mode="json") for hit in hits],
            "fusion": [item.model_dump(mode="json") for item in pool],
            "translation_mode": "bilingual" if bilingual and group.get("english_question") else "original_only",
            "retrieval_seconds": round(perf_counter() - started, 4)}

    results = await asyncio.gather(*(one_group(g, r) for g, r in prepared), return_exceptions=True)
    for result in results:
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException):
            raise result
    return results


def rank_and_select_groups(question: str, groups: list[dict], *, reranker,
                           max_chunks: int, max_characters: int) -> dict:
    """分组重排并轮流分配上下文。

    参数：question为整体问题，groups为分组召回结果，reranker为注入模型；
        max_chunks和max_characters分别限制总块数及资料字符数。
    返回值：标准上下文、分组完整排名、来源映射与覆盖记录；覆盖仅表示有入选证据，
        不证明语义充分。不比较不同问题的重排分数，不以全局前五挤掉其他问题。
    """
    ranked_groups = []
    for group in groups:
        usable = group.get("quality_evaluation", {}).get("is_usable", True)
        hits = [RagRetrievedChunk.model_validate(raw) for raw in group["candidates"]] if usable else []
        if hits:
            scores = reranker.predict([(group["query"]["question"], h.chunk.content) for h in hits],
                                      batch_size=8, show_progress_bar=False)
            if len(scores) != len(hits) or any(not math.isfinite(float(s)) for s in scores):
                raise ValueError("分组重排评分无效。")
            hits = [h.model_copy(update={"rerank_score": float(s), "final_score": float(s)})
                    for h, s in zip(hits, scores)]
            hits.sort(key=lambda hit: -hit.rerank_score)
        ranked_groups.append({**group, "ranked_candidates": [h.model_dump(mode="json") for h in hits]})
    return select_group_context(question, ranked_groups, max_chunks=max_chunks, max_characters=max_characters)


def select_group_context(question: str, ranked_groups: list[dict], *,
                         max_chunks: int, max_characters: int) -> dict:
    """按组内既有顺序轮流选证据，供实际重排与未重排基线共用。

    参数：question为完整问题；ranked_groups含每组ranked_candidates；max_chunks和
        max_characters为总预算。返回上下文、来源映射、分组覆盖及选择记录，不调用模型。
    """
    ordered, seen = [], {}
    for index in range(max((len(g["ranked_candidates"]) for g in ranked_groups), default=0)):
        for group in ranked_groups:
            if index >= len(group["ranked_candidates"]):
                continue
            hit = RagRetrievedChunk.model_validate(group["ranked_candidates"][index])
            key = (hit.chunk.doc_id, hit.chunk.chunk_id)
            if key not in seen:
                seen[key] = hit
                ordered.append(hit)
            elif seen[key].chunk.content != hit.chunk.content:
                raise ValueError("同一块标识出现不同正文。")
    context, citations, decisions = build_cited_context(question, ordered,
        max_chunks=max_chunks, max_characters=max_characters)
    coverage = []
    for group in ranked_groups:
        keys = {(raw["chunk"]["doc_id"], raw["chunk"]["chunk_id"]) for raw in group["ranked_candidates"]}
        used = [sid for sid, source in citations.items() if (source["doc_id"], source["chunk_id"]) in keys]
        coverage.append({"query_id": group["query_id"], "question": group["query"]["question"],
            "citation_ids": used, "status": "selected_evidence" if used else
                "quality_rejected" if group.get("quality_evaluation", {}).get("is_usable") is False else
                "no_candidates" if not keys else "not_selected_within_budget"})
    # 资料预算仅包含来源与正文；覆盖说明独立传给生成节点，不混进原文。
    return {"rag_context": context.model_dump(mode="json"), "groups": ranked_groups,
            "citation_map": citations, "coverage": coverage, "selection_decisions": decisions}
