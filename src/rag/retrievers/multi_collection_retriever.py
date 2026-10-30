"""多集合召回与倒数排名融合；保留各路原始结果，不混合不同集合的距离值。"""

import asyncio
from time import perf_counter

from pydantic import BaseModel, Field

from src.rag.retrievers.metadata_filter_retriever import MetadataFilterRetriever
from src.rag.schemas import RagChunk, RagQuery, RagRetrievedChunk


class RankContribution(BaseModel):
    """一条来源贡献：集合名称、集合内排名、原始检索值与该路RRF贡献。"""

    collection_name: str
    rank: int
    retrieval_score: float
    rrf_contribution: float


class FusedCandidate(BaseModel):
    """融合候选：标准块、RRF分数、来源贡献；不冒充重排结果或相关概率。"""

    chunk: RagChunk
    rrf_score: float
    contributions: list[RankContribution] = Field(default_factory=list)


class CollectionRetrievalResult(BaseModel):
    """单集合召回记录：集合名称、原始有序候选、实际检索耗时。"""

    collection_name: str
    chunks: list[RagRetrievedChunk]
    retrieval_seconds: float


class MultiCollectionResult(BaseModel):
    """多集合结果：实际问题、各路召回、全部融合候选和融合参数，不构建最终回答上下文。"""

    question: str
    rrf_k: int
    collections: list[CollectionRetrievalResult]
    fused_candidates: list[FusedCandidate]


def fuse_collection_results(
    ranked_results: dict[str, list[RagRetrievedChunk]], *, rrf_k: int = 60,
) -> list[FusedCandidate]:
    """按集合内排名融合候选，相同文档与块编号合并来源贡献。

    参数：ranked_results为集合名称到有序召回列表的映射；rrf_k为平滑常数，
        默认60且须为正整数，每路贡献为1/(rrf_k+rank)，rank从1开始。
    返回值：全部融合候选，按总分降序；同分按首次出现顺序稳定排列，不代表
        语义优劣。单路重复块只计第一次；相同编号但正文不同则报错。
        不比较原始距离，不执行权限过滤或语义去重，不修改输入对象。
    """
    if type(rrf_k) is not int or rrf_k < 1:
        raise ValueError("RRF平滑常数必须为正整数。")
    candidates: dict[tuple[str, str], FusedCandidate] = {}
    for collection_name, results in ranked_results.items():
        seen = set()
        for rank, result in enumerate(results, start=1):
            key = (result.chunk.doc_id, result.chunk.chunk_id)
            if key in candidates and candidates[key].chunk.content != result.chunk.content:
                raise ValueError("相同文档与块编号出现不同正文，不能直接合并。")
            if key in seen:
                continue
            seen.add(key)
            if key not in candidates:
                candidates[key] = FusedCandidate(chunk=result.chunk.model_copy(deep=True), rrf_score=0.0)
            contribution = 1.0 / (rrf_k + rank)
            candidates[key].rrf_score += contribution
            candidates[key].contributions.append(RankContribution(
                collection_name=collection_name, rank=rank,
                retrieval_score=result.retrieval_score, rrf_contribution=contribution,
            ))
    return sorted(candidates.values(), key=lambda item: -item.rrf_score)


class MultiCollectionRetriever:
    """组合项目现有检索器进行多集合查询；只接收上游可信配置的集合对象。"""

    def __init__(self, retrievers: dict[str, MetadataFilterRetriever], *, rrf_k: int = 60):
        """保存检索器映射；retrievers为非空集合白名单，rrf_k为正整数平滑常数；无返回值。"""
        if not retrievers or any(not name.strip() for name in retrievers):
            raise ValueError("必须提供非空且有名称的集合检索器。")
        if type(rrf_k) is not int or rrf_k < 1:
            raise ValueError("RRF平滑常数必须为正整数。")
        self.retrievers = dict(retrievers)
        self.rrf_k = rrf_k

    async def _retrieve_one(
        self, collection_name: str, retriever: MetadataFilterRetriever, query: RagQuery,
    ) -> CollectionRetrievalResult:
        """召回单集合并计时；参数为名称、检索器和问题对象；返回按原排名保存的该路结果。"""
        started_at = perf_counter()
        context = await retriever.async_retrieve(query.model_copy(deep=True))
        return CollectionRetrievalResult(
            collection_name=collection_name, chunks=context.chunks,
            retrieval_seconds=round(perf_counter() - started_at, 4),
        )

    async def async_retrieve(self, query: RagQuery) -> MultiCollectionResult:
        """并发查询已配置集合，再融合全部候选。

        参数：query为标准检索请求，top_k表示每个集合的召回数量。
        返回值：MultiCollectionResult，包含每路结果和全部融合候选；某路故障
            则整体明确失败，不把另一条成功结果伪装成完整查询。
        当前不接受filters：现有单路检索器有放宽策略，未区分硬过滤前不能接收
            权限或物种限制；公共实验无过滤，不涉及用户记忆或权限数据。
        """
        if not query.question.strip():
            raise ValueError("问题不能为空。")
        if query.filters:
            raise ValueError("当前多集合入口暂不接受过滤条件，需先落实不可放宽的硬过滤。")
        results = await asyncio.gather(*(
            self._retrieve_one(name, retriever, query)
            for name, retriever in self.retrievers.items()
        ), return_exceptions=True)
        # 等待各路结束再抛出异常，避免关闭容器时另一条查询仍在使用服务。
        for name, result in zip(self.retrievers, results):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                raise RuntimeError(f"集合{name}召回失败，本次不返回部分成功结果。") from result
        return MultiCollectionResult(
            question=query.question, rrf_k=self.rrf_k, collections=results,
            fused_candidates=fuse_collection_results(
                {result.collection_name: result.chunks for result in results}, rrf_k=self.rrf_k,
            ),
        )

    async def async_retrieve_subqueries(
        self, queries: dict[str, RagQuery],
    ) -> dict[str, MultiCollectionResult]:
        """逐个子问题调用多集合召回，保留问题到候选的映射，不跨子问题融合。

        参数：queries为非空子问题编号到检索请求的映射，可由上游人工或已验证的
            问题拆解结果构建；本方法不调用模型拆解或验证拆分是否忠实。
        返回值：相同编号到MultiCollectionResult的映射。子问题顺序执行以限制
            同时查询数量，每个子问题内部仍并发查询集合；不加载新模型。
        """
        if not queries or any(not key.strip() for key in queries):
            raise ValueError("必须提供有编号的非空子问题列表。")
        if any(not query.question.strip() or query.filters for query in queries.values()):
            raise ValueError("当前子问题必须非空且不包含过滤条件。")
        results = {}
        for query_id, query in queries.items():
            results[query_id] = await self.async_retrieve(query)
        return results

    async def async_retrieve_routed_subqueries(
        self, routes: dict[str, dict[str, RagQuery]],
    ) -> dict[str, MultiCollectionResult]:
        """按已验证的人工或上游路由分别查询集合，保持各子问题独立。

        参数：routes为子问题编号到集合名称、RagQuery的映射；集合必须已注册，
            同一子问题各路问题文本一致；带条件的检索器必须明确关闭条件放宽。
        返回值：按子问题编号保存的标准多集合结果；组内融合，组间不融合。
            所有路由在首次查询前校验；某路故障明确报错，不伪装成空结果。
            本方法不理解自然语言、不负责登录鉴权，也不是自动路由模型。
        """
        if not routes:
            raise ValueError("路由子问题不能为空。")
        for query_id, targets in routes.items():
            if not query_id.strip() or not targets:
                raise ValueError("每个子问题必须有编号和目标集合。")
            questions = set()
            for name, query in targets.items():
                if name not in self.retrievers:
                    raise ValueError(f"集合{name}不在已注册白名单中。")
                if not query.question.strip():
                    raise ValueError("问题不能为空。")
                questions.add(query.question)
                if query.filters and getattr(self.retrievers[name], "allow_filter_relaxation", None) is not False:
                    raise ValueError("带条件路由必须使用明确禁止放宽条件的检索器。")
            if len(questions) != 1:
                raise ValueError("同一子问题各集合必须使用相同问题文本。")
        output = {}
        for query_id, targets in routes.items():
            results = await asyncio.gather(*(
                self._retrieve_one(name, self.retrievers[name], query)
                for name, query in targets.items()
            ), return_exceptions=True)
            for name, result in zip(targets, results):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                if isinstance(result, BaseException):
                    raise RuntimeError(f"子问题{query_id}集合{name}召回失败。") from result
            output[query_id] = MultiCollectionResult(
                question=next(iter(targets.values())).question, rrf_k=self.rrf_k,
                collections=results,
                fused_candidates=fuse_collection_results(
                    {item.collection_name: item.chunks for item in results}, rrf_k=self.rrf_k,
                ),
            )
        return output
