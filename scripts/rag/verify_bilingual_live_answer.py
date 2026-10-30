"""真实双语召回接项目回答节点：受控案例入口，不改默认主图或知识库。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from src.graph.nodes.generate_node import build_generate_node
from src.rag.context_builders.cited_context_builder import build_cited_context, check_citation_ids
from src.rag.context_builders.generation_context_builder import resolve_generation_context
from src.rag.pipelines.bilingual_retrieval_pipeline import retrieve_bilingual
from src.rag.retrievers.metadata_filter_retriever import MetadataFilterRetriever
from src.rag.schemas import RagQuery
from src.runtime.container.core import RuntimeContainer
from src.runtime.container.providers.embedding_provider import EmbeddingProvider
from src.runtime.container.providers.llm_provider import LLMProvider
from src.runtime.container.providers.reranker_provider import RerankerProvider
from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
from src.runtime.context import RuntimeContext, runtime_ctx
from src.settings import settings


CASE_SCOPES = {"golden_exercise": "Golden Retriever", "cat_dog_escape": None}


class RecordingLLMProvider(LLMProvider):
    """只为本次实验记录真实回答提示词，沿用项目模型调用及重试逻辑。"""

    def __init__(self):
        """初始化原服务及调用记录列表；无参数；无返回值，不立即调用模型。"""
        super().__init__()
        self.calls = []

    async def safe_ainvoke(self, llm, prompt, **kwargs):
        """记录实际提示词及原始输出；llm、prompt、kwargs透传原服务；返回原模型响应。

        保留原重试和备用模型策略，一次逻辑调用可能含多个底层请求，不承诺只调用一次。
        """
        record = {"actual_prompt": prompt, "status": "running"}
        self.calls.append(record)
        try:
            response = await super().safe_ainvoke(llm, prompt, **kwargs)
            record.update(status="returned", raw_content=getattr(response, "content", response),
                          usage_metadata=getattr(response, "usage_metadata", None),
                          returned_model=(getattr(response, "response_metadata", {}) or {}).get("model_name"))
            return response
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__)
            raise


def save_report(report: dict) -> None:
    """保存阶段结果；report含report_path绝对路径；无返回值，不改源报告。"""
    Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


async def verify_live_answer(source_report: Path, case_id: str, *, allow_legacy_model: bool = False) -> dict:
    """读取受控案例已审核英文，重新召回、重排、构建上下文并调用现有回答节点。

    参数：source_report为完成并审核的原pilot_report；case_id只允许金毛运动或猫狗
        初次接触；allow_legacy_model为显式允许未知建库签名的旧集合实验。
    返回值：包含新召回、上下文决策、实际提示词、答案和引用映射的报告。
        不自动路由或翻译新问题、不读历史候选作为本轮结果；不写向量库、记忆或检查点。
    """
    if not allow_legacy_model or case_id not in CASE_SCOPES:
        raise ValueError("需明确允许旧模型实验，并选择本入口支持的固定案例。")
    if not (settings.path.CHROMA_DB_DIR / "chroma.sqlite3").is_file():
        raise FileNotFoundError("项目向量库不存在。")
    previous = json.loads(source_report.read_text(encoding="utf-8"))
    if previous.get("status") != "completed" or previous.get("human_review") != "user_approved_batch":
        raise ValueError("请提供完成且已审核英文的原始报告。")
    matches = [case for case in previous["dataset"]["cases"] if case["case_id"] == case_id]
    translations = [item for item in previous["rewrite_batch"]["items"] if item["case_id"] == case_id]
    if len(matches) != 1 or len(translations) != 1:
        raise ValueError("案例或英文文本缺失、重复。")
    case = matches[0]
    english = translations[0]["retrieval_question"].strip()
    if not english or case["collection_name"] not in {"breed_no_heading_v1", "pet_care_v1"}:
        raise ValueError("英文为空或集合不在白名单内。")
    model_config = {"provider": settings.embedding.provider, "model": settings.embedding.model_name,
                    "normalize": settings.embedding.normalize_embeddings}
    if model_config != previous["embedding_config"]:
        raise ValueError("当前向量配置与原实验不同，本次保持原配置。")
    signature = hashlib.sha256(json.dumps(model_config, sort_keys=True).encode()).hexdigest()
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "output/rag_live_answers" / uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "case_id": case_id, "question": case["question"],
              "english_retrieval_question": english, "collection_name": case["collection_name"],
              "embedding_config": model_config, "configured_reranker": "BAAI/bge-reranker-v2-m3",
              "manual_required_dog_name": CASE_SCOPES[case_id],
              "source_report": str(source_report.resolve()),
              "report_path": str(output_dir / "live_answer_report.json"),
              "review_status": "needs_human_review",
              "notice": "真实重新查询与现有回答节点接入验证，固定案例、英文与集合人工配置，不代表主图自动路由或全新问题自动改写。"}
    save_report(report)
    container = RuntimeContainer()
    embedding = EmbeddingProvider(enable_memory=False)
    container.register("embedding", embedding)
    container.register("vectors", VectorStoreProvider(
        embedding, collection_name=case["collection_name"], enable_memory=False,
        create_collection_if_not_exists=False,
    ))
    container.register("reranker", RerankerProvider(reranker_settings=settings.reranker.model_copy(
        update={"model_name": report["configured_reranker"]})))
    llm = RecordingLLMProvider()
    report["llm_calls"] = llm.calls
    runtime = RuntimeContext(trace_id=output_dir.name, user_id="rag_live_verification",
                             session_id=output_dir.name, component="verify_bilingual_live_answer")
    prior_runtime = runtime_ctx.get()
    runtime_ctx.set(runtime)
    try:
        await runtime.startup()
        await container.startup()
        # 按需提供回答模型，不启动与本实验无关的本地中文模型或记忆服务。
        container.register("llm", llm)
        database = container.get("vectors").db
        records = await asyncio.to_thread(database.get, include=["metadatas"])
        signatures = [(metadata or {}).get("embedding_signature") for metadata in records["metadatas"]]
        if not signatures or any(value is not None and value != signature for value in signatures):
            raise ValueError("集合为空或含已知不匹配的向量配置。")
        if case["collection_name"] == "pet_care_v1" and any(value is None for value in signatures):
            raise ValueError("新养护集合不允许缺少模型签名。")
        report["status"] = "retrieving"
        query = RagQuery(question=case["question"], top_k=5, filters={}, user_id=runtime.user_id)
        result = await retrieve_bilingual(
            query, english, retriever=MetadataFilterRetriever(database, allow_filter_relaxation=False),
            reranker=container.get("reranker").reranker,
        )
        report["retrieval"] = {
            "source": "fresh_database_queries", "top_k_per_channel": 5,
            "channels": {name: [hit.model_dump(mode="json") for hit in hits]
                         for name, hits in result["channels"].items()},
            "fused_candidates": [hit.model_dump(mode="json") for hit in result["fused_candidates"]],
            "reranked_candidates": [hit.model_dump(mode="json") for hit in result["reranked_candidates"]],
            "retrieval_and_fusion_seconds": result["retrieval_and_fusion_seconds"],
            "rerank_seconds": result["rerank_seconds"],
        }
        context, citations, decisions = build_cited_context(
            case["question"], result["reranked_candidates"], required_dog_name=CASE_SCOPES[case_id],
        )
        report.update(rag_context=context.model_dump(mode="json"), citation_map=citations,
                      selection_decisions=decisions)
        # 预期证据只在选择完成后检查，不参与任何选择逻辑。
        report["expected_evidence_in_context"] = any(hit.chunk.chunk_id in case["expected_chunk_ids"]
                                                     for hit in context.chunks)
        save_report(report)
        if not context.chunks:
            report["status"] = "insufficient_evidence_no_generation"
            return report
        state = {"question": case["question"], "user_id": runtime.user_id, "intent": "ask_info",
                 "current_agent": "dog_knowledge_agent", "rag_query": query.model_dump(mode="python"),
                 "rag_context": context.model_dump(mode="python"), "docs": [], "messages": [],
                 "memory_context": ""}
        actual_context, context_source = resolve_generation_context(state)
        if actual_context != context.context_text or context_source != "rag_context":
            raise RuntimeError("现有回答节点实际读取的上下文不符合预期。")
        report.update(actual_context=actual_context, context_source=context_source, status="generating")
        save_report(report)
        print("真实检索与上下文已保存，开始调用现有回答节点，可能产生费用。", flush=True)
        answer_node = build_generate_node(llm_provider=container.get("llm"))
        update = await answer_node(state)
        report.update(final_answer=update["final_answer"], answer_strategy=update["answer_strategy"])
        if report["final_answer"] == "调用 LLM 失败":
            raise RuntimeError("回答模型失败，不能把兜底文本视为成功回答。")
        if not llm.calls or actual_context not in llm.calls[-1]["actual_prompt"]:
            raise RuntimeError("实际发送提示词中没有预期上下文。")
        report["citation_check"] = check_citation_ids(report["final_answer"], citations)
        report["status"] = "generated_pending_human_review"
        report["review_questions"] = [
            "主要事实是否能在实际入选正文中找到，而不只是引用编号存在？",
            "是否错误使用其他犬种、遗漏问题的一部分或补造数字和条件？",
            "资料不足时是否明确说明，引用是否支持它所在的那句话？",
        ]
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        save_report(report)
        try:
            await container.shutdown()
        finally:
            try:
                await runtime.shutdown()
            finally:
                runtime_ctx.set(prior_runtime)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定案例真实召回、重排、上下文与现有回答节点验证。")
    parser.add_argument("--source-report", required=True, type=Path, help="原始已审核pilot_report.json路径。")
    parser.add_argument("--case", choices=list(CASE_SCOPES), default="golden_exercise", help="固定案例编号，默认金毛运动。")
    parser.add_argument("--allow-legacy-model", action="store_true", help="允许旧集合实验。")
    args = parser.parse_args()
    report = asyncio.run(verify_live_answer(args.source_report, args.case, allow_legacy_model=args.allow_legacy_model))
    print("状态：", report["status"])
    print(report.get("final_answer", "没有调用回答模型。"))
    print("引用编号检查：", json.dumps(report.get("citation_check", {}), ensure_ascii=False))
    print("报告：", report["report_path"])
