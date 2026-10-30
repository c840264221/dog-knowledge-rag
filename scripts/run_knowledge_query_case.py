"""通过原知识子图验证统一集合选择和多子查询，只读知识库，不运行工具或记忆。"""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4


async def run(question: str) -> dict:
    """运行统一知识子图；参数为自然语言问题；返回完整状态及报告路径，失败保留已完成节点。"""
    from src.settings import settings
    from src.runtime.container.core import RuntimeContainer
    from src.runtime.container.providers.embedding_provider import EmbeddingProvider
    from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
    from src.runtime.container.providers.retriever_provider import RetrieverProvider
    from src.runtime.container.providers.reranker_provider import RerankerProvider
    from src.runtime.container.providers.llm_provider import LLMProvider
    from src.runtime.context import RuntimeContext, runtime_ctx
    from src.agents.dog_knowledge_agent.agent import build_dog_knowledge_agent
    from src.graph.graph_run import create_initial_state

    run_id = uuid4().hex
    folder = Path(__file__).resolve().parents[1] / "output/learning/knowledge_queries" / run_id
    folder.mkdir(parents=True)
    report_path = folder / "report.json"
    report = {"run_id": run_id, "question": question, "status": "starting",
        "catalog": settings.rag.knowledge_collections, "bilingual_enabled": settings.rag.bilingual_enabled,
        "configured_reranker": settings.reranker.model_name,
        "report_path": str(report_path), "notice": "复用原知识子图的只读验证入口，未经过完整主图粗路由，不写向量库或账本。"}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    context = RuntimeContext(trace_id=run_id, user_id="knowledge_case_user", session_id=run_id,
                             component="knowledge_query_case")
    previous = runtime_ctx.get()
    runtime_ctx.set(context)
    container = RuntimeContainer()
    embedding = EmbeddingProvider(enable_memory=False)
    providers = {key: VectorStoreProvider(embedding, collection_name=spec["collection_name"],
        enable_memory=False, create_collection_if_not_exists=False)
        for key, spec in settings.rag.knowledge_collections.items()}
    if not providers:
        raise ValueError("知识集合目录不能为空。")
    container.register("embedding", embedding)
    retriever = RetrieverProvider(next(iter(providers.values())), collection_providers=providers,
        collection_descriptions={key: spec["description"] for key, spec in settings.rag.knowledge_collections.items()})
    container.register("retriever", retriever)
    container.register("reranker", RerankerProvider())
    print("运行编号：", run_id, "\n报告：", report_path, flush=True)
    state = create_initial_state(question, run_id, session_id=run_id)
    state["user_id"] = context.user_id
    try:
        await context.startup()
        await container.startup()
        # 只使用统一服务的主模型，避免独立验证入口预热无关的本地备用服务。
        container.register("llm", LLMProvider())
        graph = build_dog_knowledge_agent(llm_provider=container.get("llm"), retriever_provider=retriever,
            reranker_provider=container.get("reranker"))
        # 消费原图节点更新，同时保留失败前的现场，不另建查询Agent。
        async for update in graph.astream(state, stream_mode="updates"):
            for node, values in update.items():
                if isinstance(values, dict):
                    state.update(values)
                    print("完成节点：", node, flush=True)
            report["state"] = state
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        report["status"] = "completed"
        print("子查询：", json.dumps(state.get("rag_sub_queries"), ensure_ascii=False, indent=2), flush=True)
        print("证据分配：", json.dumps(state.get("rag_sub_query_coverage"), ensure_ascii=False, indent=2), flush=True)
        print("最终回答：\n", state.get("final_answer") or state.get("answer") or "", flush=True)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        report["state"] = state
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        try:
            await container.shutdown()
        finally:
            try:
                await context.shutdown()
            finally:
                runtime_ctx.set(previous)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="原知识子图的多集合、多子问题真实运行，会调用模型。")
    parser.add_argument("--question", default="金毛日常运动有什么要求？接一只新猫回家前，房间里需要准备什么？请分别回答并给出资料来源。")
    args = parser.parse_args()
    asyncio.run(run(args.question))
