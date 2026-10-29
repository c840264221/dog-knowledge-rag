"""两个独立知识步骤的并发与引用隔离案例；不改主图或知识库，不调用写入工具。"""

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4


DEFAULT_QUESTION = (
    "准备接猫回家，路上运输要准备什么、注意什么？家里还有一只狗，"
    "猫狗第一次接触要注意什么？请按这两个主题分别回答，并给出资料出处。"
)


def save_report(path: Path, report: dict) -> None:
    """保存运行现场；path为本轮路径、report为阶段数据；无返回值，不作为恢复授权。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def validate_parallel_plan(plan) -> None:
    """检查本案例覆盖目标；plan为模型正式计划；通过无返回值，否则拒绝，不替模型改计划。

    仅为两个独立知识问题的案例验收条件，不是所有任务都必须拆成两步或没有依赖。
    """
    if len(plan.steps) != 2:
        raise ValueError("本案例检查两个独立知识步骤，请检查模型是否遗漏或多拆了任务。")
    for step in plan.steps:
        if step.assigned_agent != "pet_knowledge_agent" or step.execution_mode != "capability" or step.depends_on:
            raise ValueError("本案例应为两个无依赖的知识步骤，不自动删除模型声明的依赖。")
        if not isinstance(step.input_data.get("english_retrieval_question"), str) or not step.input_data["english_retrieval_question"].strip():
            raise ValueError("知识步骤缺少需审核的英文检索文本。")


def summarize_execution_overlap(worker_runs: list[dict]) -> dict:
    """计算步骤运行区间交集。

    参数worker_runs为带单调时钟起止值的记录；返回重叠秒数和每步耗时。
    只证明执行者运行区间重叠，不证明CPU同时计算，也不是与串行实测比较的加速比。
    """
    finished = [item for item in worker_runs if "finished_seconds" in item]
    overlap = 0.0
    if len(finished) == 2:
        overlap = max(0.0, min(item["finished_seconds"] for item in finished)
                      - max(item["started_seconds"] for item in finished))
    return {"overlap_seconds": round(overlap, 4), "execution_intervals_overlap": overlap > 0,
            "worker_durations_seconds": {item["step_id"]: round(item["finished_seconds"] - item["started_seconds"], 4)
                                         for item in finished},
            "notice": "区间重叠不代表所有检索、重排和模型计算均同时发生，也不等于性能提升率。"}


async def run(question: str, collection_name: str, parallelism: int) -> None:
    """运行并发案例。

    参数：question为两个主题的需求，collection_name为宿主选定的公共集合，parallelism为并发上限。
    返回：无，保存真实检索、执行起止时间和最终聚合结果；不支持退出后恢复。
    """
    from src.settings import settings
    from src.runtime.container.core import RuntimeContainer
    from src.runtime.container.providers.embedding_provider import EmbeddingProvider
    from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
    from src.runtime.container.providers.reranker_provider import RerankerProvider
    from src.runtime.container.providers.llm_provider import LLMProvider
    from src.runtime.context import RuntimeContext, runtime_ctx
    from src.rag.retrievers.metadata_filter_retriever import MetadataFilterRetriever
    from src.agents.pet_knowledge_agent.agent import build_pet_knowledge_runner
    from src.agents.collaboration.planner import PlannerAgent
    from src.agents.collaboration.scheduler import MultiAgentTaskScheduler
    from src.agents.collaboration.aggregator import ResultAggregator
    from src.agents.collaboration.workers.graph_agent_worker_adapter import GraphAgentWorkerAdapter

    run_id = uuid4().hex
    path = Path(__file__).resolve().parents[1] / "output/learning/pet_knowledge_parallel" / run_id / "report.json"
    context = RuntimeContext(trace_id=run_id, session_id=run_id, user_id="pet_parallel_learning_user",
                             component="pet_knowledge_parallel_case")
    previous = runtime_ctx.get()
    runtime_ctx.set(context)
    container = RuntimeContainer()
    embedding = EmbeddingProvider(enable_memory=False)
    container.register("embedding", embedding)
    container.register("vectors", VectorStoreProvider(embedding, collection_name=collection_name,
                       enable_memory=False, create_collection_if_not_exists=False))
    container.register("reranker", RerankerProvider(reranker_settings=settings.reranker.model_copy(
        update={"model_name": "BAAI/bge-reranker-v2-m3"})))
    report = {"run_id": run_id, "question": question, "collection_name": collection_name,
              "parallelism": parallelism, "status": "starting", "worker_runs": [], "stages": [],
              "quote_validation_mode": "audit",
              "runtime_environment": context.time_context.as_dict(),
              "notice": "复用同一种无会话状态的知识执行者并发处理独立步骤；不是新增两个专家智能体。"}
    print(f"本轮编号：{run_id}\n报告：{path}", flush=True)
    try:
        await context.startup()
        await container.startup()
        container.register("llm", LLMProvider())
        llm = container.get("llm")
        database = container.get("vectors").db
        config = {"provider": settings.embedding.provider, "model": settings.embedding.model_name,
                  "normalize": settings.embedding.normalize_embeddings}
        signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        records = await asyncio.to_thread(database.get, include=["metadatas"])
        if not records["ids"] or any((item or {}).get("embedding_signature") != signature
                                     for item in records["metadatas"]):
            raise ValueError("集合为空或向量模型签名不匹配，停止执行。")
        report["embedding_config"] = config
        runner = build_pet_knowledge_runner(llm_provider=llm,
            retriever=MetadataFilterRetriever(database, allow_filter_relaxation=False),
            reranker=container.get("reranker").reranker, user_id=context.user_id,
            quote_validation_mode="audit")
        planner = PlannerAgent(llm_provider=llm, maximum_steps=2, maximum_plan_attempts=2,
            available_agents={"pet_knowledge_agent": "查询公共宠物养护资料，使用capability模式。"
                "每步input_data提供question、中文retrieval_question及忠实对应的english_retrieval_question。"
                "仅按当前主题查询和回答，不需要其他主题的执行结果时不声明依赖；保留条件及来源引用。"})
        plan = await planner.create_plan(objective=question, context={"tool_catalog": [],
            "knowledge_scope": {"collection_name": collection_name, "content": "公共宠物养护资料"}})
        report.update(status="plan_review", plan=plan.model_dump(mode="json"))
        save_report(path, report)
        print("\n模型生成的计划：\n" + json.dumps(report["plan"], ensure_ascii=False, indent=2), flush=True)
        if plan.status != "planned" or plan.requires_user_input:
            report["status"] = plan.status
            print("计划仍需澄清，请查看报告。", flush=True)
            return
        validate_parallel_plan(plan)
        approval = await asyncio.to_thread(input, "核对两个主题、无依赖及检索原意；输入 执行 继续：")
        if approval.strip() != "执行":
            report["status"] = "not_started"
            return
        started = perf_counter()

        async def recorded_runner(state: dict) -> dict:
            """记录单步真实运行；state为适配器创建的独立状态；返回知识结果，失败记录后继续抛出。"""
            entry = {"step_id": state["multi_agent_step_id"], "started_seconds": perf_counter() - started,
                     "question": state["question"], "retrieval_question": state["retrieval_question"],
                     "dependencies": deepcopy(state.get("multi_agent_dependency_results") or {}), "status": "running"}
            report["worker_runs"].append(entry)
            print(f"开始步骤：{entry['step_id']}，启动于第{entry['started_seconds']:.3f}秒", flush=True)
            save_report(path, report)
            try:
                result = await runner(state)
                entry.update(status="completed", result=result)
                return result
            except Exception as exc:
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                diagnostics = getattr(exc, "diagnostics", None)
                if isinstance(diagnostics, dict):
                    entry["failure_diagnostics"] = deepcopy(diagnostics)
                raise
            finally:
                entry["finished_seconds"] = perf_counter() - started
                save_report(path, report)

        worker = GraphAgentWorkerAdapter(agent_name="pet_knowledge_agent", runner=recorded_runner,
            output_fields=("final_answer", "rag_context", "citation_map", "evidence_ids", "knowledge_handoff", "limitations",
                           "evidence_validation"))
        result = await MultiAgentTaskScheduler(workers={"pet_knowledge_agent": worker},
            maximum_parallel_steps=parallelism, maximum_step_attempts=1).execute(plan)
        report["scheduling_seconds"] = round(perf_counter() - started, 4)
        report["execution_overlap"] = summarize_execution_overlap(report["worker_runs"])
        report["stages"].append(result.model_dump(mode="json"))
        report["status"] = result.status
        save_report(path, report)
        print("步骤状态：" + "，".join(f"{item.step_id}={item.status}" for item in result.task_results), flush=True)
        print("执行区间：" + json.dumps(report["execution_overlap"], ensure_ascii=False), flush=True)
        if result.status in {"running", "partial"}:
            result = await ResultAggregator(llm_provider=llm).aggregate(result)
            report["stages"].append(result.model_dump(mode="json"))
            report["status"] = result.status
            print("\n最终回答：\n" + str(result.final_answer or ""), flush=True)
        else:
            print("未完成，不自动重跑；请查看保留的报告。", flush=True)
        report["human_review_required"] = ["两个步骤是否分别回答各自主题？", "两个步骤是否存在无依据的内容？",
            "引用是否分别映射到各步骤真实来源？", "结果是否保留适用条件与限制？"]
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_report(path, report)
        try:
            await container.shutdown()
        finally:
            try:
                await context.shutdown()
            finally:
                runtime_ctx.set(previous)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="两个独立知识步骤的并发和引用隔离验证。")
    cli.add_argument("--question", default=DEFAULT_QUESTION, help="用户的两个独立知识问题")
    cli.add_argument("--collection-name", default="pet_care_v1", help="公共资料集合名称")
    cli.add_argument("--parallelism", type=int, choices=[1, 2], default=2, help="执行并发上限，1为串行对照")
    args = cli.parse_args()
    asyncio.run(run(args.question, args.collection_name, args.parallelism))
