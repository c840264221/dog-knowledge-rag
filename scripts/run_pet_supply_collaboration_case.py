"""知识查询与工具核算的真实协作案例：不写账、不改主图、不写向量库。"""

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from uuid import uuid4


DETAILED_QUESTION = (
    "我要接一只猫回家，请根据已入库的新猫到家养护指南查询需要准备哪些用品，附资料引用。"
    "然后仅对指南明确建议、且下列价格表覆盖的用品，使用预算工具计算采购预算。"
    "我的预算300元，价格均为我提供的单价：猫砂盆59.90元买1个、食水碗19.90元买2个、"
    "猫抓板29.90元买1个、躲避窝49.90元买1个。"
    "价格表不是指南推荐清单，不要不看资料就全选；语义对应不清楚的列为待核实，不擅自等同。"
    "未覆盖价格的指南用品列为未计价，未被指南支持的价格表项目不计入。"
    "最后分别报告指南建议、实际纳入核算的用品、工具合计、余额与未计价项。"
    "只是准备采购，不是已经消费，禁止记账或下单。"
)

DEFAULT_QUESTION = (
    "准备接猫回家，预算300元。我查到猫砂盆每个59.90元、食水碗每个19.90元、"
    "猫抓板每个29.90元、躲避窝每个49.90元。帮我选选需要哪些，"
    "碗考虑2个，其他各1个，算算要花多少。"
)
DEFAULT_SKILL_FILE = "skills/pet-supply-planning/SKILL.md"


def save_report(path: Path, report: dict) -> None:
    """保存独立案例报告；path为输出路径、report为本轮数据；无返回值，不作为恢复授权。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def validate_case_plan(plan) -> None:
    """检查本入口的筛选后核算案例覆盖要求。

    参数plan为模型正式计划；通过无返回值，否则拒绝执行，不重写步骤。
    本入口限定知识查询先影响核算范围，不是通用预算规则；已确定清单的独立核算不适用此案例。
    """
    knowledge = {step.step_id for step in plan.steps if step.assigned_agent == "pet_knowledge_agent"}
    tools = [step for step in plan.steps if step.assigned_agent == "tool_agent"]
    if not knowledge:
        raise ValueError("准备建议需要知识查询步骤，不能只按价格表核算。")
    by_id = {step.step_id: step for step in plan.steps}

    def ancestors(step_id: str) -> set[str]:
        """收集前置步骤；step_id为已校验无环计划中的编号；返回全部前置步骤编号集合。"""
        found = set()
        pending = list(by_id[step_id].depends_on)
        while pending:
            current = pending.pop()
            if current not in found:
                found.add(current)
                pending.extend(by_id[current].depends_on)
        return found

    for step in tools:
        if step.execution_mode != "tool" or step.tool_name != "calculate_supply_budget":
            raise ValueError("本案例只允许绑定本地预算工具，不允许写入或其他工具。")
        if not ancestors(step.step_id).intersection(knowledge):
            raise ValueError("当前案例要求先筛选再核算，预算步骤缺少知识依赖；未自动补依赖。若用户要核算已确定清单，应使用独立核算入口。")
    for step in plan.steps:
        if step.assigned_agent == "pet_knowledge_agent":
            english = step.input_data.get("english_retrieval_question")
            if not isinstance(english, str) or not english.strip():
                raise ValueError("知识步骤缺少待人工审核的英文检索文本。")


async def run(question: str, collection_name: str, *, use_skill: bool = True,
              skill_file: str = DEFAULT_SKILL_FILE) -> None:
    """运行协作案例。

    参数：question为用户需求，collection_name为宿主选择的公共集合，use_skill控制技能对照，
        skill_file为宿主配置的技能文件；无返回值，保存计划、技能及真实执行报告。
    """
    from src.settings import settings
    from src.runtime.container.core import RuntimeContainer
    from src.runtime.container.providers.embedding_provider import EmbeddingProvider
    from src.runtime.container.providers.vectorstore_provider import VectorStoreProvider
    from src.runtime.container.providers.reranker_provider import RerankerProvider
    from src.runtime.container.providers.llm_provider import LLMProvider
    from src.runtime.context import RuntimeContext, runtime_ctx
    from src.agents.pet_knowledge_agent.agent import build_pet_knowledge_runner
    from src.agents.pet_knowledge_agent.evidence import validate_evidence_bound_items
    from src.agents.collaboration.planner import PlannerAgent
    from src.agents.collaboration.scheduler import MultiAgentTaskScheduler
    from src.agents.collaboration.aggregator import ResultAggregator
    from src.agents.collaboration.workers.graph_agent_worker_adapter import GraphAgentWorkerAdapter
    from src.agents.collaboration.workers.tool_agent_worker_adapter import ToolAgentWorkerAdapter, build_tool_execution_guard
    from src.agents.tool_agent.adapters.registry_adapter import build_tool_agent_tool_catalog_state_update_with_mcp
    from src.agents.tool_agent.nodes.tool_parse_node import build_llm_tool_parser
    from src.agents.tool_agent.graph import build_tool_agent_graph
    from src.rag.retrievers.metadata_filter_retriever import MetadataFilterRetriever
    from src.skills.file_loader import load_file_skill_registry, select_agent_skill, build_file_skill_context
    from src.skills.task_context import attach_task_skill_context

    run_id = uuid4().hex
    path = Path(__file__).resolve().parents[1] / "output/learning/pet_supply_collaboration" / run_id / "report.json"
    context = RuntimeContext(trace_id=run_id, session_id=run_id, user_id="pet_supply_learning_user",
                             component="pet_supply_collaboration_case")
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
              "status": "starting", "stages": [], "worker_runs": [],
              "runtime_environment": context.time_context.as_dict(),
              "notice": "模型生成计划、真实查询和本地核算；不写账，不是主图路由或退出后恢复验证。"}
    print(f"本轮编号：{run_id}\n报告：{path}", flush=True)
    try:
        await context.startup()
        await container.startup()
        container.register("llm", LLMProvider())  # 按需提供回答模型，不启动无关本地中文模型。
        llm = container.get("llm")
        database = container.get("vectors").db
        model_config = {"provider": settings.embedding.provider, "model": settings.embedding.model_name,
                        "normalize": settings.embedding.normalize_embeddings}
        signature = hashlib.sha256(json.dumps(model_config, sort_keys=True).encode()).hexdigest()
        records = await asyncio.to_thread(database.get, include=["metadatas"])
        if not records["ids"] or any((item or {}).get("embedding_signature") != signature
                                     for item in records["metadatas"]):
            raise ValueError("集合为空、缺少向量签名或与当前向量模型配置不匹配，停止查询。")
        report["embedding_config"] = model_config
        catalog = [deepcopy(item) for item in build_tool_agent_tool_catalog_state_update_with_mcp(
            mcp_tool_definitions=[])["tool_agent_tool_catalog"] if item["name"] == "calculate_supply_budget"]
        if len(catalog) != 1:
            raise ValueError("本地预算工具未正确注册。")
        report["tool_catalog"] = catalog
        skill_context, skill_metadata = "", {}
        if use_skill:
            registry, locations = load_file_skill_registry([skill_file])
            selected = select_agent_skill(question, "multi_agent", registry, locations,
                                          {item["name"] for item in catalog})
            if selected is not None:
                # 所有执行者接收同一份任务技能；空target_agents表示不限定接收者。
                targets = locations[selected.skill_id]["target_agents"]
                if targets and not {"pet_knowledge_agent", "tool_agent"}.issubset(targets):
                    raise ValueError("当前任务技能未声明适用于案例中的步骤执行者。")
            skill_context, skill_metadata = build_file_skill_context(selected, registry, locations)
        report["task_skill"] = {"enabled": use_skill, "selected": bool(skill_context),
            "context": skill_context, "metadata": skill_metadata,
            "injection_points": ["planner_context", "knowledge_question", "tool_question"] if skill_context else [],
            "selection_mode": "existing_keyword_selector"}
        print("任务技能：" + (skill_metadata.get("skill_id", "未命中") if use_skill else "关闭（对照组）"), flush=True)
        base_parser = build_llm_tool_parser(llm_provider=llm)

        async def parser(parser_input: dict):
            """限制模型可见工具；parser_input为原规划输入；返回既有解析器输出，不引入业务专用分支。"""
            state = dict(parser_input.get("state") or {})
            state["tool_agent_tool_catalog"] = deepcopy(catalog)
            return await base_parser({**parser_input, "state": state})

        async def tool_runner(state: dict) -> dict:
            """执行受控本地工具；state为步骤状态；返回实际工具状态并保存参数与依赖记录。"""
            state = attach_task_skill_context(state, skill_context)
            binding_guard = build_tool_execution_guard(None, state.get("tool_agent_execution_contract"))

            def guard(current: dict) -> dict:
                """核对工具范围；current为执行前状态；通过返回空更新，否则抛出异常。"""
                calls = current.get("tool_calls") or []
                if len(calls) != 1 or calls[0].get("name") != "calculate_supply_budget":
                    raise ValueError("本案例只允许一次本地预算调用。")
                report["budget_arguments"] = deepcopy(calls[0])
                validate_evidence_bound_items(calls[0].get("args", {}).get("items", []),
                    state.get("multi_agent_dependency_results") or {})
                return binding_guard(current)

            graph = build_tool_agent_graph(parser=parser, llm_provider=llm, interrupt_func=None,
                                           before_execute=guard)
            result = dict(await graph.ainvoke(state))
            report["worker_runs"].append({"agent": "tool_agent", "step_id": state.get("multi_agent_step_id"),
                "dependencies": deepcopy(state.get("multi_agent_dependency_results") or {}),
                "result": {key: result.get(key) for key in ("tool_results", "tool_agent_response", "final_answer")}})
            save_report(path, report)
            return result

        knowledge_runner = build_pet_knowledge_runner(llm_provider=llm,
            retriever=MetadataFilterRetriever(database, allow_filter_relaxation=False),
            reranker=container.get("reranker").reranker, user_id=context.user_id)

        async def recorded_knowledge_runner(state: dict) -> dict:
            """记录真实知识查询；state为步骤输入；返回原运行结果，保存候选、引用和完整回答。"""
            result = await knowledge_runner(attach_task_skill_context(state, skill_context))
            report["worker_runs"].append({"agent": "pet_knowledge_agent",
                "step_id": state.get("multi_agent_step_id"), "result": result})
            save_report(path, report)
            return result

        planner = PlannerAgent(llm_provider=llm, maximum_steps=4, maximum_plan_attempts=2,
            available_agents={
                "pet_knowledge_agent": "查询宿主已选公共宠物养护集合，使用capability模式。input_data必须提供question、中文retrieval_question和忠实对应的english_retrieval_question；检索目标只包含资料需求，不混入价格表及核算任务。回答提供来源引用；不猜价格、不执行工具。",
                "tool_agent": "使用calculate_supply_budget计算用户给定价格及数量的合计与预算余额；使用tool模式绑定该工具。不提供商品价格查询、采购或记账能力。",
            })
        plan = await planner.create_plan(objective=question, context={"tool_catalog": catalog,
            "knowledge_scope": {"collection_name": collection_name, "content": "已入库的公共宠物养护资料"},
            "task_skill": {"instructions": skill_context, "metadata": skill_metadata}})
        report.update(status="plan_review", plan=plan.model_dump(mode="json"))
        save_report(path, report)
        print("\n模型生成的计划：\n" + json.dumps(report["plan"], ensure_ascii=False, indent=2), flush=True)
        if plan.status != "planned" or plan.requires_user_input:
            report["status"] = plan.status
            print("计划仍需澄清，请查看报告。", flush=True)
            return
        validate_case_plan(plan)
        approval = await asyncio.to_thread(input, "检查两种执行者、依赖和英文检索原意；输入 执行 继续：")
        if approval.strip() != "执行":
            report["status"] = "not_started"
            return
        workers = {
            "pet_knowledge_agent": GraphAgentWorkerAdapter(agent_name="pet_knowledge_agent",
                runner=recorded_knowledge_runner,
                output_fields=("final_answer", "rag_context", "citation_map", "citation_check", "evidence_ids",
                               "knowledge_handoff", "limitations")),
            "tool_agent": ToolAgentWorkerAdapter(tool_runner),
        }
        result = await MultiAgentTaskScheduler(workers=workers, maximum_parallel_steps=1,
                                              maximum_step_attempts=1).execute(plan)
        completed_agents = {item.assigned_agent for item in result.task_results if item.status == "completed"}
        report["execution_coverage"] = {
            "completed_agents": sorted(completed_agents),
            "knowledge_and_tool_completed": {"pet_knowledge_agent", "tool_agent"}.issubset(completed_agents),
            "notice": "只完成资料查询可满足缺价时的部分目标，不等同于完整核算或跨执行者协作成功。",
        }
        report["stages"].append(result.model_dump(mode="json"))
        report["status"] = result.status
        save_report(path, report)
        print("步骤状态：" + "，".join(f"{item.step_id}={item.status}" for item in result.task_results), flush=True)
        if result.status in {"running", "partial"}:
            result = await ResultAggregator(llm_provider=llm).aggregate(result)
            report["stages"].append(result.model_dump(mode="json"))
            report["status"] = result.status
            print("\n最终回答：\n" + str(result.final_answer or ""), flush=True)
        else:
            print("未完成，报告已保留；本入口不自动补参或重跑。", flush=True)
        report["human_review_required"] = ["用品确实来自入选指南吗？", "与价格表的语义对应是否正确？",
            "是否只核算交集、保留单价和数量？", "来源引用和未计价说明是否准确？"]
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
    cli = argparse.ArgumentParser(description="知识查询与本地预算工具的真实依赖协作案例。")
    cli.add_argument("--question", default=DEFAULT_QUESTION, help="用户需求，默认新猫到家用品预算")
    cli.add_argument("--collection-name", default="pet_care_v1", help="宿主选择的公共资料集合名称")
    cli.add_argument("--without-skill", action="store_true", help="不加载任务技能，用于同问题对照")
    cli.add_argument("--detailed", action="store_true", help="使用原详细需求作为对照，不能与--question同时使用")
    cli.add_argument("--skill-file", default=DEFAULT_SKILL_FILE, help="宿主配置的任务技能文件路径")
    args = cli.parse_args()
    if args.detailed and args.question != DEFAULT_QUESTION:
        cli.error("--detailed不能与自定义--question同时使用。")
    asyncio.run(run(DETAILED_QUESTION if args.detailed else args.question, args.collection_name,
                    use_skill=not args.without_skill, skill_file=args.skill_file))
