from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage

from src.agents.collaboration.aggregator import ResultAggregator
from src.agents.collaboration.graph import build_multi_agent_entry_node
from src.agents.collaboration.orchestrator import MultiAgentOrchestrator
from src.agents.collaboration.planner import PlannerAgent
from src.agents.collaboration.scheduler import MultiAgentTaskScheduler
from src.agents.tool_agent.graph import build_tool_agent_graph
from src.evaluation.schemas import AgentEvaluationCase
from src.evaluation.scenarios.multi_agent_orchestration_scenario_runtime import (
    EvaluationOrchestrationMessage,
    EvaluationOrchestrationLLMProvider,
)
from src.evaluation.scenarios.multi_agent_scenario_runtime import (
    EvaluationMultiAgentWorker,
)
from src.evaluation.scenarios.dog_knowledge_scenario_runtime import (
    EvaluationDogQueryParser,
    EvaluationMetadataFilterRetriever,
    EvaluationRerankerModel,
    EvaluationRerankerProvider,
    EvaluationRetrieverProvider,
)
from src.evaluation.scenarios.tool_agent_scenario_runtime import (
    EvaluationToolExecutor,
    EvaluationToolParser,
    build_evaluation_tool_registry,
)
from src.rag.schemas import RagContext
from src.runtime.context import RuntimeContext, runtime_ctx
from src.runtime.services.graph_runtime_service import GraphRuntimeService


class EvaluationMainGraphLLMProvider:
    """
    为真实主图提供确定性的 LLM Provider（大语言模型服务提供者）。

    功能：
        根据真实节点生成的 Prompt（提示词）识别调用用途，分别返回固定的
        Memory 抽取结果、GeneralAgent（通用问答智能体）答案、
        DogKnowledgeAgent 答案和 ToolAgent 格式化答案，并记录分类调用轨迹。

    参数含义：
        general_answer:
            GeneralAgent 最终回答节点需要返回的固定文本。
        dog_answer:
            DogKnowledgeAgent 答案生成节点需要返回的固定文本。
        tool_answer:
            ToolAgent 工具结果格式化节点需要返回的固定文本。

    返回值含义：
        EvaluationMainGraphLLMProvider:
            同时提供 main_llm、backup_llm、chinese_llm 和 safe_ainvoke 的
            确定性 LLM Provider。
    """

    def __init__(
        self,
        general_answer: str,
        dog_answer: str,
        tool_answer: str,
    ) -> None:
        """
        初始化主图评估 LLM Provider。

        参数含义：
            general_answer:
                GeneralAgent 固定回答文本。
            dog_answer:
                DogKnowledgeAgent 固定回答文本。
            tool_answer:
                ToolAgent 固定格式化回答文本。

        返回值含义：
            None。
        """

        self.general_answer = general_answer
        self.dog_answer = dog_answer
        self.tool_answer = tool_answer
        self.main_llm = object()
        self.backup_llm = object()
        self.chinese_llm = object()
        self.calls: list[dict[str, str]] = []

    async def safe_ainvoke(
        self,
        llm: Any,
        prompt: Any,
        fallback_response: str | None = None,
        max_attempts: int | None = None,
    ) -> AIMessage:
        """
        按 Prompt 用途返回确定性 LLM 响应。

        参数含义：
            llm:
                真实节点传入的模型对象，本评估环境不访问外部模型。
            prompt:
                真实业务节点构建的 Prompt 或 PromptValue（提示词对象）。
            fallback_response:
                业务节点声明的失败兜底文本；无法识别 Prompt 时使用。
            max_attempts:
                最大尝试次数；评估环境不执行真实重试。

        返回值含义：
            AIMessage:
                包含当前调用类型对应固定文本的 LangChain AI 消息。
        """

        _ = llm, max_attempts
        prompt_text = (
            prompt.to_string()
            if hasattr(prompt, "to_string")
            else str(prompt)
        )
        call_type, response_text = self._resolve_response(
            prompt_text=prompt_text,
            fallback_response=fallback_response,
        )
        self.calls.append(
            {
                "call_type": call_type,
                "prompt": prompt_text,
            }
        )
        return AIMessage(content=response_text)

    def count_calls(self, call_type: str) -> int:
        """
        统计指定用途的 LLM 调用次数。

        参数含义：
            call_type:
                调用分类名称，例如 memory_extract、dog_answer。

        返回值含义：
            int:
                调用轨迹中分类名称相同的记录数量。
        """

        return sum(
            1
            for call in self.calls
            if call.get("call_type") == call_type
        )

    def _resolve_response(
        self,
        prompt_text: str,
        fallback_response: str | None,
    ) -> tuple[str, str]:
        """
        根据真实 Prompt 中的稳定职责标识选择响应。

        参数含义：
            prompt_text:
                转换成字符串后的完整 Prompt。
            fallback_response:
                无法识别调用用途时的兜底文本。

        返回值含义：
            tuple[str, str]:
                第一项是调用分类，第二项是需要返回给业务节点的文本。
        """

        if "长期记忆提取器" in prompt_text:
            return (
                "memory_extract",
                """
{
  "should_save": false,
  "memory_type": "preference",
  "content": "",
  "confidence": 0.0,
  "importance": 0.0,
  "reason": "主图评估不保存长期记忆"
}
""".strip(),
            )

        if "狗狗百科" in prompt_text and "调度员" in prompt_text:
            normalized_prompt = prompt_text.lower()
            decision = (
                "finish"
                if '"has_answer": true' in normalized_prompt
                else "answer_gen"
            )
            return "general_supervisor", decision

        if "ToolAgent 的最终答案格式化器" in prompt_text:
            return "tool_answer", self.tool_answer

        if "只能基于提供信息回答的助手" in prompt_text:
            return "general_answer", self.general_answer

        if "Dog Agent Framework 的犬种" in prompt_text:
            return "dog_answer", self.dog_answer

        return "fallback", str(fallback_response or "评估模型未识别调用类型")


class EvaluationMainGraphPlanningProvider(
    EvaluationOrchestrationLLMProvider
):
    """
    为主图多 Agent 评估返回匹配本轮运行编号的固定计划。

    功能：
        保留黄金用例声明的步骤结构，同时从真实 Planner 提示词提取程序
        本轮生成的 plan_id 和原始 objective，避免固定测试数据与随机运行
        编号冲突，并继续接受真实 Planner 的完整输出校验。

    参数含义：
        plan_template:
            黄金用例声明的固定计划模板。

    返回值含义：
        EvaluationMainGraphPlanningProvider:
            可注入真实 PlannerAgent 的确定性评估 Provider。
    """

    def __init__(self, plan_template: dict[str, Any]) -> None:
        """
        初始化主图多 Agent 计划评估 Provider。

        参数含义：
            plan_template:
                需要保留步骤、依赖和 Agent 分配的计划模板。

        返回值含义：
            None。
        """

        super().__init__([])
        self.plan_template = dict(plan_template)

    async def safe_ainvoke(
        self,
        *,
        llm: Any,
        prompt: str,
        fallback_response: str | None = None,
    ) -> EvaluationOrchestrationMessage:
        """
        根据真实 Planner 提示词生成本轮合法的确定性计划响应。

        参数含义：
            llm:
                Planner 选择的模型对象，评估环境不会访问该模型。
            prompt:
                包含本轮 plan_id 和原始目标的真实 Planner 提示词。
            fallback_response:
                真实 Provider 的可选兜底文本，本评估替身不会使用。

        返回值含义：
            EvaluationOrchestrationMessage:
                包含动态运行编号和固定步骤结构的 JSON 消息。
        """

        _ = llm, fallback_response
        self.prompts.append(prompt)
        plan_data = dict(self.plan_template)
        plan_data["plan_id"] = _extract_planner_plan_id(prompt)
        plan_data["objective"] = _extract_planner_objective(prompt)
        return EvaluationOrchestrationMessage(
            json.dumps(plan_data, ensure_ascii=False)
        )


def _extract_planner_plan_id(prompt: str) -> str:
    """
    从真实 Planner 提示词提取程序生成的计划编号。

    参数含义：
        prompt:
            build_planner_prompt 生成的完整提示词。

    返回值含义：
        str:
            Planner 要求 LLM 原样返回的 plan_id。
    """

    match = re.search(
        r"plan_id 必须原样返回为 (.+?)。",
        prompt,
    )
    if match is None:
        raise ValueError("主图评估无法从 Planner 提示词提取 plan_id")
    try:
        plan_id = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Planner 提示词中的 plan_id 格式不合法") from exc
    if not isinstance(plan_id, str) or not plan_id:
        raise ValueError("Planner 提示词中的 plan_id 不能为空")
    return plan_id


def _extract_planner_objective(prompt: str) -> str:
    """
    从真实 Planner 提示词提取未改写的用户目标。

    参数含义：
        prompt:
            build_planner_prompt 生成的完整提示词。

    返回值含义：
        str:
            位于用户目标开始和结束标记之间的原始目标。
    """

    match = re.search(
        r"用户目标开始：\s*(.*?)\s*用户目标结束。",
        prompt,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("主图评估无法从 Planner 提示词提取 objective")
    objective = match.group(1).strip()
    if not objective:
        raise ValueError("Planner 提示词中的 objective 不能为空")
    return objective


def build_main_graph_evaluation_initial_state(
    question: str,
    user_id: str,
    trace_id: str,
) -> dict[str, Any]:
    """
    构建不依赖全局 Container（容器）的主图评估初始状态。

    功能：
        初始化真实主图、RootAgent 和三个下游 Agent 共同使用的核心字段，
        保持每条用例从干净状态开始，同时避免导入 graph_run 时触发生产容器注册。

    参数含义：
        question:
            当前黄金用例的用户问题。
        user_id:
            当前评估用例使用的隔离用户编号。
        trace_id:
            当前评估用例的链路追踪编号，同时作为 session_id。

    返回值含义：
        dict[str, Any]:
            可直接交给真实 Main Graph 执行的基础 DogState 字典。
    """

    return {
        "question": question,
        "messages": [],
        "user_id": user_id,
        "session_id": trace_id,
        "trace_id": trace_id,
        "intent": "",
        "strategy": None,
        "next_agent": "",
        "current_agent": "",
        "next_worker": "",
        "route_decision": {},
        "filters": {},
        "tags": [],
        "features": [],
        "dog_name": None,
        "top_k": 5,
        "docs": [],
        "rag_query": None,
        "rag_context": None,
        "retrieval_ok": False,
        "retrieval_evaluated": False,
        "retrieval_quality": None,
        "retrieval_failure_type": None,
        "retrieval_retry_strategy": None,
        "retry_count": 0,
        "answer": "",
        "final_answer": "",
        "answer_strategy": {},
        "user_feedback": "",
        "has_asked_user": False,
        "pending_prompt": "",
        "waiting_user_input": False,
        "task_relation_decision": {},
        "task_relation_pending_kind": "",
        "task_relation_requires_confirmation": False,
        "tool_calls": [],
        "tool_results": [],
        "need_tool": False,
        "tool_round": 0,
        "tool_confirmed": "",
        "tool_executed": False,
        "tool_agent_llm_answer_used": False,
        "memory_context": "",
        "memory_recall_result": {},
        "memory_saved": False,
        "memory_extract_result": {},
        "memory_save_result": None,
        "multi_agent_task_result": {},
        "multi_agent_resume_action": "none",
        "multi_agent_resume_inputs": {},
        "multi_agent_resume_ready": False,
        "multi_agent_pending_prompt": "",
    }


class EvaluationMainGraphRuntimeService(GraphRuntimeService):
    """
    使用真实 GraphRuntimeService 主图构建逻辑的评估运行时服务。

    功能：
        只重写 ToolAgent 子图依赖组装，将确定性 Parser（解析器）、
        ToolRegistry（工具注册表）和 Executor（执行器）注入真实 ToolAgent；
        主图节点、条件边和其他 Agent 构建过程继续复用生产实现。

    参数含义：
        tool_parser:
            返回黄金用例预设工具调用的确定性解析器。
        tool_executor:
            返回固定工具结果并记录执行轨迹的确定性执行器。
        multi_agent_node:
            可选确定性多 Agent 入口；未提供时继续使用生产组装逻辑。
        其他参数:
            继续沿用 GraphRuntimeService 的 Provider 注入参数。

    返回值含义：
        EvaluationMainGraphRuntimeService:
            可构建真实主图的评估专用运行时服务。
    """

    def __init__(
        self,
        *,
        tool_parser: EvaluationToolParser,
        tool_executor: EvaluationToolExecutor,
        multi_agent_node: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化主图评估运行时服务。

        参数含义：
            tool_parser:
                ToolAgent 使用的确定性工具解析器。
            tool_executor:
                ToolAgent 使用的确定性工具执行器。
            multi_agent_node:
                多 Agent 黄金用例使用的确定性主图入口。
            **kwargs:
                传给 GraphRuntimeService 的其他 Provider 依赖。

        返回值含义：
            None。
        """

        super().__init__(
            tool_parser=tool_parser,
            **kwargs,
        )
        self.evaluation_tool_executor = tool_executor
        self.evaluation_multi_agent_node = multi_agent_node

    def _build_tool_agent_node(self) -> Any:
        """
        构建注入确定性外部依赖的真实 ToolAgent 子图。

        参数含义：
            无。

        返回值含义：
            Any:
                build_tool_agent_graph 返回的真实编译子图。
        """

        return build_tool_agent_graph(
            parser=self.tool_parser,
            llm_provider=self.llm_provider,
            tool_registry=build_evaluation_tool_registry(),
            executor=self.evaluation_tool_executor,
            checkpoint_manager=None,
            runtime_context_getter=runtime_ctx.get,
            interrupt_func=None,
        )

    def _build_multi_agent_node(
        self,
        *,
        dog_knowledge_agent: Any,
        general_agent: Any,
    ) -> Any:
        """
        为主图评估选择确定性或生产多 Agent 入口。

        功能：
            多 Agent 黄金用例返回注入确定性依赖的真实入口；其他主图用例
            继续调用父类生产组装逻辑，避免改变既有评估行为。

        参数含义：
            dog_knowledge_agent:
                父类生产组装需要的狗狗知识子图。
            general_agent:
                父类生产组装需要的通用问答子图。

        返回值含义：
            Any:
                可注册到主图 multi_agent 节点的异步入口。
        """

        if self.evaluation_multi_agent_node is not None:
            return self.evaluation_multi_agent_node
        return super()._build_multi_agent_node(
            dog_knowledge_agent=dog_knowledge_agent,
            general_agent=general_agent,
        )


@dataclass
class MainGraphScenarioRuntime:
    """
    保存一条 Main Graph（主图）行为评估的运行环境和调用轨迹。

    参数含义：
        graph:
            GraphRuntimeService 构建的真实编译主图。
        initial_state:
            已移除 evaluation_* 配置字段的真实主图初始状态。
        llm_provider:
            确定性 LLM Provider 及其分类调用轨迹。
        dog_parser、dog_retriever、dog_reranker:
            DogKnowledgeAgent 的确定性外部依赖。
        tool_parser、tool_executor:
            ToolAgent 的确定性外部依赖。
        multi_agent_worker:
            可选确定性多 Agent Worker 及其步骤调用轨迹。
        multi_agent_planning_provider、multi_agent_aggregation_provider:
            可选 Planner 与 Aggregator 固定响应及提示词调用轨迹。
        runtime_context:
            当前评估用例独享的真实 RuntimeContext（运行时上下文）。

    返回值含义：
        MainGraphScenarioRuntime:
            可执行真实主图并读取各外部依赖调用轨迹的场景对象。
    """

    graph: Any
    initial_state: dict[str, Any]
    llm_provider: EvaluationMainGraphLLMProvider
    dog_parser: EvaluationDogQueryParser
    dog_retriever: EvaluationMetadataFilterRetriever
    dog_reranker: EvaluationRerankerModel
    tool_parser: EvaluationToolParser
    tool_executor: EvaluationToolExecutor
    multi_agent_worker: EvaluationMultiAgentWorker | None = None
    multi_agent_planning_provider: (
        EvaluationOrchestrationLLMProvider | None
    ) = None
    multi_agent_aggregation_provider: (
        EvaluationOrchestrationLLMProvider | None
    ) = None
    runtime_context: RuntimeContext = field(default_factory=RuntimeContext)

    async def invoke(self) -> dict[str, Any]:
        """
        在隔离 RuntimeContext 中执行真实 Main Graph。

        参数含义：
            无。

        返回值含义：
            dict[str, Any]:
                主图从 memory_extract 到目标 Agent 执行完成后的最终 DogState。
        """

        previous_context = runtime_ctx.get()
        runtime_ctx.set(self.runtime_context)
        try:
            return await self.graph.ainvoke(
                dict(self.initial_state)
            )
        finally:
            runtime_ctx.set(previous_context)


async def build_main_graph_scenario_runtime(
    eval_case: AgentEvaluationCase,
) -> MainGraphScenarioRuntime:
    """
    根据黄金用例构建真实 Main Graph 确定性评估场景。

    功能：
        提取 evaluation_* 评估配置，构建确定性 LLM、RAG 和 Tool 依赖，
        再调用真实 GraphRuntimeService._build_graph 构建完整主图。

    参数含义：
        eval_case:
            包含主图输入、外部依赖预设结果和黄金期望的统一评估用例。

    返回值含义：
        MainGraphScenarioRuntime:
            包含真实主图、干净初始状态和依赖调用轨迹的运行环境。
    """

    raw_state = dict(eval_case.input_state)
    parser_filters = raw_state.pop("evaluation_parser_filters", {})
    raw_rag_context = raw_state.pop(
        "evaluation_rag_context",
        {
            "question": eval_case.question,
            "context_text": "",
            "chunks": [],
            "source_count": 0,
            "status": "empty",
        },
    )
    raw_tool_parser_result = raw_state.pop(
        "evaluation_tool_parser_result",
        {
            "need_tool": False,
            "tool_calls": [],
        },
    )
    general_answer = str(
        raw_state.pop(
            "evaluation_general_answer",
            "这是主图评估使用的通用回答。",
        )
    )
    dog_answer = str(
        raw_state.pop(
            "evaluation_dog_answer",
            "当前狗狗知识库没有足够信息。",
        )
    )
    tool_answer = str(
        raw_state.pop(
            "evaluation_tool_answer",
            "工具已经执行完成。",
        )
    )
    raw_multi_agent_plan = raw_state.pop(
        "evaluation_multi_agent_plan",
        None,
    )
    raw_multi_agent_pending_result = raw_state.pop(
        "evaluation_multi_agent_pending_result",
        None,
    )
    raw_multi_agent_behaviors = raw_state.pop(
        "evaluation_multi_agent_worker_behaviors",
        {},
    )
    raw_multi_agent_aggregation = raw_state.pop(
        "evaluation_multi_agent_aggregation_response",
        None,
    )

    if not isinstance(parser_filters, dict):
        raise ValueError("evaluation_parser_filters 必须是 dict")
    if not isinstance(raw_rag_context, dict):
        raise ValueError("evaluation_rag_context 必须是 dict")
    if not isinstance(raw_tool_parser_result, dict):
        raise ValueError("evaluation_tool_parser_result 必须是 dict")
    if (
        raw_multi_agent_plan is not None
        and not isinstance(raw_multi_agent_plan, dict)
    ):
        raise ValueError("evaluation_multi_agent_plan 必须是 dict")
    if (
        raw_multi_agent_pending_result is not None
        and not isinstance(raw_multi_agent_pending_result, dict)
    ):
        raise ValueError(
            "evaluation_multi_agent_pending_result 必须是 dict"
        )
    if not isinstance(raw_multi_agent_behaviors, dict):
        raise ValueError(
            "evaluation_multi_agent_worker_behaviors 必须是 dict"
        )
    if (
        raw_multi_agent_aggregation is not None
        and not isinstance(raw_multi_agent_aggregation, dict)
    ):
        raise ValueError(
            "evaluation_multi_agent_aggregation_response 必须是 dict"
        )

    llm_provider = EvaluationMainGraphLLMProvider(
        general_answer=general_answer,
        dog_answer=dog_answer,
        tool_answer=tool_answer,
    )
    dog_parser = EvaluationDogQueryParser(parser_filters)
    dog_retriever = EvaluationMetadataFilterRetriever(
        RagContext.model_validate(raw_rag_context)
    )
    dog_reranker = EvaluationRerankerModel()
    tool_parser = EvaluationToolParser(raw_tool_parser_result)
    tool_executor = EvaluationToolExecutor()
    (
        multi_agent_node,
        multi_agent_worker,
        multi_agent_planning_provider,
        multi_agent_aggregation_provider,
    ) = _build_main_graph_multi_agent_evaluation_dependencies(
        raw_plan=raw_multi_agent_plan,
        raw_pending_result=raw_multi_agent_pending_result,
        raw_behaviors=raw_multi_agent_behaviors,
        raw_aggregation=raw_multi_agent_aggregation,
    )

    graph_runtime = EvaluationMainGraphRuntimeService(
        llm_provider=llm_provider,
        memory_provider=None,
        checkpoint_provider=None,
        retriever_provider=EvaluationRetrieverProvider(
            dog_parser,
            dog_retriever,
        ),
        reranker_provider=EvaluationRerankerProvider(dog_reranker),
        sqlite_mcp_provider=None,
        tool_parser=tool_parser,
        tool_executor=tool_executor,
        multi_agent_node=multi_agent_node,
    )
    graph = await graph_runtime._build_graph()

    trace_id = f"evaluation-{eval_case.case_id}"
    evaluation_user_id = str(
        raw_state.get("user_id", "evaluation_user")
    )
    initial_state = build_main_graph_evaluation_initial_state(
        question=eval_case.question,
        user_id=evaluation_user_id,
        trace_id=trace_id,
    )
    initial_state.update(raw_state)
    initial_state.update(
        {
            "question": eval_case.question,
            "user_id": evaluation_user_id,
            "session_id": trace_id,
            "trace_id": trace_id,
            "messages": list(raw_state.get("messages", [])),
        }
    )
    if raw_multi_agent_pending_result is not None:
        initial_state["multi_agent_task_result"] = dict(
            raw_multi_agent_pending_result
        )

    return MainGraphScenarioRuntime(
        graph=graph,
        initial_state=initial_state,
        llm_provider=llm_provider,
        dog_parser=dog_parser,
        dog_retriever=dog_retriever,
        dog_reranker=dog_reranker,
        tool_parser=tool_parser,
        tool_executor=tool_executor,
        multi_agent_worker=multi_agent_worker,
        multi_agent_planning_provider=multi_agent_planning_provider,
        multi_agent_aggregation_provider=(
            multi_agent_aggregation_provider
        ),
        runtime_context=RuntimeContext(
            trace_id=trace_id,
            user_id=initial_state["user_id"],
            component="main_graph_evaluation",
        ),
    )


def _build_main_graph_multi_agent_evaluation_dependencies(
    *,
    raw_plan: dict[str, Any] | None,
    raw_pending_result: dict[str, Any] | None,
    raw_behaviors: dict[str, Any],
    raw_aggregation: dict[str, Any] | None,
) -> tuple[
    Any | None,
    EvaluationMultiAgentWorker | None,
    EvaluationOrchestrationLLMProvider | None,
    EvaluationOrchestrationLLMProvider | None,
]:
    """
    为主图多 Agent 黄金用例组装确定性入口依赖。

    功能：
        创建真实 Planner、Scheduler、Aggregator、Orchestrator 和入口节点，
        只把 LLM 响应与 Worker 输出替换为黄金用例中的确定性数据。
        普通主图用例没有多 Agent 配置时返回四个 None。

    参数含义：
        raw_plan:
            新任务或 replan 使用的固定 Planner 输出。
        raw_pending_result:
            resume、replan 或澄清场景使用的暂停任务结果。
        raw_behaviors:
            各步骤的确定性 Worker 行为。
        raw_aggregation:
            可选固定聚合响应。

    返回值含义：
        tuple:
            依次返回入口节点、Worker、Planner Provider 和 Aggregator
            Provider；非多 Agent 用例全部为 None。
    """

    if raw_plan is None and raw_pending_result is None:
        return None, None, None, None

    plan_data = raw_plan
    if plan_data is None and raw_pending_result is not None:
        pending_plan = raw_pending_result.get("plan")
        if not isinstance(pending_plan, dict):
            raise ValueError("暂停任务缺少合法 plan")
        plan_data = dict(pending_plan)

    raw_steps = plan_data.get("steps") if plan_data else None
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("主图多 Agent 评估计划缺少 steps")
    available_agents = {
        str(step.get("assigned_agent") or "").strip(): (
            f"执行步骤：{str(step.get('title') or '').strip()}"
        )
        for step in raw_steps
        if isinstance(step, dict)
        and str(step.get("assigned_agent") or "").strip()
    }
    if not available_agents:
        raise ValueError("主图多 Agent 评估计划缺少可用 Agent")

    planning_provider = EvaluationMainGraphPlanningProvider(
        plan_data
    )
    aggregation_response = raw_aggregation or {
        "final_answer": "主图多 Agent 评估完成。",
        "used_step_ids": [
            str(step.get("step_id"))
            for step in raw_steps
            if isinstance(step, dict)
        ],
        "limitations": [],
    }
    aggregation_provider = EvaluationOrchestrationLLMProvider(
        [json.dumps(aggregation_response, ensure_ascii=False)]
    )
    worker = EvaluationMultiAgentWorker(raw_behaviors)
    planner = PlannerAgent(
        llm_provider=planning_provider,
        available_agents=available_agents,
        maximum_plan_attempts=1,
    )
    scheduler = MultiAgentTaskScheduler(
        workers={
            agent_name: worker
            for agent_name in available_agents
        },
        maximum_step_attempts=1,
    )
    aggregator = ResultAggregator(
        llm_provider=aggregation_provider,
        maximum_aggregation_attempts=1,
    )
    entry_node = build_multi_agent_entry_node(
        orchestrator=MultiAgentOrchestrator(
            planner=planner,
            scheduler=scheduler,
            result_aggregator=aggregator,
        )
    )
    return (
        entry_node,
        worker,
        planning_provider,
        aggregation_provider,
    )
