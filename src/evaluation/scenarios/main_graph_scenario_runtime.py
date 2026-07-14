from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage

from src.agents.tool_agent.graph import build_tool_agent_graph
from src.evaluation.schemas import AgentEvaluationCase
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
        **kwargs: Any,
    ) -> None:
        """
        初始化主图评估运行时服务。

        参数含义：
            tool_parser:
                ToolAgent 使用的确定性工具解析器。
            tool_executor:
                ToolAgent 使用的确定性工具执行器。
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

    if not isinstance(parser_filters, dict):
        raise ValueError("evaluation_parser_filters 必须是 dict")
    if not isinstance(raw_rag_context, dict):
        raise ValueError("evaluation_rag_context 必须是 dict")
    if not isinstance(raw_tool_parser_result, dict):
        raise ValueError("evaluation_tool_parser_result 必须是 dict")

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

    return MainGraphScenarioRuntime(
        graph=graph,
        initial_state=initial_state,
        llm_provider=llm_provider,
        dog_parser=dog_parser,
        dog_retriever=dog_retriever,
        dog_reranker=dog_reranker,
        tool_parser=tool_parser,
        tool_executor=tool_executor,
        runtime_context=RuntimeContext(
            trace_id=trace_id,
            user_id=initial_state["user_id"],
            component="main_graph_evaluation",
        ),
    )
