from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage

from src.agents.dog_knowledge_agent.agent import build_dog_knowledge_agent
from src.evaluation.schemas import AgentEvaluationCase
from src.rag.schemas import RagContext, RagQuery
from src.runtime.context import RuntimeContext, runtime_ctx


class EvaluationDogQueryParser:
    """
    DogKnowledgeAgent 评估专用查询解析器。

    功能：
        返回黄金用例预设的 metadata filters（元数据过滤条件），
        同时记录真实 retrieve_node（检索节点）传入的解析参数。

    参数含义：
        filters:
            当前评估用例希望查询解析器返回的过滤条件。

    返回值含义：
        EvaluationDogQueryParser:
            支持 parse 方法并保存调用轨迹的确定性解析器。
    """

    def __init__(self, filters: dict[str, Any]) -> None:
        self.filters = dict(filters)
        self.inputs: list[dict[str, Any]] = []

    def parse(
        self,
        question: str,
        user_id: str,
        top_k: int,
        intent: str,
    ) -> RagQuery:
        """
        将预设过滤条件包装成 RagQuery（RAG 查询对象）。

        参数含义：
            question:
                用户问题。
            user_id:
                用户唯一编号。
            top_k:
                本轮最多召回的文本块数量。
            intent:
                当前业务意图。

        返回值含义：
            RagQuery:
                包含预设过滤条件的标准 RAG 查询对象。
        """

        self.inputs.append(
            {
                "question": question,
                "user_id": user_id,
                "top_k": top_k,
                "intent": intent,
            }
        )
        return RagQuery(
            question=question,
            user_id=user_id,
            top_k=top_k,
            filters=dict(self.filters),
            intent=intent,
        )


class EvaluationMetadataFilterRetriever:
    """
    DogKnowledgeAgent 评估专用 RAG 检索器。

    功能：
        接收真实 retrieve_node 生成的 RagQuery，返回黄金用例预设的
        RagContext（RAG 检索上下文），避免访问真实 Chroma 向量数据库。

    参数含义：
        rag_context:
            当前评估场景需要返回的确定性检索上下文。

    返回值含义：
        EvaluationMetadataFilterRetriever:
            支持 async_retrieve 方法并记录查询轨迹的检索器。
    """

    def __init__(self, rag_context: RagContext) -> None:
        self.rag_context = rag_context
        self.queries: list[dict[str, Any]] = []

    async def async_retrieve(self, query: RagQuery) -> RagContext:
        """
        返回当前场景预设的 RAG 上下文。

        参数含义：
            query:
                真实 retrieve_node 传入的标准 RAG 查询对象。

        返回值含义：
            RagContext:
                当前黄金用例配置的确定性检索结果副本。
        """

        self.queries.append(query.model_dump(mode="python"))
        return RagContext.model_validate(
            self.rag_context.model_dump(mode="python")
        )


class EvaluationRetrieverProvider:
    """
    向真实 DogKnowledgeAgent 提供确定性解析器和检索器。

    参数含义：
        parser:
            评估专用查询解析器。
        retriever:
            评估专用 RAG 检索器。

    返回值含义：
        EvaluationRetrieverProvider:
            符合 retrieve_node Provider（服务提供者）读取约定的对象。
    """

    def __init__(
        self,
        parser: EvaluationDogQueryParser,
        retriever: EvaluationMetadataFilterRetriever,
    ) -> None:
        self.dog_query_filter_parser = parser
        self.metadata_filter_retriever = retriever


class EvaluationRerankerModel:
    """
    DogKnowledgeAgent 评估专用重排序模型。

    功能：
        为真实 rerank_node（重排序节点）返回稳定且按原顺序递减的分数，
        并记录节点实际提交的 query-document pairs（查询与文档组合）。

    参数含义：
        无。

    返回值含义：
        EvaluationRerankerModel:
            支持 predict 方法的确定性重排序模型。
    """

    def __init__(self) -> None:
        self.calls: list[list[Any]] = []

    def predict(self, pairs: list[Any]) -> list[float]:
        """
        为每个候选文本块生成稳定重排分数。

        参数含义：
            pairs:
                rerank_node 构造的查询与文档组合列表。

        返回值含义：
            list[float]:
                与输入数量一致、从高到低排列的原始重排分数。
        """

        self.calls.append(list(pairs))
        return [
            5.0 - (index * 0.1)
            for index in range(len(pairs))
        ]


class EvaluationRerankerProvider:
    """
    向真实 rerank_node 提供确定性重排序模型。

    参数含义：
        model:
            评估专用重排序模型。

    返回值含义：
        EvaluationRerankerProvider:
            具有 reranker 属性的服务提供者。
    """

    def __init__(self, model: EvaluationRerankerModel) -> None:
        self.reranker = model


class EvaluationLLMProvider:
    """
    DogKnowledgeAgent 评估专用 LLM Provider（大语言模型提供者）。

    功能：
        让真实 generate_node（答案生成节点）继续构建 Prompt（提示词），
        但不访问外部模型，而是返回黄金用例预设的答案文本。

    参数含义：
        answer:
            当前评估场景固定返回的自然语言答案。

    返回值含义：
        EvaluationLLMProvider:
            支持 main_llm 和 safe_ainvoke 的确定性模型提供者。
    """

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.main_llm = object()
        self.prompts: list[str] = []

    async def safe_ainvoke(
        self,
        llm: Any,
        prompt: str,
        fallback_response: str | None = None,
        max_attempts: int | None = None,
    ) -> AIMessage:
        """
        记录真实生成节点构造的 Prompt 并返回固定 AIMessage。

        参数含义：
            llm:
                generate_node 传入的主模型对象，本评估环境不实际调用。
            prompt:
                generate_node 构造的完整提示词。
            fallback_response:
                外部模型失败时的备用文本，本评估环境不会使用。
            max_attempts:
                最大尝试次数，本评估环境不会执行重试。

        返回值含义：
            AIMessage:
                包含预设答案的 LangChain AI 消息对象。
        """

        _ = llm, fallback_response, max_attempts
        self.prompts.append(str(prompt))
        return AIMessage(content=self.answer)


@dataclass
class DogKnowledgeScenarioRuntime:
    """
    单条 DogKnowledgeAgent 评估用例的确定性运行环境。

    参数含义：
        graph:
            build_dog_knowledge_agent 构建的真实编译子图。
        initial_state:
            清除评估配置字段后的真实 DogState 初始数据。
        parser、retriever、reranker、llm_provider:
            外部依赖替身及其调用轨迹。
        runtime_context:
            当前评估用例独享的运行时上下文。

    返回值含义：
        DogKnowledgeScenarioRuntime:
            可以通过 invoke 执行真实子图的场景对象。
    """

    graph: Any
    initial_state: dict[str, Any]
    parser: EvaluationDogQueryParser
    retriever: EvaluationMetadataFilterRetriever
    reranker: EvaluationRerankerModel
    llm_provider: EvaluationLLMProvider
    runtime_context: RuntimeContext = field(default_factory=RuntimeContext)

    async def invoke(self) -> dict[str, Any]:
        """
        在隔离的 RuntimeContext 中执行真实 DogKnowledgeAgent 子图。

        参数含义：
            无。

        返回值含义：
            dict[str, Any]:
                DogKnowledgeAgent 完整执行后的最终 DogState。
        """

        previous_context = runtime_ctx.get()
        runtime_ctx.set(self.runtime_context)
        try:
            return await self.graph.ainvoke(dict(self.initial_state))
        finally:
            runtime_ctx.set(previous_context)


def build_dog_knowledge_scenario_runtime(
    eval_case: AgentEvaluationCase,
) -> DogKnowledgeScenarioRuntime:
    """
    根据黄金用例构建真实 DogKnowledgeAgent 确定性场景。

    功能：
        从 input_state 提取 evaluation_* 评估配置，构造外部依赖替身，
        再调用 build_dog_knowledge_agent 构建真实编译子图。

    参数含义：
        eval_case:
            当前统一 Agent 评估用例。

    返回值含义：
        DogKnowledgeScenarioRuntime:
            包含真实子图、初始状态和依赖调用轨迹的运行环境。
    """

    raw_state = dict(eval_case.input_state)
    parser_filters = raw_state.pop("evaluation_parser_filters", {})
    raw_rag_context = raw_state.pop("evaluation_rag_context", None)
    llm_answer = str(
        raw_state.pop(
            "evaluation_llm_answer",
            "我暂时无法基于当前狗狗知识库可靠回答这个问题。",
        )
    )

    if not isinstance(parser_filters, dict):
        raise ValueError("evaluation_parser_filters 必须是 dict")
    if not isinstance(raw_rag_context, dict):
        raise ValueError("evaluation_rag_context 必须是 dict")

    parser = EvaluationDogQueryParser(parser_filters)
    retriever = EvaluationMetadataFilterRetriever(
        RagContext.model_validate(raw_rag_context)
    )
    reranker = EvaluationRerankerModel()
    llm_provider = EvaluationLLMProvider(llm_answer)

    graph = build_dog_knowledge_agent(
        llm_provider=llm_provider,
        memory_provider=None,
        checkpoint_provider=None,
        retriever_provider=EvaluationRetrieverProvider(parser, retriever),
        reranker_provider=EvaluationRerankerProvider(reranker),
    )
    initial_state = {
        **raw_state,
        "question": eval_case.question,
        "user_id": str(raw_state.get("user_id", "evaluation_user")),
        "messages": list(raw_state.get("messages", [])),
    }

    return DogKnowledgeScenarioRuntime(
        graph=graph,
        initial_state=initial_state,
        parser=parser,
        retriever=retriever,
        reranker=reranker,
        llm_provider=llm_provider,
        runtime_context=RuntimeContext(
            trace_id=f"evaluation-{eval_case.case_id}",
            user_id=initial_state["user_id"],
            component="dog_knowledge_evaluation",
        ),
    )
