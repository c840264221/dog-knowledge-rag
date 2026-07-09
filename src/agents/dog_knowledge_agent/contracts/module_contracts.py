"""
DogKnowledgeAgent 内部模块职责契约。

功能：
    定义 DogKnowledgeAgent 在 V1.7.2 之后应该遵守的内部职责分层。

    该模块不执行业务逻辑，不调用 LLM，不调用 Retriever，
    只用于描述和固定 DogKnowledgeAgent 的内部架构边界。

设计目标：
    1. 明确 DogKnowledgeAgent 是狗狗知识统一入口。
    2. 明确内部模块职责，避免一个文件承担过多职责。
    3. 固定 RAG 子链路顺序：
       RagQuery -> Retrieval -> Rerank -> Quality -> RagContext。
    4. 明确 MemoryContext 不属于 RagContext，而是在生成阶段和 RagContext 一起注入 Prompt。
    5. 为后续代码收拢、测试、文档、面试复习提供标准。
    6. 作为结构契约测试的数据来源。

专业名词：
    Contract：契约，表示模块之间约定好的结构和职责。
    Module：模块，表示一个代码职责单元。
    Responsibility：职责，表示该模块应该负责什么事情。
    Retrieval：检索，从知识库中召回候选内容。
    Rerank：重排，对召回结果重新排序。
    Quality：质量评估，判断召回结果是否足够支持回答。
    Context Builder：上下文构建器，把合格证据整理成 RagContext。
    Memory Context：记忆上下文，表示用户长期记忆召回结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DogKnowledgeLayer = Literal[
    "entry",
    "query_builder",
    "retrieval",
    "rerank",
    "quality",
    "context_builder",
    "memory_context",
    "strategy",
    "generation",
    "debug_report",
]


@dataclass(frozen=True)
class DogKnowledgeModuleContract:
    """
    DogKnowledgeAgent 内部模块职责契约。

    功能：
        描述 DogKnowledgeAgent 内部某一层的职责、输入、输出和设计原因。

    参数：
        layer:
            模块层级名称。
            例如 entry、query_builder、retrieval。

        module_name:
            模块英文名称。

        chinese_name:
            模块中文名称。

        responsibility:
            模块职责说明。

        expected_input:
            该模块预期输入。

        expected_output:
            该模块预期输出。

        should_not_do:
            该模块不应该承担的职责。

        enterprise_reason:
            企业级项目中为什么需要该职责拆分。

    返回值：
        DogKnowledgeModuleContract:
            一个不可变的模块职责契约对象。
    """

    layer: DogKnowledgeLayer
    module_name: str
    chinese_name: str
    responsibility: str
    expected_input: str
    expected_output: str
    should_not_do: tuple[str, ...]
    enterprise_reason: str


DOG_KNOWLEDGE_MODULE_CONTRACTS: tuple[
    DogKnowledgeModuleContract,
    ...
] = (
    DogKnowledgeModuleContract(
        layer="entry",
        module_name="entry_node",
        chinese_name="入口节点",
        responsibility=(
            "接收 RootAgent 路由过来的狗狗知识类问题，"
            "协调 DogKnowledgeAgent 内部流程。"
        ),
        expected_input=(
            "DogState，至少包含 question、user_id、session_id、trace_id、"
            "route_decision 等字段。"
        ),
        expected_output=(
            "更新后的 DogState 字段，例如 rag_query、retrieved_chunks、"
            "reranked_chunks、retrieval_quality、rag_context、memory_context、"
            "answer_strategy、final_answer、debug metadata。"
        ),
        should_not_do=(
            "不直接写死复杂 RAG filters。",
            "不直接拼接最终回答。",
            "不直接依赖旧 exact_search_agent。",
            "不直接依赖旧 recommendation_agent。",
            "不直接调用旧 query_parse 链路。",
            "不把所有 RAG 细节都堆在入口节点里。",
        ),
        enterprise_reason=(
            "入口层只负责流程编排，避免业务细节全部堆在入口节点中。"
        ),
    ),
    DogKnowledgeModuleContract(
        layer="query_builder",
        module_name="rag_query_builder",
        chinese_name="RAG 查询构建器",
        responsibility=(
            "根据用户问题、RootAgent route_decision、用户记忆和上下文，"
            "构建标准 RagQuery。"
        ),
        expected_input=(
            "用户问题、用户 ID、RootAgent route_decision、可选上下文信息。"
        ),
        expected_output=(
            "RagQuery，包含 question、user_id、top_k、filters、intent 等字段。"
        ),
        should_not_do=(
            "不执行向量检索。",
            "不执行 rerank。",
            "不判断召回质量。",
            "不生成最终回答。",
            "不直接操作 Chroma collection。",
            "不返回旧版查询解析结果对象。",
        ),
        enterprise_reason=(
            "查询构建层单独拆分后，可以独立测试 filters、intent、top_k 等逻辑。"
        ),
    ),
    DogKnowledgeModuleContract(
        layer="retrieval",
        module_name="retrieval_runner",
        chinese_name="检索执行器",
        responsibility=(
            "接收 RagQuery，调用 Retriever 执行初始检索，返回候选召回结果。"
        ),
        expected_input=(
            "RagQuery。"
        ),
        expected_output=(
            "retrieved_chunks，类型通常是 list[RagRetrievedChunk]。"
        ),
        should_not_do=(
            "不解析用户原始意图。",
            "不执行最终 rerank 策略选择。",
            "不决定最终回答风格。",
            "不拼接用户可见答案。",
            "不修改 RootAgent 路由决策。",
            "不直接构建最终 RagContext。",
        ),
        enterprise_reason=(
            "检索执行层独立后，可以单独替换 Retriever、增加 hybrid search 或修改向量库实现。"
        ),
    ),
    DogKnowledgeModuleContract(
        layer="rerank",
        module_name="reranker",
        chinese_name="重排器",
        responsibility=(
            "对 Retriever 返回的候选 chunks 进行重新排序，"
            "让更相关的内容排在前面。"
        ),
        expected_input=(
            "用户问题、RagQuery、retrieved_chunks。"
        ),
        expected_output=(
            "reranked_chunks，类型通常是 list[RagRetrievedChunk]。"
        ),
        should_not_do=(
            "不构建 RagQuery。",
            "不执行初始向量检索。",
            "不构建最终 RagContext。",
            "不生成最终答案。",
            "不直接决定 fallback。",
        ),
        enterprise_reason=(
            "重排层独立后，可以替换 cross-encoder、LLM rerank 或规则 rerank，"
            "不会影响 Retriever 和 AnswerGenerator。"
        ),
    ),
    DogKnowledgeModuleContract(
        layer="quality",
        module_name="retrieval_quality_evaluator",
        chinese_name="检索质量评估器",
        responsibility=(
            "在 rerank 之后判断 reranked_chunks 是否足够支持回答，"
            "例如是否为空、是否相关、是否需要 retry、是否需要 fallback。"
        ),
        expected_input=(
            "RagQuery 和 reranked_chunks。"
        ),
        expected_output=(
            "retrieval_quality 字段，例如 good、empty、low_confidence、"
            "need_retry、need_fallback。"
        ),
        should_not_do=(
            "不执行真实检索。",
            "不执行 rerank。",
            "不拼接 context_text。",
            "不直接调用 LLM 生成答案。",
            "不修改向量库。",
        ),
        enterprise_reason=(
            "企业 RAG 不能只看是否有返回 chunk，还要判断召回结果是否足够支撑回答。"
        ),
    ),
    DogKnowledgeModuleContract(
        layer="context_builder",
        module_name="rag_context_builder",
        chinese_name="RAG 上下文构建器",
        responsibility=(
            "在检索质量合格后，把 reranked_chunks 整理成最终可以注入 Prompt 的 RagContext。"
        ),
        expected_input=(
            "RagQuery、reranked_chunks、retrieval_quality。"
        ),
        expected_output=(
            "RagContext，包含 question、context_text、chunks、source_count、status 等字段。"
        ),
        should_not_do=(
            "不执行初始检索。",
            "不执行 rerank。",
            "不判断用户长期记忆。",
            "不生成最终回答。",
            "不把 memory_context 混入 RagContext。",
            "不修改 RootAgent route。",
        ),
        enterprise_reason=(
            "RagContext 应该代表通过质量检查后可用于回答的知识库证据，"
            "单独拆分可以避免低质量召回内容直接进入 Prompt。"
        ),
    ),
    DogKnowledgeModuleContract(
        layer="memory_context",
        module_name="memory_context_provider",
        chinese_name="记忆上下文提供器",
        responsibility=(
            "根据用户 ID、用户问题和可选上下文，检索用户长期记忆，"
            "并格式化为 answer_generator 可以注入 Prompt 的 memory_context。"
        ),
        expected_input=(
            "user_id、question、可选 route_decision、可选 RagQuery。"
        ),
        expected_output=(
            "memory_context 字符串或结构化 memory_context dict。"
        ),
        should_not_do=(
            "不构建 RagContext。",
            "不执行 RAG 文档检索。",
            "不执行 rerank。",
            "不生成最终回答。",
            "不修改用户长期记忆。",
            "不替代 AnswerStrategy。",
        ),
        enterprise_reason=(
            "用户长期记忆和 RAG 文档证据来源不同，单独拆分 memory_context "
            "可以避免知识库上下文和个性化上下文混在一起。"
        ),
    ),
    DogKnowledgeModuleContract(
        layer="strategy",
        module_name="answer_strategy_selector",
        chinese_name="回答策略选择器",
        responsibility=(
            "根据 query_type、retrieval_quality、RagContext 状态和 memory_context，"
            "选择回答策略。"
        ),
        expected_input=(
            "RagQuery、RagContext、retrieval_quality、memory_context。"
        ),
        expected_output=(
            "answer_strategy，例如 grounded_answer、fallback_answer、clarify_question。"
        ),
        should_not_do=(
            "不直接执行向量检索。",
            "不执行 rerank。",
            "不直接写 debug report 文件。",
            "不直接修改 RootAgent route。",
            "不直接生成最终答案。",
        ),
        enterprise_reason=(
            "回答策略单独拆分后，可以让不同类型问题走不同生成策略，避免生成逻辑混乱。"
        ),
    ),
    DogKnowledgeModuleContract(
        layer="generation",
        module_name="answer_generator",
        chinese_name="答案生成器",
        responsibility=(
            "根据用户问题、RagContext、memory_context 和 answer_strategy 生成最终回答。"
        ),
        expected_input=(
            "用户问题、RagContext、memory_context、answer_strategy、可选 debug metadata。"
        ),
        expected_output=(
            "final_answer 或 answer draft。"
        ),
        should_not_do=(
            "不重新解析 route。",
            "不重新构建 RagQuery。",
            "不重新执行检索。",
            "不重新执行 rerank。",
            "不直接修改检索结果。",
            "不绕过 answer_strategy。",
        ),
        enterprise_reason=(
            "答案生成层只关注表达和生成，避免和检索、路由、评估逻辑耦合。"
        ),
    ),
    DogKnowledgeModuleContract(
        layer="debug_report",
        module_name="dog_knowledge_debug_report",
        chinese_name="DogKnowledgeAgent 调试报告",
        responsibility=(
            "整理 DogKnowledgeAgent 内部 RAG 查询、检索、重排、质量判断、"
            "RagContext、记忆上下文、回答策略等调试字段。"
        ),
        expected_input=(
            "DogState 中的 rag_query、retrieved_chunks、reranked_chunks、"
            "retrieval_quality、rag_context、memory_context、answer_strategy 等字段。"
        ),
        expected_output=(
            "dog_knowledge_debug_report 或 Markdown debug report section。"
        ),
        should_not_do=(
            "不执行主业务流程。",
            "不改变检索结果。",
            "不改变 rerank 结果。",
            "不改变最终回答。",
            "不影响用户请求成功或失败。",
        ),
        enterprise_reason=(
            "Debug Report 独立后，开发者可以复盘 RAG 链路，而不会污染主业务逻辑。"
        ),
    ),
)


def get_dog_knowledge_module_contracts() -> tuple[
    DogKnowledgeModuleContract,
    ...
]:
    """
    获取 DogKnowledgeAgent 内部模块职责契约。

    功能：
        返回 DogKnowledgeAgent 在 V1.7.2 之后应该遵守的内部职责分层。

    参数：
        无。

    返回值：
        tuple[DogKnowledgeModuleContract, ...]:
            DogKnowledgeAgent 内部模块职责契约列表。
    """

    return DOG_KNOWLEDGE_MODULE_CONTRACTS


def get_contract_by_layer(
        layer: DogKnowledgeLayer,
) -> DogKnowledgeModuleContract:
    """
    根据 layer 获取 DogKnowledgeAgent 模块契约。

    功能：
        通过模块层级名称查找对应的职责契约。

    参数：
        layer:
            模块层级名称。
            例如 entry、query_builder、retrieval、rerank。

    返回值：
        DogKnowledgeModuleContract:
            匹配到的模块职责契约。

    异常：
        ValueError:
            如果没有找到对应 layer，则抛出异常。
    """

    for contract in DOG_KNOWLEDGE_MODULE_CONTRACTS:
        if contract.layer == layer:
            return contract

    raise ValueError(
        f"未知的 DogKnowledgeAgent 模块层级: {layer}"
    )


def get_expected_dog_knowledge_layers() -> tuple[
    DogKnowledgeLayer,
    ...
]:
    """
    获取 DogKnowledgeAgent 预期内部层级。

    功能：
        返回 V1.7.2 之后 DogKnowledgeAgent 应该逐步具备的职责层列表。

    参数：
        无。

    返回值：
        tuple[DogKnowledgeLayer, ...]:
            预期内部职责层。
    """

    return tuple(
        contract.layer
        for contract in DOG_KNOWLEDGE_MODULE_CONTRACTS
    )


def render_dog_knowledge_contract_markdown() -> str:
    """
    渲染 DogKnowledgeAgent 模块职责契约 Markdown。

    功能：
        将模块职责契约渲染成 Markdown 文本，
        方便写入项目文档或 Debug Report 附录。

    参数：
        无。

    返回值：
        str:
            Markdown 格式的 DogKnowledgeAgent 模块职责说明。
    """

    lines = [
        "# DogKnowledgeAgent 内部模块职责契约",
        "",
        "该文档由 module_contracts.py 中的结构契约生成。",
        "",
        "## 标准执行顺序",
        "",
        (
            "entry -> query_builder -> retrieval -> rerank -> quality -> "
            "context_builder -> memory_context -> strategy -> generation -> debug_report"
        ),
        "",
    ]

    for contract in DOG_KNOWLEDGE_MODULE_CONTRACTS:
        lines.extend(
            [
                f"## {contract.chinese_name} / {contract.module_name}",
                "",
                f"- 层级: `{contract.layer}`",
                f"- 职责: {contract.responsibility}",
                f"- 输入: {contract.expected_input}",
                f"- 输出: {contract.expected_output}",
                f"- 企业级设计原因: {contract.enterprise_reason}",
                "- 不应该做:",
            ]
        )

        for item in contract.should_not_do:
            lines.append(
                f"  - {item}"
            )

        lines.append(
            ""
        )

    return "\n".join(
        lines,
    )
