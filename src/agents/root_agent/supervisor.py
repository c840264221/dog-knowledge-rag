from __future__ import annotations

from typing import (
    Any,
)

from langchain_core.messages import HumanMessage

from src.agents.root_agent.schemas import (
    RootRouteDecision,
)
from src.common.decorators.safe_node import (
    safe_node,
)
from src.common.decorators.state_validation import (
    validate_state,
)
from src.common.decorators.validation_input import (
    validate_question,
)
from src.graph.states.dog_state import (
    DogState,
)
from src.logger import logger
from src.runtime.context import (
    runtime_ctx,
)

from src.agents.root_agent.observability import (
    build_root_route_observability_payload,
    record_root_route_timeline,
)

from src.agents.root_agent.debug_report import (
    build_root_debug_report_fields,
)


DOG_DOMAIN_KEYWORDS = [
    "狗",
    "犬",
    "犬种",
    "品种",
    "幼犬",
    "狗狗",
    "宠物狗",
    "金毛",
    "拉布拉多",
    "柯基",
    "哈士奇",
    "边牧",
    "泰迪",
    "贵宾",
    "柴犬",
    "萨摩耶",
    "阿拉斯加",
    "比熊",
    "博美",
    "雪纳瑞",
    "德牧",
    "dog",
    "dogs",
    "puppy",
    "breed",
    "breeds",
    "golden retriever",
    "labrador",
    "corgi",
    "husky",
    "poodle",
    "shiba",
]


DOG_RECOMMENDATION_KEYWORDS = [
    "推荐",
    "适合养什么",
    "适合我养",
    "帮我选",
    "哪种狗适合",
    "什么狗适合",
    "新手养什么",
    "公寓养什么",
    "recommend",
    "suggest",
    "which dog",
    "what dog",
]


DOG_KNOWLEDGE_KEYWORDS = [
    "性格",
    "寿命",
    "身高",
    "体重",
    "掉毛",
    "爱叫",
    "吠叫",
    "训练",
    "护理",
    "喂养",
    "适合新手",
    "适合公寓",
    "区别",
    "对比",
    "比较",
    "temperament",
    "lifespan",
    "height",
    "weight",
    "shedding",
    "barking",
    "training",
    "grooming",
    "compare",
]


TOOL_KEYWORDS = [
    "天气",
    "气温",
    "下雨",
    "日期",
    "今天几号",
    "现在几点",
    "几点",
    "时间",
    "搜索",
    "查一下",
    "weather",
    "temperature",
    "date",
    "time",
    "search",
]


FINISH_KEYWORDS = [
    "不用了",
    "先这样",
    "结束",
    "finish",
    "stop",
]


def normalize_question(
        question: Any,
) -> str:
    """
    归一化用户问题。

    功能：
        将任意输入安全转换为去除首尾空白的字符串。

    参数：
        question:
            原始用户问题。

    返回值：
        str:
            归一化后的用户问题文本。
    """

    return str(
        question or ""
    ).strip()


def find_matched_keywords(
        question: str,
        keywords: list[str],
) -> list[str]:
    """
    查找命中的关键词。

    功能：
        在用户问题中查找指定关键词列表的命中项。
        当前使用简单字符串包含关系，后续可以升级为 Embedding 路由或 LLM Router。

    参数：
        question:
            用户问题。

        keywords:
            候选关键词列表。

    返回值：
        list[str]:
            命中的关键词列表。
    """

    lowered_question = question.lower()

    matched_keywords = [
        keyword
        for keyword in keywords
        if keyword.lower() in lowered_question
    ]

    return matched_keywords


def decide_root_route(
        question: str,
        state: DogState | None = None,
) -> RootRouteDecision:
    """
    决定 RootAgent 主图路由。

    功能：
        根据用户问题做主图粗路由：
        1. 结束类问题 -> FINISH
        2. 狗狗推荐问题 -> dog_knowledge_agent
        3. 狗狗知识问题 -> dog_knowledge_agent
        4. 工具类问题 -> tool_agent
        5. 其他问题 -> general_agent

        注意：
            本函数不调用旧版 query_parse，也不调用 LLM。
            Root 层只做粗路由，不生成 RagQuery，也不解析 filters。

    参数：
        question:
            用户问题。

        state:
            当前 DogState。
            第一版规则路由暂时不强依赖 state，保留参数方便后续使用 memory_context、history 等信息。

    返回值：
        RootRouteDecision:
            RootAgent 路由决策对象。
    """

    # 澄清适配器已补全上一轮工具参数时，优先恢复 ToolAgent 调用。
    clarification_resolution = (
        state.get(
            "tool_agent_clarification_resolution",
            {},
        )
        if state
        else {}
    )
    if state and (
        state.get(
            "tool_agent_clarification_resume_ready"
        )
        or (
            isinstance(
                clarification_resolution,
                dict,
            )
            and clarification_resolution.get(
                "action"
            ) == "partial"
        )
    ):
        return RootRouteDecision(
            route="tool_agent",
            query_type="tool_request",
            confidence=1.0,
            reason="用户输入匹配待补全工具参数，继续上一轮工具调用。",
            requires_rag=False,
            requires_tool=True,
            requires_memory=False,
            hints={
                "clarification_resume": True,
            },
        )

    finish_matches = find_matched_keywords(
        question=question,
        keywords=FINISH_KEYWORDS,
    )

    if finish_matches:
        return RootRouteDecision(
            route="FINISH",
            query_type="finish",
            confidence=0.95,
            reason="用户问题命中结束类表达，可以直接结束主图。",
            requires_rag=False,
            requires_tool=False,
            requires_memory=False,
            hints={
                "matched_keywords": finish_matches,
                "root_layer_policy": "Root 只做结束意图识别，不处理业务细节。",
            },
        )

    recommendation_matches = find_matched_keywords(
        question=question,
        keywords=DOG_RECOMMENDATION_KEYWORDS,
    )

    domain_matches = find_matched_keywords(
        question=question,
        keywords=DOG_DOMAIN_KEYWORDS,
    )

    knowledge_matches = find_matched_keywords(
        question=question,
        keywords=DOG_KNOWLEDGE_KEYWORDS,
    )

    if recommendation_matches:
        return RootRouteDecision(
            route="dog_knowledge_agent",
            query_type="dog_recommendation",
            confidence=0.90,
            reason="用户问题命中犬种推荐关键词，需要进入 dog_knowledge_agent。",
            requires_rag=True,
            requires_tool=False,
            requires_memory=True,
            hints={
                "matched_recommendation_keywords": recommendation_matches,
                "matched_domain_keywords": domain_matches,
                "root_layer_policy": "Root 只判断推荐大类，不解析推荐 filters。",
            },
        )

    if domain_matches or knowledge_matches:
        confidence = 0.88

        if domain_matches and knowledge_matches:
            confidence = 0.92

        return RootRouteDecision(
            route="dog_knowledge_agent",
            query_type="dog_knowledge",
            confidence=confidence,
            reason="用户问题命中狗狗领域关键词，需要进入 dog_knowledge_agent 进行知识库问答。",
            requires_rag=True,
            requires_tool=False,
            requires_memory=True,
            hints={
                "matched_domain_keywords": domain_matches,
                "matched_knowledge_keywords": knowledge_matches,
                "root_layer_policy": "Root 不生成 RagQuery，RAG 细解析交给 dog_knowledge_agent。",
            },
        )

    tool_matches = find_matched_keywords(
        question=question,
        keywords=TOOL_KEYWORDS,
    )

    if tool_matches:
        return RootRouteDecision(
            route="tool_agent",
            query_type="tool_request",
            confidence=0.86,
            reason="用户问题命中工具请求关键词，需要进入工具处理链路。",
            requires_rag=False,
            requires_tool=True,
            requires_memory=True,
            hints={
                "matched_keywords": tool_matches,
                "note": "V1.8 起 tool_agent 路由进入新版 ToolAgent 独立子图。",
            },
        )

    return RootRouteDecision(
        route="general_agent",
        query_type="general_chat",
        confidence=0.70,
        reason="用户问题没有命中狗狗知识或工具关键词，进入 general_agent。",
        requires_rag=False,
        requires_tool=False,
        requires_memory=True,
        hints={
            "root_layer_policy": "默认兜底到 general_agent。",
        },
    )


def record_runtime_route(
        decision: RootRouteDecision,
) -> None:
    """
    记录 RootAgent 路由信息到 Runtime Context。

    功能：
        将路由结果写入 runtime state 和 timeline。
        如果当前测试环境没有 runtime_ctx，本函数只记录 debug，不中断主流程。

    参数：
        decision:
            RootAgent 路由决策对象。

    返回值：
        None:
            只执行副作用记录。
    """

    try:
        runtime_context = runtime_ctx.get()

        if runtime_context is None:
            return

        runtime_context.state().set_node(
            "root_supervisor_node",
        )

        runtime_context.state().set_agent(
            decision.route,
        )

        runtime_context.timeline().add_event(
            event_type="node",
            name="root_supervisor_node",
        )

        runtime_context.timeline().add_event(
            event_type="agent",
            name=decision.route,
        )

    except Exception as exc:
        logger.debug(
            f"RootAgent 写入 runtime context 失败: {exc}"
        )


@safe_node(
    fallback=lambda state, e: {
        "next_agent": "general_agent",
        "current_agent": "root_agent",
        "route_decision": RootRouteDecision(
            route="general_agent",
            query_type="general_chat",
            confidence=0.0,
            reason="root_supervisor_node 执行失败，safe_node fallback 兜底到 general_agent。",
            requires_rag=False,
            requires_tool=False,
            requires_memory=True,
            hints={
                "error": str(e),
                "fallback": True,
            },
        ).model_dump(),
        "error": str(e),
    }
)
@validate_question
@validate_state(
    required_keys=[
        "question",
    ]
)
async def root_supervisor_node(
        state: DogState,
) -> dict[str, Any]:
    """
    Root Supervisor 根调度节点。

    功能：
        判断用户问题应该进入哪个主 Agent。

        V1.7 设计原则：
        1. 不使用旧版查询解析模块。
        2. 不输出旧版查询解析结果模型。
        3. 不在 Root 层解析 filters、tags、features、dog_name。
        4. 不在 Root 层创建 RagQuery。
        5. 只做主图粗路由。
        6. 狗狗知识细解析交给 dog_knowledge_agent 内部 extractor。

    参数：
        state:
            DogState，当前 LangGraph 状态。
            至少需要包含 question。

    返回值：
        dict[str, Any]:
            LangGraph Partial State（局部状态）。
            主要包含：
            - route_decision
            - next_agent
            - current_agent
            - messages

    输出格式：
        {
            "route_decision": {...},
            "next_agent": "dog_knowledge_agent",
            "current_agent": "root_agent",
            "messages": [HumanMessage(content=question)]
        }

    专业名词：
        Root Supervisor：
            根调度器。主图最外层的路由节点，负责把请求分发到不同子 Agent。

        Rule-based Router：
            规则路由器。用关键词和状态字段做稳定分类，不依赖 LLM。

        Coarse Routing：
            粗路由。只判断大方向，不解析业务细节。
    """

    question = normalize_question(
        state.get(
            "question",
            "",
        )
    )

    decision = decide_root_route(
        question=question,
        state=state,
    )

    root_observability = build_root_route_observability_payload(
        question=question,
        decision=decision,
    )

    timeline_recorded = record_root_route_timeline(
        payload=root_observability,
    )

    root_observability[
        "timeline_recorded"
    ] = timeline_recorded

    root_debug_report = build_root_debug_report_fields(
        root_observability=root_observability,
    )

    logger.info(
        "[root_supervisor_node] "
        f"route={decision.route}, "
        f"query_type={decision.query_type}, "
        f"confidence={decision.confidence}, "
        f"reason={decision.reason}, "
        f"timeline_recorded={timeline_recorded}"
    )

    return {
        "route_decision": decision.model_dump(),
        "root_observability": root_observability,
        "root_debug_report": root_debug_report,
        "next_agent": decision.route,
        "current_agent": "root_agent",
    }
