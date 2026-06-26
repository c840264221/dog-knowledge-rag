from typing import Literal

from pydantic import BaseModel, Field


KnowledgeTaskType = Literal[
    "exact_info",
    "recommendation",
    "comparison",
    "care_advice",
    "general_dog_qa",
]


AnswerStyle = Literal[
    "direct_fact",
    "ranked_recommendation",
    "comparison",
    "step_by_step_advice",
    "general_explanation",
]


class AnswerStrategy(BaseModel):
    """
    回答策略对象。

    功能：
        表示 generate_node 在生成答案前需要采用哪种回答策略。

        它不负责调用 LLM，也不负责检索。
        它只负责描述：
        1. 当前问题属于哪种知识任务。
        2. 应该使用哪种回答风格。
        3. 是否必须基于 RAG 上下文回答。
        4. 是否需要给出推荐理由。
        5. 是否需要给出注意事项。

    技术名词：
        Answer Strategy：
            回答策略。表示生成答案时采用的格式、语气、结构。

        Knowledge Task Type：
            知识任务类型。表示用户问题属于精确信息、推荐、对比、护理建议等哪一类。

        Answer Style：
            回答风格。表示答案应该是直接事实回答、推荐列表、对比表述、步骤建议等。

        RAG Grounding：
            RAG 依据约束。表示答案必须基于检索上下文，不允许凭空编造。

    字段：
        task_type:
            知识任务类型。

        answer_style:
            回答风格。

        must_use_context:
            是否必须基于 RAG 上下文回答。

        include_sources:
            是否需要在答案中体现依据来源。

        include_cautions:
            是否需要包含注意事项。

        include_recommendation_reason:
            是否需要包含推荐理由。

        reason:
            为什么选择这个回答策略。
            用于 Debug、Trace、Evaluation。
    """

    task_type: KnowledgeTaskType = Field(
        ...,
        description="知识任务类型"
    )

    answer_style: AnswerStyle = Field(
        ...,
        description="回答风格"
    )

    must_use_context: bool = Field(
        default=True,
        description="是否必须基于 RAG 上下文回答"
    )

    include_sources: bool = Field(
        default=True,
        description="是否需要体现来源依据"
    )

    include_cautions: bool = Field(
        default=True,
        description="是否需要包含注意事项"
    )

    include_recommendation_reason: bool = Field(
        default=False,
        description="是否需要包含推荐理由"
    )

    reason: str = Field(
        default="",
        description="策略选择原因"
    )