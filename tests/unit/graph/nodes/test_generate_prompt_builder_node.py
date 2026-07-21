"""
DogKnowledgeAgent 回答提示词构建测试。

功能：
    验证长期记忆进入回答提示词时包含清楚的使用边界。
"""

from src.graph.nodes.generate_prompt_builder_node import (
    MEMORY_USAGE_RULES,
    build_generation_prompt,
)
from src.graph.schemas.answer_strategy import AnswerStrategy


def test_generation_prompt_does_not_treat_favorite_dog_as_pet_profile() -> None:
    """
    测试喜欢的犬种不能被自动当成当前宠物资料。

    功能：
        构建普通狗狗问答提示词，确认其中明确区分偏好记忆和当前宠物资料。

    参数：
        无。

    返回值：
        None：
            pytest 根据断言判断测试是否通过。
    """

    prompt = build_generation_prompt(
        state={
            "question": "请结合我的狗狗资料制定健康方案",
        },
        answer_strategy=AnswerStrategy(
            task_type="general_dog_qa",
            answer_style="general_explanation",
            reason="测试普通问答提示词。",
        ),
        context="测试检索上下文",
        context_source="rag_context",
        memory_text="- 用户喜欢的狗狗：Golden Retriever",
        history_text="",
    )

    assert MEMORY_USAGE_RULES in prompt
    assert "只代表偏好，不代表用户当前饲养该犬种" in prompt
    assert "不能用偏好记忆自动补全" in prompt
