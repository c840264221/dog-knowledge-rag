"""默认 Skill 目录和确定性选择器测试。"""

from __future__ import annotations

from src.skills import (
    SkillDefinition,
    SkillRegistry,
    SkillSelector,
    build_default_skill_registry,
)


def test_default_catalog_should_register_dog_training_plan() -> None:
    """
    测试默认目录包含狗狗训练计划技能。

    功能：
        验证 V1.23 第一个真实技能已经进入默认注册表。

    参数含义：
        无。

    返回值含义：
        None。
    """

    registry = build_default_skill_registry()

    skill = registry.require("dog-training-plan")

    assert skill.name == "狗狗训练计划"
    assert skill.instructions
    assert skill.allowed_tools == []


def test_selector_should_select_training_plan_from_user_question() -> None:
    """
    测试训练计划问题会选择狗狗训练技能。

    功能：
        验证用户自然语言可以通过 activation_hints 命中默认技能。

    参数含义：
        无。

    返回值含义：
        None。
    """

    selector = SkillSelector(build_default_skill_registry())

    result = selector.select("帮我为6岁的金毛制定训练计划")

    assert result.selected_skill_id == "dog-training-plan"
    assert result.matched_hints == ["制定训练计划", "训练计划"]
    assert result.candidate_skill_ids == ["dog-training-plan"]


def test_selector_should_match_planner_rewritten_training_step() -> None:
    """
    测试 Planner 在“制定”和“训练计划”之间插入目标后仍能命中技能。

    功能：
        覆盖真实多智能体步骤“制定等待与召回训练计划”，避免选择器只识别
        完全连续的“制定训练计划”而漏掉语义相同的步骤。

    参数含义：
        无。

    返回值含义：
        None。
    """

    selector = SkillSelector(build_default_skill_registry())

    result = selector.select("为6岁金毛制定等待与召回训练计划")

    assert result.selected_skill_id == "dog-training-plan"
    assert result.matched_hints == ["训练计划"]


def test_selector_should_not_force_skill_when_question_does_not_match() -> None:
    """
    测试普通问题不会被强行分配技能。

    功能：
        防止确定性选择器在没有命中依据时猜测技能。

    参数含义：
        无。

    返回值含义：
        None。
    """

    selector = SkillSelector(build_default_skill_registry())

    result = selector.select("成都今天的天气怎么样")

    assert result.selected_skill_id is None
    assert result.matched_hints == []
    assert result.candidate_skill_ids == []


def test_selector_should_use_stable_ranking_for_multiple_candidates() -> None:
    """
    测试多个技能命中时使用稳定排名规则。

    功能：
        验证命中提示更多的技能优先，避免依赖注册先后顺序。

    参数含义：
        无。

    返回值含义：
        None。
    """

    specific_skill = SkillDefinition(
        skill_id="specific-training-plan",
        name="精细训练计划",
        description="生成精细训练计划。",
        activation_hints=["训练计划", "金毛训练计划"],
        instructions=["生成计划"],
        output_contract="输出计划。",
    )
    general_skill = SkillDefinition(
        skill_id="general-training-plan",
        name="通用训练计划",
        description="生成通用训练计划。",
        activation_hints=["训练计划"],
        instructions=["生成计划"],
        output_contract="输出计划。",
    )
    selector = SkillSelector(
        SkillRegistry([general_skill, specific_skill])
    )

    result = selector.select("请生成一份金毛训练计划")

    assert result.selected_skill_id == "specific-training-plan"
    assert result.candidate_skill_ids == [
        "general-training-plan",
        "specific-training-plan",
    ]
