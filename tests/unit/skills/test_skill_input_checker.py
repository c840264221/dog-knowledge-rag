"""Skill 结构化输入检查器测试。"""

from __future__ import annotations

import pytest

from src.skills import (
    SkillDefinition,
    SkillInputChecker,
    SkillInputRequirement,
    SkillLoader,
    SkillRegistry,
    build_default_skill_registry,
)


def build_checker() -> SkillInputChecker:
    """
    构建使用默认技能目录的输入检查器。

    功能：
        统一测试所需的 Registry、Loader 和 Checker 装配过程。

    参数含义：
        无。

    返回值含义：
        SkillInputChecker:
            可以检查默认训练技能输入的测试对象。
    """

    registry = build_default_skill_registry()
    return SkillInputChecker(SkillLoader(registry))


def test_input_checker_should_report_missing_and_empty_inputs() -> None:
    """
    测试检查器会区分缺失字段和空值字段。

    功能：
        breed 完全不存在，current_behavior 虽然存在但为空，两者都应阻止执行。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = build_checker().check(
        "dog-training-plan",
        {
            "age": "6岁",
            "current_behavior": "   ",
            "training_goal": "学习等待指令",
            "unrelated_state": "不应传给 Skill",
        },
    )

    assert result.is_ready is False
    assert result.available_input_ids == ["age", "training_goal"]
    assert result.missing_input_ids == ["breed", "current_behavior"]
    assert [
        (
            requirement.input_id,
            requirement.name,
            requirement.requirement_level,
            requirement.source_mappings,
        )
        for requirement in result.missing_input_requirements
    ] == [
        (
            "breed",
            "犬种",
            "degradable",
            {"pet_profile": "breed"},
        ),
        (
            "current_behavior",
            "当前行为基础",
            "degradable",
            {},
        ),
    ]
    assert result.missing_hard_required_input_ids == []
    assert result.missing_degradable_input_ids == [
        "breed",
        "current_behavior",
    ]
    assert result.missing_optional_input_ids == []
    assert result.can_run_degraded is True
    assert result.empty_input_ids == ["current_behavior"]
    assert result.accepted_inputs == {
        "age": "6岁",
        "training_goal": "学习等待指令",
    }
    assert "犬种" in result.clarification_prompt
    assert "当前行为基础" in result.clarification_prompt
    assert "breed" not in result.clarification_prompt


def test_input_checker_should_be_ready_when_all_inputs_are_available() -> None:
    """
    测试所有必需输入具备后允许执行 Skill。

    功能：
        验证完整档案会得到 is_ready=True，并且不再生成澄清提示。

    参数含义：
        无。

    返回值含义：
        None。
    """

    inputs = {
        "breed": "金毛",
        "age": "6岁",
        "current_behavior": "会坐下，还不会等待",
        "training_goal": "学习等待和召回",
    }

    result = build_checker().check("dog-training-plan", inputs)

    assert result.is_ready is True
    assert result.missing_input_ids == []
    assert result.missing_input_requirements == []
    assert result.missing_hard_required_input_ids == []
    assert result.missing_degradable_input_ids == []
    assert result.missing_optional_input_ids == []
    assert result.can_run_degraded is False
    assert result.empty_input_ids == []
    assert result.accepted_inputs == inputs
    assert result.clarification_prompt == ""


def test_input_checker_should_treat_zero_and_false_as_valid_values() -> None:
    """
    测试数字零和布尔 False 不会被误判为空。

    功能：
        防止使用简单的 if not value 检查时丢失合法业务值。

    参数含义：
        无。

    返回值含义：
        None。
    """

    assert SkillInputChecker._is_empty(0) is False
    assert SkillInputChecker._is_empty(False) is False
    assert SkillInputChecker._is_empty([]) is True
    assert SkillInputChecker._is_empty(None) is True


def test_input_checker_should_classify_all_requirement_levels() -> None:
    """
    验证检查器会按输入缺失影响级别分类。

    功能：
        强制必填和可简化字段会阻止标准执行，可选字段只记录而不触发澄清；
        只要仍缺强制字段，就不能进入简化执行。

    参数含义：
        无。

    返回值含义：
        None。
    """

    skill = SkillDefinition(
        skill_id="requirement-level-test",
        name="输入级别测试",
        description="验证三类技能输入的检查行为。",
        required_inputs=[
            SkillInputRequirement(
                input_id="hard_value",
                name="强制资料",
            ),
            SkillInputRequirement(
                input_id="degradable_value",
                name="可简化资料",
                requirement_level="degradable",
            ),
            SkillInputRequirement(
                input_id="optional_value",
                name="可选资料",
                requirement_level="optional",
            ),
        ],
        instructions=["验证输入。"],
        output_contract="输出验证结果。",
    )
    checker = SkillInputChecker(
        SkillLoader(SkillRegistry([skill]))
    )

    result = checker.check("requirement-level-test", {})

    assert result.is_ready is False
    assert result.missing_input_ids == [
        "hard_value",
        "degradable_value",
    ]
    assert result.missing_hard_required_input_ids == ["hard_value"]
    assert result.missing_degradable_input_ids == ["degradable_value"]
    assert result.missing_optional_input_ids == ["optional_value"]
    assert result.can_run_degraded is False
    assert "强制资料" in result.clarification_prompt
    assert "可简化资料" in result.clarification_prompt
    assert "可选资料" not in result.clarification_prompt


def test_input_checker_should_allow_degraded_choice_without_hard_missing() -> None:
    """验证只缺可简化字段时会明确标记可以选择简化执行。"""

    result = build_checker().check(
        "dog-training-plan",
        {
            "age": "6岁",
            "training_goal": "学习等待和召回",
        },
    )

    assert result.is_ready is False
    assert result.missing_hard_required_input_ids == []
    assert result.missing_degradable_input_ids == [
        "breed",
        "current_behavior",
    ]
    assert result.can_run_degraded is True


def test_input_checker_should_be_ready_when_degradable_inputs_are_ignored() -> None:
    """验证用户同意后，可简化缺失字段不再阻止 Skill 执行。"""

    result = build_checker().check(
        "dog-training-plan",
        {
            "age": "6岁",
            "training_goal": "学习等待和召回",
        },
        ignored_input_ids=["breed", "current_behavior"],
    )

    assert result.is_ready is True
    assert result.missing_input_ids == []
    assert result.ignored_degradable_input_ids == [
        "breed",
        "current_behavior",
    ]
    assert result.can_run_degraded is False
    assert result.clarification_prompt == ""


def test_input_checker_should_reject_ignoring_hard_required_input() -> None:
    """验证简化模式不能绕过年龄等强制必填输入。"""

    with pytest.raises(ValueError, match="只有可简化输入允许被忽略"):
        build_checker().check(
            "dog-training-plan",
            {"training_goal": "学习等待"},
            ignored_input_ids=["age"],
        )
