"""Skill 结构化输入检查器测试。"""

from __future__ import annotations

from src.skills import (
    SkillInputChecker,
    SkillLoader,
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
