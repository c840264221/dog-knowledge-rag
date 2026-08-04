"""Skill 基础 Schema 测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.skills import SkillDefinition, SkillInputRequirement


def build_skill(**overrides: object) -> SkillDefinition:
    """
    构建测试使用的标准技能定义。

    功能：
        提供一份合法默认数据，并允许单个测试覆盖目标字段。

    参数含义：
        overrides:
            需要覆盖的技能字段。

    返回值含义：
        SkillDefinition:
            经过校验的测试技能定义。
    """

    data = {
        "skill_id": "dog-training-plan",
        "name": "狗狗训练计划",
        "description": "根据狗狗档案生成分阶段训练计划。",
        "activation_hints": ["训练计划", "行为训练"],
        "required_inputs": [
            SkillInputRequirement(input_id="breed", name="犬种"),
            SkillInputRequirement(input_id="age", name="年龄"),
            SkillInputRequirement(
                input_id="training_goal",
                name="训练目标",
            ),
        ],
        "instructions": ["检查档案", "查询训练知识", "生成计划"],
        "allowed_tools": ["dog_knowledge_search"],
        "output_contract": "输出目标、每日安排和注意事项。",
        "guardrails": ["不替代兽医诊断"],
        "version": "1.0.0",
    }
    data.update(overrides)
    return SkillDefinition.model_validate(data)


def test_skill_definition_should_keep_structured_required_inputs() -> None:
    """
        测试技能会保存机器字段和中文名称分离的输入要求。

    功能：
        验证后续输入检查可以使用 input_id，同时保留用户可读名称。

    参数含义：
        无。

    返回值含义：
        None。
    """

    skill = build_skill()

    assert skill.required_inputs[0].input_id == "breed"
    assert skill.required_inputs[0].name == "犬种"


@pytest.mark.parametrize(
    "skill_id",
    ["DogTraining", "dog_training", "dog--training", "dog training"],
)
def test_skill_definition_should_reject_invalid_skill_id(
    skill_id: str,
) -> None:
    """
    测试技能编号只接受稳定的连字符格式。

    功能：
        避免同一技能因大小写、下划线或空格产生多个身份。

    参数含义：
        skill_id:
            当前准备验证的非法编号。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError, match="skill_id"):
        build_skill(skill_id=skill_id)


def test_skill_definition_should_require_at_least_one_instruction() -> None:
    """
    测试技能必须包含执行步骤。

    功能：
        防止只有名称和描述、实际没有工作流程的 Prompt 文件冒充 Skill。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError):
        build_skill(instructions=[])


def test_skill_definition_should_reject_duplicate_list_items() -> None:
    """
    测试技能列表字段不能包含重复内容。

    功能：
        避免重复步骤或边界浪费上下文并干扰 Agent。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError, match="重复值"):
        build_skill(allowed_tools=["weather", "weather"])


def test_skill_definition_should_reject_duplicate_required_input_ids() -> None:
    """
    测试技能不能重复声明同一个输入编号。

    功能：
        避免输入检查器重复检查和重复询问同一个字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValidationError, match="重复 input_id"):
        build_skill(
            required_inputs=[
                SkillInputRequirement(input_id="age", name="年龄"),
                SkillInputRequirement(input_id="age", name="狗狗年龄"),
            ]
        )
