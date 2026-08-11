"""Skill 注册表和加载器测试。"""

from __future__ import annotations

import pytest

from src.skills import (
    DisabledSkillError,
    DuplicateSkillError,
    SkillDefinition,
    SkillInputRequirement,
    SkillLoader,
    SkillNotFoundError,
    SkillRegistry,
)


def build_skill(
    skill_id: str,
    *,
    enabled: bool = True,
) -> SkillDefinition:
    """
    构建注册表测试使用的技能。

    功能：
        根据指定编号和启用状态生成最小合法技能定义。

    参数含义：
        skill_id:
            测试技能编号。
        enabled:
            技能是否启用。

    返回值含义：
        SkillDefinition:
            可注册的测试技能。
    """

    return SkillDefinition(
        skill_id=skill_id,
        name=f"测试技能 {skill_id}",
        description="用于验证 Skill 基础链路。",
        required_inputs=[
            SkillInputRequirement(
                input_id="question",
                name="用户问题",
            )
        ],
        instructions=["读取问题", "生成回答"],
        allowed_tools=["dog_knowledge_search"],
        output_contract="输出结构清晰的中文回答。",
        guardrails=["不得编造资料"],
        enabled=enabled,
    )


def test_registry_should_register_and_require_skill() -> None:
    """
    测试注册表可以注册并严格读取技能。

    功能：
        验证 SkillDefinition 从创建到被注册表取出的基本链路。

    参数含义：
        无。

    返回值含义：
        None。
    """

    skill = build_skill("dog-training-plan")
    registry = SkillRegistry()

    registry.register(skill)

    assert registry.require("dog-training-plan") is skill


def test_registry_should_reject_duplicate_skill_id() -> None:
    """
    测试重复技能编号不会静默覆盖。

    功能：
        保证技能版本或来源冲突能够尽早暴露。

    参数含义：
        无。

    返回值含义：
        None。
    """

    registry = SkillRegistry([build_skill("dog-training-plan")])

    with pytest.raises(DuplicateSkillError, match="已经注册"):
        registry.register(build_skill("dog-training-plan"))


def test_registry_should_list_only_enabled_skills_in_stable_order() -> None:
    """
    测试注册表只列出启用技能并稳定排序。

    功能：
        确保后续选择器看到的候选目录不包含停用技能，且顺序可预测。

    参数含义：
        无。

    返回值含义：
        None。
    """

    registry = SkillRegistry(
        [
            build_skill("z-skill"),
            build_skill("disabled-skill", enabled=False),
            build_skill("a-skill"),
        ]
    )

    assert [
        skill.skill_id
        for skill in registry.list_enabled()
    ] == ["a-skill", "z-skill"]


def test_loader_should_reject_missing_and_disabled_skills() -> None:
    """
    测试加载器拒绝不存在或已停用技能。

    功能：
        验证加载阶段会阻止无效技能进入 Agent 上下文。

    参数含义：
        无。

    返回值含义：
        None。
    """

    loader = SkillLoader(
        SkillRegistry([build_skill("disabled-skill", enabled=False)])
    )

    with pytest.raises(SkillNotFoundError, match="技能不存在"):
        loader.load("missing-skill")
    with pytest.raises(DisabledSkillError, match="技能已停用"):
        loader.load("disabled-skill")


def test_loader_should_render_deterministic_agent_context() -> None:
    """
    测试加载器可以生成稳定的 Agent 上下文。

    功能：
        验证技能结构会按照固定顺序转换成包含步骤、工具和边界的文本。

    参数含义：
        无。

    返回值含义：
        None。
    """

    loader = SkillLoader(
        SkillRegistry([build_skill("dog-training-plan")])
    )

    context = loader.render_context("dog-training-plan")

    assert "技能：测试技能 dog-training-plan（dog-training-plan@1.0.0）" in context
    assert "输入要求：用户问题（question，必须提供）" in context
    assert "执行步骤：\n1. 读取问题\n2. 生成回答" in context
    assert "允许工具：dog_knowledge_search" in context
    assert "输出要求：输出结构清晰的中文回答。" in context
    assert "执行边界：不得编造资料" in context


def test_registry_catalog_should_not_expose_full_instructions() -> None:
    """
    测试精简目录不会暴露完整技能步骤。

    功能：
        验证发现阶段只拿到选择所需元数据，完整 instructions 仍需按需加载。

    参数含义：
        无。

    返回值含义：
        None。
    """

    registry = SkillRegistry([build_skill("dog-training-plan")])

    catalog_dump = registry.list_catalog()[0].model_dump()

    assert catalog_dump["skill_id"] == "dog-training-plan"
    assert "instructions" not in catalog_dump
    assert "output_contract" not in catalog_dump
    assert "guardrails" not in catalog_dump


def test_loader_should_render_compact_catalog_without_full_steps() -> None:
    """
    测试加载器只用简短字段渲染技能目录。

    功能：
        确认目录文本不会提前加载完整执行步骤，体现渐进式加载边界。

    参数含义：
        无。

    返回值含义：
        None。
    """

    loader = SkillLoader(
        SkillRegistry([build_skill("dog-training-plan")])
    )

    catalog_text = loader.render_catalog()

    assert "dog-training-plan@1.0.0" in catalog_text
    assert "用于验证 Skill 基础链路" in catalog_text
    assert "读取问题" not in catalog_text
    assert "不得编造资料" not in catalog_text
