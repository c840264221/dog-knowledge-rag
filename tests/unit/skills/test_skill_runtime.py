"""Skill 统一运行器测试。"""

from __future__ import annotations

from src.skills import (
    SkillDefinition,
    SkillInputRequirement,
    SkillRegistry,
    build_default_skill_runtime,
)


def test_runtime_should_return_no_skill_for_unmatched_question() -> None:
    """
    测试普通问题不会被强行套用 Skill。

    功能：
        没有命中技能提示时应直接返回 no_skill，并且不执行输入准备。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = build_default_skill_runtime().prepare(
        user_text="成都今天的天气怎么样",
    )

    assert result.status == "no_skill"
    assert result.selection.selected_skill_id is None
    assert result.extraction is None
    assert result.input_check is None
    assert result.skill_context == ""


def test_runtime_should_wait_when_required_inputs_are_missing() -> None:
    """
    测试命中技能但资料不足时进入等待输入状态。

    功能：
        犬种和年龄虽然已经提取，但行为基础和训练目标缺失，因此不能提前
        加载完整 Skill 上下文。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = build_default_skill_runtime().prepare(
        user_text="帮我为6岁的金毛制定训练计划。",
    )

    assert result.status == "awaiting_input"
    assert result.selection.selected_skill_id == "dog-training-plan"
    assert result.extraction is not None
    assert result.extraction.merged_inputs == {
        "breed": "Golden Retriever",
        "age": "6岁",
    }
    assert result.input_check is not None
    assert result.input_check.missing_input_ids == [
        "current_behavior",
        "training_goal",
    ]
    assert result.skill_context == ""


def test_runtime_should_load_context_when_inputs_are_ready() -> None:
    """
    测试输入完整后生成可注入 Agent 的技能上下文。

    功能：
        一轮文本同时包含全部必需信息时，运行器应返回 ready 和完整说明。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = build_default_skill_runtime().prepare(
        user_text=(
            "帮我为6岁的金毛制定训练计划，它目前会坐下，"
            "希望学习等待和召回。"
        ),
    )

    assert result.status == "ready"
    assert result.input_check is not None
    assert result.input_check.is_ready is True
    assert "技能：狗狗训练计划" in result.skill_context
    assert "执行步骤" in result.skill_context


def test_training_step_title_should_extract_goal_without_fake_behavior() -> None:
    """
    测试计划器生成的训练步骤标题可以提取目标且不会误识别普通词语。

    功能：
        覆盖真实日志中的“制定等待和召回训练计划”以及“社会化保持等内容”，
        防止把“社会”中的“会”错误当成狗狗已经掌握的行为。

    参数含义：
        无。

    返回值含义：
        None，pytest 根据结构化提取结果判断是否通过。
    """

    runtime = build_default_skill_runtime()
    title_extraction = runtime.extract_inputs(
        skill_id="dog-training-plan",
        user_text="制定等待和召回训练计划",
    )
    generated_text_extraction = runtime.extract_inputs(
        skill_id="dog-training-plan",
        user_text=(
            "制定日常训练方案，包括体能训练、脑力游戏、"
            "行为巩固及社会化保持等内容。"
        ),
    )

    assert title_extraction.extracted_inputs == {
        "training_goal": "等待和召回",
    }
    assert "current_behavior" not in (
        generated_text_extraction.extracted_inputs
    )

    input_check = runtime.check_inputs(
        skill_id="dog-training-plan",
        provided_inputs={
            "age": "6岁",
            **title_extraction.extracted_inputs,
        },
    )
    assert input_check.missing_hard_required_input_ids == []
    assert input_check.missing_degradable_input_ids == [
        "breed",
        "current_behavior",
    ]
    assert input_check.can_run_degraded is True
    assert "简化执行" in input_check.clarification_prompt


def test_runtime_should_resume_with_saved_skill_and_inputs() -> None:
    """
    测试恢复时使用上一轮 Skill 编号和已保存输入继续执行。

    功能：
        第二轮回答本身没有“训练计划”触发词，因此必须使用上一轮 skill_id，
        并把本轮补充字段与已有犬种、年龄合并后进入 ready。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = build_default_skill_runtime().prepare(
        user_text="它目前会坐下，希望学习等待和召回。",
        selected_skill_id="dog-training-plan",
        existing_inputs={
            "breed": "Golden Retriever",
            "age": "6岁",
        },
    )

    assert result.status == "ready"
    assert result.selection.source == "provided_skill_id"
    assert result.extraction is not None
    assert result.extraction.merged_inputs == {
        "breed": "Golden Retriever",
        "age": "6岁",
        "current_behavior": "坐下",
        "training_goal": "学习等待和召回",
    }


def test_runtime_should_use_pet_profile_as_default_inputs() -> None:
    """
    测试宠物档案可以补全 Skill 输入，但用户本轮表达仍具有最高优先级。

    参数含义：无。

    返回值含义：None，pytest 根据合并后的技能输入判断是否通过。
    """

    result = build_default_skill_runtime().prepare(
        user_text=(
            "帮我制定训练计划，它目前会坐下，希望学习等待和召回。"
        ),
        available_input_sources={
            "pet_profile": {
                "breed": "金毛",
                "age_years": "6岁",
                "training_goal": "改善随行",
            }
        },
    )

    assert result.status == "ready"
    assert result.extraction is not None
    assert result.extraction.merged_inputs == {
        "breed": "金毛",
        "age": "6岁",
        "current_behavior": "坐下",
        "training_goal": "学习等待和召回",
    }


def test_runtime_should_normalize_profile_age_for_skill_input() -> None:
    """
    测试宠物档案中的无单位年龄会转换成 Skill 使用的带单位文本。

    参数含义：无。
    返回值含义：None，pytest 根据最终技能输入判断是否通过。
    """

    result = build_default_skill_runtime().prepare(
        user_text=(
            "帮我制定训练计划，它目前会坐下，希望学习等待。"
        ),
        available_input_sources={
            "pet_profile": {
                "breed": "金毛",
                "age_years": 6,
            }
        },
    )

    assert result.status == "ready"
    assert result.extraction is not None
    assert result.extraction.merged_inputs["age"] == "6岁"


def test_runtime_should_ignore_invalid_historical_profile_age() -> None:
    """
    测试历史脏年龄不会中断 Skill，而是继续作为缺失字段等待补充。

    参数含义：无。
    返回值含义：None，pytest 根据等待状态和缺失字段判断是否通过。
    """

    result = build_default_skill_runtime().prepare(
        user_text=(
            "帮我制定训练计划，它目前会坐下，希望学习等待。"
        ),
        available_input_sources={
            "pet_profile": {
                "breed": "金毛",
                "age_years": "年龄未知",
            }
        },
    )

    assert result.status == "awaiting_input"
    assert result.input_check is not None
    assert result.input_check.missing_input_ids == ["age"]


def test_runtime_should_request_optional_source_field_for_enrichment() -> None:
    """
    验证可选输入会尝试从外部数据源补全。

    功能：
        可选字段可以提高 Skill 执行质量，因此应该进入宠物档案查询范围；
        它只是在查询不到时不阻止执行，也不要求用户补充。

    参数含义：
        无。

    返回值含义：
        None。
    """

    skill = SkillDefinition(
        skill_id="dog-training-plan",
        name="狗狗训练计划",
        description="根据资料生成训练计划。",
        required_inputs=[
            SkillInputRequirement(
                input_id="age",
                name="年龄",
                source_mappings={"pet_profile": "age_years"},
            ),
            SkillInputRequirement(
                input_id="preferred_reward",
                name="偏好的奖励",
                requirement_level="optional",
                source_mappings={
                    "pet_profile": "preferred_reward"
                },
            ),
        ],
        instructions=["生成训练计划。"],
        output_contract="输出训练计划。",
    )
    runtime = build_default_skill_runtime(SkillRegistry([skill]))

    assert runtime.get_source_required_fields(
        skill_id="dog-training-plan",
        source_name="pet_profile",
    ) == ["age_years", "preferred_reward"]
    assert runtime.get_missing_source_required_fields(
        skill_id="dog-training-plan",
        source_name="pet_profile",
        provided_inputs={"age": "6岁"},
    ) == ["preferred_reward"]

    input_check = runtime.check_inputs(
        skill_id="dog-training-plan",
        provided_inputs={"age": "6岁"},
    )
    assert input_check.is_ready is True
    assert input_check.missing_optional_input_ids == [
        "preferred_reward"
    ]
    assert input_check.clarification_prompt == ""
