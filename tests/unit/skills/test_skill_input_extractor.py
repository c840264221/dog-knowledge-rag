"""Skill 自然语言输入提取器测试。"""

from __future__ import annotations

from src.skills import (
    SkillInputExtractor,
    SkillLoader,
    build_default_skill_input_extractor,
    build_default_skill_registry,
)


def test_training_extractor_should_extract_explicit_breed_and_age() -> None:
    """
    测试训练技能可以提取明确的犬种和年龄。

    功能：
        验证“6岁的金毛”会转换为与 RAG 一致的标准犬种名和规范年龄文本。

    参数含义：
        无。

    返回值含义：
        None。
    """

    extractor = build_default_skill_input_extractor()

    result = extractor.extract(
        skill_id="dog-training-plan",
        user_text="帮我为6岁的金毛制定训练计划。",
    )

    assert result.extracted_inputs == {
        "breed": "Golden Retriever",
        "age": "6岁",
    }
    assert result.merged_inputs == result.extracted_inputs


def test_training_extractor_should_use_aliases_from_json_file() -> None:
    """
    测试训练技能会使用 alias_dog_name.json 的完整犬种别名。

    功能：
        Affenpinscher 不在代码 fallback 表中，本用例用于证明 Skill 读取的是
        Parser 根据 JSON 构建的完整别名索引。

    参数含义：
        无。

    返回值含义：
        None。
    """

    extractor = build_default_skill_input_extractor()

    result = extractor.extract(
        skill_id="dog-training-plan",
        user_text="我家养的是Affenpinscher，想制定训练计划。",
    )

    assert result.extracted_inputs["breed"] == "Affenpinscher"


def test_training_extractor_should_extract_behavior_and_goal() -> None:
    """
    测试训练技能可以提取明确的当前行为和训练目标。

    功能：
        验证带明显提示词的自然语言能够转换成 Skill 需要的结构化字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    extractor = build_default_skill_input_extractor()

    result = extractor.extract(
        skill_id="dog-training-plan",
        user_text="它目前会坐下，希望学习等待和召回。",
    )

    assert result.extracted_inputs == {
        "current_behavior": "坐下",
        "training_goal": "学习等待和召回",
    }


def test_training_extractor_should_read_planner_quoted_fields() -> None:
    """
    测试训练技能可以识别 Planner 使用引号改写的行为和目标。

    功能：
        覆盖真实步骤描述中的“现有‘坐下’技能”和“‘等待’、‘召回’科目”，
        防止信息明明写在步骤中却再次向用户提问。

    参数含义：
        无。

    返回值含义：
        None。
    """

    extractor = build_default_skill_input_extractor()

    result = extractor.extract(
        skill_id="dog-training-plan",
        user_text=(
            "针对6岁金毛犬，基于其现有“坐下”技能，"
            "为“等待”和“召回”两个科目制定分阶段训练计划。"
        ),
    )

    assert result.extracted_inputs == {
        "breed": "Golden Retriever",
        "age": "6岁",
        "current_behavior": "坐下",
        "training_goal": "等待和召回",
    }


def test_training_extractor_should_read_parenthesized_labeled_fields() -> None:
    """
    测试训练技能可以识别括号内使用字段名标注的输入。

    功能：
        支持 Planner 或人工模板生成的“当前行为基础：...；训练目标：...”格式。

    参数含义：
        无。

    返回值含义：
        None。
    """

    extractor = build_default_skill_input_extractor()

    result = extractor.extract(
        skill_id="dog-training-plan",
        user_text=(
            "为6岁金毛制定训练计划"
            "（当前行为基础：坐下；训练目标：等待和召回）。"
        ),
    )

    assert result.extracted_inputs == {
        "breed": "Golden Retriever",
        "age": "6岁",
        "current_behavior": "坐下",
        "training_goal": "等待和召回",
    }


def test_training_extractor_should_not_treat_negative_behavior_as_mastered() -> None:
    """
    测试否定表达不会被误判为已经掌握的行为。

    功能：
        “不会等待”表示缺少该能力，不能提取成 current_behavior="等待"。

    参数含义：
        无。

    返回值含义：
        None。
    """

    extractor = build_default_skill_input_extractor()

    result = extractor.extract(
        skill_id="dog-training-plan",
        user_text="它还没学会召回，也没有掌握等待，希望学习等待。",
    )

    assert "current_behavior" not in result.extracted_inputs
    assert result.extracted_inputs["training_goal"] == "学习等待"


def test_extractor_should_merge_new_inputs_over_existing_values() -> None:
    """
    测试本轮用户补充信息可以更新上一轮字段。

    功能：
        验证恢复执行时会保留旧字段，同时允许用户纠正已有年龄。

    参数含义：
        无。

    返回值含义：
        None。
    """

    extractor = build_default_skill_input_extractor()

    result = extractor.extract(
        skill_id="dog-training-plan",
        user_text="不是5岁，是6岁。",
        existing_inputs={
            "breed": "Golden Retriever",
            "age": "5岁",
            "trace_id": "不应进入 Skill 输入",
        },
    )

    assert result.extracted_inputs == {"age": "6岁"}
    assert result.merged_inputs == {
        "breed": "Golden Retriever",
        "age": "6岁",
    }


def test_extractor_should_filter_fields_not_declared_by_skill() -> None:
    """
    测试提取规则不能绕过 Skill 输入契约。

    功能：
        即使错误规则返回 trace_id，未在 required_inputs 声明的字段也会被过滤。

    参数含义：
        无。

    返回值含义：
        None。
    """

    registry = build_default_skill_registry()
    extractor = SkillInputExtractor(
        loader=SkillLoader(registry),
        rules={
            "dog-training-plan": lambda _: {
                "breed": "Golden Retriever",
                "trace_id": "trace_001",
            }
        },
    )

    result = extractor.extract(
        skill_id="dog-training-plan",
        user_text="金毛",
    )

    assert result.extracted_inputs == {
        "breed": "Golden Retriever",
    }
    assert "trace_id" not in result.merged_inputs


def test_extractor_should_report_when_skill_has_no_registered_rule() -> None:
    """
    测试没有提取规则时返回明确来源而不是报错。

    功能：
        为未来 LLM Extractor 或人工输入兜底保留稳定扩展边界。

    参数含义：
        无。

    返回值含义：
        None。
    """

    registry = build_default_skill_registry()
    extractor = SkillInputExtractor(
        loader=SkillLoader(registry),
    )

    result = extractor.extract(
        skill_id="dog-training-plan",
        user_text="6岁的金毛",
        existing_inputs={"age": "6岁"},
    )

    assert result.extracted_inputs == {}
    assert result.merged_inputs == {"age": "6岁"}
    assert result.source == "no_registered_rule"
