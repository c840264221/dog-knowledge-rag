"""Skill 统一运行器测试。"""

from __future__ import annotations

from src.skills import build_default_skill_runtime


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
