"""等待任务与本轮输入关系判断测试。"""

import pytest

from src.runtime.resume import classify_pending_task_relation


@pytest.mark.parametrize(
    ("user_input", "expected_relation", "expected_input"),
    [
        ("取消", "cancel", "取消"),
        (
            "新问题：金毛每天需要运动多久？",
            "new_task",
            "金毛每天需要运动多久？",
        ),
        (
            "继续任务：它目前会坐下，希望学习等待。",
            "resume",
            "它目前会坐下，希望学习等待。",
        ),
    ],
)
def test_explicit_relation_should_have_highest_priority(
    user_input: str,
    expected_relation: str,
    expected_input: str,
) -> None:
    """
    测试用户明确表达任务关系时优先采用其指令。

    参数含义：
        user_input:
            用户本轮输入。
        expected_relation:
            预期任务关系。
        expected_input:
            去掉业务前缀后的预期文本。

    返回值含义：
        None。
    """

    decision = classify_pending_task_relation(user_input)

    assert decision.relation == expected_relation
    assert decision.normalized_input == expected_input
    assert decision.confidence == 1.0
    assert decision.source == "explicit"


@pytest.mark.parametrize(
    "user_input",
    [
        "6岁，体重30公斤",
        "是一只金毛，健康状况良好",
        "它目前会坐下，希望学习等待和召回。",
        "y",
        '{"step_profile": "6岁"}',
    ],
)
def test_answer_shaped_input_should_resume_pending_task(
    user_input: str,
) -> None:
    """
    测试档案、现状、确认词和 JSON 回答会继续暂停任务。

    参数含义：
        user_input:
            适合作为澄清答案的本轮文字。

    返回值含义：
        None。
    """

    decision = classify_pending_task_relation(user_input)

    assert decision.relation == "resume"
    assert decision.source == "rule"


@pytest.mark.parametrize(
    "user_input",
    [
        "请使用多个智能体制定一份训练计划。",
        "帮我查一下成都今天的天气",
        "金毛和拉布拉多有什么区别？",
    ],
)
def test_complete_request_should_start_new_task(
    user_input: str,
) -> None:
    """
    测试具有完整请求形式的文字不会被误当成旧任务补充。

    参数含义：
        user_input:
            可以独立成立的新请求。

    返回值含义：
        None。
    """

    decision = classify_pending_task_relation(user_input)

    assert decision.relation == "new_task"
    assert decision.source == "rule"


def test_uncertain_input_should_not_risk_automatic_resume() -> None:
    """
    测试无法判断的文字会返回 ambiguous，而不是冒险恢复旧任务。

    参数含义：
        无。

    返回值含义：
        None。
    """

    decision = classify_pending_task_relation("成都天气")

    assert decision.relation == "ambiguous"
    assert decision.source == "fallback"
    assert decision.confidence == 0.50


@pytest.mark.parametrize(
    "user_input",
    [
        "",
        "   ",
        "新问题：",
        "继续任务：",
    ],
)
def test_empty_business_input_should_be_rejected(user_input: str) -> None:
    """
    测试空输入或只有业务前缀时会被拒绝。

    参数含义：
        user_input:
            无法用于任务关系判断的输入。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError):
        classify_pending_task_relation(user_input)
