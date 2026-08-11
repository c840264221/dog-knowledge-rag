from src.memory.memory_retention_policy import MemoryRetentionPolicy


def test_retention_policy_should_accept_qualified_preference() -> None:
    """
    验证达到门槛的长期偏好可以进入持久化层。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    decision = MemoryRetentionPolicy().evaluate(
        {
            "should_save": True,
            "memory_type": "preference",
            "content": "用户希望技术名词附带中文解释",
            "confidence": 0.90,
            "importance": 0.80,
        }
    )

    assert decision.is_accepted is True
    assert decision.action == "accepted"


def test_retention_policy_should_reject_low_confidence_candidate() -> None:
    """
    验证 LLM 即使建议保存，低可信度候选仍会被策略拒绝。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    decision = MemoryRetentionPolicy().evaluate(
        {
            "should_save": True,
            "memory_type": "preference",
            "content": "用户可能喜欢简短回答",
            "confidence": 0.65,
            "importance": 0.90,
        }
    )

    assert decision.is_accepted is False
    assert "可信度" in decision.reason


def test_retention_policy_should_reject_low_importance_profile() -> None:
    """
    验证高风险档案类型没有达到重要度门槛时不会长期保存。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    decision = MemoryRetentionPolicy().evaluate(
        {
            "should_save": True,
            "memory_type": "profile",
            "content": "用户今天在成都",
            "confidence": 0.95,
            "importance": 0.40,
        }
    )

    assert decision.is_accepted is False
    assert "重要度" in decision.reason


def test_retention_policy_should_keep_llm_rejection_reason() -> None:
    """
    验证 LLM 本身不建议保存时保留其业务原因，便于日志排查。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    decision = MemoryRetentionPolicy().evaluate(
        {
            "should_save": False,
            "memory_type": "preference",
            "content": "",
            "confidence": 0.0,
            "importance": 0.0,
            "reason": "这是一次性天气查询。",
        }
    )

    assert decision.is_accepted is False
    assert decision.reason == "这是一次性天气查询。"
