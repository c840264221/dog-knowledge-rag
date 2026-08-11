"""宠物档案 Prompt 上下文格式化测试。"""

from src.memory.pet_profile_context import format_pet_profile_context


def test_format_pet_profile_context_should_use_stable_labels() -> None:
    """验证结构化档案会按照固定中文标签转换成文本。"""

    result = format_pet_profile_context(
        {
            "status": "applied",
            "pet_name": "豆豆",
            "facts": {
                "weight_kg": "30",
                "breed": "金毛",
                "age_years": "6",
            },
        }
    )

    assert result.splitlines() == [
        "- 宠物名称：豆豆",
        "- 品种：金毛",
        "- 年龄：6",
        "- 体重：30",
    ]


def test_format_pet_profile_context_should_ignore_unapplied_result() -> None:
    """验证对象不明确的档案不会进入 LLM 上下文。"""

    assert format_pet_profile_context(
        {"status": "ambiguous", "facts": {"breed": "金毛"}}
    ) == ""
