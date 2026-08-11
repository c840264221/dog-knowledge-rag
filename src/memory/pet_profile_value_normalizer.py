"""宠物档案属性值的确定性归一化工具。"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


AGE_YEARS_PATTERN = re.compile(
    r"^(?P<number>\d+(?:\.\d+)?)\s*(?:岁|年)?$"
)


def normalize_pet_profile_value(
    *,
    attribute: str,
    value: Any,
) -> Any:
    """
    按宠物档案属性归一化外部输入值。

    功能：
        当前对 age_years（年龄年数）执行严格归一化；其他属性保持原值，
        避免本次年龄修复意外改变其他档案字段的既有校验行为。

    参数含义：
        attribute：宠物档案属性名称。
        value：LLM 或其他外部来源提供的原始属性值。

    返回值含义：
        Any：年龄返回无单位数字字符串，其他属性返回原值。
    """

    if str(attribute or "").strip() != "age_years":
        return value
    return normalize_age_years(value)


def normalize_age_years(value: Any) -> str:
    """
    将年龄年数统一成无单位数字字符串。

    功能：
        接受数字、数字字符串以及带“岁”或“年”的字符串，并统一输出
        数据库存储使用的无单位格式。标准结果再次传入时保持不变。

    参数含义：
        value：原始年龄，例如 6、"6"、"6.0" 或 "6岁"。

    返回值含义：
        str：归一化后的年龄年数，例如 "6" 或 "0.5"。
    """

    if value is None or isinstance(value, bool):
        raise ValueError("年龄年数不能为空或布尔值")

    text = str(value).strip()
    match = AGE_YEARS_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(f"无法识别年龄年数格式: {value}")

    try:
        number = Decimal(match.group("number"))
    except InvalidOperation as exc:
        raise ValueError(f"无法识别年龄年数格式: {value}") from exc
    if not number.is_finite() or number < 0:
        raise ValueError("年龄年数必须是非负有限数字")

    normalized_number = format(number.normalize(), "f")
    if "." in normalized_number:
        normalized_number = normalized_number.rstrip("0").rstrip(".")
    return normalized_number


def normalize_age_years_for_skill(value: Any) -> str:
    """
    将宠物档案年龄转换成 Skill 使用的带单位文本。

    功能：
        先复用档案年龄归一化，再添加一次“岁”单位。输入已经带单位时
        不会重复追加，因此该转换可以安全地重复执行。

    参数含义：
        value：数据库或历史数据中的年龄值。

    返回值含义：
        str：Skill 可以直接使用的年龄文本，例如 "6岁"。
    """

    return f"{normalize_age_years(value)}岁"
