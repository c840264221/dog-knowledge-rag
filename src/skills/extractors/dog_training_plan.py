"""狗狗训练计划 Skill 的确定性输入提取规则。"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from src.rag.query_parsers.dog_query_filter_parser import (
    DogQueryFilterParser,
)


AGE_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>岁|个月|月龄)",
    re.IGNORECASE,
)
CURRENT_BEHAVIOR_PATTERNS = [
    re.compile(
        r"(?:目前|现在|已经)?"
        r"(?<!不)(?<!没)(?<!未)(?<!太)(?<!学)(?<!没有)"
        r"(?:会|掌握了?|学会了)"
        r"(?P<value>[^，。；,;]{1,40})"
    ),
]
TRAINING_GOAL_PATTERNS = [
    re.compile(
        r"(?:希望(?:它|狗狗)?|想让(?:它|狗狗)?|训练目标(?:是|为)?|目标(?:是|为)?)"
        r"(?P<value>[^。；;]{1,60})"
    ),
    re.compile(
        r"想(?:训练|教)(?:它|狗狗)?"
        r"(?P<value>[^。；;]{1,60})"
    ),
]


def extract_dog_training_plan_inputs(user_text: str) -> dict[str, Any]:
    """
    从用户自然语言中提取狗狗训练计划所需的明确字段。

    功能：
        使用保守规则识别犬种、年龄、当前已掌握行为和训练目标。只有规则明确
        命中的内容才会返回，无法确定的字段保持缺失并交给 Checker 询问用户。

    参数含义：
        user_text:
            用户本轮提供的原始自然语言。

    返回值含义：
        dict[str, Any]:
            本轮明确提取出的训练技能输入字段。
    """

    # 去除用户文本首尾空白，后续所有规则都基于这份文本进行匹配。
    normalized_text = str(user_text or "").strip()

    # 仅保存规则明确识别出的字段，不会为缺失信息编造默认值。
    extracted_inputs: dict[str, Any] = {}
    if not normalized_text:
        return extracted_inputs

    breed = _extract_breed(normalized_text)
    if breed is not None:
        extracted_inputs["breed"] = breed

    age = _extract_age(normalized_text)
    if age is not None:
        extracted_inputs["age"] = age

    current_behavior = _extract_first_match(
        text=normalized_text,
        patterns=CURRENT_BEHAVIOR_PATTERNS,
    )
    if current_behavior is not None:
        extracted_inputs["current_behavior"] = current_behavior

    training_goal = _extract_first_match(
        text=normalized_text,
        patterns=TRAINING_GOAL_PATTERNS,
    )
    if training_goal is not None:
        extracted_inputs["training_goal"] = training_goal

    return extracted_inputs


def _extract_breed(text: str) -> str | None:
    """
    从文本中提取标准犬种名称。

    功能：
        复用现有 RAG Parser 的犬种别名知识，并优先匹配更长别名，避免短别名
        抢先命中。返回标准英文名称，便于后续与 RAG 元数据保持一致。

    参数含义：
        text:
            已去除首尾空白的用户文本。

    返回值含义：
        str | None:
            命中时返回标准犬种名称，否则返回 None。
    """

    lowered_text = text.lower()

    # 这里使用由 alias_dog_name.json 和 fallback 共同构建的完整别名索引。
    # 索引已经在 _load_breed_alias_map 中缓存，不会每次提取都重新读取文件。
    breed_alias_map = _load_breed_alias_map()

    # 按别名长度从长到短检查，例如优先匹配“金毛寻回犬”而不是“金毛”。
    ordered_aliases = sorted(
        breed_alias_map.items(),
        key=lambda item: (-len(item[0]), item[0]),
    )
    for alias, canonical_name in ordered_aliases:
        if alias.lower() in lowered_text:
            return canonical_name
    return None


@lru_cache(maxsize=1)
def _load_breed_alias_map() -> dict[str, str]:
    """
    加载并缓存完整犬种别名索引。

    功能：
        创建 DogQueryFilterParser，让它读取 alias_dog_name.json、转换别名方向
        并合并 fallback。结果只构建一次，后续 Skill 提取直接复用缓存。

    参数含义：
        无。

    返回值含义：
        dict[str, str]:
            小写犬种别名到标准英文犬种名称的映射。
    """

    # Parser 初始化后生成完整的 alias -> canonical name 索引。
    # 它内部读取 JSON 失败时会自动退回代码中的 fallback 别名表。
    parser = DogQueryFilterParser()
    return dict(parser.breed_alias_map)


def _extract_age(text: str) -> str | None:
    """
    从文本中提取狗狗年龄。

    功能：
        识别“6岁”“8个月”“3月龄”等明确表达，并统一去除数字和单位间空格。

    参数含义：
        text:
            已去除首尾空白的用户文本。

    返回值含义：
        str | None:
            命中时返回规范化年龄文本，否则返回 None。
    """

    # 用户可能先否定旧年龄、再给出新年龄，例如“不是5岁，是6岁”。
    # 取最后一次明确表达，让后面的纠正值覆盖前面的旧值。
    matches = list(AGE_PATTERN.finditer(text))
    if not matches:
        return None

    match = matches[-1]
    value = match.group("value")
    unit = match.group("unit")
    return f"{value}{unit}"


def _extract_first_match(
    *,
    text: str,
    patterns: list[re.Pattern[str]],
) -> str | None:
    """
    使用一组规则提取第一个非空内容。

    功能：
        按规则优先级依次匹配，并清理命中内容首尾空白和常见逗号。

    参数含义：
        text:
            已去除首尾空白的用户文本。
        patterns:
            按优先级排列、包含 value 命名分组的正则表达式列表。

    返回值含义：
        str | None:
            第一个有效命中内容；全部未命中时返回 None。
    """

    for pattern in patterns:
        match = pattern.search(text)
        if match is None:
            continue

        value = match.group("value").strip(" ，,：:")
        if value:
            return value
    return None
