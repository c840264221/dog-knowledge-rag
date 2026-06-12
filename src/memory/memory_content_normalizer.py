from typing import Any

from src.retrieval.alias_loader import (
    get_alias_dict
)


_alias_cache = get_alias_dict()


def normalize_memory_content(
        memory_type: str,
        content: Any
) -> str:
    """
    归一化 Memory 内容。

    功能：
    - 将不同表达形式的同一条 Memory 归一成稳定内容
    - 例如“用户喜欢金毛”“我喜欢金毛”“金毛”统一成“金毛”
    - 例如“用户不喜欢哈士奇”“讨厌哈士奇”“哈士奇”统一成“哈士奇”
    - 如果命中 alias_dict，则转换成标准名称
    - 用于 Memory 保存、冲突判断、去重、旧数据清理

    参数：
    - memory_type: str
      Memory 类型。
      中文释义：记忆类型，例如 favorite_dog、dislike、preference。

    - content: Any
      原始 Memory 内容。
      中文释义：LLM 抽取出来准备保存的记忆内容。

    返回值：
    - str
      归一化后的 Memory 内容。
    """

    clean_content = str(
        content
        or ""
    ).strip()

    if not clean_content:
        return ""

    if memory_type == "favorite_dog":
        remove_words = [
            "用户最喜欢",
            "用户喜欢",
            "我最喜欢",
            "我喜欢",
            "最喜欢",
            "喜欢",
            "狗狗",
            "犬种",
            "是",
            "：",
            ":",
        ]

        for word in remove_words:
            clean_content = clean_content.replace(
                word,
                ""
            )

    elif memory_type == "dislike":
        remove_words = [
            "用户不喜欢",
            "用户讨厌",
            "我不喜欢",
            "我讨厌",
            "不喜欢",
            "讨厌",
            "不想要",
            "狗狗",
            "犬种",
            "是",
            "：",
            ":",
        ]

        for word in remove_words:
            clean_content = clean_content.replace(
                word,
                ""
            )

    clean_content = clean_content.strip()

    for standard_name, aliases in _alias_cache.items():

        if clean_content == standard_name:
            return standard_name

        if clean_content in aliases:
            return standard_name

    return clean_content


def build_memory_identity_key(
        user_id: str,
        memory_type: str,
        content: Any
) -> str:
    """
    构建 Memory 身份 key。

    功能：
    - 使用 user_id、memory_type、归一化 content 组成唯一判断 key
    - 用于判断两条 Memory 是否表示同一个事实
    - 不用于数据库主键，只用于业务去重和冲突判断

    参数：
    - user_id: str
      用户唯一标识。

    - memory_type: str
      Memory 类型。

    - content: Any
      Memory 内容。

    返回值：
    - str
      Memory 身份 key。
    """

    normalized_content = normalize_memory_content(
        memory_type=memory_type,
        content=content
    )

    return (
        f"{user_id}:"
        f"{memory_type}:"
        f"{normalized_content}"
    )