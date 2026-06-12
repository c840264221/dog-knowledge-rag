from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.logger import logger
from src.retrieval.alias_loader import (
    get_alias_dict
)


from src.memory.memory_content_normalizer import (
    normalize_memory_content
)


_alias_cache = get_alias_dict()

parser = JsonOutputParser()


MEMORY_PROMPT = ChatPromptTemplate.from_template(
    """
你是一个长期记忆提取器。

你的任务：
从用户输入中判断是否存在“长期有效、以后还会影响回答”的信息。

只保存以下长期信息：
1. 用户喜欢的狗狗品种
2. 用户不喜欢的狗狗品种
3. 用户长期偏好，例如希望回答方式、语言风格、解释习惯
4. 用户长期兴趣
5. 用户稳定的个人背景信息

不要保存以下内容：
1. 一次性的临时问题
2. 普通知识查询
3. 没有长期价值的闲聊
4. 不确定、含糊、开玩笑的信息
5. 助手自己推测出来的信息

提取memory_type时的规则如下：
1.如果用户表达“喜欢/最喜欢/偏爱 + 狗狗品种”，memory_type 必须是 favorite_dog。

2.如果用户表达“不喜欢/讨厌/不想要 + 狗狗品种”，memory_type 必须是 dislike。

3.不要把“喜欢某个狗狗品种”分类为 hobby 或 preference。

4.memory_type 只能是以下值之一：
- favorite_dog：用户喜欢的狗狗品种
- dislike：用户不喜欢、讨厌、避免的东西
- preference：用户长期偏好，例如回答方式、语言偏好、解释习惯
- hobby：用户长期兴趣爱好
- profile：用户稳定的个人背景信息

content 规则：
1. 如果 memory_type 是 favorite_dog，content 只写狗狗品种核心实体，例如“金毛”
2. 如果 memory_type 是 dislike，content 只写用户不喜欢的核心实体，例如“哈士奇”
3. 如果 memory_type 是 preference，content 写成简洁中文短句，例如“用户希望技术名词附带中文解释”
4. content 不要写长句
5. content 不要包含无关解释
6. content 必须能长期保存

confidence 规则：
- 取值范围必须是 0 到 1
- 用户明确表达时，通常是 0.85 到 0.98
- 用户表达含糊时，低于 0.7
- 不值得保存时，confidence 可以是 0.0

请严格输出 JSON，不要输出 Markdown，不要输出解释文字。

JSON 格式必须是：
{{
  "should_save": true 或 false,
  "memory_type": "favorite_dog/dislike/preference/hobby/profile",
  "content": "要保存的内容",
  "confidence": 0.0 到 1.0,
  "reason": "简短原因"
}}

示例 1：
用户输入：
我最喜欢金毛了

输出：
{{
  "should_save": true,
  "memory_type": "favorite_dog",
  "content": "金毛",
  "confidence": 0.95,
  "reason": "用户明确表达了喜欢的狗狗品种"
}}

示例 2：
用户输入：
我不喜欢哈士奇

输出：
{{
  "should_save": true,
  "memory_type": "dislike",
  "content": "哈士奇",
  "confidence": 0.95,
  "reason": "用户明确表达了不喜欢的对象"
}}

示例 3：
用户输入：
以后所有技术名词都帮我加中文解释

输出：
{{
  "should_save": true,
  "memory_type": "preference",
  "content": "用户希望技术名词附带中文解释",
  "confidence": 0.95,
  "reason": "用户表达了长期回答偏好"
}}

示例 4：
用户输入：
金毛寿命是多少

输出：
{{
  "should_save": false,
  "memory_type": "preference",
  "content": "",
  "confidence": 0.0,
  "reason": "这是普通知识问题，不是用户长期偏好"
}}

用户输入：
{question}
"""
)


def normalize_memory_type(
        memory_type: Any
) -> str:
    """
    归一化 memory_type。

    功能：
    - 校验 LLM 返回的 memory_type 是否合法
    - 将非法 memory_type 兜底为 preference
    - 避免非法类型写入数据库

    参数：
    - memory_type: Any
      LLM 返回的原始 memory_type。

    返回值：
    - str
      合法的 memory_type。
    """

    valid_types = {
        "favorite_dog",
        "dislike",
        "preference",
        "hobby",
        "profile",
    }

    clean_type = str(
        memory_type
        or "preference"
    ).strip()

    if clean_type not in valid_types:
        logger.warning(
            f"非法 memory_type，已兜底为 preference: {clean_type!r}"
        )

        return "preference"

    return clean_type


def normalize_confidence(
        confidence: Any
) -> float:
    """
    归一化 confidence 置信度。

    功能：
    - 将 LLM 返回的 confidence 转换为 float
    - 限制范围在 0 到 1
    - 如果转换失败，则返回 0.0

    参数：
    - confidence: Any
      LLM 返回的原始 confidence。

    返回值：
    - float
      归一化后的置信度。
    """

    try:
        value = float(
            confidence
        )

    except Exception:
        return 0.0

    return max(
        0.0,
        min(
            value,
            1.0
        )
    )

def normalize_memory_result(
        result: dict[str, Any]
) -> dict[str, Any]:
    """
    归一化 Memory 抽取结果。

    功能：
    - 修复 LLM 输出中的字段类型问题
    - 修复非法 memory_type
    - 修复 confidence 范围
    - 修复 content 空值
    - 当 should_save 为 true 但 content 为空时，强制改为不保存

    参数：
    - result: dict[str, Any]
      LLM 返回的原始 JSON 结果。

    返回值：
    - dict[str, Any]
      可供 memory_extract_node 使用的稳定结果。
    """

    should_save = bool(
        result.get(
            "should_save",
            False
        )
    )

    memory_type = normalize_memory_type(
        result.get(
            "memory_type",
            "preference"
        )
    )

    content = normalize_memory_content(
        memory_type=memory_type,
        content=result.get(
            "content",
            ""
        )
    )

    confidence = normalize_confidence(
        result.get(
            "confidence",
            0.0
        )
    )

    reason = str(
        result.get(
            "reason",
            ""
        )
        or ""
    ).strip()

    if should_save and not content:
        logger.warning(
            "Memory 抽取结果 should_save=True 但 content 为空，已改为不保存"
        )

        should_save = False

        reason = (
            reason
            or "content 为空，不能保存"
        )

    if should_save and confidence <= 0:
        confidence = 0.5

    return {
        "should_save": should_save,
        "memory_type": memory_type,
        "content": content,
        "confidence": confidence,
        "reason": reason,
    }


def default_memory_result(
        reason: str
) -> dict[str, Any]:
    """
    创建默认 Memory 抽取结果。

    功能：
    - 当 LLM 调用失败、JSON 解析失败、用户输入为空时使用
    - 返回 should_save=False，保证不会错误保存记忆

    参数：
    - reason: str
      不保存的原因。

    返回值：
    - dict[str, Any]
      默认 Memory 抽取结果。
    """

    return {
        "should_save": False,
        "memory_type": "preference",
        "content": "",
        "confidence": 0.0,
        "reason": reason,
    }


async def extract_memory(
        llm_provider,
        question: str
) -> dict[str, Any]:
    """
    从用户输入中抽取长期记忆。

    功能：
    - 使用 LCEL 构建 Memory 抽取链
    - 使用 RunnableLambda 包装 llm_provider.safe_ainvoke
    - 保留统一 LLM 调用入口、fallback、日志和异常处理
    - 使用 JsonOutputParser 将 LLM 输出解析成 dict
    - 对解析结果做归一化处理

    参数：
    - llm_provider:
      LLMProvider 实例。
      中文释义：统一管理大语言模型调用的 Provider。

    - question: str
      用户当前输入。

    返回值：
    - dict[str, Any]
      Memory 抽取结果。
    """

    clean_question = question.strip()

    if not clean_question:
        return default_memory_result(
            "用户输入为空"
        )

    fallback_response = """
{
  "should_save": false,
  "memory_type": "preference",
  "content": "",
  "confidence": 0.0,
  "reason": "LLM 调用失败"
}
"""

    async def safe_llm_call(
            prompt_value
    ):
        """
        安全调用 LLM。

        功能：
        - 接收 LCEL 上游生成的 PromptValue
        - 转换成字符串 Prompt
        - 调用 llm_provider.safe_ainvoke
        - 返回 LLM 响应文本，供 JsonOutputParser 解析

        参数：
        - prompt_value:
          LCEL 生成的 PromptValue。
          中文释义：Prompt 模板格式化后的对象。

        返回值：
        - str
          LLM 输出文本。
        """

        prompt_text = prompt_value.to_string()

        response = await llm_provider.safe_ainvoke(
            llm=llm_provider.chinese_llm,
            prompt=prompt_text,
            fallback_response=fallback_response
        )

        return str(
            getattr(
                response,
                "content",
                response
            )
        )

    safe_llm = RunnableLambda(
        safe_llm_call
    )

    chain = (
        MEMORY_PROMPT
        | safe_llm
        | parser
    )

    try:
        raw_result = await chain.ainvoke(
            {
                "question": clean_question
            }
        )

        result = normalize_memory_result(
            raw_result
        )

        logger.info(
            f"Memory 抽取完成: {result}"
        )

        return result

    except Exception as e:
        logger.warning(
            f"Memory 抽取失败，已跳过保存: {e}"
        )

        return default_memory_result(
            "Memory 抽取失败"
        )