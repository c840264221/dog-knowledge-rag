from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval.alias_loader import get_alias_dict

_alias_cache = get_alias_dict()

MEMORY_PROMPT = ChatPromptTemplate.from_template(
    """
   你是一个记忆提取器。

   请从用户输入中提取长期有效的信息。

   只提取：
   - 用户偏好
   - 用户厌恶
   - 长期兴趣

   不要提取：
   - 临时问题
   - 普通聊天

   # memory_type 只能是：

   - favorite_dog
   - dislike
   - hobby
   - profile

   # content 要求：

   1. content必须简洁
   2. 不要包含“喜欢”“讨厌”等描述词
   3. content必须是核心实体
   4. 尽量使用英文标准名称
   
   # confidence 置信度：
   1.将推理的置信度存入
   2.取值范围为0~1浮点数
   
   # reason 为什么这么提取的原因
   1.为字符串类型
   2.简洁明了，不要详细描述

   例如：

   用户：
   我最喜欢金毛了

   输出：
   {{
     "should_save": true,
     "memory_type": "preference",
     "content": "Golden Retriever",
     "confidence": 0.64,
     "reason": "用户明确表示喜欢金毛"
   }}

   用户输入：
   {question}
   """
)


parser = JsonOutputParser()


def extract_memory(llm, question: str):

    chain = MEMORY_PROMPT | llm | parser

    result = chain.invoke({
        "question": question
    })
    for k, v in _alias_cache.items():
        if result["content"] in v:
            result['content'] = k
            break
    return result


if __name__ == "__main__":
    from src.models.llm import get_chinese_llm
    from src.logger import logger
    from src.memory.memory_schema import MemoryOutput

    llm = get_chinese_llm()
    structured_llm = llm.with_structured_output(MemoryOutput)
    result = extract_memory(llm, "我觉得拉布拉多挺不错")
    logger.info(result)