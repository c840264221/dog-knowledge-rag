from langchain_ollama import ChatOllama
from src.logger import logger
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

_llm_instance = None
_llm_backup = None
_llm_chinese = None

def get_instance_llm():
    global _llm_instance

    if _llm_instance is None:
        logger.info("🚀  初始化instance LLM......")

        # _llm_instance = ChatOllama(
        #     # model="deepseek-r1:latest",
        #     model="deepseek-r1:14b",
        #     temperature=0
        # )
        _llm_instance = ChatOpenAI(
            model="deepseek-v4-pro",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
            temperature=0,
        )
    return _llm_instance

def get_chinese_llm():
    global _llm_chinese
    if _llm_chinese is None:
        logger.info("🚀  初始化_llm_chinese......")
        _llm_chinese = ChatOllama(
            model="qwen2.5:7b",
            # model="qwen2:7b",
            temperature=0
        )
    return _llm_chinese

def get_backup_llm():
    global _llm_backup
    if _llm_backup is None:
        logger.info("🚀  初始化back up LLM......")
        # _llm_backup = ChatOllama(
        #     # model="deepseek-r1:latest",
        #     model="qwen2:7b",
        #     temperature=0
        # )
        _llm_backup = ChatOpenAI(
            model="deepseek-v4-flash",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
            temperature=0,
        )
    return _llm_backup

async def safe_llm_ainvoke(llm, prompt, backup_llm=get_backup_llm(), fallback_response=None, max_attempts=3):
    """
       安全调用 LLM，支持重试和降级响应
       :param backup_llm: 降级的LLM模型
       :param llm: LangChain 的 BaseChatModel 实例（需有 invoke 方法）
       :param prompt: 字符串 prompt
       :param fallback_response: 所有重试失败后返回的备用内容（若为 None 则抛出异常）
       :param max_attempts: 最大重试次数
       :return: LLM 响应的 content 字符串，或 fallback_response
       """
    for attempt in range(1, max_attempts + 1):

        try:
            logger.info(
                f"主模型调用尝试 "
                f"{attempt}/{max_attempts}"
            )

            response = await llm.ainvoke(prompt)

            return response

        except Exception as e:

            logger.warning(
                f"主模型调用失败: {e}"
            )

        # 主模型彻底失败
    logger.warning("主模型不可用，切换备用模型")

    # ===== fallback llm =====
    if backup_llm:

        try:

            response = await backup_llm.ainvoke(prompt)

            return response

        except Exception as e:

            logger.error(
                f"备用模型调用失败: {e}"
            )

    # ===== fallback response =====
    if fallback_response is not None:
        logger.warning("使用降级响应")

        return fallback_response

    raise RuntimeError("所有 LLM 调用失败")

