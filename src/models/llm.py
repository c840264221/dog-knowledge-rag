from langchain_ollama import ChatOllama
from src.logger import logger

_llm_instance = None
_llm_backup = ChatOllama(
            # model="deepseek-r1:latest",
            model="qwen2:7b",
            temperature=0
        )

def get_llm():
    global _llm_instance

    if _llm_instance is None:
        logger.info("🚀  初始化LLM......")

        _llm_instance = ChatOllama(
            # model="deepseek-r1:latest",
            model="deepseek-r1:14b",
            temperature=0
        )


    return _llm_instance

def safe_llm_invoke(llm, prompt, backup_llm=_llm_backup, fallback_response=None, max_attempts=3):
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

            response = llm.invoke(prompt)

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

            response = backup_llm.invoke(prompt)

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