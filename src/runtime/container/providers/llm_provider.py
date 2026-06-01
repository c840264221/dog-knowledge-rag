from langchain_openai import ChatOpenAI

from langchain_ollama import ChatOllama

from src.settings import settings

from src.logger import logger


class LLMProvider:

    def __init__(self):

        self._main_llm = None

        self._backup_llm = None

        self._chinese_llm = None

    @property
    def main_llm(self):

        if self._main_llm is None:

            logger.info(
                "🚀 初始化主LLM..."
            )

            self._main_llm = ChatOpenAI(

                model=settings.llm.main_model,

                api_key=settings.llm.deepseek_api_key.get_secret_value(),

                base_url=(
                    settings.llm.deepseek_base_url
                ),

                temperature=(
                    settings.llm.temperature
                )
            )

        return self._main_llm

    @property
    def backup_llm(self):

        if self._backup_llm is None:

            logger.info(
                "🚀 初始化备用LLM..."
            )

            self._backup_llm = ChatOpenAI(

                model=settings.llm.backup_model,

                api_key=(
                    settings.llm
                    .deepseek_api_key
                    .get_secret_value()
                ),

                base_url=(
                    settings.llm.deepseek_base_url
                ),

                temperature=(
                    settings.llm.temperature
                )
            )

        return self._backup_llm

    @property
    def chinese_llm(self):

        if self._chinese_llm is None:

            logger.info(
                "🚀 初始化中文LLM..."
            )

            self._chinese_llm = ChatOllama(

                model=settings.llm.chinese_model,

                base_url=(
                    settings.llm.ollama_base_url
                ),

                temperature=0
            )

        return self._chinese_llm

    async def safe_ainvoke(self,llm,prompt,fallback_response=None,max_attempts=None):

        if max_attempts is None:
            max_attempts = (
                settings.runtime.max_retries
            )

        for attempt in range(
                1,
                max_attempts + 1
        ):

            try:

                logger.info(
                    f"LLM调用尝试 "
                    f"{attempt}/{max_attempts}"
                )

                response = await llm.ainvoke(
                    prompt
                )

                return response

            except Exception as e:

                logger.warning(
                    f"LLM调用失败: {e}"
                )

        logger.warning(
            "主模型失败，切换备用模型"
        )

        try:

            response = await (
                self.backup_llm.ainvoke(prompt)
            )

            return response

        except Exception as e:

            logger.error(
                f"备用模型失败: {e}"
            )

        if fallback_response:
            return fallback_response

        raise RuntimeError(
            "所有LLM调用失败"
        )

    async def startup(self):

        # 提前初始化
        _ = self.main_llm
        _ = self.backup_llm
        _ = self.chinese_llm

        logger.info(
            "LLMProvider 启动完成"
        )

    async def shutdown(self):

        logger.info(
            "LLMProvider 已关闭"
        )