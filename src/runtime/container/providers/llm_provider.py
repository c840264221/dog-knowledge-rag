import time

from langchain_openai import ChatOpenAI

from langchain_ollama import ChatOllama

from src.settings import settings

from src.logger import logger
from src.runtime.context import runtime_ctx
from src.runtime.scopes.metrics_scope import MetricsScope


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

        started_at = time.perf_counter()

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

                self._record_llm_metrics(
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                    failed=False,
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

            self._record_llm_metrics(
                latency_ms=(time.perf_counter() - started_at) * 1000,
                failed=False,
            )

            return response

        except Exception as e:

            logger.error(
                f"备用模型失败: {e}"
            )

        self._record_llm_metrics(
            latency_ms=(time.perf_counter() - started_at) * 1000,
            failed=True,
        )

        if fallback_response:
            return fallback_response

        raise RuntimeError(
            "所有LLM调用失败"
        )

    @staticmethod
    def _record_llm_metrics(
        *,
        latency_ms: float,
        failed: bool,
    ) -> None:
        """
        把一次完整 LLM 逻辑调用写入当前请求的运行时指标。

        功能：
            无论主模型内部重试多少次，都把一次 safe_ainvoke 记为一次逻辑
            调用；累加从首次尝试到最终结果的总耗时，全部模型失败时再增加
            error_count。指标不可用时安静跳过，不能影响正常回答。

        参数含义：
            latency_ms:
                本次逻辑调用从开始到最终成功或失败的毫秒耗时。
            failed:
                主模型和备用模型是否全部失败。

        返回值含义：
            None:
                只更新 MetricsScope，不返回业务数据。
        """

        try:
            runtime_context = runtime_ctx.get()
            if runtime_context is None:
                return
            metrics_scope = runtime_context.service(MetricsScope)
            if not metrics_scope.get_metrics():
                return

            metrics_scope.increment("llm_count")
            current_latency = metrics_scope.get_metrics().get(
                "llm_latency",
                0,
            )
            metrics_scope.update(
                "llm_latency",
                current_latency + max(0.0, latency_ms),
            )
            if failed:
                metrics_scope.increment("error_count")
        except Exception as exc:
            logger.debug(f"记录 LLM Runtime Metrics 失败: {exc}")

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
