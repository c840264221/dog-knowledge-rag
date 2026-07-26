import time
from collections.abc import Mapping

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
                    token_usage=self._extract_llm_token_usage(response),
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
                token_usage=self._extract_llm_token_usage(response),
            )

            return response

        except Exception as e:

            logger.error(
                f"备用模型失败: {e}"
            )

        self._record_llm_metrics(
            latency_ms=(time.perf_counter() - started_at) * 1000,
            failed=True,
            token_usage=None,
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
        token_usage: Mapping[str, int] | None,
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
            token_usage:
                成功响应中提取的输入、输出和总 Token；模型未返回 usage
                信息或调用失败时为 None。

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
            if token_usage:
                metrics_scope.increment(
                    "llm_input_tokens",
                    token_usage.get("input_tokens", 0),
                )
                metrics_scope.increment(
                    "llm_output_tokens",
                    token_usage.get("output_tokens", 0),
                )
                metrics_scope.increment(
                    "llm_total_tokens",
                    token_usage.get("total_tokens", 0),
                )
        except Exception as exc:
            logger.debug(f"记录 LLM Runtime Metrics 失败: {exc}")

    @staticmethod
    def _extract_llm_token_usage(
        response,
    ) -> dict[str, int] | None:
        """
        从不同 LangChain 模型响应中提取统一 Token 用量。

        功能：
            优先读取标准 usage_metadata；没有时兼容
            response_metadata.token_usage。字段名称同时兼容 input/output
            tokens 和 prompt/completion tokens。无法识别时返回 None。

        参数含义：
            response:
                LLM ainvoke 返回的消息对象或映射。

        返回值含义：
            dict[str, int] | None:
                统一的 input_tokens、output_tokens、total_tokens；响应没有
                可用 Token 数据时返回 None。
        """

        usage_metadata = getattr(response, "usage_metadata", None)
        response_metadata = getattr(response, "response_metadata", None)
        if isinstance(response, Mapping):
            usage_metadata = (
                usage_metadata
                or response.get("usage_metadata")
            )
            response_metadata = (
                response_metadata
                or response.get("response_metadata")
            )

        raw_usage = usage_metadata
        if (
            not isinstance(raw_usage, Mapping)
            and isinstance(response_metadata, Mapping)
        ):
            raw_usage = (
                response_metadata.get("token_usage")
                or response_metadata.get("usage")
            )
        if not isinstance(raw_usage, Mapping):
            return None

        input_tokens = _read_non_negative_token_count(
            raw_usage,
            "input_tokens",
            "prompt_tokens",
        )
        output_tokens = _read_non_negative_token_count(
            raw_usage,
            "output_tokens",
            "completion_tokens",
        )
        total_tokens = _read_non_negative_token_count(
            raw_usage,
            "total_tokens",
        )
        if total_tokens is None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        if (
            input_tokens is None
            and output_tokens is None
            and total_tokens == 0
        ):
            return None
        return {
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "total_tokens": total_tokens,
        }

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


def _read_non_negative_token_count(
    usage: Mapping,
    *field_names: str,
) -> int | None:
    """
    从模型 usage 映射读取第一个合法的非负 Token 数。

    功能：
        兼容不同模型供应商使用的字段名称，并拒绝布尔值、负数和无法
        转换成整数的数据。

    参数含义：
        usage:
            模型返回的 Token usage 映射。
        *field_names:
            按优先级尝试读取的字段名称。

    返回值含义：
        int | None:
            找到合法数值时返回非负整数，否则返回 None。
    """

    for field_name in field_names:
        raw_value = usage.get(field_name)
        if isinstance(raw_value, bool):
            continue
        if isinstance(raw_value, (int, float)) and raw_value >= 0:
            return int(raw_value)
    return None
