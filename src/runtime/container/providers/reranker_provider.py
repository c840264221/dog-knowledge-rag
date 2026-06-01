from sentence_transformers import CrossEncoder

from src.settings import settings

from src.logger import logger


class RerankerProvider:

    def __init__(self):

        self._reranker = None

    @property
    def reranker(self):

        if self._reranker is None:

            logger.info(
                "🚀 初始化 Reranker..."
            )

            model_kwargs = {}

            token = (
                settings.reranker.huggingface_token
            )

            if token:

                model_kwargs["token"] = (
                    token.get_secret_value()
                )

            self._reranker = CrossEncoder(

                settings.reranker.model_name,

                model_kwargs=model_kwargs,

                device=settings.reranker.device
            )

        return self._reranker

    async def startup(self):

        # 提前初始化
        _ = self._reranker

        logger.info(
            "RerankerProvider 启动完成"
        )

    async def shutdown(self):

        logger.info(
            "RerankerProvider 已关闭"
        )