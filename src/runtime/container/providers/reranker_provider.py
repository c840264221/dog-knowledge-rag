from sentence_transformers import CrossEncoder
from threading import Lock

from src.settings import settings

from src.logger import logger


class RerankerProvider:

    def __init__(self):

        self._reranker = None
        self._initialization_lock = Lock()

    @property
    def reranker(self):

        if self._reranker is None:
            # 并发 Worker 可能同时首次访问，锁内需要再次检查实例是否已创建。
            with self._initialization_lock:
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
        _ = self.reranker

        logger.info(
            "RerankerProvider 启动完成"
        )

    async def shutdown(self):

        logger.info(
            "RerankerProvider 已关闭"
        )
