from typing import Any

from src.logger import logger


class RuntimeContainer:

    def __init__(self):

        # 服务注册表
        self._services = {}

        # 生命周期状态
        self._started = False

    # =========================
    # 注册服务
    # =========================

    def register(self,name: str,service: Any):

        logger.info(
            f"注册服务: {name}"
        )

        self._services[name] = service

    # =========================
    # 获取服务
    # =========================

    def get(self, name: str):

        if name not in self._services:

            raise ValueError(
                f"服务不存在: {name}"
            )

        return self._services[name]

    # =========================
    # 启动容器
    # =========================

    async def startup(self):

        if self._started:

            logger.warning(
                "Container 已启动"
            )

            return

        logger.info(
            "🚀 Container 启动中..."
        )

        for name, service in self._services.items():

            startup = getattr(
                service,
                "startup",
                None
            )

            if callable(startup):

                logger.info(
                    f"启动服务: {name}"
                )

                await startup()

        self._started = True

        logger.info(
            "✅ Container 启动完成"
        )

    # =========================
    # 关闭容器
    # =========================

    async def shutdown(self):

        logger.info(
            "🛑 Container 关闭中..."
        )

        for name, service in self._services.items():

            shutdown = getattr(
                service,
                "shutdown",
                None
            )

            if callable(shutdown):

                logger.info(
                    f"关闭服务: {name}"
                )

                await shutdown()

        logger.info(
            "✅ Container 已关闭"
        )