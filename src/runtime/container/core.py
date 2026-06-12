from typing import Any

from src.logger import logger


class RuntimeContainer:
    """
       RuntimeContainer 是运行时容器。

       功能：
       - 统一注册和获取项目中的 Provider / Service
       - 管理服务的 startup / shutdown 生命周期
       - 避免业务模块到处手动创建依赖对象

       技术名词：
       - Container：容器，统一保存和管理对象的地方
       - Provider：提供者，负责创建和提供某类服务
       - Service：服务，承载具体业务能力的对象
       - Lifecycle：生命周期，指服务启动、运行、关闭的过程
       """

    def __init__(self):
        """
              初始化运行时容器。

              功能：
              - 创建服务注册表
              - 初始化容器启动状态

              参数：
              - 无

              返回值：
              - None
                初始化函数不返回业务数据。
              """

        # 服务注册表
        self._services = {}

        # 生命周期状态
        self._started = False

    # =========================
    # 注册服务
    # =========================

    def register(self,name: str,service: Any):
        """
           注册服务到容器。

           功能：
           - 将 Provider / Service 按名称保存到容器中
           - 后续可以通过 get 方法统一获取

           参数：
           - name: str
             服务名称，例如 llm、memory、checkpoint。
           - service: Any
             服务实例，可以是 Provider，也可以是普通 Service。

           返回值：
           - None
             只执行注册动作。
           """
        if name in self._services:
            logger.warning(
                f"服务已存在，将被覆盖: {name}"
            )

        logger.info(
            f"注册服务: {name}"
        )

        self._services[name] = service

    # =========================
    # 获取服务
    # =========================

    def get(self, name: str):
        """
           从容器中获取服务。

           功能：
           - 根据服务名称获取已经注册的 Provider / Service

           参数：
           - name: str
             服务名称。

           返回值：
           - Any
             对应的服务实例。
           """

        if name not in self._services:

            raise ValueError(
                f"服务不存在: {name}"
            )

        return self._services[name]

    # =========================
    # 启动容器
    # =========================

    async def startup(self):
        """
           启动容器中的所有服务。

           功能：
           - 遍历所有已注册服务
           - 如果服务实现了 startup 方法，则自动调用
           - 避免业务入口手动启动每个服务

           参数：
           - 无

           返回值：
           - None
             只执行启动流程。
       """

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
        """
           关闭容器中的所有服务。

           功能：
           - 反向遍历所有已注册服务
           - 如果服务实现了 shutdown 方法，则自动调用
           - 释放数据库连接、文件连接、运行时资源等

           参数：
           - 无

           返回值：
           - None
             只执行关闭流程。
       """
        if not self._started:
            logger.warning(
                "Container 尚未启动，无需关闭"
            )

            return

        logger.info(
            "🛑 Container 关闭中..."
        )

        for name, service in reversed(
                list(self._services.items())
        ):

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

        self._started = False

        logger.info(
            "✅ Container 已关闭"
        )