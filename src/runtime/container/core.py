from typing import Any

from src.logger import logger

import inspect


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


    async def _call_lifecycle_method(
            self,
            service_name: str,
            service,
            method_name: str,
    ):
        """
        调用服务的生命周期方法。

        功能：
        - 根据 method_name 获取 service 上对应的生命周期方法
        - 如果该方法不存在，则直接跳过
        - 如果该方法是同步函数，则直接执行
        - 如果该方法返回 awaitable（可等待对象），则使用 await 等待执行完成
        - 统一兼容同步 Provider 和异步 Provider

        参数：
        - service_name：服务名称，字符串格式，用于日志输出
        - service：服务实例，可以是 provider、runtime service 或其他容器管理对象
        - method_name：生命周期方法名称，字符串格式，例如 startup 或 shutdown

        返回值：
        - None
          只负责执行生命周期方法，不返回业务数据。

        专业名词：
        - lifecycle method（生命周期方法）：服务启动或关闭时自动执行的方法
        - awaitable（可等待对象）：可以被 await 的对象，例如 coroutine、Task、Future
        - coroutine（协程）：async 函数调用后返回的异步执行对象
        """

        lifecycle_method = getattr(
            service,
            method_name,
            None,
        )

        if not callable(lifecycle_method):
            return

        logger.info(
            f"{method_name} 服务: {service_name}"
        )

        result = lifecycle_method()

        if inspect.isawaitable(result):
            await result


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
        - 同时支持同步 startup 和异步 startup
        - 避免业务入口手动启动每个服务

        参数：
        - 无

        返回值：
        - None
          只执行启动流程，不返回业务数据。

        专业名词：
        - container startup（容器启动）：统一初始化容器中注册的服务
        - provider startup（提供者启动）：provider 初始化自身资源的过程
        - sync startup（同步启动）：普通 def startup 方法
        - async startup（异步启动）：async def startup 方法
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
            await self._call_lifecycle_method(
                service_name=name,
                service=service,
                method_name="startup",
            )

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
        - 遍历所有已注册服务
        - 如果服务实现了 shutdown 方法，则自动调用
        - 同时支持同步 shutdown 和异步 shutdown
        - 按注册顺序的反方向关闭服务

        参数：
        - 无

        返回值：
        - None
          只执行关闭流程，不返回业务数据。

        专业名词：
        - container shutdown（容器关闭）：统一释放容器中服务占用的资源
        - provider shutdown（提供者关闭）：provider 释放自身资源的过程
        - reverse order shutdown（反序关闭）：按照注册顺序的反方向关闭服务
        """

        if not self._started:
            logger.warning(
                "Container 未启动"
            )

            return

        logger.info(
            "🛑 Container 关闭中..."
        )

        for name, service in reversed(
                list(self._services.items())
        ):
            await self._call_lifecycle_method(
                service_name=name,
                service=service,
                method_name="shutdown",
            )

        self._started = False

        logger.info(
            "✅ Container 关闭完成"
        )