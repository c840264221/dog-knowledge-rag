"""把多个MCP服务合并为一个工具目录和调用入口，不创建新的Agent。"""

from collections.abc import Mapping


class McpToolDispatcher:
    """按工具名分发服务调用；当前同名工具拒绝装配，避免静默覆盖。"""

    def __init__(self, providers: Mapping[str, object] | None = None):
        """保存服务集合。

        参数：providers为宿主装配的服务标识到Provider的映射，每个Provider提供tool_definitions和tool_client。
        返回值：无；复制映射以固定本实例注册范围，不启动或关闭内部服务，不依赖具体业务服务名称。
        """
        self.providers = dict(providers or {})
        if any(not isinstance(name, str) or not name.strip() for name in self.providers):
            raise ValueError("MCP服务标识必须是非空字符串。")

    def _enabled_providers(self):
        """遍历已启用服务；无参数；产生服务标识与Provider，旧Provider未声明enabled时按启用兼容。"""
        for service_id, provider in self.providers.items():
            if provider is not None and getattr(provider, "enabled", True):
                yield service_id, provider

    @property
    def allowed_databases(self) -> dict:
        """合并可选数据库白名单；无参数；返回副本，供既有数据库工具目录兼容使用。

        不要求所有服务提供白名单；同一别名对应不同资源时拒绝合并，不扩大任何Provider自身权限。
        """
        aliases = {}
        for _, provider in self._enabled_providers():
            for alias, target in getattr(provider, "allowed_databases", {}).items():
                if alias in aliases and aliases[alias] != target:
                    raise ValueError(f"MCP数据库白名单别名冲突：{alias}")
                aliases[alias] = target
        return aliases

    @property
    def tool_client(self):
        """提供兼容现有ToolAgent的客户端入口；无参数；返回当前分发器。"""
        return self

    def _routes(self) -> dict:
        """读取服务持有的目录；无参数；返回工具名到定义、客户端、服务标识的映射，重名抛错。"""
        routes = {}
        for service_id, provider in self._enabled_providers():
            client = provider.tool_client
            if not callable(getattr(client, "call_tool", None)):
                raise TypeError(f"MCP服务{service_id}未提供可调用的tool_client.call_tool。")
            for definition in provider.tool_definitions:
                if definition.name in routes:
                    previous_service = routes[definition.name][2]
                    raise ValueError(f"MCP工具{definition.name}在服务{previous_service}和{service_id}中重名，请先配置唯一名称。")
                routes[definition.name] = (definition, client, service_id)
        return routes

    async def startup(self) -> None:
        """校验已启动服务的目录和别名；无参数、无返回值；服务生命周期仍由宿主容器管理。"""
        self._routes()
        _ = self.allowed_databases

    @property
    def tool_definitions(self) -> list:
        """构建所有已启用服务的合并目录；无参数；返回独立定义副本，不暴露连接对象。"""
        return [definition.model_copy(deep=True) for definition, _, _ in self._routes().values()]

    async def call_tool(self, name: str, arguments: dict):
        """按工具名选择服务；name为目录名称、arguments为已校验参数；返回对应服务业务结果。"""
        routes = self._routes()
        if name not in routes:
            raise ValueError(f"工具不在当前MCP目录中：{name}")
        _, client, _ = routes[name]
        return await client.call_tool(name, arguments)
