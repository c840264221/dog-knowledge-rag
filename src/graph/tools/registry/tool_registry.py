class ToolRegistry:
    """
    工具注册表。

    功能：
        保存工具名称到工具实例的映射，并提供注册和查询能力。

    参数：
        无。

    返回值：
        ToolRegistry:
            工具注册表实例。
    """

    def __init__(self):
        """
        初始化工具注册表。

        功能：
            创建内部 tools 字典，用于保存工具实例。

        参数：
            无。

        返回值：
            None。
        """

        self.tools = {}

    def register(self, tool):
        """
        注册工具实例。

        功能：
            读取 tool.metadata.name，并将工具保存到注册表。

        参数：
            tool:
                工具实例，需要提供 metadata.name 字段。

        返回值：
            None。
        """

        name = tool.metadata.name

        self.tools[name] = tool

    def get_tool(self, name):
        """
        根据工具名获取工具实例。

        功能：
            从注册表中读取指定名称的工具。

        参数：
            name:
                工具名称。

        返回值：
            Any | None:
                找到时返回工具实例，找不到时返回 None。
        """

        return self.tools.get(name)
