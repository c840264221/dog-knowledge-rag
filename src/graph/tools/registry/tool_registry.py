from src.graph.tools.implementations.weather_tool import WeatherTool
from src.graph.tools.implementations.date_tool import DateTool


class ToolRegistry:

    def __init__(self):

        self.tools = {}

    def register(self, tool):
        name = tool.metadata.name

        self.tools[name] = tool

    def get_tool(self, name):

        return self.tools.get(name)


registry = ToolRegistry()

registry.register(WeatherTool())
registry.register(DateTool())