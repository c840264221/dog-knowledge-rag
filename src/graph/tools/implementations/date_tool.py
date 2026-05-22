from src.graph.tools.base.base_tool import BaseTool
import datetime
from src.graph.tools.schemas.tool_metadata import ToolMetadata


class DateTool(BaseTool):
    metadata=ToolMetadata(
        name="date",
        description='获取当前日期',
        timeout=5,
        retries=3
    )

    def run(self, args):
        return datetime.date.today().strftime("%Y-%m-%d")