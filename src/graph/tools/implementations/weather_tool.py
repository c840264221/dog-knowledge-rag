from src.graph.tools.base.base_tool import BaseTool
from src.graph.tools.schemas.tool_metadata import ToolMetadata
from src.graph.tools.schemas.weather_args import WeatherArgs
from src.logger import logger


class WeatherTool(BaseTool):

    metadata = ToolMetadata(
        name='weather',
        description='查询天气',
        timeout=5,
        retries=3
    )

    args_schema = WeatherArgs

    def run(self, args):

        if not isinstance(args, dict):

            logger.error( f"weather args必须是dict，当前是: {type(args)}")

            raise Exception(
                f"weather args必须是dict，当前是: {type(args)}"
            )

        validated_args = self.args_schema(
            **args
        )

        city = validated_args.city

        return f"{city}今天晴天"