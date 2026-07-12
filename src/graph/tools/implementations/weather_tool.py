from src.graph.tools.base.base_tool import BaseTool
from src.graph.tools.schemas.tool_metadata import ToolMetadata
from src.graph.tools.schemas.weather_args import WeatherArgs

import aiohttp


class WeatherTool(BaseTool):

    metadata = ToolMetadata(
        name='weather',
        description='查询指定城市的当前天气',
        timeout=5,
        retries=3,
        require_confirm=True,
        input_schema=WeatherArgs.model_json_schema(),
    )

    args_schema = WeatherArgs

    async def run(self, args):

        validated_args = self.args_schema(
            **args
        )

        city = validated_args.city

        # ========= 地理编码 =========

        geocode_url = (
            "https://geocoding-api.open-meteo.com/"
            f"v1/search?name={city}"
            "&count=1"
            "&language=zh"
            "&format=json"
        )

        async with aiohttp.ClientSession() as session:

            async with session.get(
                    geocode_url
            ) as resp:
                geo_data = await resp.json()

        if not geo_data.get("results"):
            raise Exception(
                f"未找到城市: {city}"
            )

        result = geo_data["results"][0]

        lat = result["latitude"]

        lon = result["longitude"]

        city_name = result["name"]

        country = result["country"]

        # ========= 天气接口 =========

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}"
            f"&longitude={lon}"
            "&current_weather=true"
            "&timezone=auto"
        )

        async with aiohttp.ClientSession() as session:

            async with session.get(
                    weather_url
            ) as resp:
                weather_data = await resp.json()

        current = weather_data.get(
            "current_weather",
            {}
        )

        if not current:
            raise Exception(
                f"{city}天气查询失败"
            )

        temp = current.get(
            "temperature"
        )

        wind_speed = current.get(
            "windspeed"
        )

        return (

            f"{city_name}，{country}，"

            f"温度 {temp}°C，"

            f"风速 {wind_speed} km/h"
        )
