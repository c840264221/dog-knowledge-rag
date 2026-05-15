import aiohttp
import asyncio
from src.common.decorators.safe_llm_invoke import retry_async


@retry_async(max_attempts=3, delay=1, backoff=2, exceptions=(ConnectionError,TimeoutError))
async def get_weather_by_city(city: str) -> str:
    """根据城市名获取实时天气"""
    # 1. 地理编码：将城市名转换为经纬度
    geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=zh&format=json"
    async with aiohttp.ClientSession() as session:
        async with session.get(geocode_url) as resp:
            geo_data = await resp.json()
            if not geo_data.get("results"):
                return f"未找到城市 '{city}' 的天气信息"

            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            city_name_cn = geo_data["results"][0]["name"]
            country = geo_data["results"][0]["country"]
            admin1 = geo_data["results"][0].get("admin1", "")  # 省/州

    # 2. 调用天气API，获取实时数据
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&timezone=auto"
    async with aiohttp.ClientSession() as session:
        async with session.get(weather_url) as resp:
            weather_data = await resp.json()
            current = weather_data.get("current_weather", {})
            if not current:
                return f"获取 '{city}' 的天气数据失败"

            temp = current.get("temperature")
            wind_speed = current.get("windspeed")
            weather_code = current.get("weathercode")

            # 简单的天气代码映射
            weather_map = {0: "未知", 1: "晴朗", 2: "部分多云", 3: "阴天", 45: "有雾", 51: "小雨", 61: "雨"}
            condition = weather_map.get(weather_code, "未知")

            return (f"{admin1}{city_name_cn}，{country}，"
                    f"当前天气：{condition}，温度：{temp}°C，"
                    f"风速：{wind_speed} km/h")