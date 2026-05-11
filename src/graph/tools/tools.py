from src.graph.tools.get_time_tool import get_current_time, get_today_date
from src.graph.tools.get_weather_by_city import get_weather_by_city

# 工具分发字典
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_today_date": get_today_date,
    # 可以在这里继续添加更多工具，比如 search_akc_breed
    "get_weather": get_weather_by_city,
}