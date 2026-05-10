import datetime
from typing import List, Dict, Any

# 工具函数：获取当前时间
def get_current_time() -> str:
    now = datetime.datetime.now()
    return f"当前时间是 {now.strftime('%Y-%m-%d %H:%M:%S')}"

# 工具函数：获取今天的日期
def get_today_date() -> str:
    return datetime.date.today().strftime("%Y-%m-%d")
