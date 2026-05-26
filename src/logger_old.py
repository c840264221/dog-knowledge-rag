import sys
import os
from loguru import logger
from src.config import LOG_PATH

# 确保日志目录存在
# LOG_PATH = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_PATH, exist_ok=True)

# 自定义格式（包含时间、级别、代码位置、消息）
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {module}:{function}:{line} - {message}"

# 移除默认 handler
logger.remove()

# 控制台输出（彩色，级别 DEBUG 及以上）
logger.add(sys.stdout,
           level="DEBUG",
           format=LOG_FORMAT,
           colorize=True)

# 文件输出（滚动，级别 INFO 及以上，避免文件太大）
logger.add(os.path.join(LOG_PATH, "app.log"),
           level="INFO",
           format=LOG_FORMAT,
           rotation="10 MB",
           retention="7 days",      # 保留最近7天的日志文件
           encoding="utf-8")

# 可选：单独的 error 日志文件（只记录 ERROR 以上）
logger.add(os.path.join(LOG_PATH, "error.log"),
           level="ERROR",
           format=LOG_FORMAT,
           rotation="10 MB",
           retention="30 days",
           encoding="utf-8")

# 对外暴露
app_logger = logger