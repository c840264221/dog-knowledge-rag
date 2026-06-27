import os
import sys

from loguru import logger as base_logger

# from src.runtime.trace import trace_ctx

from src.runtime.context import runtime_ctx

from src.config import LOG_PATH


# 创建日志目录
os.makedirs(LOG_PATH, exist_ok=True)


# 注入 Context
def inject_context(record):

    # record["extra"]["trace_id"] = (
    #     trace_ctx.get_trace_id()
    # )
    #
    # record["extra"]["user_id"] = (
    #     trace_ctx.get_user_id()
    # )
    #
    # record["extra"]["session_id"] = (
    #     trace_ctx.get_session_id()
    # )
    #
    # record["extra"]["component"] = (
    #     trace_ctx.get_component()
    # )

    ctx = runtime_ctx.get()

    record["extra"]["trace_id"] = (
        ctx.trace_id if ctx else None
    )

    record["extra"]["user_id"] = (
        ctx.user_id if ctx else None
    )

    record["extra"]["session_id"] = (
        ctx.session_id if ctx else None
    )

    record["extra"]["component"] = (
        ctx.component if ctx else None
    )


logger = base_logger.patch(
    inject_context
)


CONSOLE_LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper()

LOG_DIAGNOSE = os.getenv(
    "LOG_DIAGNOSE",
    "false",
).lower() == "true"

LOG_BACKTRACE = os.getenv(
    "LOG_BACKTRACE",
    "false",
).lower() == "true"

# 移除默认 Handler
logger.remove()


# Console Logger
logger.add(

    sys.stdout,

    level=CONSOLE_LOG_LEVEL,

    colorize=True,

    enqueue=True,

    backtrace=LOG_BACKTRACE,

    diagnose=LOG_DIAGNOSE,

    format=(

        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "

        "<level>{level: <8}</level> | "

        "<cyan>{module}:{function}:{line}</cyan> | "

        "trace_id=<yellow>{extra[trace_id]}</yellow> | "

        "<level>{message}</level>"
    )
)


# App JSON Log
logger.add(

    os.path.join(
        LOG_PATH,
        "app.log"
    ),

    level="INFO",

    rotation="10 MB",

    retention="7 days",

    encoding="utf-8",

    enqueue=True,

    serialize=True
)


# Error JSON Log
logger.add(

    os.path.join(
        LOG_PATH,
        "error.log"
    ),

    level="ERROR",

    rotation="10 MB",

    retention="30 days",

    encoding="utf-8",

    enqueue=True,

    serialize=True
)