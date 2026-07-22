from pydantic import Field

from src.settings.base import BaseAppSettings


class RuntimeSettings(BaseAppSettings):

    tool_timeout: int = 30

    retry_delay: float = 1

    max_retries: int = 3

    # 工具并发数量
    tool_concurrency: int = 10

    # 是否开启链路追踪
    enable_trace: bool = True

    # 是否开启事件总线
    enable_event_bus: bool = True

    # 是否开启结构化日志
    enable_structured_logging: bool = True

    # 是否开启重试机制
    enable_retry: bool = True

    # 是否开启超时机制
    enable_timeout: bool = True

    # 流式响应中每个数据块的超时时间
    stream_chunk_timeout: int = 120

    # =========================
    # Multi-Agent Scheduler
    # =========================

    # 同一批多 Agent 步骤最多并发执行数量
    multi_agent_maximum_parallel_steps: int = Field(default=4, ge=1)

    # 单个异步 Worker 最长执行秒数
    multi_agent_step_timeout_seconds: float = Field(default=120.0, gt=0)

    # 单个 Worker 总尝试次数，包含第一次执行
    multi_agent_maximum_step_attempts: int = Field(default=2, ge=1)

    # =========================
    # Middleware
    # =========================

    enable_retry_middleware: bool = True

    enable_timeout_middleware: bool = True

    enable_trace_middleware: bool = True

    enable_logging_middleware: bool = True

    enable_async_middleware: bool = True
