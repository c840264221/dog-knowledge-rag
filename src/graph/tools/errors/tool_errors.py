from src.graph.tools.errors.base import (
    BaseRuntimeError
)


class ToolNotFoundError(BaseRuntimeError):

    def __init__(self,tool_name: str):

        super().__init__(
            f"工具不存在: {tool_name}",
            retryable=False
        )


class ToolTimeoutError(BaseRuntimeError):

    def __init__(self,tool_name: str):

        super().__init__(
            f"工具执行超时: {tool_name}",
            retryable=True
        )


class ToolValidationError(BaseRuntimeError):

    def __init__(self,message: str):

        super().__init__(
            f"工具参数错误: {message}",
            retryable=False
        )


class ToolExecutionError(BaseRuntimeError):

    def __init__(self,message: str):

        super().__init__(
            f"工具执行失败: {message}",
            retryable=True
        )


class FatalToolError(BaseRuntimeError):

    def __init__(self,message: str):

        super().__init__(
            f"致命错误: {message}",
            retryable=False
        )