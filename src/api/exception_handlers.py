from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.schemas import ApiErrorDetail, ApiErrorResponse
from src.core.errors.base import DogAgentError


def register_exception_handlers(app: FastAPI) -> None:
    """
    为 FastAPI 注册统一异常处理器。

    功能：
        把请求参数错误、HTTP 业务错误、DogAgentError 和未知系统异常转换成
        相同 JSON 契约，避免把内部堆栈和敏感配置直接暴露给调用方。

    参数含义：
        app:
            需要注册异常处理器的 FastAPI 应用。

    返回值含义：
        None。
    """

    app.add_exception_handler(
        RequestValidationError,
        _handle_request_validation_error,
    )
    app.add_exception_handler(
        HTTPException,
        _handle_http_exception,
    )
    app.add_exception_handler(
        DogAgentError,
        _handle_dog_agent_error,
    )
    app.add_exception_handler(
        Exception,
        _handle_unexpected_exception,
    )


async def _handle_request_validation_error(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    """
    处理 Pydantic 请求参数校验错误。

    参数含义：
        request:
            当前 FastAPI 请求。
        error:
            FastAPI 在调用路由前捕获的参数校验异常。

    返回值含义：
        JSONResponse:
            HTTP 422 和不包含原始输入值的统一错误 JSON。
    """

    details = [
        {
            "location": [
                str(part)
                for part in item.get("loc", ())
            ],
            "message": str(item.get("msg") or ""),
            "type": str(item.get("type") or ""),
        }
        for item in error.errors()
    ]
    return _build_error_response(
        request=request,
        status_code=422,
        code="REQUEST_VALIDATION_FAILED",
        message="请求参数校验失败",
        details=details,
    )


async def _handle_http_exception(
    request: Request,
    error: HTTPException,
) -> JSONResponse:
    """
    处理路由主动抛出的 HTTP 业务异常。

    参数含义：
        request:
            当前 FastAPI 请求。
        error:
            包含 HTTP 状态码和公开 detail 的异常。

    返回值含义：
        JSONResponse:
            保留原状态码的统一错误 JSON。
    """

    error_codes = {
        404: "RESOURCE_NOT_FOUND",
        503: "SERVICE_NOT_READY",
    }
    return _build_error_response(
        request=request,
        status_code=error.status_code,
        code=error_codes.get(
            error.status_code,
            "HTTP_REQUEST_FAILED",
        ),
        message=str(error.detail),
    )


async def _handle_dog_agent_error(
    request: Request,
    error: DogAgentError,
) -> JSONResponse:
    """
    处理项目内可识别的 Agent 业务异常。

    参数含义：
        request:
            当前 FastAPI 请求。
        error:
            包含 message 和 recoverable 的 DogAgentError。

    返回值含义：
        JSONResponse:
            可恢复异常返回 400；不可恢复异常返回 500 和通用公开说明。
    """

    recoverable = bool(error.recoverable)
    return _build_error_response(
        request=request,
        status_code=400 if recoverable else 500,
        code=(
            "DOG_AGENT_REQUEST_FAILED"
            if recoverable
            else "DOG_AGENT_INTERNAL_ERROR"
        ),
        message=(
            error.message
            if recoverable
            else "Agent 服务执行失败"
        ),
    )


async def _handle_unexpected_exception(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """
    处理未被业务层识别的系统异常。

    参数含义：
        request:
            当前 FastAPI 请求。
        error:
            未知异常对象；异常原文不会写入 HTTP 响应。

    返回值含义：
        JSONResponse:
            HTTP 500 和不暴露内部实现的统一错误 JSON。
    """

    _ = error
    return _build_error_response(
        request=request,
        status_code=500,
        code="INTERNAL_SERVER_ERROR",
        message="服务器内部错误",
    )


def _build_error_response(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    """
    构建统一 JSON 错误响应。

    参数含义：
        request:
            当前 FastAPI 请求，用于读取 trace_id。
        status_code:
            HTTP 状态码。
        code:
            稳定机器错误码。
        message:
            可安全展示的错误说明。
        details:
            可选字段级错误详情。

    返回值含义：
        JSONResponse:
            带 X-Trace-ID 响应头的统一错误响应。
    """

    trace_id = str(
        getattr(request.state, "trace_id", "unknown")
        or "unknown"
    )
    body = ApiErrorResponse(
        error=ApiErrorDetail(
            code=code,
            message=message,
            details=list(details or []),
        ),
        trace_id=trace_id,
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers={
            "X-Trace-ID": trace_id,
        },
    )
