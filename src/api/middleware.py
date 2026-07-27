from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.logger import logger


TRACE_HEADER_NAME = "X-Trace-ID"


class ApiRequestLoggingMiddleware:
    """
    为每个 HTTP 请求建立 trace_id 并记录完整响应耗时。

    功能：
        从 X-Trace-ID 请求头读取调用方链路编号，缺失时生成 UUID；把编号
        写入 request.state 和响应头，并在最后一个响应片段发送时记录状态码
        与完整耗时。纯 ASGI 实现可以覆盖 SSE 的整个传输周期。

    参数含义：
        app:
            被当前中间件包装的下一层 ASGI 应用。

    返回值含义：
        ApiRequestLoggingMiddleware:
            可注册到 FastAPI 的 ASGI 请求日志中间件。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """
        处理一次 ASGI 调用。

        参数含义：
            scope:
                ASGI 请求元数据，包含类型、路径、方法、请求头和 state。
            receive:
                接收客户端请求消息的异步函数。
            send:
                向客户端发送响应消息的异步函数。

        返回值含义：
            None:
                响应通过 send 函数逐段发送，不使用普通返回值。
        """

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        trace_id = str(
            headers.get(TRACE_HEADER_NAME)
            or uuid4()
        ).strip()
        state = scope.setdefault("state", {})
        state["trace_id"] = trace_id
        state["trace_id_source"] = (
            "header"
            if headers.get(TRACE_HEADER_NAME)
            else "generated"
        )

        started_at = time.perf_counter()
        status_code = 500
        response_finished = False

        async def send_with_trace(message: Message) -> None:
            """
            为响应增加 trace 头，并在最后一个响应片段记录完成日志。

            参数含义：
                message:
                    ASGI 应用准备发送的响应开始或响应正文消息。

            返回值含义：
                None:
                    修改消息后继续调用原始 send。
            """

            nonlocal status_code, response_finished
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                response_headers = MutableHeaders(scope=message)
                response_headers[TRACE_HEADER_NAME] = str(
                    state.get("trace_id") or trace_id
                )

            if (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                response_finished = True
                _log_request_finished(
                    scope=scope,
                    trace_id=str(state.get("trace_id") or trace_id),
                    status_code=status_code,
                    started_at=started_at,
                )

            await send(message)

        try:
            await self.app(scope, receive, send_with_trace)
        except Exception:
            logger.exception(
                "API 请求执行异常: "
                f"method={scope.get('method')} "
                f"path={scope.get('path')} "
                f"trace_id={state.get('trace_id') or trace_id}"
            )
            raise
        finally:
            if not response_finished:
                logger.warning(
                    "API 请求未完成响应发送: "
                    f"method={scope.get('method')} "
                    f"path={scope.get('path')} "
                    f"trace_id={state.get('trace_id') or trace_id}"
                )


def _log_request_finished(
    *,
    scope: Scope,
    trace_id: str,
    status_code: int,
    started_at: float,
) -> None:
    """
    记录一次 HTTP 请求完成日志。

    参数含义：
        scope:
            当前 ASGI HTTP 请求元数据。
        trace_id:
            当前请求最终使用的链路编号。
        status_code:
            HTTP 响应状态码。
        started_at:
            time.perf_counter 生成的请求开始时间。

    返回值含义：
        None。
    """

    latency_ms = max(
        0.0,
        (time.perf_counter() - started_at) * 1000,
    )
    logger.info(
        "API 请求完成: "
        f"method={scope.get('method')} "
        f"path={scope.get('path')} "
        f"status_code={status_code} "
        f"latency_ms={latency_ms:.2f} "
        f"trace_id={trace_id}"
    )
