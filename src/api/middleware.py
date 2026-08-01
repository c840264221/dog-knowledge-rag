from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from ipaddress import (
    IPv4Address,
    IPv4Network,
    IPv6Address,
    IPv6Network,
    ip_address,
    ip_network,
)
from typing import Any
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.api.schemas import ApiErrorDetail, ApiErrorResponse
from src.logger import logger


TRACE_HEADER_NAME = "X-Trace-ID"
IpAddress = IPv4Address | IPv6Address
IpNetwork = IPv4Network | IPv6Network


class _RequestBodyTooLargeError(RuntimeError):
    """表示未声明总长度的请求在分块接收过程中超过字节上限。"""


class ApiRequestBodyLimitMiddleware:
    """
    在请求体进入 JSON、Pydantic 和 Agent 之前限制最大字节数。

    功能：
        优先检查 Content-Length；没有可信总长度时包装 ASGI receive 并累计
        http.request 分块大小，超限后返回统一 HTTP 413 错误响应。

    参数含义：
        app:
            被当前中间件包装的下一层 ASGI 应用。
        max_body_bytes:
            单次 HTTP 请求体允许占用的最大字节数。

    返回值含义：
        ApiRequestBodyLimitMiddleware:
            可注册到 FastAPI 的请求资源保护中间件。
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int,
    ) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """
        检查一次 ASGI HTTP 请求的请求体大小。

        参数含义：
            scope:
                ASGI 请求元数据，包含类型、路径、请求头和共享 state。
            receive:
                从服务器接收请求体消息的异步函数。
            send:
                向客户端发送响应消息的异步函数。

        返回值含义：
            None:
                合法请求交给下一层；超限请求直接发送 HTTP 413。
        """

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = _parse_content_length(scope)
        if (
            content_length is not None
            and content_length > self.max_body_bytes
        ):
            await _send_request_body_too_large(
                scope=scope,
                receive=receive,
                send=send,
                max_body_bytes=self.max_body_bytes,
            )
            return

        received_body_bytes = 0

        async def receive_with_limit() -> Message:
            """
            累计一个请求的 ASGI 请求体分块。

            参数含义：
                无。

            返回值含义：
                Message:
                    未超过上限时返回原始 ASGI 消息。
            """

            nonlocal received_body_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_body_bytes += len(message.get("body", b""))
                if received_body_bytes > self.max_body_bytes:
                    raise _RequestBodyTooLargeError
            return message

        try:
            await self.app(scope, receive_with_limit, send)
        except _RequestBodyTooLargeError:
            await _send_request_body_too_large(
                scope=scope,
                receive=receive,
                send=send,
                max_body_bytes=self.max_body_bytes,
            )


@dataclass(frozen=True)
class RateLimitDecision:
    """
    保存一次客户端限流判断结果。

    参数含义：
        allowed:
            当前请求是否允许继续执行。
        remaining:
            当前窗口内还可以接受的请求数量。
        retry_after_seconds:
            被拒绝时建议客户端等待的秒数；放行时为 0。
    """

    allowed: bool
    remaining: int
    retry_after_seconds: int


class InMemoryApiRateLimiter:
    """
    使用进程内滑动时间窗口统计每个客户端的请求频率。

    功能：
        保存客户端最近请求时间，移除窗口外记录，并在并发锁保护下生成
        放行或拒绝决定；定期清理长期不再访问的客户端，限制内存增长。

    参数含义：
        request_limit:
            一个窗口内允许的最大请求数量。
        window_seconds:
            滑动统计窗口秒数。
        clock:
            返回单调递增秒数的函数，测试可注入确定性时钟。

    返回值含义：
        InMemoryApiRateLimiter:
            只在当前 Python 进程内共享状态的基础限流器。
    """

    def __init__(
        self,
        *,
        request_limit: int,
        window_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.request_limit = request_limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._requests_by_client: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup_at = 0.0

    async def acquire(self, client_id: str) -> RateLimitDecision:
        """
        为一个客户端登记请求并返回当前限流决定。

        参数含义：
            client_id:
                当前请求使用的稳定客户端标识，基础版本使用连接来源 IP。

        返回值含义：
            RateLimitDecision:
                是否放行、剩余额度和建议重试等待时间。
        """

        now = self._clock()
        cutoff = now - self.window_seconds
        async with self._lock:
            self._cleanup_expired_clients(
                now=now,
                cutoff=cutoff,
            )
            request_times = self._requests_by_client.setdefault(
                client_id,
                deque(),
            )
            while request_times and request_times[0] <= cutoff:
                request_times.popleft()

            if len(request_times) >= self.request_limit:
                retry_after_seconds = max(
                    1,
                    math.ceil(
                        request_times[0]
                        + self.window_seconds
                        - now
                    ),
                )
                return RateLimitDecision(
                    allowed=False,
                    remaining=0,
                    retry_after_seconds=retry_after_seconds,
                )

            request_times.append(now)
            return RateLimitDecision(
                allowed=True,
                remaining=self.request_limit - len(request_times),
                retry_after_seconds=0,
            )

    def _cleanup_expired_clients(
        self,
        *,
        now: float,
        cutoff: float,
    ) -> None:
        """
        定期删除窗口外请求以及不再活跃的客户端。

        参数含义：
            now:
                当前单调时钟秒数。
            cutoff:
                当前窗口最早允许保留的时间。

        返回值含义：
            None。
        """

        if now - self._last_cleanup_at < self.window_seconds:
            return
        for client_id, request_times in list(
            self._requests_by_client.items()
        ):
            while request_times and request_times[0] <= cutoff:
                request_times.popleft()
            if not request_times:
                del self._requests_by_client[client_id]
        self._last_cleanup_at = now


class ApiRateLimitMiddleware:
    """
    在业务路由执行前限制单个客户端的请求频率。

    功能：
        只统计 /v1 业务请求，跳过健康检查和 CORS OPTIONS 预检；放行响应
        增加剩余额度头，超限时直接返回统一 HTTP 429。

    参数含义：
        app:
            被当前中间件包装的下一层 ASGI 应用。
        request_limit:
            单个窗口内允许的请求数量。
        window_seconds:
            请求统计窗口秒数。
        trusted_proxy_cidrs:
            允许提供代理转发头的反向代理 IP 或 CIDR；默认空列表表示不信任
            任何代理头，继续使用直接连接地址。

    返回值含义：
        ApiRateLimitMiddleware:
            可注册到 FastAPI 的单进程基础限流中间件。
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        request_limit: int,
        window_seconds: int,
        trusted_proxy_cidrs: list[str] | tuple[str, ...] = (),
    ) -> None:
        self.app = app
        self.request_limit = request_limit
        self.window_seconds = window_seconds
        self.limiter = InMemoryApiRateLimiter(
            request_limit=request_limit,
            window_seconds=window_seconds,
        )
        self.trusted_proxy_networks = tuple(
            ip_network(cidr, strict=False)
            for cidr in trusted_proxy_cidrs
        )

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """
        对一次 ASGI 请求执行限流判断。

        参数含义：
            scope:
                ASGI 请求元数据，包含路径、方法和客户端连接地址。
            receive:
                接收请求消息的异步函数。
            send:
                发送响应消息的异步函数。

        返回值含义：
            None:
                放行请求交给下一层；超限请求直接发送 HTTP 429。
        """

        if not _should_apply_rate_limit(scope):
            await self.app(scope, receive, send)
            return

        decision = await self.limiter.acquire(
            _resolve_rate_limit_client_id(
                scope,
                trusted_proxy_networks=self.trusted_proxy_networks,
            )
        )
        if not decision.allowed:
            await _send_rate_limit_exceeded(
                scope=scope,
                receive=receive,
                send=send,
                request_limit=self.request_limit,
                window_seconds=self.window_seconds,
                retry_after_seconds=decision.retry_after_seconds,
            )
            return

        async def send_with_rate_limit_headers(
            message: Message,
        ) -> None:
            """
            为正常业务响应增加当前限流额度响应头。

            参数含义：
                message:
                    下一层准备发送的 ASGI 响应消息。

            返回值含义：
                None。
            """

            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-RateLimit-Limit"] = str(
                    self.request_limit
                )
                headers["X-RateLimit-Remaining"] = str(
                    decision.remaining
                )
            await send(message)

        await self.app(
            scope,
            receive,
            send_with_rate_limit_headers,
        )


def _should_apply_rate_limit(scope: Scope) -> bool:
    """
    判断当前请求是否属于需要限流的业务接口。

    参数含义：
        scope:
            当前 ASGI 请求元数据。

    返回值含义：
        bool:
            /v1 HTTP 业务请求且不是 OPTIONS 时返回 True。
    """

    if scope["type"] != "http":
        return False
    if str(scope.get("method") or "").upper() == "OPTIONS":
        return False
    path = str(scope.get("path") or "")
    return path == "/v1" or path.startswith("/v1/")


def _resolve_rate_limit_client_id(
    scope: Scope,
    *,
    trusted_proxy_networks: tuple[IpNetwork, ...] = (),
) -> str:
    """
    从直接连接或可信代理链解析限流客户端编号。

    功能：
        默认使用 ASGI server 看到的连接来源 IP；只有直接来源命中可信代理
        网络时才读取 X-Forwarded-For，并从右向左跳过可信代理，选择最靠近
        服务的非代理地址。非法 Header 会退回直接来源，避免伪造绕过限流。

    参数含义：
        scope:
            当前 ASGI 请求元数据。
        trusted_proxy_networks:
            已完成校验的可信 IPv4 / IPv6 代理网络元组。

    返回值含义：
        str:
            安全解析后的客户端来源 IP；缺失时使用 unknown-client。
    """

    client = scope.get("client")
    if not isinstance(client, tuple) or not client:
        return "unknown-client"

    direct_client_id = str(client[0])
    direct_client_ip = _parse_ip_address(direct_client_id)
    if (
        direct_client_ip is None
        or not _is_trusted_proxy(
            direct_client_ip,
            trusted_proxy_networks,
        )
    ):
        return direct_client_id

    forwarded_for = Headers(scope=scope).get("x-forwarded-for")
    if not forwarded_for:
        return direct_client_id

    forwarded_ips: list[IpAddress] = []
    for raw_address in forwarded_for.split(","):
        forwarded_ip = _parse_ip_address(raw_address.strip())
        if forwarded_ip is None:
            return direct_client_id
        forwarded_ips.append(forwarded_ip)

    for forwarded_ip in reversed(forwarded_ips):
        if not _is_trusted_proxy(
            forwarded_ip,
            trusted_proxy_networks,
        ):
            return str(forwarded_ip)

    return str(forwarded_ips[0])


def _parse_ip_address(value: str) -> IpAddress | None:
    """
    把字符串解析为标准 IPv4 或 IPv6 地址。

    功能：
        使用标准库拒绝主机名、端口和格式非法的代理头片段，使调用方可以
        在解析失败时采取保守回退策略。

    参数含义：
        value:
            需要解析的单个地址字符串。

    返回值含义：
        IpAddress | None:
            合法时返回标准 IP 对象，非法时返回 None。
    """

    try:
        return ip_address(value)
    except ValueError:
        return None


def _is_trusted_proxy(
    address: IpAddress,
    trusted_proxy_networks: tuple[IpNetwork, ...],
) -> bool:
    """
    判断一个地址是否位于明确配置的可信代理网络中。

    参数含义：
        address:
            当前准备检查的 IPv4 或 IPv6 地址。
        trusted_proxy_networks:
            允许运行时信任的代理网络元组。

    返回值含义：
        bool:
            地址属于任一同版本代理网络时返回 True，否则返回 False。
    """

    return any(
        address.version == network.version and address in network
        for network in trusted_proxy_networks
    )


async def _send_rate_limit_exceeded(
    *,
    scope: Scope,
    receive: Receive,
    send: Send,
    request_limit: int,
    window_seconds: int,
    retry_after_seconds: int,
) -> None:
    """
    发送统一请求频率超限错误响应。

    参数含义：
        scope:
            当前 ASGI 请求元数据，用于读取 trace_id。
        receive:
            原始 ASGI 请求接收函数。
        send:
            当前 ASGI 响应发送函数。
        request_limit:
            当前窗口允许的最大请求数量。
        window_seconds:
            当前统计窗口秒数。
        retry_after_seconds:
            建议客户端等待后重试的秒数。

    返回值含义：
        None:
            HTTP 429 通过 ASGI send 发送。
    """

    trace_id = str(
        scope.get("state", {}).get("trace_id") or "unknown"
    )
    body = ApiErrorResponse(
        error=ApiErrorDetail(
            code="RATE_LIMIT_EXCEEDED",
            message="请求过于频繁，请稍后重试",
            details=[
                {
                    "request_limit": request_limit,
                    "window_seconds": window_seconds,
                    "retry_after_seconds": retry_after_seconds,
                }
            ],
        ),
        trace_id=trace_id,
    )
    response = JSONResponse(
        status_code=429,
        content=body.model_dump(mode="json"),
        headers={
            TRACE_HEADER_NAME: trace_id,
            "Retry-After": str(retry_after_seconds),
            "X-RateLimit-Limit": str(request_limit),
            "X-RateLimit-Remaining": "0",
        },
    )
    await response(scope, receive, send)


def _parse_content_length(scope: Scope) -> int | None:
    """
    从 ASGI 请求头解析客户端声明的请求体字节数。

    参数含义：
        scope:
            当前 ASGI HTTP 请求元数据。

    返回值含义：
        int | None:
            合法非负 Content-Length；缺失或格式非法时返回 None，后续改用
            实际分块累计结果判断。
    """

    raw_content_length = Headers(scope=scope).get("content-length")
    if raw_content_length is None:
        return None
    try:
        content_length = int(raw_content_length)
    except ValueError:
        return None
    return content_length if content_length >= 0 else None


async def _send_request_body_too_large(
    *,
    scope: Scope,
    receive: Receive,
    send: Send,
    max_body_bytes: int,
) -> None:
    """
    发送统一请求体超限错误响应。

    参数含义：
        scope:
            当前 ASGI HTTP 请求元数据，用于读取 trace_id。
        receive:
            原始 ASGI 接收函数，交给 JSONResponse 完成响应调用契约。
        send:
            当前响应发送函数。
        max_body_bytes:
            当前服务允许的最大请求体字节数。

    返回值含义：
        None:
            HTTP 413 通过 ASGI send 发送，不返回业务对象。
    """

    trace_id = str(
        scope.get("state", {}).get("trace_id") or "unknown"
    )
    body = ApiErrorResponse(
        error=ApiErrorDetail(
            code="REQUEST_BODY_TOO_LARGE",
            message="请求体超过服务允许的最大大小",
            details=[
                {
                    "max_body_bytes": max_body_bytes,
                }
            ],
        ),
        trace_id=trace_id,
    )
    response = JSONResponse(
        status_code=413,
        content=body.model_dump(mode="json"),
        headers={
            TRACE_HEADER_NAME: trace_id,
        },
    )
    await response(scope, receive, send)


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
