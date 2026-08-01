from __future__ import annotations

import json
from ipaddress import ip_network

import pytest
from starlette.types import Message, Receive, Scope, Send

from src.api.middleware import (
    ApiRequestBodyLimitMiddleware,
    InMemoryApiRateLimiter,
    _resolve_rate_limit_client_id,
)


def _build_http_scope(
    *,
    client_ip: str,
    forwarded_for: str | None = None,
) -> Scope:
    """
    构建包含直连地址和可选代理头的最小 HTTP Scope。

    参数含义：
        client_ip:
            ASGI Server 直接看到的连接来源地址。
        forwarded_for:
            可选 X-Forwarded-For 原始文本。

    返回值含义：
        Scope:
            可以交给限流客户端识别函数的测试请求元数据。
    """

    headers: list[tuple[bytes, bytes]] = []
    if forwarded_for is not None:
        headers.append(
            (b"x-forwarded-for", forwarded_for.encode("ascii"))
        )
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat",
        "raw_path": b"/v1/chat",
        "query_string": b"",
        "headers": headers,
        "client": (client_ip, 12345),
        "server": ("testserver", 80),
        "state": {},
    }


def test_untrusted_client_should_not_control_forwarded_for() -> None:
    """
    验证非可信直连方伪造 X-Forwarded-For 时仍使用直连地址。

    参数含义：
        无。

    返回值含义：
        None。
    """

    scope = _build_http_scope(
        client_ip="198.51.100.20",
        forwarded_for="203.0.113.99",
    )

    client_id = _resolve_rate_limit_client_id(
        scope,
        trusted_proxy_networks=(ip_network("10.0.0.0/8"),),
    )

    assert client_id == "198.51.100.20"


def test_trusted_proxy_should_forward_real_client_ip() -> None:
    """
    验证可信直连代理可以提供单个真实客户端地址。

    参数含义：
        无。

    返回值含义：
        None。
    """

    scope = _build_http_scope(
        client_ip="10.0.0.10",
        forwarded_for="203.0.113.25",
    )

    client_id = _resolve_rate_limit_client_id(
        scope,
        trusted_proxy_networks=(ip_network("10.0.0.0/8"),),
    )

    assert client_id == "203.0.113.25"


def test_trusted_proxy_chain_should_be_resolved_from_right() -> None:
    """
    验证多层代理链从右向左跳过可信代理并找到真实客户端。

    参数含义：
        无。

    返回值含义：
        None。
    """

    scope = _build_http_scope(
        client_ip="10.0.0.10",
        forwarded_for="203.0.113.25, 10.0.0.20",
    )

    client_id = _resolve_rate_limit_client_id(
        scope,
        trusted_proxy_networks=(ip_network("10.0.0.0/8"),),
    )

    assert client_id == "203.0.113.25"


def test_spoofed_leftmost_address_should_not_replace_nearest_client() -> None:
    """
    验证代理追加真实来源后不会盲目信任客户端伪造的最左地址。

    参数含义：
        无。

    返回值含义：
        None。
    """

    scope = _build_http_scope(
        client_ip="10.0.0.10",
        forwarded_for="192.0.2.99, 198.51.100.30",
    )

    client_id = _resolve_rate_limit_client_id(
        scope,
        trusted_proxy_networks=(ip_network("10.0.0.0/8"),),
    )

    assert client_id == "198.51.100.30"


def test_invalid_forwarded_for_should_fall_back_to_direct_proxy() -> None:
    """
    验证代理头包含非法地址时保守退回直接连接地址。

    参数含义：
        无。

    返回值含义：
        None。
    """

    scope = _build_http_scope(
        client_ip="10.0.0.10",
        forwarded_for="203.0.113.25, invalid-address",
    )

    client_id = _resolve_rate_limit_client_id(
        scope,
        trusted_proxy_networks=(ip_network("10.0.0.0/8"),),
    )

    assert client_id == "10.0.0.10"


@pytest.mark.asyncio
async def test_body_limit_should_count_chunked_request_without_length() -> None:
    """
    测试没有 Content-Length 的分块请求仍会按实际接收字节累计限流。

    参数含义：
        无。

    返回值含义：
        None。
    """

    request_messages: list[Message] = [
        {
            "type": "http.request",
            "body": b"1234",
            "more_body": True,
        },
        {
            "type": "http.request",
            "body": b"5678",
            "more_body": False,
        },
    ]
    sent_messages: list[Message] = []
    downstream_completed = False

    async def receive() -> Message:
        """
        按顺序返回测试请求体分块。

        返回值含义：
            Message:
                下一条 ASGI http.request 消息。
        """

        return request_messages.pop(0)

    async def send(message: Message) -> None:
        """
        保存中间件发送的响应消息。

        参数含义：
            message:
                当前 ASGI 响应消息。

        返回值含义：
            None。
        """

        sent_messages.append(message)

    async def downstream_app(
        scope: Scope,
        receive_from_middleware: Receive,
        send_from_middleware: Send,
    ) -> None:
        """
        模拟会完整读取请求体的下游 ASGI 应用。

        参数含义：
            scope:
                当前测试请求元数据。
            receive_from_middleware:
                已被请求体限制中间件包装的接收函数。
            send_from_middleware:
                当前响应发送函数。

        返回值含义：
            None。
        """

        nonlocal downstream_completed
        _ = scope, send_from_middleware
        while True:
            message = await receive_from_middleware()
            if not message.get("more_body", False):
                break
        downstream_completed = True

    middleware = ApiRequestBodyLimitMiddleware(
        downstream_app,
        max_body_bytes=5,
    )
    scope: Scope = {
        "type": "http",
        "asgi": {
            "version": "3.0",
        },
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat",
        "raw_path": b"/v1/chat",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 123),
        "server": ("testserver", 80),
        "state": {
            "trace_id": "trace-chunked-body",
        },
    }

    await middleware(scope, receive, send)

    assert downstream_completed is False
    assert sent_messages[0]["type"] == "http.response.start"
    assert sent_messages[0]["status"] == 413
    response_body = json.loads(sent_messages[1]["body"])
    assert response_body["error"]["code"] == (
        "REQUEST_BODY_TOO_LARGE"
    )
    assert response_body["trace_id"] == "trace-chunked-body"


@pytest.mark.asyncio
async def test_rate_limiter_should_allow_again_after_window() -> None:
    """
    测试滑动窗口过期后旧请求会被移除并恢复客户端额度。

    参数含义：
        无。

    返回值含义：
        None。
    """

    current_time = [100.0]
    limiter = InMemoryApiRateLimiter(
        request_limit=2,
        window_seconds=10,
        clock=lambda: current_time[0],
    )

    first = await limiter.acquire("client-a")
    second = await limiter.acquire("client-a")
    rejected = await limiter.acquire("client-a")

    assert first.allowed is True
    assert first.remaining == 1
    assert second.allowed is True
    assert second.remaining == 0
    assert rejected.allowed is False
    assert rejected.retry_after_seconds == 10

    current_time[0] = 110.1
    recovered = await limiter.acquire("client-a")

    assert recovered.allowed is True
    assert recovered.remaining == 1
