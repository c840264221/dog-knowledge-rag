"""V1.22 Nginx 代理部署验收脚本单元测试。"""

from __future__ import annotations

import json
from email.message import Message
from urllib.error import URLError

import pytest

from scripts.deployment import verify_v122_proxy_deployment as verifier


class _FakeHttpResponse:
    """提供部署验收测试所需的最小 HTTP 响应上下文。"""

    def __init__(
        self,
        *,
        status: int,
        server_header: str,
        payload: object,
    ) -> None:
        self.status = status
        self.headers = Message()
        self.headers["Server"] = server_header
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_verify_proxy_deployment_should_accept_healthy_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证健康、就绪和直连隔离全部正常时验收通过。"""

    responses: list[object] = [
        _FakeHttpResponse(
            status=200,
            server_header="nginx/1.30.4",
            payload={"status": "ok"},
        ),
        _FakeHttpResponse(
            status=200,
            server_header="nginx/1.30.4",
            payload={"status": "ready"},
        ),
        URLError("connection refused"),
    ]

    def fake_urlopen(*_: object, **__: object) -> object:
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(verifier, "urlopen", fake_urlopen)

    results = verifier.verify_proxy_deployment(
        base_url="http://proxy:8080/",
        direct_api_url="http://api:8000/",
        timeout_seconds=1,
    )

    assert [result.payload["status"] for result in results] == [
        "ok",
        "ready",
    ]
    assert responses == []


def test_verify_proxy_endpoint_should_reject_non_nginx_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证代理端点没有 Nginx 响应头时验收失败。"""

    monkeypatch.setattr(
        verifier,
        "urlopen",
        lambda *_args, **_kwargs: _FakeHttpResponse(
            status=200,
            server_header="uvicorn",
            payload={"status": "ok"},
        ),
    )

    with pytest.raises(RuntimeError, match="未经过 Nginx"):
        verifier.verify_proxy_endpoint(
            url="http://proxy:8080/health",
            expected_status_value="ok",
            timeout_seconds=1,
        )


def test_verify_direct_api_should_reject_any_http_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 API 直连地址只要返回 HTTP 响应就判定隔离失败。"""

    monkeypatch.setattr(
        verifier,
        "urlopen",
        lambda *_args, **_kwargs: _FakeHttpResponse(
            status=200,
            server_header="uvicorn",
            payload={"status": "ok"},
        ),
    )

    with pytest.raises(RuntimeError, match="直连端口仍可访问"):
        verifier.verify_direct_api_is_unreachable(
            url="http://api:8000/health",
            timeout_seconds=1,
        )
