"""
验证 V1.22 Nginx 反向代理部署是否满足基础安全与可用性契约。

功能：
    通过 Nginx 检查 API 存活和就绪状态，确认响应来自 Nginx，并验证宿主机
    无法绕过代理直接访问 FastAPI 的 8000 端口。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ProxyEndpointResult:
    """
    保存一次代理 HTTP 端点验收结果。

    参数含义：
        url:
            本次实际请求的完整 URL。
        status_code:
            Nginx 返回的 HTTP 状态码。
        server_header:
            响应中的 Server 头，用于确认请求经过 Nginx。
        payload:
            已解析的 JSON 响应体。

    返回值含义：
        ProxyEndpointResult:
            可供主验收流程继续检查的结构化结果。
    """

    url: str
    status_code: int
    server_header: str
    payload: dict[str, Any]


def verify_proxy_endpoint(
    *,
    url: str,
    expected_status_value: str,
    timeout_seconds: float,
) -> ProxyEndpointResult:
    """
    验证一个健康端点通过 Nginx 返回预期 JSON。

    功能：
        请求指定端点，检查 HTTP 200、Nginx Server 响应头以及响应体中的
        status 字段，任一条件不满足时立即抛出 RuntimeError。

    参数含义：
        url:
            需要通过代理访问的完整健康检查 URL。
        expected_status_value:
            JSON 响应体中 status 字段的预期值。
        timeout_seconds:
            单次 HTTP 请求最多等待的秒数。

    返回值含义：
        ProxyEndpointResult:
            已通过校验的代理端点结果。
    """

    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = int(response.status)
            server_header = str(response.headers.get("Server") or "")
            raw_body = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"代理端点无法访问: {url}: {exc}") from exc

    if status_code != 200:
        raise RuntimeError(
            f"代理端点状态码异常: {url}: {status_code}"
        )
    if not server_header.lower().startswith("nginx"):
        raise RuntimeError(
            f"响应未经过 Nginx: {url}: Server={server_header!r}"
        )
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"代理端点没有返回合法 JSON: {url}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"代理端点响应体不是 JSON 对象: {url}")
    if payload.get("status") != expected_status_value:
        raise RuntimeError(
            "代理端点业务状态异常: "
            f"{url}: expected={expected_status_value!r}, "
            f"actual={payload.get('status')!r}"
        )

    return ProxyEndpointResult(
        url=url,
        status_code=status_code,
        server_header=server_header,
        payload=payload,
    )


def verify_direct_api_is_unreachable(
    *,
    url: str,
    timeout_seconds: float,
) -> None:
    """
    验证宿主机不能绕过 Nginx 直接访问 FastAPI。

    功能：
        尝试访问 API 直连地址；连接失败表示端口隔离生效，任何 HTTP 响应都
        表示端口仍然暴露并抛出 RuntimeError。

    参数含义：
        url:
            预期无法访问的 FastAPI 宿主机直连 URL。
        timeout_seconds:
            等待连接失败的最长秒数。

    返回值含义：
        None:
            直连端口不可访问时正常结束。
    """

    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = int(response.status)
    except HTTPError as exc:
        raise RuntimeError(
            f"FastAPI 直连端口仍可访问: {url}: HTTP {exc.code}"
        ) from exc
    except (URLError, TimeoutError):
        return

    raise RuntimeError(
        f"FastAPI 直连端口仍可访问: {url}: HTTP {status_code}"
    )


def verify_proxy_deployment(
    *,
    base_url: str,
    direct_api_url: str,
    timeout_seconds: float,
) -> list[ProxyEndpointResult]:
    """
    执行完整的 Nginx 代理部署基础验收。

    功能：
        依次检查代理存活、应用就绪和 API 直连隔离，全部通过后返回两个代理
        端点结果，供命令行输出或其他自动化流程复用。

    参数含义：
        base_url:
            用户实际访问的 Nginx 基础地址。
        direct_api_url:
            预期被隔离的 FastAPI 宿主机基础地址。
        timeout_seconds:
            每次网络探测的超时秒数。

    返回值含义：
        list[ProxyEndpointResult]:
            `/health` 和 `/ready` 的结构化验收结果。
    """

    normalized_base_url = base_url.rstrip("/")
    normalized_direct_url = direct_api_url.rstrip("/")
    results = [
        verify_proxy_endpoint(
            url=f"{normalized_base_url}/health",
            expected_status_value="ok",
            timeout_seconds=timeout_seconds,
        ),
        verify_proxy_endpoint(
            url=f"{normalized_base_url}/ready",
            expected_status_value="ready",
            timeout_seconds=timeout_seconds,
        ),
    ]
    verify_direct_api_is_unreachable(
        url=f"{normalized_direct_url}/health",
        timeout_seconds=timeout_seconds,
    )
    return results


def _parse_args() -> argparse.Namespace:
    """
    解析代理部署验收脚本命令行参数。

    参数含义：
        无。

    返回值含义：
        argparse.Namespace:
            包含代理地址、API 直连地址和超时设置的参数对象。
    """

    parser = argparse.ArgumentParser(
        description="验证 V1.22 Nginx 代理部署是否可用且不可绕过。"
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="用户访问的 Nginx 基础地址。",
    )
    parser.add_argument(
        "--direct-api-url",
        default="http://127.0.0.1:8000",
        help="预期无法从宿主机访问的 FastAPI 基础地址。",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="每次网络探测的超时秒数。",
    )
    return parser.parse_args()


def main() -> None:
    """
    运行命令行代理部署验收并输出中文结果。

    参数含义：
        无。

    返回值含义：
        None:
            全部检查通过时正常退出，失败时由异常令进程返回非零状态码。
    """

    args = _parse_args()
    results = verify_proxy_deployment(
        base_url=args.base_url,
        direct_api_url=args.direct_api_url,
        timeout_seconds=args.timeout_seconds,
    )
    for result in results:
        print(
            f"通过: {result.url} -> HTTP {result.status_code}, "
            f"Server={result.server_header}"
        )
    print("通过: FastAPI 宿主机直连端口不可访问。")
    print("V1.22 Nginx 代理部署基础验收通过。")


if __name__ == "__main__":
    main()
