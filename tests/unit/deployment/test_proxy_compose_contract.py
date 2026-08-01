"""
Nginx 反向代理部署契约测试。

功能：
    验证代理部署会隐藏 API 宿主机端口、只公开 Nginx，并通过固定 Docker
    网络安全传递客户端地址和流式响应。
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROXY_COMPOSE_PATH = PROJECT_ROOT / "compose.proxy.yaml"
NGINX_CONFIG_PATH = PROJECT_ROOT / "deployment" / "nginx" / "nginx.conf"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"
CI_WORKFLOW_PATH = (
    PROJECT_ROOT / ".github" / "workflows" / "v121-ci.yml"
)


def _read_text(path: Path) -> str:
    """
    使用 UTF-8 读取代理部署契约文件。

    参数含义：
        path:
            需要读取的仓库文件绝对路径。

    返回值含义：
        str:
            文件的完整文本内容。
    """

    return path.read_text(encoding="utf-8")


def test_proxy_compose_should_expose_only_nginx() -> None:
    """
    验证代理模式清除 API 端口并只公开 Nginx HTTP 入口。

    参数含义：
        无。

    返回值含义：
        None。
    """

    compose = _read_text(PROXY_COMPOSE_PATH)

    assert "ports: !reset []" in compose
    assert 'NGINX_HTTP_PORT:-8080}:8080' in compose
    assert 'expose:\n      - "8000"' in compose
    assert "condition: service_healthy" in compose


def test_proxy_compose_should_use_an_exact_trusted_proxy_address() -> None:
    """
    验证 API 只信任固定 Nginx 地址而不是整个互联网来源。

    参数含义：
        无。

    返回值含义：
        None。
    """

    compose = _read_text(PROXY_COMPOSE_PATH)

    assert "172.30.0.2/32" in compose
    assert "ipv4_address: 172.30.0.2" in compose
    assert "ipv4_address: 172.30.0.3" in compose
    assert "subnet: 172.30.0.0/24" in compose
    assert "0.0.0.0/0" not in compose


def test_nginx_should_forward_client_context_and_preserve_sse() -> None:
    """
    验证 Nginx 追加代理链信息并关闭 SSE 响应缓冲。

    参数含义：
        无。

    返回值含义：
        None。
    """

    nginx = _read_text(NGINX_CONFIG_PATH)

    assert "server api:8000" in nginx
    assert "X-Forwarded-For $proxy_add_x_forwarded_for" in nginx
    assert "X-Forwarded-Proto $scheme" in nginx
    assert "location /v1/chat/stream" in nginx
    assert "proxy_buffering off" in nginx
    assert "X-Accel-Buffering no" in nginx
    assert "client_max_body_size 64k" in nginx


def test_proxy_deployment_should_be_documented_and_ci_validated() -> None:
    """
    验证环境模板声明代理端口且 CI 校验三层 Compose 合并结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    env_example = _read_text(ENV_EXAMPLE_PATH)
    workflow = _read_text(CI_WORKFLOW_PATH)

    assert "NGINX_HTTP_PORT=8080" in env_example
    assert "-f compose.proxy.yaml" in workflow
    assert "Validate Nginx configuration" in workflow
    assert '--add-host "api:127.0.0.1"' in workflow
    assert "nginx:1.30.4-alpine" in workflow
    assert "nginx -t" in workflow
