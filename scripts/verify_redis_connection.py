"""验证 Dog Agent Redis Provider 与本机 Redis 服务连通。"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from src.runtime.container.providers.redis_provider import RedisProvider
from src.settings.redis import RedisSettings


SMOKE_KEY = "dog-agent:smoke:redis-provider"


async def verify_redis_connection(redis_url: str) -> dict[str, Any]:
    """
    依次执行 PING、SET、GET、TTL 和 DELETE 连通性验证。

    参数含义：
        redis_url:
            需要验证的 Redis 连接地址，例如 ``redis://localhost:6379/0``。

    返回值含义：
        dict[str, Any]:
            包含连接状态、读取值、剩余 TTL 和删除数量的结构化结果。
    """

    provider = RedisProvider(
        RedisSettings(
            enabled=True,
            url=redis_url,
            _env_file=None,
        )
    )
    await provider.startup()
    client = provider.client
    try:
        await client.set(SMOKE_KEY, "connected", ex=30)
        stored_value = await client.get(SMOKE_KEY)
        remaining_ttl_seconds = await client.ttl(SMOKE_KEY)
        deleted_count = await client.delete(SMOKE_KEY)
        return {
            "connected": True,
            "key": SMOKE_KEY,
            "stored_value": stored_value,
            "remaining_ttl_seconds": remaining_ttl_seconds,
            "deleted_count": deleted_count,
        }
    finally:
        # 前面任一读取失败时仍尝试清理测试 Key，再关闭连接池。
        await client.delete(SMOKE_KEY)
        await provider.shutdown()


def parse_args() -> argparse.Namespace:
    """
    解析 Redis 连通性脚本命令行参数。

    返回值含义：
        argparse.Namespace:
            包含 ``redis_url`` 的命令行参数对象。
    """

    parser = argparse.ArgumentParser(
        description="验证 Dog Agent Redis 异步连接与基础命令。"
    )
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379/0",
        help="Redis 连接地址，默认连接本机 Docker Redis。",
    )
    return parser.parse_args()


def main() -> int:
    """
    运行 Redis 连通性验证并输出 JSON 结果。

    返回值含义：
        int:
            验证成功时返回 0；连接或命令失败时由异常产生非零退出状态。
    """

    args = parse_args()
    result = asyncio.run(verify_redis_connection(args.redis_url))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
