"""
Reranker Provider 初始化测试。

功能：
    验证容器启动会真正预加载重排模型，并发首次访问也只初始化一个实例。
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.runtime.container.providers.reranker_provider import (
    RerankerProvider,
)


def test_reranker_startup_should_preload_model(monkeypatch: Any) -> None:
    """
    检查 startup 是否通过公开属性真正创建 Reranker。

    参数含义：
        monkeypatch:
            pytest 提供的临时替换工具，用于避免加载真实模型。

    返回值含义：
        None。
    """

    created_models: list[object] = []

    def fake_cross_encoder(*args: Any, **kwargs: Any) -> object:
        """记录一次模型创建并返回固定对象。"""

        _ = args, kwargs
        model = object()
        created_models.append(model)
        return model

    monkeypatch.setattr(
        "src.runtime.container.providers.reranker_provider.CrossEncoder",
        fake_cross_encoder,
    )
    provider = RerankerProvider()

    asyncio.run(provider.startup())

    assert provider.reranker is created_models[0]
    assert len(created_models) == 1


def test_reranker_should_initialize_once_under_concurrency(
    monkeypatch: Any,
) -> None:
    """
    检查多个 Worker 同时首次访问时是否共享一个 Reranker 实例。

    参数含义：
        monkeypatch:
            pytest 提供的临时替换工具，用于模拟较慢的模型初始化。

    返回值含义：
        None。
    """

    created_models: list[object] = []

    def fake_cross_encoder(*args: Any, **kwargs: Any) -> object:
        """短暂等待以扩大并发竞争窗口并记录模型创建。"""

        _ = args, kwargs
        time.sleep(0.02)
        model = object()
        created_models.append(model)
        return model

    monkeypatch.setattr(
        "src.runtime.container.providers.reranker_provider.CrossEncoder",
        fake_cross_encoder,
    )
    provider = RerankerProvider()

    with ThreadPoolExecutor(max_workers=4) as executor:
        models = list(executor.map(lambda _: provider.reranker, range(4)))

    assert len(created_models) == 1
    assert all(model is created_models[0] for model in models)
