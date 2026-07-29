from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from scripts import api_run


def test_api_run_should_forward_validated_settings_to_uvicorn(
    monkeypatch,
) -> None:
    """
    验证 API 启动脚本会把统一配置传给 Uvicorn。

    参数含义：
        monkeypatch:
            pytest 提供的临时替换工具，用于阻止测试真正启动 HTTP 服务。

    返回值含义：
        None。
    """

    captured: dict[str, Any] = {}

    def fake_uvicorn_run(app_path: str, **kwargs: Any) -> None:
        """
        记录 Uvicorn 启动参数而不创建真实服务器。

        参数含义：
            app_path:
                Uvicorn 需要导入的 ASGI 应用路径。
            **kwargs:
                启动脚本传入的监听和日志参数。

        返回值含义：
            None。
        """

        captured["app_path"] = app_path
        captured.update(kwargs)

    monkeypatch.setattr(
        api_run.settings,
        "api",
        SimpleNamespace(
            host="0.0.0.0",
            port=9000,
            workers=1,
            reload=False,
            log_level="warning",
            access_log=False,
        ),
    )
    monkeypatch.setattr(api_run.uvicorn, "run", fake_uvicorn_run)

    api_run.main()

    assert captured == {
        "app_path": "src.api.app:app",
        "host": "0.0.0.0",
        "port": 9000,
        "workers": 1,
        "reload": False,
        "log_level": "warning",
        "access_log": False,
    }
