"""
Runtime Settings 测试。

功能：
    验证多 Agent Scheduler 配置默认值和正数边界，不启动真实运行时。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.settings.runtime import RuntimeSettings


def test_runtime_settings_should_provide_multi_agent_defaults() -> None:
    """
    检查多 Agent Scheduler 是否具有可直接使用的默认配置。

    参数：
        无。

    返回值：
        None。
    """

    runtime_settings = RuntimeSettings()

    assert runtime_settings.multi_agent_maximum_parallel_steps == 4
    assert runtime_settings.multi_agent_step_timeout_seconds == 120.0
    assert runtime_settings.multi_agent_maximum_step_attempts == 2


@pytest.mark.parametrize(
    "field_name, invalid_value",
    [
        ("multi_agent_maximum_parallel_steps", 0),
        ("multi_agent_step_timeout_seconds", 0),
        ("multi_agent_maximum_step_attempts", 0),
    ],
)
def test_runtime_settings_should_reject_non_positive_multi_agent_values(
    field_name: str,
    invalid_value: int,
) -> None:
    """
    检查多 Agent 数值配置为零时是否被 Pydantic 拒绝。

    参数：
        field_name:
            当前需要验证的 Runtime Settings 字段名称。
        invalid_value:
            pytest 传入的非法配置值。

    返回值：
        None。
    """

    with pytest.raises(ValidationError):
        RuntimeSettings(**{field_name: invalid_value})
