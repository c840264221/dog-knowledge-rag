"""验证通用请求时间、Planner自动注入与工具解析共享基准，不调用外部服务。"""

import asyncio
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from src.runtime.context import RuntimeContext, RuntimeTimeContext, get_runtime_time_context, runtime_ctx


def test_request_time_is_isolated_and_readers_cannot_modify_snapshot():
    """验证时区与快照隔离；无参数、无返回值，不同请求使用各自时区，读取副本不能改写环境。"""
    now = datetime(2026, 10, 24, 23, 30, tzinfo=timezone.utc)
    hong_kong = RuntimeContext(time_context=RuntimeTimeContext.capture(now=now))
    new_york = RuntimeContext(time_context=RuntimeTimeContext.capture(now=now, timezone_name="America/New_York"))
    previous = runtime_ctx.get()
    try:
        runtime_ctx.set(hong_kong)
        value = get_runtime_time_context()
        assert value["reference_date"] == "2026-10-25"
        value["reference_date"] = "1900-01-01"
        assert get_runtime_time_context()["reference_date"] == "2026-10-25"
        runtime_ctx.set(new_york)
        assert get_runtime_time_context()["reference_date"] == "2026-10-24"
        with pytest.raises(FrozenInstanceError):
            hong_kong.time_context.timezone_name = "UTC"
        with pytest.raises(ValueError, match="带时区"):
            RuntimeTimeContext.capture(now=datetime(2026, 10, 25))
    finally:
        runtime_ctx.set(previous)


def test_generic_planner_injects_runtime_environment_without_case_date_helper():
    """验证通用规划提示；无参数、无返回值，不同业务都自动携带时间，业务上下文不覆盖环境段。"""
    from src.agents.collaboration.planner.prompts import build_planner_prompt

    context = RuntimeContext(time_context=RuntimeTimeContext.capture(
        now=datetime(2026, 10, 24, 23, 30, tzinfo=timezone.utc)))
    previous = runtime_ctx.get()
    runtime_ctx.set(context)
    try:
        for objective in ["整理明天的活动安排", "汇总昨天的消费"]:
            prompt = build_planner_prompt(objective=objective, plan_id="test-plan",
                available_agents={"tool_agent": "使用已注册工具"}, context={"reference_date": "1900-01-01"})
            environment = prompt.split("宿主运行环境（由程序自动提供，日期仅用于解释相对时间，不是默认业务日期）：\n", 1)[1]
            environment = json.loads(environment.split("\n\n可用 Agent：", 1)[0])
            assert environment == get_runtime_time_context()
            assert environment["reference_date"] == "2026-10-25"
            assert "前置结果尚未产生时声明依赖" in prompt
    finally:
        runtime_ctx.set(previous)


def test_tool_parser_uses_same_request_time_as_planner():
    """验证真实工具解析器取用统一时间；无参数、无返回值，模型替身仅记录提示，不调用网络。"""
    from src.agents.tool_agent.nodes.tool_parse_node import build_llm_tool_parser

    prompts = []

    async def fake_call(**kwargs):
        """记录模型输入；kwargs为真实解析器调用参数；返回合法的固定工具提议JSON。"""
        prompts.append(kwargs["prompt"].to_string())
        return '{"need_tool":false,"tool_calls":[],"response":"仅检查上下文"}'

    context = RuntimeContext(time_context=RuntimeTimeContext.capture(
        now=datetime(2026, 10, 24, 23, 30, tzinfo=timezone.utc), timezone_name="America/New_York"))
    previous = runtime_ctx.get()
    runtime_ctx.set(context)
    try:
        parser = build_llm_tool_parser(llm_provider=SimpleNamespace(backup_llm=object(), safe_ainvoke=fake_call))
        asyncio.run(parser({"question": "今天有哪些可用工具", "state": {"tool_agent_tool_catalog": []}}))
        assert "2026-10-24" in prompts[0] and "America/New_York" in prompts[0]
        metadata = context.metadata["tool_planning_skill"]
        for key, value in get_runtime_time_context().items():
            assert metadata[key] == value
    finally:
        runtime_ctx.set(previous)
