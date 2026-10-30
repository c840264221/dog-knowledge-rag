"""根据宿主工具契约补充运行参数；与任何具体技能或工具名称无关。"""

from uuid import uuid4

from src.graph.tools.schemas.tool_call_schema import ToolCall, ToolParseResult


def bind_runtime_arguments(proposal: ToolParseResult, catalog: list) -> ToolParseResult:
    """为新提议绑定程序生成参数。

    参数：proposal为尚未确认的新工具提议；catalog为程序构建的真实工具目录。
    返回值：独立的ToolParseResult；顶层字段标记x-runtime-generated=uuid时忽略模型值并生成动作键。
    仅在解析新调用后使用；已有tool_calls的恢复分支不会进入这里，保证沿用原键。
    """
    schemas = {item["name"]: item.get("input_schema", {}) for item in catalog}
    calls = []
    for call in proposal.tool_calls:
        arguments = dict(call.args)
        for field, spec in schemas.get(call.name, {}).get("properties", {}).items():
            generator = spec.get("x-runtime-generated")
            if generator is None:
                continue
            if generator != "uuid":
                raise ValueError(f"不支持的运行参数生成方式：{generator}")
            arguments[field] = "tool-action:" + uuid4().hex
        calls.append(ToolCall(name=call.name, args=arguments))
    return ToolParseResult(need_tool=proposal.need_tool, tool_calls=calls, response=proposal.response)
