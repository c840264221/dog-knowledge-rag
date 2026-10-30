"""工具执行绑定的通用检查；不负责选择工具、补参或授予执行权限。"""

from collections.abc import Mapping


def bound_tool_name(contract: Mapping | None) -> str | None:
    """读取执行绑定；contract为宿主写入的步骤契约；返回绑定名称，能力模式返回None，非法契约抛异常。"""
    if not contract:
        return None
    if not isinstance(contract, Mapping):
        raise ValueError("工具执行契约必须是字典对象。")
    mode = contract.get("execution_mode", "capability")
    name = contract.get("tool_name")
    if mode == "capability" and name is None:
        return None
    if mode != "tool" or not isinstance(name, str) or not name.strip():
        raise ValueError("工具执行契约不合法。")
    return name.strip()


def validate_bound_tool_calls(calls: list, contract: Mapping | None) -> None:
    """检查提议遵守绑定；calls为调用列表、contract为步骤执行契约；通过无返回值，空调用或换工具抛异常。"""
    name = bound_tool_name(contract)
    if name is None:
        return
    if not calls:
        raise ValueError(f"计划绑定工具 {name}，但没有生成对应调用；步骤不能视为完成。")
    if len(calls) != 1 or any(call.get("name") != name for call in calls):
        raise ValueError(f"实际工具调用与计划绑定 {name} 不一致；需要修订计划，不能静默更换工具。")
