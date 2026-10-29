"""工具补参的通用字段操作；字段路径例如 items.0.amount，不包含业务分类规则。"""

import json
from copy import deepcopy

from jsonschema import Draft202012Validator


def field_schema_at_path(schema: dict, path: str) -> dict:
    """查找字段结构；schema为工具输入契约，path为点分字段路径；返回含本地引用定义的字段契约。"""
    current = schema
    for part in path.split("."):
        while "$ref" in current:
            reference = current["$ref"]
            if not reference.startswith("#/$defs/"):
                raise ValueError("补参只支持工具契约中的本地定义引用。")
            current = schema["$defs"][reference[len("#/$defs/"):]]
        current = current.get("items", {}) if part.isdecimal() else current.get("properties", {}).get(part, {})
    while "$ref" in current:
        reference = current["$ref"]
        if not reference.startswith("#/$defs/"):
            raise ValueError("补参只支持工具契约中的本地定义引用。")
        current = schema["$defs"][reference[len("#/$defs/"):]]
    result = deepcopy(current)
    if "$defs" in schema:
        result["$defs"] = deepcopy(schema["$defs"])
    return result


def replace_argument_field(arguments: dict, path: str, value: object) -> dict:
    """按路径补值；arguments为已保存参数，path为缺参路径，value为补充值；返回独立副本，不修改原快照。"""
    result = deepcopy(arguments)
    target = result
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    if isinstance(target, list):
        target[int(parts[-1])] = value
    else:
        target[parts[-1]] = value
    return result


def parse_clarification_value(value: object, schema: dict) -> tuple[bool, object]:
    """校验补充值；value为用户提供的值，schema为字段契约；返回是否合法及转换后的值，不猜测自然语言单位。"""
    if isinstance(value, str) and schema.get("type") in ("number", "integer", "boolean", "array", "object"):
        try:
            value = json.loads(value)
        except ValueError:
            return False, value
    return Draft202012Validator(schema).is_valid(value), value
