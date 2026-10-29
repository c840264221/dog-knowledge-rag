import copy
import hashlib
import json
from typing import Any


def build_agent_action_fingerprint(
    *,
    action_name: str,
    action_arguments: dict[str, Any],
) -> str:
    """
    根据动作名称和动作参数生成稳定的 Agent 动作指纹。

    功能：
        检查 action_name 必须是字符串，
        清理首尾空格并转换为小写后不能为空。

        检查 action_arguments 必须是字典。

        使用规范化后的动作名称和动作参数
        构造动作指纹载荷。

        通过 json.dumps 将载荷转换成规范 JSON：
        - sort_keys=True，使字典键顺序不影响结果；
        - separators=(",", ":")，去除无意义空格；
        - ensure_ascii=False，保留可读中文；
        - allow_nan=False，拒绝 NaN 和正负无穷。

        将规范 JSON 编码为 UTF-8 字节，
        使用 SHA-256 计算 64 位十六进制动作指纹。

        本函数只根据动作的业务身份生成指纹，
        调用方不应把 trace_id、时间戳、请求 ID、
        重试编号等每次都会变化的运行噪声
        放入 action_arguments。

        字典键顺序不会影响动作指纹，
        但列表顺序会被保留，因为列表顺序
        通常可能具有业务含义。

        本函数不修改传入的 action_arguments。

        当前指纹用于重复动作检测，
        不是密码学身份认证或数字签名。

    参数：
        action_name:
            动作或工具名称。

            例如：reason、call_tool、weather。
            清理并转换为小写后不能为空。

        action_arguments:
            决定动作业务身份的参数字典。

            其中的键和值必须能够被 JSON 序列化，
            并且不能包含 NaN 或正负无穷。

    返回值：
        str:
            SHA-256 生成的 64 位十六进制动作指纹。

            对于相同的规范化动作名称和参数内容，
            即使字典键的插入顺序不同，
            也会返回相同指纹。

    异常：
        TypeError:
            当 action_name 不是字符串，
            或 action_arguments 不是字典时抛出。

        ValueError:
            当规范化动作名称为空，
            或动作参数无法转换为规范 JSON 时抛出。
    """
    if not isinstance(action_name, str):
        raise TypeError("action_name 必须是字符串")
    normalized_action_name = action_name.strip().lower()
    if not normalized_action_name:
        raise ValueError("action_name 清理后不能为空")

    if not isinstance(action_arguments, dict):
        raise TypeError("action_arguments 必须是字典")

    fingerprint_payload = {
        "action_name": normalized_action_name,
        "action_arguments": action_arguments,
    }
    try:
        canonical_action_json = json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "action_arguments 必须能够转换为不含特殊浮点值的规范 JSON"
        ) from exc

    action_fingerprint = hashlib.sha256(
        canonical_action_json.encode("utf-8")
    ).hexdigest()
    return action_fingerprint


def build_agent_progress_fingerprint(
    *,
    progress_snapshot: dict[str, Any],
) -> str:
    """
    根据稳定的业务进展快照生成 Agent 进展指纹。

    功能：
        检查 progress_snapshot 必须是非空字典。

        使用 json.dumps 将业务进展快照转换成规范 JSON：

        - sort_keys=True，使字典键顺序不影响结果；
        - separators=(",", ":")，去除无意义空格；
        - ensure_ascii=False，保留可读中文；
        - allow_nan=False，拒绝 NaN 和正负无穷。

        将规范 JSON 编码为 UTF-8 字节，
        使用 SHA-256 计算 64 位十六进制进展指纹。

        进展指纹回答的是“任务推进到了哪里”，
        不回答“本轮准备执行什么动作”。

        progress_snapshot 应当只包含能够代表
        真实业务进展的稳定字段，例如：

        - 当前任务阶段；
        - 已完成的步骤；
        - 已获得的关键事实；
        - 尚未完成的目标；
        - 是否已经生成最终答案。

        progress_snapshot 不应包含每轮都会变化、
        但不代表真实任务推进的运行噪声，例如：

        - 时间戳；
        - trace_id；
        - request_id；
        - 当前循环次数；
        - 随机值；
        - 日志序号。

        如果把运行噪声放入快照，
        每轮都会生成不同的进展指纹，
        即使任务完全没有推进，系统也会误判为有进展。

        本函数不修改传入的 progress_snapshot。

    参数：
        progress_snapshot:
            当前任务的稳定业务进展快照。

            必须是非空字典，
            其中的键和值必须能够被 JSON 序列化，
            并且不能包含 NaN 或正负无穷。

    返回值：
        str:
            SHA-256 生成的 64 位十六进制进展指纹。

            两份业务含义相同、仅字典键插入顺序不同的
            进展快照会得到相同指纹。

    异常：
        TypeError:
            当 progress_snapshot 不是字典时抛出。

        ValueError:
            当 progress_snapshot 为空，
            或无法转换为不含特殊浮点值的规范 JSON时抛出。
    """
    if not isinstance(progress_snapshot, dict):
        raise TypeError("progress_snapshot 必须是字典")
    if not progress_snapshot:
        raise ValueError("progress_snapshot 不能为空")

    try:
        canonical_progress_json = json.dumps(
            progress_snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "progress_snapshot 必须能够转换为"
            "不含特殊浮点值的规范 JSON"
        ) from exc

    progress_fingerprint = hashlib.sha256(
        canonical_progress_json.encode("utf-8")
    ).hexdigest()
    return progress_fingerprint


def build_no_progress_detection_state(
    *,
    max_consecutive_no_progress: int,
) -> dict[str, Any]:
    """
    创建 Agent 无进展检测使用的初始控制状态。

    功能：
        检查 max_consecutive_no_progress
        必须是真正的整数，不能使用布尔值冒充整数，
        并且必须大于 0。

        创建独立的无进展检测控制状态，保存：

        - 上一次业务进展指纹；
        - 当前连续无进展次数；
        - 允许的连续无进展次数阈值；
        - 已经观察过的进展快照数量；
        - 生命周期状态；
        - 终止原因。

        第一次记录业务进展指纹时，
        系统只能建立 Baseline（比较基线），
        还没有前一个指纹可供比较，
        因此不计为一次无进展。

        从第二次进展观察开始：

        - 新指纹等于上一次指纹时，
          consecutive_no_progress_count 加 1；
        - 新指纹不同于上一次指纹时，
          说明业务发生进展，计数重置为 0。

        无进展检测发生在动作执行并产生观察结果之后。
        因此达到停止阈值时，导致本次无进展的动作
        已经执行完成；停止决定用于阻止下一轮继续运行。

        当前函数只创建初始状态，
        不接收业务进展快照、不生成进展指纹，
        不比较进展，也不执行终止判断。

    参数：
        max_consecutive_no_progress:
            连续无进展并触发停止的阈值。

            必须是真正的整数，并且大于 0。

            例如设置为 3，表示建立第一次基线之后，
            如果连续三个执行后快照都没有变化，
            则停止后续 Agent 循环。

    返回值：
        dict[str, Any]:
            新创建的无进展检测控制状态，格式如下：

            {
                "status": "running",
                "last_progress_fingerprint": None,
                "consecutive_no_progress_count": 0,
                "max_consecutive_no_progress": 3,
                "progress_observation_count": 0,
                "termination_reason": None,
            }

            status:
                当前无进展检测生命周期状态。
                初始值固定为 running。

            last_progress_fingerprint:
                上一次稳定业务进展快照的 SHA-256 指纹。
                尚未建立基线时为 None。

            consecutive_no_progress_count:
                当前连续无进展次数。
                初始值为 0。

            max_consecutive_no_progress:
                连续无进展的停止阈值。

            progress_observation_count:
                已经记录过多少次业务进展快照。
                初始值为 0。

            termination_reason:
                无进展检测导致的终止原因。
                初始运行状态下为 None。

    异常：
        TypeError:
            当 max_consecutive_no_progress
            不是真正的整数时抛出。

        ValueError:
            当 max_consecutive_no_progress
            小于或等于 0 时抛出。
    """
    if type(max_consecutive_no_progress) is not int:
        raise TypeError(
            "max_consecutive_no_progress 必须是真正的整数"
        )
    if max_consecutive_no_progress <= 0:
        raise ValueError(
            "max_consecutive_no_progress 必须大于 0"
        )

    no_progress_state = {
        "status": "running",
        "last_progress_fingerprint": None,
        "consecutive_no_progress_count": 0,
        "max_consecutive_no_progress": (
            max_consecutive_no_progress
        ),
        "progress_observation_count": 0,
        "termination_reason": None,
    }
    return no_progress_state


def record_progress_observation(
    *,
    no_progress_state: dict[str, Any],
    progress_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """
    记录一次动作执行后的业务进展观察。

    功能：
        检查 no_progress_state 的必要字段、
        生命周期状态、字段类型和计数约束。

        调用 build_agent_progress_fingerprint，
        根据 progress_snapshot 生成稳定进展指纹。

        如果尚未记录过进展指纹，
        将本次指纹保存为第一次比较基线，
        consecutive_no_progress_count 保持为 0。

        如果本次指纹等于上一次指纹，
        说明本轮动作执行后业务状态没有推进，
        consecutive_no_progress_count 加 1。

        如果本次指纹不同于上一次指纹，
        说明本轮动作执行后产生了真实业务进展，
        consecutive_no_progress_count 重置为 0。

        无论本次是否产生进展，
        progress_observation_count 都增加 1，
        last_progress_fingerprint 都更新为本次指纹。

        本函数只记录进展观察事实，
        不判断是否达到无进展停止阈值，
        不修改 status 和 termination_reason，
        也不调用模型或执行工具。

        本函数使用深复制，
        不修改调用方传入的原状态。

    参数：
        no_progress_state:
            当前无进展检测控制状态。

            必须由 build_no_progress_detection_state
            创建或遵循相同结构，
            并且 status 必须为 running。

        progress_snapshot:
            本轮动作执行完成后的稳定业务进展快照。

            必须符合 build_agent_progress_fingerprint
            对非空字典和 JSON 序列化的要求。

    返回值：
        dict[str, Any]:
            记录本次进展观察后的状态深复制版本。

            原 no_progress_state 不会被修改。

    异常：
        TypeError:
            当 no_progress_state 不是字典、
            状态字段类型不正确，
            或 progress_snapshot 类型不正确时抛出。

        ValueError:
            当状态缺少字段、状态不是 running、
            状态字段违反约束，
            或 progress_snapshot 无法生成指纹时抛出。
    """
    if not isinstance(no_progress_state, dict):
        raise TypeError("no_progress_state 必须是字典")

    required_state_fields = {
        "status",
        "last_progress_fingerprint",
        "consecutive_no_progress_count",
        "max_consecutive_no_progress",
        "progress_observation_count",
        "termination_reason",
    }
    missing_state_fields = required_state_fields.difference(
        no_progress_state
    )
    if missing_state_fields:
        missing_fields_text = ", ".join(
            sorted(missing_state_fields)
        )
        raise ValueError(
            f"no_progress_state 缺少字段：{missing_fields_text}"
        )

    if no_progress_state["status"] != "running":
        raise ValueError("只有 running 状态可以记录进展观察")
    if no_progress_state["termination_reason"] is not None:
        raise ValueError(
            "running 状态的 termination_reason 必须是 None"
        )

    last_progress_fingerprint = no_progress_state[
        "last_progress_fingerprint"
    ]
    if (
        last_progress_fingerprint is not None
        and not isinstance(last_progress_fingerprint, str)
    ):
        raise TypeError(
            "last_progress_fingerprint 必须是字符串或 None"
        )
    if (
        isinstance(last_progress_fingerprint, str)
        and not last_progress_fingerprint.strip()
    ):
        raise ValueError(
            "last_progress_fingerprint 不能为空字符串"
        )

    integer_field_constraints = {
        "consecutive_no_progress_count": 0,
        "max_consecutive_no_progress": 1,
        "progress_observation_count": 0,
    }
    for field_name, minimum_value in (
        integer_field_constraints.items()
    ):
        field_value = no_progress_state[field_name]
        if type(field_value) is not int:
            raise TypeError(
                f"{field_name} 必须是真正的整数"
            )
        if field_value < minimum_value:
            raise ValueError(
                f"{field_name} 不能小于 {minimum_value}"
            )

    progress_observation_count = no_progress_state[
        "progress_observation_count"
    ]
    consecutive_no_progress_count = no_progress_state[
        "consecutive_no_progress_count"
    ]

    if progress_observation_count == 0:
        if last_progress_fingerprint is not None:
            raise ValueError(
                "尚无进展观察时 last_progress_fingerprint 必须是 None"
            )
        if consecutive_no_progress_count != 0:
            raise ValueError(
                "尚无进展观察时无进展次数必须为 0"
            )
    else:
        if last_progress_fingerprint is None:
            raise ValueError(
                "已有进展观察时必须包含 last_progress_fingerprint"
            )
        if (
            consecutive_no_progress_count
            > progress_observation_count - 1
        ):
            raise ValueError(
                "连续无进展次数不能超过可比较的观察次数"
            )

    progress_fingerprint = build_agent_progress_fingerprint(
        progress_snapshot=progress_snapshot,
    )
    updated_state = copy.deepcopy(no_progress_state)

    if last_progress_fingerprint is None:
        updated_no_progress_count = 0
    elif progress_fingerprint == last_progress_fingerprint:
        updated_no_progress_count = (
            consecutive_no_progress_count + 1
        )
    else:
        updated_no_progress_count = 0

    updated_state["last_progress_fingerprint"] = (
        progress_fingerprint
    )
    updated_state["consecutive_no_progress_count"] = (
        updated_no_progress_count
    )
    updated_state["progress_observation_count"] = (
        progress_observation_count + 1
    )
    return updated_state


def evaluate_no_progress_detection_state(
    *,
    no_progress_state: dict[str, Any],
) -> dict[str, Any]:
    """
    评估当前无进展检测状态是否要求停止 Agent 循环。

    功能：
        检查 no_progress_state 的必要字段、
        生命周期状态、字段类型和计数约束。

        判断 consecutive_no_progress_count 是否
        大于或等于 max_consecutive_no_progress。

        如果达到阈值，返回 should_stop=True、
        control_action="stop"，终止原因为
        consecutive_no_progress。

        如果没有达到阈值，返回 should_stop=False、
        control_action="continue"，终止原因为 None。

        如果传入状态已经是 stopped，
        保留并返回状态中已有的终止原因，
        不重新选择新的终止原因。

        本函数只读取并分析状态，
        不修改 no_progress_state，
        不增加观察次数，
        也不调用模型或执行工具。

    参数：
        no_progress_state:
            已经记录进展观察的无进展检测状态。

            可以是 running 状态，
            也可以是已经 stopped 的终态。

    返回值：
        dict[str, Any]:
            无进展检测判定结果，格式如下：

            {
                "should_stop": False,
                "control_action": "continue",
                "termination_reason": None,
                "no_progress_threshold_reached": False,
            }

            should_stop:
                是否应该停止后续 Agent 循环。

            control_action:
                Harness 下一步控制动作。
                取值为 continue 或 stop。

            termination_reason:
                稳定终止原因。
                循环可以继续时为 None。

            no_progress_threshold_reached:
                当前连续无进展次数是否已经达到阈值。

    异常：
        TypeError:
            当 no_progress_state 不是字典，
            或所需状态字段类型不正确时抛出。

        ValueError:
            当状态缺少字段、状态值不受支持、
            数值字段违反约束，
            进展指纹与观察次数不一致，
            或生命周期与终止原因不一致时抛出。
    """
    if not isinstance(no_progress_state, dict):
        raise TypeError("no_progress_state 必须是字典")

    required_state_fields = {
        "status",
        "last_progress_fingerprint",
        "consecutive_no_progress_count",
        "max_consecutive_no_progress",
        "progress_observation_count",
        "termination_reason",
    }
    missing_state_fields = required_state_fields.difference(
        no_progress_state
    )
    if missing_state_fields:
        missing_fields_text = ", ".join(
            sorted(missing_state_fields)
        )
        raise ValueError(
            f"no_progress_state 缺少字段：{missing_fields_text}"
        )

    status = no_progress_state["status"]
    if not isinstance(status, str):
        raise TypeError("status 必须是字符串")
    if status not in {"running", "stopped"}:
        raise ValueError("status 必须是 running 或 stopped")

    termination_reason = no_progress_state["termination_reason"]
    if termination_reason is not None and not isinstance(
        termination_reason,
        str,
    ):
        raise TypeError("termination_reason 必须是字符串或 None")
    if status == "running" and termination_reason is not None:
        raise ValueError(
            "running 状态的 termination_reason 必须是 None"
        )
    if (
        status == "stopped"
        and termination_reason != "consecutive_no_progress"
    ):
        raise ValueError(
            "stopped 状态的 termination_reason 必须是 "
            "consecutive_no_progress"
        )

    last_progress_fingerprint = no_progress_state[
        "last_progress_fingerprint"
    ]
    if (
        last_progress_fingerprint is not None
        and not isinstance(last_progress_fingerprint, str)
    ):
        raise TypeError(
            "last_progress_fingerprint 必须是字符串或 None"
        )
    if (
        isinstance(last_progress_fingerprint, str)
        and not last_progress_fingerprint.strip()
    ):
        raise ValueError(
            "last_progress_fingerprint 不能为空字符串"
        )

    integer_field_constraints = {
        "consecutive_no_progress_count": 0,
        "max_consecutive_no_progress": 1,
        "progress_observation_count": 0,
    }
    for field_name, minimum_value in (
        integer_field_constraints.items()
    ):
        field_value = no_progress_state[field_name]
        if type(field_value) is not int:
            raise TypeError(
                f"{field_name} 必须是真正的整数"
            )
        if field_value < minimum_value:
            raise ValueError(
                f"{field_name} 不能小于 {minimum_value}"
            )

    progress_observation_count = no_progress_state[
        "progress_observation_count"
    ]
    consecutive_no_progress_count = no_progress_state[
        "consecutive_no_progress_count"
    ]

    if progress_observation_count == 0:
        if last_progress_fingerprint is not None:
            raise ValueError(
                "尚无进展观察时 last_progress_fingerprint 必须是 None"
            )
        if consecutive_no_progress_count != 0:
            raise ValueError(
                "尚无进展观察时无进展次数必须为 0"
            )
    else:
        if last_progress_fingerprint is None:
            raise ValueError(
                "已有进展观察时必须包含 last_progress_fingerprint"
            )
        if (
            consecutive_no_progress_count
            > progress_observation_count - 1
        ):
            raise ValueError(
                "连续无进展次数不能超过可比较的观察次数"
            )

    no_progress_threshold_reached = (
        consecutive_no_progress_count
        >= no_progress_state["max_consecutive_no_progress"]
    )
    if status == "stopped" and not no_progress_threshold_reached:
        raise ValueError(
            "stopped 状态的连续无进展次数必须达到停止阈值"
        )

    common_result = {
        "no_progress_threshold_reached": (
            no_progress_threshold_reached
        ),
    }

    if status == "stopped":
        return {
            **common_result,
            "should_stop": True,
            "control_action": "stop",
            "termination_reason": termination_reason,
        }

    if no_progress_threshold_reached:
        return {
            **common_result,
            "should_stop": True,
            "control_action": "stop",
            "termination_reason": "consecutive_no_progress",
        }

    return {
        **common_result,
        "should_stop": False,
        "control_action": "continue",
        "termination_reason": None,
    }


def apply_no_progress_detection_decision(
    *,
    no_progress_state: dict[str, Any],
    no_progress_decision: dict[str, Any],
) -> dict[str, Any]:
    """
    将无进展检测判定结果应用到状态复制版本中。

    功能：
        检查 no_progress_decision 必须包含：

        - should_stop
        - control_action
        - termination_reason
        - no_progress_threshold_reached

        根据当前 no_progress_state 重新调用
        evaluate_no_progress_detection_state，
        得到当前状态下的权威判定结果。

        只比较判定结果所要求的标准字段，
        允许调用方携带额外的调试或追踪字段。

        如果传入判定与重新计算结果不一致，
        拒绝应用，避免伪造或过期判定
        被错误写入已经变化的状态。

        如果状态已经是 stopped，
        返回内容不变的深复制状态，
        保持终态幂等性。

        如果权威判定要求停止，
        将复制状态的 status 更新为 stopped，
        并写入 consecutive_no_progress 终止原因。

        如果权威判定允许继续，
        返回内容不变的深复制状态。

        本函数不修改传入的 no_progress_state，
        不增加观察次数，
        不生成新的进展指纹，
        也不调用模型或执行工具。

    参数：
        no_progress_state:
            当前无进展检测控制状态。

            可以是 running 状态，
            也可以是已经 stopped 的终态。

        no_progress_decision:
            evaluate_no_progress_detection_state
            针对当前状态生成的判定结果。

            可以包含额外诊断字段，
            但四个标准字段必须与重新计算结果一致。

    返回值：
        dict[str, Any]:
            应用判定后的无进展检测状态深复制版本。

            should_stop 为 True 时，返回状态中的：

            - status 为 stopped；
            - termination_reason 为 consecutive_no_progress。

            should_stop 为 False 时，
            返回与原状态内容相同的复制版本。

    异常：
        TypeError:
            当 no_progress_decision 不是字典，
            或判定字段类型不正确时抛出。

        ValueError:
            当判定缺少标准字段、字段取值不受支持、
            终止原因与 should_stop 不一致，
            判定结果与当前状态重新计算结果不一致，
            或 no_progress_state 本身不合法时抛出。
    """
    if not isinstance(no_progress_decision, dict):
        raise TypeError("no_progress_decision 必须是字典")

    required_decision_fields = {
        "should_stop",
        "control_action",
        "termination_reason",
        "no_progress_threshold_reached",
    }
    missing_decision_fields = required_decision_fields.difference(
        no_progress_decision
    )
    if missing_decision_fields:
        missing_fields_text = ", ".join(
            sorted(missing_decision_fields)
        )
        raise ValueError(
            f"no_progress_decision 缺少字段：{missing_fields_text}"
        )

    if type(no_progress_decision["should_stop"]) is not bool:
        raise TypeError("should_stop 必须是真正的布尔值")
    if not isinstance(no_progress_decision["control_action"], str):
        raise TypeError("control_action 必须是字符串")
    if no_progress_decision["control_action"] not in {
        "continue",
        "stop",
    }:
        raise ValueError("control_action 必须是 continue 或 stop")

    decision_termination_reason = no_progress_decision[
        "termination_reason"
    ]
    if (
        decision_termination_reason is not None
        and not isinstance(decision_termination_reason, str)
    ):
        raise TypeError("termination_reason 必须是字符串或 None")

    if type(
        no_progress_decision["no_progress_threshold_reached"]
    ) is not bool:
        raise TypeError(
            "no_progress_threshold_reached 必须是真正的布尔值"
        )

    if no_progress_decision["should_stop"]:
        if no_progress_decision["control_action"] != "stop":
            raise ValueError(
                "should_stop 为 True 时 control_action 必须是 stop"
            )
        if (
            decision_termination_reason
            != "consecutive_no_progress"
        ):
            raise ValueError(
                "should_stop 为 True 时 termination_reason 必须是 "
                "consecutive_no_progress"
            )
    else:
        if no_progress_decision["control_action"] != "continue":
            raise ValueError(
                "should_stop 为 False 时 control_action 必须是 continue"
            )
        if decision_termination_reason is not None:
            raise ValueError(
                "should_stop 为 False 时 termination_reason 必须是 None"
            )

    expected_decision = evaluate_no_progress_detection_state(
        no_progress_state=no_progress_state,
    )
    actual_decision = {
        field_name: no_progress_decision[field_name]
        for field_name in required_decision_fields
    }
    if actual_decision != expected_decision:
        raise ValueError(
            "no_progress_decision 与当前状态重新计算的结果不一致"
        )

    updated_state = copy.deepcopy(no_progress_state)
    if no_progress_state["status"] == "stopped":
        return updated_state

    if expected_decision["should_stop"]:
        updated_state["status"] = "stopped"
        updated_state["termination_reason"] = (
            expected_decision["termination_reason"]
        )

    return updated_state


def run_no_progress_detection_step(
    *,
    no_progress_state: dict[str, Any],
    progress_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """
    对一次动作执行后的业务快照执行完整无进展检测。

    功能：
        首先评估传入的 no_progress_state。

        如果状态已经停止，或者现有记录已经达到
        无进展停止阈值，则不处理新的进展快照，
        直接应用当前停止决定并返回。

        如果状态仍然允许记录观察，调用
        record_progress_observation：

        - 生成当前业务进展指纹；
        - 与上一次进展指纹比较；
        - 更新连续无进展次数；
        - 增加进展观察总次数。

        随后统一调用：

        - evaluate_no_progress_detection_state
        - apply_no_progress_detection_decision

        得到本轮无进展决定和更新后的控制状态。

        第一次进展观察只建立比较基线，
        progress_changed 返回 None。

        从第二次进展观察开始：

        - 新旧指纹不同时，progress_changed 为 True；
        - 新旧指纹相同时，progress_changed 为 False。

        本函数只编排动作执行后的无进展检测流程，
        不修改传入的 no_progress_state，
        不调用模型，也不执行真实工具。

    参数：
        no_progress_state:
            本轮进展观察开始前的无进展检测控制状态。

        progress_snapshot:
            本轮动作执行完成后的稳定业务进展快照。

            必须符合 build_agent_progress_fingerprint
            对非空字典和 JSON 序列化的要求。

    返回值：
        dict[str, Any]:
            本轮无进展检测步骤结果，格式如下：

            {
                "observation_processed": True,
                "progress_fingerprint": "...",
                "progress_changed": False,
                "no_progress_decision": {...},
                "updated_state": {...},
            }

            observation_processed:
                是否实际记录了本轮进展快照。
                进入函数前已经应当停止时为 False。

            progress_fingerprint:
                本轮进展快照生成的 SHA-256 指纹。
                未处理快照时为 None。

            progress_changed:
                True 表示相对上一次观察发生了进展；
                False 表示相对上一次观察没有进展；
                第一次建立基线或未处理快照时为 None。

            no_progress_decision:
                本轮最终无进展检测决定。

            updated_state:
                应用本轮决定后的无进展检测控制状态。

    异常：
        TypeError:
            当 no_progress_state、progress_snapshot
            或其中字段类型不正确时抛出。

        ValueError:
            当状态结构不合法、快照无法生成指纹、
            状态决定无法应用或状态约束冲突时抛出。
    """
    current_decision = evaluate_no_progress_detection_state(
        no_progress_state=no_progress_state,
    )
    if current_decision["should_stop"]:
        stopped_state = apply_no_progress_detection_decision(
            no_progress_state=no_progress_state,
            no_progress_decision=current_decision,
        )
        return {
            "observation_processed": False,
            "progress_fingerprint": None,
            "progress_changed": None,
            "no_progress_decision": current_decision,
            "updated_state": stopped_state,
        }

    previous_progress_fingerprint = no_progress_state[
        "last_progress_fingerprint"
    ]
    recorded_state = record_progress_observation(
        no_progress_state=no_progress_state,
        progress_snapshot=progress_snapshot,
    )
    current_progress_fingerprint = recorded_state[
        "last_progress_fingerprint"
    ]

    if previous_progress_fingerprint is None:
        progress_changed: bool | None = None
    else:
        progress_changed = (
            current_progress_fingerprint
            != previous_progress_fingerprint
        )

    no_progress_decision = evaluate_no_progress_detection_state(
        no_progress_state=recorded_state,
    )
    updated_state = apply_no_progress_detection_decision(
        no_progress_state=recorded_state,
        no_progress_decision=no_progress_decision,
    )

    return {
        "observation_processed": True,
        "progress_fingerprint": current_progress_fingerprint,
        "progress_changed": progress_changed,
        "no_progress_decision": no_progress_decision,
        "updated_state": updated_state,
    }


def has_repeated_action_suffix_pattern(
    *,
    action_fingerprints: list[str],
    pattern_length: int,
) -> bool:
    """
    判断动作指纹列表末尾是否连续出现两个相同的动作模式。

    功能：
        根据 pattern_length 指定的模式长度，
        只检查动作历史列表末尾的两个相邻模式，
        不扫描更早位置曾经出现过的历史循环。

        例如 pattern_length 为 3 时：

        ["A", "A", "B", "A", "A", "B"]

        末尾两个模式分别是：

        - 第一个模式：["A", "A", "B"]
        - 第二个模式：["A", "A", "B"]

        两个模式相同，因此返回 True。

        如果列表变为：

        ["A", "A", "B", "A", "A", "B", "xxx"]

        那么当前末尾的两个长度为 3 的模式不再相同，
        因此返回 False。

        本函数只负责识别模式，不修改动作历史，
        不更新重复检测状态，也不决定是否终止 Agent 循环。

        长度为 1 的连续相同动作由
        consecutive_same_action_count 负责检测；
        本函数只处理长度大于或等于 2 的循环模式，
        避免两个检测机制职责重叠。

    参数：
        action_fingerprints:
            按动作提议顺序保存的指纹列表。

            列表中的每个元素应当是由
            build_agent_action_fingerprint 生成的字符串指纹。

        pattern_length:
            单个循环模式包含的动作数量，
            必须是真正的整数，并且大于或等于 2。

            例如传入 3，表示检查列表末尾是否存在：
            模式 A + 模式 A，且每个模式包含 3 个动作。

    返回值：
        bool:
            如果列表末尾两个相邻模式完全相同，返回 True；
            如果动作数量不足，或者两个模式不同，返回 False。

    异常：
        TypeError:
            当 action_fingerprints 不是列表，
            或 pattern_length 不是真正的整数时抛出。

        ValueError:
            当 pattern_length 小于 2 时抛出。
    """
    if not isinstance(action_fingerprints, list):
        raise TypeError("action_fingerprints 必须是列表")

    if type(pattern_length) is not int:
        raise TypeError("pattern_length 必须是真正的整数")
    if pattern_length < 2:
        raise ValueError("pattern_length 必须大于或等于 2")

    required_action_count = pattern_length * 2
    if len(action_fingerprints) < required_action_count:
        return False

    first_pattern = action_fingerprints[
        -required_action_count:-pattern_length
    ]
    second_pattern = action_fingerprints[-pattern_length:]

    return first_pattern == second_pattern


def build_repetition_detection_state(
    *,
    max_consecutive_same_actions: int,
    max_consecutive_invalid_actions: int,
    repeated_pattern_length: int,
) -> dict[str, Any]:
    """
    创建 Agent 重复动作检测使用的初始控制状态。

    功能：
        检查 max_consecutive_same_actions
        必须是真正的整数，不能使用布尔值冒充整数，
        并且必须大于 0。

        检查 max_consecutive_invalid_actions
        必须是真正的整数，不能使用布尔值冒充整数，
        并且必须大于 0。

        检查 repeated_pattern_length
        必须是真正的整数，不能使用布尔值冒充整数，
        并且必须大于或等于 2。

        创建重复动作检测控制状态，
        保存上一次合法动作指纹、
        当前相同合法动作连续出现次数、
        连续非法动作次数、最近动作指纹历史、
        循环模式长度、动作历史容量、对应阈值、
        生命周期状态和终止原因。

        consecutive_same_action_count 的语义是：
        当前相同合法动作连续出现了多少次。
        第一次出现某个合法动作时记为 1，
        后续连续出现相同指纹时继续增加。

        consecutive_invalid_action_count 的语义是：
        连续出现了多少次无法通过动作校验、
        JSON 规范化或指纹生成的非法动作。

        合法动作重复计数和非法动作计数相互独立，
        避免指纹生成失败时绕过无限循环保护。

        recent_action_fingerprints 只保存检测当前循环
        所需要的最近动作指纹，不保存无限增长的完整历史。

        当前规则使用两个相邻的相同模式判定循环，
        因此 max_action_history_size 自动计算为：

        repeated_pattern_length * 2

        例如 repeated_pattern_length 为 3 时，
        最多保留最近 6 个动作指纹。

        当前函数只创建初始状态，
        不接收动作、不生成指纹，
        也不执行终止判断。

    参数：
        max_consecutive_same_actions:
            同一个合法动作连续出现并触发停止的阈值。

            例如设置为 3，表示同一动作
            被连续提议到第 3 次时达到停止阈值，
            后续准入判断会在真实执行前拦截第 3 次动作。

        max_consecutive_invalid_actions:
            非法动作连续出现并触发停止的阈值。

            例如设置为 2，表示连续第 2 次
            无法校验或生成指纹时达到停止阈值。

        repeated_pattern_length:
            单个循环模式包含的动作数量。

            必须大于或等于 2。
            长度为 1 的连续相同动作由
            max_consecutive_same_actions 负责处理。

            例如设置为 3，可以检测：
            A、A、B、A、A、B。

    返回值：
        dict[str, Any]:
            新创建的重复检测控制状态，格式如下：

            {
                "status": "running",
                "last_action_fingerprint": None,
                "consecutive_same_action_count": 0,
                "max_consecutive_same_actions": 3,
                "consecutive_invalid_action_count": 0,
                "max_consecutive_invalid_actions": 2,
                "recent_action_fingerprints": [],
                "repeated_pattern_length": 3,
                "max_action_history_size": 6,
                "termination_reason": None,
            }

            status:
                当前重复检测生命周期状态。
                初始值固定为 running。

            last_action_fingerprint:
                上一次合法动作的 SHA-256 指纹。
                尚未记录合法动作时为 None。

            consecutive_same_action_count:
                当前相同合法动作连续出现次数。
                初始值为 0。

            max_consecutive_same_actions:
                相同合法动作连续出现的停止阈值。

            consecutive_invalid_action_count:
                连续非法动作次数。
                初始值为 0。

            max_consecutive_invalid_actions:
                连续非法动作的停止阈值。

            recent_action_fingerprints:
                用于检测当前循环模式的最近动作指纹。
                初始值为空列表。

            repeated_pattern_length:
                单个循环模式包含的动作数量。

            max_action_history_size:
                最近动作指纹列表允许保留的最大长度。
                等于 repeated_pattern_length * 2。

            termination_reason:
                重复检测导致的终止原因。
                初始运行状态下为 None。

    异常：
        TypeError:
            当任一整数参数不是真正的整数时抛出。

        ValueError:
            当任一连续次数阈值小于或等于 0，
            或 repeated_pattern_length 小于 2 时抛出。
    """
    if type(max_consecutive_same_actions) is not int:
        raise TypeError(
            "max_consecutive_same_actions 必须是真正的整数"
        )
    if max_consecutive_same_actions <= 0:
        raise ValueError(
            "max_consecutive_same_actions 必须大于 0"
        )

    if type(max_consecutive_invalid_actions) is not int:
        raise TypeError(
            "max_consecutive_invalid_actions 必须是真正的整数"
        )
    if max_consecutive_invalid_actions <= 0:
        raise ValueError(
            "max_consecutive_invalid_actions 必须大于 0"
        )

    if type(repeated_pattern_length) is not int:
        raise TypeError(
            "repeated_pattern_length 必须是真正的整数"
        )
    if repeated_pattern_length < 2:
        raise ValueError(
            "repeated_pattern_length 必须大于或等于 2"
        )

    max_action_history_size = repeated_pattern_length * 2

    repetition_state = {
        "status": "running",
        "last_action_fingerprint": None,
        "consecutive_same_action_count": 0,
        "max_consecutive_same_actions": (
            max_consecutive_same_actions
        ),
        "consecutive_invalid_action_count": 0,
        "max_consecutive_invalid_actions": (
            max_consecutive_invalid_actions
        ),
        "recent_action_fingerprints": [],
        "repeated_pattern_length": repeated_pattern_length,
        "max_action_history_size": max_action_history_size,
        "termination_reason": None,
    }
    return repetition_state


def build_agent_stagnation_guard_state(
    *,
    max_consecutive_same_actions: int,
    max_consecutive_invalid_actions: int,
    repeated_pattern_length: int,
    max_consecutive_no_progress: int,
) -> dict[str, Any]:
    """
    创建组合重复动作与无进展检测的停滞保护状态。

    功能：
        分别调用：

        - build_repetition_detection_state
        - build_no_progress_detection_state

        创建两个职责独立的子控制状态，
        再将它们放入统一的 Agent 停滞保护状态中。

        repetition_state 用于动作执行前：

        - 检测连续相同动作；
        - 检测动作模式循环；
        - 检测连续非法动作；
        - 在产生真实副作用前拦截可疑动作。

        no_progress_state 用于动作执行后：

        - 记录稳定业务进展快照；
        - 比较前后进展指纹；
        - 检测连续无进展；
        - 决定是否阻止下一轮继续运行。

        父级 status 是 Harness 判断整个停滞保护器
        是否仍可运行的最终权威状态。

        后续任一子保护状态停止时，
        父状态会同步为 stopped，
        termination_source 用于区分停止来源：

        - repetition：重复动作或连续非法动作；
        - no_progress：连续无业务进展。

        当前函数只创建组合初始状态，
        不接收动作、不记录进展、
        不执行判定，也不调用模型或工具。

    参数：
        max_consecutive_same_actions:
            同一合法动作连续出现的停止阈值。

            具体校验规则由
            build_repetition_detection_state 负责。

        max_consecutive_invalid_actions:
            非法动作连续出现的停止阈值。

            具体校验规则由
            build_repetition_detection_state 负责。

        repeated_pattern_length:
            单个循环动作模式包含的动作数量。

            具体校验规则由
            build_repetition_detection_state 负责。

        max_consecutive_no_progress:
            连续无业务进展的停止阈值。

            具体校验规则由
            build_no_progress_detection_state 负责。

    返回值：
        dict[str, Any]:
            新创建的 Agent 停滞保护状态，格式如下：

            {
                "status": "running",
                "repetition_state": {...},
                "no_progress_state": {...},
                "termination_source": None,
                "termination_reason": None,
            }

            status:
                整个停滞保护器的父级生命周期状态。
                初始值固定为 running。

            repetition_state:
                动作执行前使用的重复检测子状态。

            no_progress_state:
                动作执行后使用的无进展检测子状态。

            termination_source:
                导致父状态停止的子保护器来源。
                初始运行状态下为 None。

            termination_reason:
                子保护器产生的稳定停止原因。
                初始运行状态下为 None。

    异常：
        TypeError:
            当任一阈值参数类型不正确时，
            由对应子状态构造函数抛出。

        ValueError:
            当任一阈值参数不符合数值范围时，
            由对应子状态构造函数抛出。
    """
    repetition_state = build_repetition_detection_state(
        max_consecutive_same_actions=(
            max_consecutive_same_actions
        ),
        max_consecutive_invalid_actions=(
            max_consecutive_invalid_actions
        ),
        repeated_pattern_length=repeated_pattern_length,
    )
    no_progress_state = build_no_progress_detection_state(
        max_consecutive_no_progress=(
            max_consecutive_no_progress
        ),
    )

    stagnation_guard_state = {
        "status": "running",
        "repetition_state": repetition_state,
        "no_progress_state": no_progress_state,
        "termination_source": None,
        "termination_reason": None,
    }
    return stagnation_guard_state


def record_valid_action_fingerprint(
    *,
    repetition_state: dict[str, Any],
    action_fingerprint: str,
) -> dict[str, Any]:
    """
    将一个合法动作指纹记录到重复检测状态中。

    功能：
        接收已经成功生成的动作指纹，
        复制传入的重复检测状态，
        在复制版本中更新连续相同动作次数、
        上一次合法动作指纹和最近动作指纹历史。

        如果本次指纹与上一次合法动作指纹相同，
        consecutive_same_action_count 加 1；
        如果不同，则开始一段新的连续动作，
        consecutive_same_action_count 重置为 1。

        因为本次是合法动作，
        consecutive_invalid_action_count 重置为 0。

        本次指纹会被追加到 recent_action_fingerprints。
        如果列表长度超过 max_action_history_size，
        只保留列表末尾仍然参与当前循环检测的指纹。

        本函数记录的是通过基础校验后、
        准备进入重复准入判断的动作提议。
        记录动作不代表工具或模型已经真实执行成功。

        本函数只更新记录，不判断是否达到重复阈值，
        不修改 status 和 termination_reason，
        也不执行工具或模型。

        本函数使用深复制，不修改调用方传入的原状态，
        避免 recent_action_fingerprints 列表被新旧状态共享。

    参数：
        repetition_state:
            当前重复检测控制状态。

            必须是由 build_repetition_detection_state
            创建或遵循相同结构的字典，
            并且 status 必须为 running。

        action_fingerprint:
            已成功生成的合法动作指纹。

            必须是清理首尾空格后不为空的字符串。
            正常情况下应由
            build_agent_action_fingerprint 生成。

    返回值：
        dict[str, Any]:
            更新后的重复检测状态复制版本。

            原 repetition_state 不会被修改。

    异常：
        TypeError:
            当 repetition_state 不是字典、
            action_fingerprint 不是字符串，
            或所需状态字段类型不正确时抛出。

        ValueError:
            当 action_fingerprint 清理后为空、
            状态不是 running，
            状态缺少所需字段，
            或状态字段的数值不符合约束时抛出。
    """
    if not isinstance(repetition_state, dict):
        raise TypeError("repetition_state 必须是字典")

    if not isinstance(action_fingerprint, str):
        raise TypeError("action_fingerprint 必须是字符串")
    normalized_action_fingerprint = action_fingerprint.strip()
    if not normalized_action_fingerprint:
        raise ValueError("action_fingerprint 清理后不能为空")

    required_state_fields = {
        "status",
        "last_action_fingerprint",
        "consecutive_same_action_count",
        "consecutive_invalid_action_count",
        "recent_action_fingerprints",
        "max_action_history_size",
    }
    missing_state_fields = required_state_fields.difference(
        repetition_state
    )
    if missing_state_fields:
        missing_fields_text = ", ".join(
            sorted(missing_state_fields)
        )
        raise ValueError(
            f"repetition_state 缺少字段：{missing_fields_text}"
        )

    if repetition_state["status"] != "running":
        raise ValueError("只有 running 状态可以记录合法动作")

    last_action_fingerprint = repetition_state[
        "last_action_fingerprint"
    ]
    if (
        last_action_fingerprint is not None
        and not isinstance(last_action_fingerprint, str)
    ):
        raise TypeError(
            "last_action_fingerprint 必须是字符串或 None"
        )

    consecutive_same_action_count = repetition_state[
        "consecutive_same_action_count"
    ]
    if type(consecutive_same_action_count) is not int:
        raise TypeError(
            "consecutive_same_action_count 必须是真正的整数"
        )
    if consecutive_same_action_count < 0:
        raise ValueError(
            "consecutive_same_action_count 不能小于 0"
        )

    consecutive_invalid_action_count = repetition_state[
        "consecutive_invalid_action_count"
    ]
    if type(consecutive_invalid_action_count) is not int:
        raise TypeError(
            "consecutive_invalid_action_count 必须是真正的整数"
        )
    if consecutive_invalid_action_count < 0:
        raise ValueError(
            "consecutive_invalid_action_count 不能小于 0"
        )

    recent_action_fingerprints = repetition_state[
        "recent_action_fingerprints"
    ]
    if not isinstance(recent_action_fingerprints, list):
        raise TypeError(
            "recent_action_fingerprints 必须是列表"
        )

    max_action_history_size = repetition_state[
        "max_action_history_size"
    ]
    if type(max_action_history_size) is not int:
        raise TypeError(
            "max_action_history_size 必须是真正的整数"
        )
    if max_action_history_size <= 0:
        raise ValueError(
            "max_action_history_size 必须大于 0"
        )

    updated_state = copy.deepcopy(repetition_state)

    if normalized_action_fingerprint == last_action_fingerprint:
        updated_same_action_count = (
            consecutive_same_action_count + 1
        )
    else:
        updated_same_action_count = 1

    updated_action_history = [
        *recent_action_fingerprints,
        normalized_action_fingerprint,
    ]
    updated_action_history = updated_action_history[
        -max_action_history_size:
    ]

    updated_state["last_action_fingerprint"] = (
        normalized_action_fingerprint
    )
    updated_state["consecutive_same_action_count"] = (
        updated_same_action_count
    )
    updated_state["consecutive_invalid_action_count"] = 0
    updated_state["recent_action_fingerprints"] = (
        updated_action_history
    )
    return updated_state


def record_invalid_action_attempt(
    *,
    repetition_state: dict[str, Any],
) -> dict[str, Any]:
    """
    在重复检测状态中记录一次非法动作提议。

    功能：
        复制传入的重复检测状态，
        将 consecutive_invalid_action_count 加 1，
        表示又出现了一次无法通过基础校验、
        JSON 规范化或动作指纹生成的非法动作。

        非法动作会打断合法动作的连续关系，
        因此同时执行以下重置：

        - last_action_fingerprint 重置为 None；
        - consecutive_same_action_count 重置为 0；
        - recent_action_fingerprints 重置为空列表。

        例如 A、非法动作、A 不能被判断为
        两个连续相同的合法动作。

        同理，A、B、非法动作、A、B
        也不能跨过非法动作拼接成重复循环模式。

        本函数只更新非法动作记录，
        不判断是否达到非法动作停止阈值，
        不修改 status 和 termination_reason，
        也不执行任何工具或模型。

        本函数使用深复制，
        不修改调用方传入的原状态。

    参数：
        repetition_state:
            当前重复检测控制状态。

            必须是由 build_repetition_detection_state
            创建或遵循相同结构的字典，
            并且 status 必须为 running。

    返回值：
        dict[str, Any]:
            记录本次非法动作后的状态复制版本。

            原 repetition_state 不会被修改。

    异常：
        TypeError:
            当 repetition_state 不是字典，
            或所需状态字段类型不正确时抛出。

        ValueError:
            当状态不是 running、
            状态缺少所需字段，
            或计数字段小于 0 时抛出。
    """
    if not isinstance(repetition_state, dict):
        raise TypeError("repetition_state 必须是字典")

    required_state_fields = {
        "status",
        "last_action_fingerprint",
        "consecutive_same_action_count",
        "consecutive_invalid_action_count",
        "recent_action_fingerprints",
    }
    missing_state_fields = required_state_fields.difference(
        repetition_state
    )
    if missing_state_fields:
        missing_fields_text = ", ".join(
            sorted(missing_state_fields)
        )
        raise ValueError(
            f"repetition_state 缺少字段：{missing_fields_text}"
        )

    if repetition_state["status"] != "running":
        raise ValueError("只有 running 状态可以记录非法动作")

    last_action_fingerprint = repetition_state[
        "last_action_fingerprint"
    ]
    if (
        last_action_fingerprint is not None
        and not isinstance(last_action_fingerprint, str)
    ):
        raise TypeError(
            "last_action_fingerprint 必须是字符串或 None"
        )

    consecutive_same_action_count = repetition_state[
        "consecutive_same_action_count"
    ]
    if type(consecutive_same_action_count) is not int:
        raise TypeError(
            "consecutive_same_action_count 必须是真正的整数"
        )
    if consecutive_same_action_count < 0:
        raise ValueError(
            "consecutive_same_action_count 不能小于 0"
        )

    consecutive_invalid_action_count = repetition_state[
        "consecutive_invalid_action_count"
    ]
    if type(consecutive_invalid_action_count) is not int:
        raise TypeError(
            "consecutive_invalid_action_count 必须是真正的整数"
        )
    if consecutive_invalid_action_count < 0:
        raise ValueError(
            "consecutive_invalid_action_count 不能小于 0"
        )

    recent_action_fingerprints = repetition_state[
        "recent_action_fingerprints"
    ]
    if not isinstance(recent_action_fingerprints, list):
        raise TypeError(
            "recent_action_fingerprints 必须是列表"
        )

    updated_state = copy.deepcopy(repetition_state)
    updated_state["last_action_fingerprint"] = None
    updated_state["consecutive_same_action_count"] = 0
    updated_state["consecutive_invalid_action_count"] = (
        consecutive_invalid_action_count + 1
    )
    updated_state["recent_action_fingerprints"] = []
    return updated_state


def evaluate_repetition_detection_state(
    *,
    repetition_state: dict[str, Any],
) -> dict[str, Any]:
    """
    评估当前重复检测状态是否要求停止 Agent 循环。

    功能：
        读取已经记录本轮合法或非法动作提议的状态，
        依次检查以下停止条件：

        1. 状态是否已经不再运行；
        2. 连续非法动作是否达到停止阈值；
        3. 连续相同合法动作是否达到停止阈值；
        4. 最近合法动作是否形成两个相同的相邻模式。

        如果任一条件要求停止，返回 should_stop=True
        和稳定的 termination_reason；
        否则返回 control_action="continue"，
        表示 Harness 可以继续 Agent 循环。

        control_action="continue" 只表示循环可以继续，
        不表示某个非法动作可以被真实执行。

        判定条件按固定优先级执行，
        保证同一个状态始终得到确定的终止原因：

        - consecutive_invalid_actions
        - consecutive_same_actions
        - repeated_action_pattern

        本函数只读取状态并返回判定结果，
        不修改 repetition_state，
        不写入 status 或 termination_reason，
        也不执行工具或模型。

    参数：
        repetition_state:
            已经记录当前动作提议的重复检测状态。

            可以是 running 状态，
            也可以是已经 stopped 的终态。

    返回值：
        dict[str, Any]:
            重复检测判定结果，格式如下：

            {
                "should_stop": False,
                "control_action": "continue",
                "termination_reason": None,
                "repeated_pattern_detected": False,
            }

            should_stop:
                是否应该停止 Agent 循环。

            control_action:
                Harness 下一步控制动作。
                取值为 continue 或 stop。

            termination_reason:
                稳定的终止原因。
                循环可以继续时为 None。

            repeated_pattern_detected:
                当前短期动作历史末尾是否形成
                两个相同的相邻模式。

    异常：
        TypeError:
            当 repetition_state 不是字典，
            或所需状态字段类型不正确时抛出。

        ValueError:
            当状态缺少字段、状态值不受支持、
            数值字段超出允许范围，
            或终止原因与生命周期不一致时抛出。
    """
    if not isinstance(repetition_state, dict):
        raise TypeError("repetition_state 必须是字典")

    required_state_fields = {
        "status",
        "consecutive_same_action_count",
        "max_consecutive_same_actions",
        "consecutive_invalid_action_count",
        "max_consecutive_invalid_actions",
        "recent_action_fingerprints",
        "repeated_pattern_length",
        "max_action_history_size",
        "termination_reason",
    }
    missing_state_fields = required_state_fields.difference(
        repetition_state
    )
    if missing_state_fields:
        missing_fields_text = ", ".join(
            sorted(missing_state_fields)
        )
        raise ValueError(
            f"repetition_state 缺少字段：{missing_fields_text}"
        )

    status = repetition_state["status"]
    if not isinstance(status, str):
        raise TypeError("status 必须是字符串")
    if status not in {"running", "stopped"}:
        raise ValueError("status 必须是 running 或 stopped")

    termination_reason = repetition_state["termination_reason"]
    if termination_reason is not None and not isinstance(
        termination_reason,
        str,
    ):
        raise TypeError("termination_reason 必须是字符串或 None")
    if status == "running" and termination_reason is not None:
        raise ValueError(
            "running 状态的 termination_reason 必须是 None"
        )
    if status == "stopped" and not termination_reason:
        raise ValueError(
            "stopped 状态必须包含 termination_reason"
        )

    integer_field_constraints = {
        "consecutive_same_action_count": 0,
        "max_consecutive_same_actions": 1,
        "consecutive_invalid_action_count": 0,
        "max_consecutive_invalid_actions": 1,
        "repeated_pattern_length": 2,
        "max_action_history_size": 1,
    }
    for field_name, minimum_value in (
        integer_field_constraints.items()
    ):
        field_value = repetition_state[field_name]
        if type(field_value) is not int:
            raise TypeError(
                f"{field_name} 必须是真正的整数"
            )
        if field_value < minimum_value:
            raise ValueError(
                f"{field_name} 不能小于 {minimum_value}"
            )

    repeated_pattern_length = repetition_state[
        "repeated_pattern_length"
    ]
    max_action_history_size = repetition_state[
        "max_action_history_size"
    ]
    expected_history_size = repeated_pattern_length * 2
    if max_action_history_size != expected_history_size:
        raise ValueError(
            "max_action_history_size 必须等于 "
            "repeated_pattern_length * 2"
        )

    recent_action_fingerprints = repetition_state[
        "recent_action_fingerprints"
    ]
    if not isinstance(recent_action_fingerprints, list):
        raise TypeError(
            "recent_action_fingerprints 必须是列表"
        )
    if len(recent_action_fingerprints) > max_action_history_size:
        raise ValueError(
            "recent_action_fingerprints 超过允许的最大长度"
        )
    if any(
        not isinstance(action_fingerprint, str)
        or not action_fingerprint.strip()
        for action_fingerprint in recent_action_fingerprints
    ):
        raise ValueError(
            "recent_action_fingerprints 必须只包含非空字符串"
        )

    repeated_pattern_detected = (
        has_repeated_action_suffix_pattern(
            action_fingerprints=recent_action_fingerprints,
            pattern_length=repeated_pattern_length,
        )
    )
    common_result = {
        "repeated_pattern_detected": (
            repeated_pattern_detected
        ),
    }

    if status == "stopped":
        return {
            **common_result,
            "should_stop": True,
            "control_action": "stop",
            "termination_reason": termination_reason,
        }

    if (
        repetition_state["consecutive_invalid_action_count"]
        >= repetition_state["max_consecutive_invalid_actions"]
    ):
        return {
            **common_result,
            "should_stop": True,
            "control_action": "stop",
            "termination_reason": "consecutive_invalid_actions",
        }

    if (
        repetition_state["consecutive_same_action_count"]
        >= repetition_state["max_consecutive_same_actions"]
    ):
        return {
            **common_result,
            "should_stop": True,
            "control_action": "stop",
            "termination_reason": "consecutive_same_actions",
        }

    if repeated_pattern_detected:
        return {
            **common_result,
            "should_stop": True,
            "control_action": "stop",
            "termination_reason": "repeated_action_pattern",
        }

    return {
        **common_result,
        "should_stop": False,
        "control_action": "continue",
        "termination_reason": None,
    }


def apply_repetition_detection_decision(
    *,
    repetition_state: dict[str, Any],
    repetition_decision: dict[str, Any],
) -> dict[str, Any]:
    """
    将重复检测判定结果应用到状态复制版本中。

    功能：
        检查 repetition_decision 必须包含：

        - should_stop
        - control_action
        - termination_reason
        - repeated_pattern_detected

        根据当前 repetition_state 重新调用
        evaluate_repetition_detection_state，
        得到当前状态下的权威判定结果。

        只比较判定结果所要求的标准字段，
        允许调用方在 repetition_decision 中
        携带额外的调试或追踪字段。

        如果传入判定与重新计算结果不一致，
        拒绝应用，避免伪造判定或旧判定
        被错误地写入已经变化的状态。

        如果状态已经是 stopped，
        返回内容不变的深复制状态，
        保持终态幂等性。

        如果权威判定要求停止，
        将复制状态的 status 更新为 stopped，
        并写入稳定的 termination_reason。

        如果权威判定允许继续，
        返回内容不变的深复制状态。

        本函数不修改传入的 repetition_state，
        不增加任何动作计数，
        不记录新的动作指纹，
        也不执行工具或模型。

    参数：
        repetition_state:
            当前重复检测控制状态。

            可以是 running 状态，
            也可以是已经 stopped 的终态。

        repetition_decision:
            evaluate_repetition_detection_state
            针对当前状态生成的重复检测判定结果。

            可以包含额外诊断字段，
            但四个标准字段必须与重新计算结果一致。

    返回值：
        dict[str, Any]:
            应用判定后的重复检测状态深复制版本。

            should_stop 为 True 时，返回状态中的：

            - status 为 stopped；
            - termination_reason 为对应稳定终止原因。

            should_stop 为 False 时，
            返回与原状态内容相同的复制版本。

    异常：
        TypeError:
            当 repetition_decision 不是字典，
            或判定字段类型不正确时抛出。

        ValueError:
            当判定缺少标准字段、字段取值不受支持、
            终止原因与 should_stop 不一致，
            判定结果与当前状态重新计算的结果不一致，
            或 repetition_state 本身不合法时抛出。
    """
    if not isinstance(repetition_decision, dict):
        raise TypeError("repetition_decision 必须是字典")

    required_decision_fields = {
        "should_stop",
        "control_action",
        "termination_reason",
        "repeated_pattern_detected",
    }
    missing_decision_fields = required_decision_fields.difference(
        repetition_decision
    )
    if missing_decision_fields:
        missing_fields_text = ", ".join(
            sorted(missing_decision_fields)
        )
        raise ValueError(
            f"repetition_decision 缺少字段：{missing_fields_text}"
        )

    if type(repetition_decision["should_stop"]) is not bool:
        raise TypeError("should_stop 必须是真正的布尔值")
    if not isinstance(repetition_decision["control_action"], str):
        raise TypeError("control_action 必须是字符串")
    if repetition_decision["control_action"] not in {
        "continue",
        "stop",
    }:
        raise ValueError("control_action 必须是 continue 或 stop")

    decision_termination_reason = repetition_decision[
        "termination_reason"
    ]
    if (
        decision_termination_reason is not None
        and not isinstance(decision_termination_reason, str)
    ):
        raise TypeError("termination_reason 必须是字符串或 None")

    if type(
        repetition_decision["repeated_pattern_detected"]
    ) is not bool:
        raise TypeError(
            "repeated_pattern_detected 必须是真正的布尔值"
        )

    if repetition_decision["should_stop"]:
        if repetition_decision["control_action"] != "stop":
            raise ValueError(
                "should_stop 为 True 时 control_action 必须是 stop"
            )
        if not decision_termination_reason:
            raise ValueError(
                "should_stop 为 True 时必须包含 termination_reason"
            )
    else:
        if repetition_decision["control_action"] != "continue":
            raise ValueError(
                "should_stop 为 False 时 control_action 必须是 continue"
            )
        if decision_termination_reason is not None:
            raise ValueError(
                "should_stop 为 False 时 termination_reason 必须是 None"
            )

    expected_decision = evaluate_repetition_detection_state(
        repetition_state=repetition_state,
    )
    actual_decision = {
        field_name: repetition_decision[field_name]
        for field_name in required_decision_fields
    }
    if actual_decision != expected_decision:
        raise ValueError(
            "repetition_decision 与当前状态重新计算的结果不一致"
        )

    updated_state = copy.deepcopy(repetition_state)
    if repetition_state["status"] == "stopped":
        return updated_state

    if expected_decision["should_stop"]:
        updated_state["status"] = "stopped"
        updated_state["termination_reason"] = (
            expected_decision["termination_reason"]
        )

    return updated_state


def run_repetition_detection_step(
    *,
    repetition_state: dict[str, Any],
    action_name: str,
    action_arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    对一个动作提议执行完整的重复检测控制步骤。

    功能：
        首先评估传入的 repetition_state。

        如果状态已经停止，或者现有记录已经达到
        某个重复停止条件，则不处理新的动作提议，
        直接应用当前停止决定并返回。

        如果状态仍然允许接收动作，尝试调用
        build_agent_action_fingerprint
        生成稳定动作指纹。

        如果动作名称、动作参数或 JSON 规范化不合法，
        捕获 TypeError 或 ValueError，调用
        record_invalid_action_attempt
        记录一次非法动作提议。

        如果指纹生成成功，调用
        record_valid_action_fingerprint
        记录合法动作指纹。

        最后统一调用：

        - evaluate_repetition_detection_state
        - apply_repetition_detection_decision

        得到本轮重复检测决定和更新后的控制状态。

        try-except 只包围动作指纹生成过程。
        状态结构损坏、判定错误或应用错误不会被
        错误归类为普通非法动作，而是继续向上抛出，
        便于发现程序自身的问题。

        本函数只编排重复检测流程，
        不修改传入的 repetition_state，
        不调用模型，也不执行真实工具。

    参数：
        repetition_state:
            本轮开始前的重复检测控制状态。

        action_name:
            本轮模型或上游决策器提出的动作名称。

            该值会交给 build_agent_action_fingerprint
            进行类型、空值和规范化检查。

        action_arguments:
            决定动作业务身份的参数字典。

            该值会交给 build_agent_action_fingerprint
            进行类型和 JSON 规范化检查。

    返回值：
        dict[str, Any]:
            本轮重复检测步骤结果，格式如下：

            {
                "action_processed": True,
                "action_valid": True,
                "action_fingerprint": "...",
                "action_error": None,
                "repetition_decision": {...},
                "updated_state": {...},
            }

            action_processed:
                是否实际检查并记录了本轮动作提议。
                进入函数前已经应当停止时为 False。

            action_valid:
                True 表示成功生成动作指纹；
                False 表示动作不合法；
                未处理动作时为 None。

            action_fingerprint:
                合法动作生成的 SHA-256 指纹。
                非法或未处理动作时为 None。

            action_error:
                指纹生成失败时的稳定错误信息。
                合法或未处理动作时为 None。

            repetition_decision:
                本轮最终重复检测决定。

            updated_state:
                应用本轮决定后的重复检测控制状态。

    异常：
        TypeError:
            当 repetition_state 或其字段类型不正确时抛出。

        ValueError:
            当 repetition_state 结构不合法、
            状态决定无法应用或状态约束冲突时抛出。

            action_name 或 action_arguments 导致的
            TypeError、ValueError 不会继续抛出，
            而会被记录成一次非法动作提议。
    """
    current_decision = evaluate_repetition_detection_state(
        repetition_state=repetition_state,
    )
    if current_decision["should_stop"]:
        stopped_state = apply_repetition_detection_decision(
            repetition_state=repetition_state,
            repetition_decision=current_decision,
        )
        return {
            "action_processed": False,
            "action_valid": None,
            "action_fingerprint": None,
            "action_error": None,
            "repetition_decision": current_decision,
            "updated_state": stopped_state,
        }

    action_fingerprint: str | None = None
    action_error: str | None = None
    try:
        action_fingerprint = build_agent_action_fingerprint(
            action_name=action_name,
            action_arguments=action_arguments,
        )
    except (TypeError, ValueError) as exc:
        action_error = f"{type(exc).__name__}: {exc}"
        recorded_state = record_invalid_action_attempt(
            repetition_state=repetition_state,
        )
        action_valid = False
    else:
        recorded_state = record_valid_action_fingerprint(
            repetition_state=repetition_state,
            action_fingerprint=action_fingerprint,
        )
        action_valid = True

    repetition_decision = evaluate_repetition_detection_state(
        repetition_state=recorded_state,
    )
    updated_state = apply_repetition_detection_decision(
        repetition_state=recorded_state,
        repetition_decision=repetition_decision,
    )

    return {
        "action_processed": True,
        "action_valid": action_valid,
        "action_fingerprint": action_fingerprint,
        "action_error": action_error,
        "repetition_decision": repetition_decision,
        "updated_state": updated_state,
    }


def run_stagnation_guard_before_action(
    *,
    stagnation_guard_state: dict[str, Any],
    action_name: str,
    action_arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    在动作真实执行前运行组合停滞保护检查。

    功能：
        检查组合停滞保护状态必须包含：

        - status
        - repetition_state
        - no_progress_state
        - termination_source
        - termination_reason

        如果父状态已经是 stopped，
        不处理新的动作提议，直接返回父状态复制版本。

        如果父状态仍为 running，
        先评估 no_progress_state。
        如果上一轮执行后的无进展记录已经要求停止，
        先把无进展停止结果同步到父状态，
        不再处理本轮新动作。

        如果无进展状态允许继续，调用
        run_repetition_detection_step
        检查本轮动作提议是否合法、是否连续重复、
        是否形成循环模式。

        如果重复检测子状态停止，
        将父状态同步为 stopped，
        termination_source 设置为 repetition，
        termination_reason 保存子状态停止原因。

        如果动作非法但尚未达到整体停止阈值，
        父状态仍保持 running，
        但 action_can_proceed 返回 False，
        表示当前动作不能进入真实执行阶段。

        只有动作合法、重复检测没有停止、
        父状态仍为 running 时，
        action_can_proceed 才返回 True。

        本函数使用深复制，
        不修改传入的 stagnation_guard_state，
        不调用模型，也不执行真实工具。

    参数：
        stagnation_guard_state:
            当前组合停滞保护状态。

            通常由 build_agent_stagnation_guard_state 创建，
            或由上一次执行前、执行后组合入口返回。

        action_name:
            本轮模型或上游决策器提出的动作名称。

        action_arguments:
            决定本轮动作业务身份的参数字典。

    返回值：
        dict[str, Any]:
            动作执行前的组合停滞检查结果：

            {
                "action_can_proceed": True,
                "action_block_reason": None,
                "repetition_step_result": {...},
                "updated_state": {...},
            }

            action_can_proceed:
                当前动作是否通过停滞保护检查，
                可以继续进入其他准入检查和真实执行阶段。

            action_block_reason:
                当前动作不能继续时的稳定原因。

                父状态停止时使用 termination_reason；
                动作非法但循环仍可继续时为 invalid_action；
                动作可以继续时为 None。

            repetition_step_result:
                本轮重复检测子步骤的完整结果。

                因父状态或无进展状态提前停止，
                尚未处理动作时为 None。

            updated_state:
                同步子状态结果后的组合停滞保护状态。

    异常：
        TypeError:
            当 stagnation_guard_state 不是字典，
            或组合状态字段类型不正确时抛出。

        ValueError:
            当组合状态缺少字段、生命周期字段不一致，
            子状态不合法，或子状态停止结果无法应用时抛出。
    """
    if not isinstance(stagnation_guard_state, dict):
        raise TypeError("stagnation_guard_state 必须是字典")

    required_state_fields = {
        "status",
        "repetition_state",
        "no_progress_state",
        "termination_source",
        "termination_reason",
    }
    missing_state_fields = required_state_fields.difference(
        stagnation_guard_state
    )
    if missing_state_fields:
        missing_fields_text = ", ".join(
            sorted(missing_state_fields)
        )
        raise ValueError(
            f"stagnation_guard_state 缺少字段：{missing_fields_text}"
        )

    status = stagnation_guard_state["status"]
    if not isinstance(status, str):
        raise TypeError("status 必须是字符串")
    if status not in {"running", "stopped"}:
        raise ValueError("status 必须是 running 或 stopped")

    repetition_state = stagnation_guard_state[
        "repetition_state"
    ]
    no_progress_state = stagnation_guard_state[
        "no_progress_state"
    ]
    if not isinstance(repetition_state, dict):
        raise TypeError("repetition_state 必须是字典")
    if not isinstance(no_progress_state, dict):
        raise TypeError("no_progress_state 必须是字典")

    termination_source = stagnation_guard_state[
        "termination_source"
    ]
    termination_reason = stagnation_guard_state[
        "termination_reason"
    ]

    if status == "running":
        if termination_source is not None:
            raise ValueError(
                "running 状态的 termination_source 必须是 None"
            )
        if termination_reason is not None:
            raise ValueError(
                "running 状态的 termination_reason 必须是 None"
            )
    else:
        if termination_source not in {
            "repetition",
            "no_progress",
        }:
            raise ValueError(
                "stopped 状态的 termination_source 必须是 "
                "repetition 或 no_progress"
            )
        if not isinstance(termination_reason, str):
            raise TypeError(
                "stopped 状态的 termination_reason 必须是字符串"
            )
        if not termination_reason.strip():
            raise ValueError(
                "stopped 状态的 termination_reason 不能为空"
            )

    updated_guard_state = copy.deepcopy(stagnation_guard_state)
    if status == "stopped":
        return {
            "action_can_proceed": False,
            "action_block_reason": termination_reason,
            "repetition_step_result": None,
            "updated_state": updated_guard_state,
        }

    no_progress_decision = evaluate_no_progress_detection_state(
        no_progress_state=no_progress_state,
    )
    if no_progress_decision["should_stop"]:
        applied_no_progress_state = (
            apply_no_progress_detection_decision(
                no_progress_state=no_progress_state,
                no_progress_decision=no_progress_decision,
            )
        )
        updated_guard_state["no_progress_state"] = (
            applied_no_progress_state
        )
        updated_guard_state["status"] = "stopped"
        updated_guard_state["termination_source"] = "no_progress"
        updated_guard_state["termination_reason"] = (
            applied_no_progress_state["termination_reason"]
        )
        return {
            "action_can_proceed": False,
            "action_block_reason": updated_guard_state[
                "termination_reason"
            ],
            "repetition_step_result": None,
            "updated_state": updated_guard_state,
        }

    repetition_step_result = run_repetition_detection_step(
        repetition_state=repetition_state,
        action_name=action_name,
        action_arguments=action_arguments,
    )
    updated_repetition_state = repetition_step_result[
        "updated_state"
    ]
    updated_guard_state["repetition_state"] = (
        updated_repetition_state
    )

    if updated_repetition_state["status"] == "stopped":
        updated_guard_state["status"] = "stopped"
        updated_guard_state["termination_source"] = "repetition"
        updated_guard_state["termination_reason"] = (
            updated_repetition_state["termination_reason"]
        )
        action_can_proceed = False
        action_block_reason = updated_guard_state[
            "termination_reason"
        ]
    elif not repetition_step_result["action_valid"]:
        action_can_proceed = False
        action_block_reason = "invalid_action"
    else:
        action_can_proceed = True
        action_block_reason = None

    return {
        "action_can_proceed": action_can_proceed,
        "action_block_reason": action_block_reason,
        "repetition_step_result": repetition_step_result,
        "updated_state": updated_guard_state,
    }


def run_stagnation_guard_after_action(
    *,
    stagnation_guard_state: dict[str, Any],
    progress_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """
    在动作真实执行后运行组合停滞保护检查。

    功能：
        检查组合停滞保护状态必须包含：

        - status
        - repetition_state
        - no_progress_state
        - termination_source
        - termination_reason

        如果父状态已经是 stopped，
        不记录新的进展快照，
        直接返回父状态复制版本。

        如果父状态仍为 running，
        首先重新评估 repetition_state。
        如果重复检测已经要求停止，
        说明当前动作本应在执行前被拦截，
        此时调用执行后入口属于 Harness 顺序错误，
        因此抛出 ValueError，避免把错误执行
        伪装成一次正常的进展观察。

        如果执行前状态一致，调用
        run_no_progress_detection_step：

        - 记录动作执行后的业务进展快照；
        - 比较新旧进展指纹；
        - 更新连续无进展次数；
        - 生成并应用无进展停止决定。

        如果无进展子状态停止，
        将父状态同步为 stopped，
        termination_source 设置为 no_progress，
        termination_reason 保存子状态停止原因。

        loop_can_continue 回答的是当前动作执行完成后，
        Harness 是否还可以进入下一轮；
        它不同于执行前入口的 action_can_proceed。

        本函数使用深复制，
        不修改传入的 stagnation_guard_state，
        不调用模型，也不执行真实工具。

    参数：
        stagnation_guard_state:
            当前组合停滞保护状态。

            应当是本轮执行前入口返回的 updated_state，
            并且本轮动作确实已经通过其他准入检查、
            完成真实执行。

        progress_snapshot:
            根据本轮动作执行结果提取的
            稳定业务进展快照。

            必须符合 build_agent_progress_fingerprint
            对非空字典和 JSON 序列化的要求。

    返回值：
        dict[str, Any]:
            动作执行后的组合停滞检查结果：

            {
                "loop_can_continue": True,
                "loop_stop_reason": None,
                "no_progress_step_result": {...},
                "updated_state": {...},
            }

            loop_can_continue:
                当前动作执行完成后，
                是否允许 Harness 进入下一轮。

            loop_stop_reason:
                不允许进入下一轮时的稳定停止原因。
                可以继续时为 None。

            no_progress_step_result:
                本轮无进展检测子步骤的完整结果。

                父状态进入函数前已经停止时为 None。

            updated_state:
                同步无进展子状态后的
                组合停滞保护状态。

    异常：
        TypeError:
            当 stagnation_guard_state 不是字典，
            或组合状态字段类型不正确时抛出。

        ValueError:
            当组合状态缺少字段、生命周期字段不一致、
            子状态不合法、progress_snapshot 不合法，
            或重复检测本应在执行前阻止动作时抛出。
    """
    if not isinstance(stagnation_guard_state, dict):
        raise TypeError("stagnation_guard_state 必须是字典")

    required_state_fields = {
        "status",
        "repetition_state",
        "no_progress_state",
        "termination_source",
        "termination_reason",
    }
    missing_state_fields = required_state_fields.difference(
        stagnation_guard_state
    )
    if missing_state_fields:
        missing_fields_text = ", ".join(
            sorted(missing_state_fields)
        )
        raise ValueError(
            f"stagnation_guard_state 缺少字段：{missing_fields_text}"
        )

    status = stagnation_guard_state["status"]
    if not isinstance(status, str):
        raise TypeError("status 必须是字符串")
    if status not in {"running", "stopped"}:
        raise ValueError("status 必须是 running 或 stopped")

    repetition_state = stagnation_guard_state[
        "repetition_state"
    ]
    no_progress_state = stagnation_guard_state[
        "no_progress_state"
    ]
    if not isinstance(repetition_state, dict):
        raise TypeError("repetition_state 必须是字典")
    if not isinstance(no_progress_state, dict):
        raise TypeError("no_progress_state 必须是字典")

    termination_source = stagnation_guard_state[
        "termination_source"
    ]
    termination_reason = stagnation_guard_state[
        "termination_reason"
    ]

    if status == "running":
        if termination_source is not None:
            raise ValueError(
                "running 状态的 termination_source 必须是 None"
            )
        if termination_reason is not None:
            raise ValueError(
                "running 状态的 termination_reason 必须是 None"
            )
    else:
        if termination_source not in {
            "repetition",
            "no_progress",
        }:
            raise ValueError(
                "stopped 状态的 termination_source 必须是 "
                "repetition 或 no_progress"
            )
        if not isinstance(termination_reason, str):
            raise TypeError(
                "stopped 状态的 termination_reason 必须是字符串"
            )
        if not termination_reason.strip():
            raise ValueError(
                "stopped 状态的 termination_reason 不能为空"
            )

    updated_guard_state = copy.deepcopy(stagnation_guard_state)
    if status == "stopped":
        return {
            "loop_can_continue": False,
            "loop_stop_reason": termination_reason,
            "no_progress_step_result": None,
            "updated_state": updated_guard_state,
        }

    repetition_decision = evaluate_repetition_detection_state(
        repetition_state=repetition_state,
    )
    if repetition_decision["should_stop"]:
        raise ValueError(
            "repetition_state 已要求停止，当前动作不应进入执行后阶段"
        )

    no_progress_step_result = run_no_progress_detection_step(
        no_progress_state=no_progress_state,
        progress_snapshot=progress_snapshot,
    )
    updated_no_progress_state = no_progress_step_result[
        "updated_state"
    ]
    updated_guard_state["no_progress_state"] = (
        updated_no_progress_state
    )

    if updated_no_progress_state["status"] == "stopped":
        updated_guard_state["status"] = "stopped"
        updated_guard_state["termination_source"] = "no_progress"
        updated_guard_state["termination_reason"] = (
            updated_no_progress_state["termination_reason"]
        )
        loop_can_continue = False
        loop_stop_reason = updated_guard_state[
            "termination_reason"
        ]
    else:
        loop_can_continue = True
        loop_stop_reason = None

    return {
        "loop_can_continue": loop_can_continue,
        "loop_stop_reason": loop_stop_reason,
        "no_progress_step_result": no_progress_step_result,
        "updated_state": updated_guard_state,
    }


def run_day18_repetition_detection_demo() -> dict[str, Any]:
    """
    演示 A、A、B、A、A、B 循环模式的完整检测流程。

    功能：
        创建一份重复检测初始状态，配置：

        - 连续相同动作停止阈值为 3；
        - 连续非法动作停止阈值为 2；
        - 循环模式长度为 3。

        准备两个稳定动作提议：

        - A：reason 动作；
        - B：call_tool 动作。

        按照 A、A、B、A、A、B 的顺序，
        逐个调用 run_repetition_detection_step。

        A 和 B 是为了方便阅读而设置的演示标签，
        不会进入动作指纹载荷。
        真正参与重复检测的是动作名称与动作参数
        生成的 SHA-256 指纹。

        每一步都把上一步返回的 updated_state
        作为下一步 repetition_state，
        保证动作历史和连续次数不会丢失。

        第六个动作 B 被记录后，最近六个指纹形成：

        A、A、B、A、A、B

        因此重复检测决定应当要求停止，
        最终状态的 termination_reason 应当为
        repeated_action_pattern。

        本演示只执行重复检测控制代码，
        不调用模型，也不执行 B 所代表的真实工具。

    参数：
        无。

    返回值：
        dict[str, Any]:
            Day 18 重复模式演示报告，包含：

            {
                "action_pattern": [
                    "A", "A", "B", "A", "A", "B"
                ],
                "initial_state": {...},
                "steps": [...],
                "final_state": {...},
                "stopped_on_sixth_action": True,
                "initial_state_unchanged": True,
            }

            action_pattern:
                本次演示使用的人类可读动作标签顺序。

            initial_state:
                演示开始前的重复检测状态快照。

            steps:
                每个动作提议对应的处理结果摘要，
                包含动作标签、动作内容、动作指纹、
                重复检测决定和应用后的状态快照。

            final_state:
                第六个动作处理完成后的最终控制状态。

            stopped_on_sixth_action:
                是否恰好处理六个动作，
                并因 repeated_action_pattern 停止。

            initial_state_unchanged:
                最初创建的状态对象内容是否仍然
                等于演示开始前的快照，
                用于验证各步骤没有原地修改初始状态。
    """
    initial_state = build_repetition_detection_state(
        max_consecutive_same_actions=3,
        max_consecutive_invalid_actions=2,
        repeated_pattern_length=3,
    )
    initial_state_snapshot = copy.deepcopy(initial_state)

    action_a = {
        "label": "A",
        "action_name": "reason",
        "action_arguments": {
            "topic": "dog_care_plan",
        },
    }
    action_b = {
        "label": "B",
        "action_name": "call_tool",
        "action_arguments": {
            "tool_name": "dog_knowledge_search",
            "query": "dog care plan",
        },
    }
    action_proposals = [
        action_a,
        action_a,
        action_b,
        action_a,
        action_a,
        action_b,
    ]

    current_state = initial_state
    step_records: list[dict[str, Any]] = []

    for step_number, action_proposal in enumerate(
        action_proposals,
        start=1,
    ):
        step_result = run_repetition_detection_step(
            repetition_state=current_state,
            action_name=action_proposal["action_name"],
            action_arguments=action_proposal["action_arguments"],
        )
        current_state = step_result["updated_state"]

        step_records.append(
            {
                "step_number": step_number,
                "action_label": action_proposal["label"],
                "action_name": action_proposal["action_name"],
                "action_arguments": copy.deepcopy(
                    action_proposal["action_arguments"]
                ),
                "action_processed": step_result[
                    "action_processed"
                ],
                "action_valid": step_result["action_valid"],
                "action_fingerprint": step_result[
                    "action_fingerprint"
                ],
                "action_error": step_result["action_error"],
                "repetition_decision": copy.deepcopy(
                    step_result["repetition_decision"]
                ),
                "state_after_step": copy.deepcopy(current_state),
            }
        )

        if current_state["status"] == "stopped":
            break

    stopped_on_sixth_action = (
        len(step_records) == 6
        and current_state["status"] == "stopped"
        and current_state["termination_reason"]
        == "repeated_action_pattern"
    )

    return {
        "action_pattern": [
            action_proposal["label"]
            for action_proposal in action_proposals
        ],
        "initial_state": initial_state_snapshot,
        "steps": step_records,
        "final_state": copy.deepcopy(current_state),
        "stopped_on_sixth_action": stopped_on_sixth_action,
        "initial_state_unchanged": (
            initial_state == initial_state_snapshot
        ),
    }


def run_day18_invalid_action_detection_demo() -> dict[str, Any]:
    """
    演示无法生成指纹的动作仍会受到连续非法次数保护。

    功能：
        创建一份重复检测初始状态，
        将连续非法动作停止阈值设置为 2。

        构造一个 action_arguments 中包含 object 实例的
        call_tool 动作提议。
        object 实例不能被 json.dumps 规范化，
        因此 build_agent_action_fingerprint
        无法为该动作生成稳定指纹。

        连续两次把该非法动作提议交给
        run_repetition_detection_step：

        - 第一次记录非法次数 1，循环仍可继续；
        - 第二次记录非法次数 2，达到停止阈值；
        - 最终停止原因为 consecutive_invalid_actions。

        演示报告不会保存不可 JSON 序列化的 object 实例，
        只保存参数类型说明，避免演示报告自身
        在后续 json.dumps 时再次失败。

        本演示不调用模型，也不执行真实工具。

    参数：
        无。

    返回值：
        dict[str, Any]:
            Day 18 连续非法动作演示报告，包含：

            {
                "initial_state": {...},
                "steps": [...],
                "final_state": {...},
                "stopped_on_second_invalid_action": True,
                "initial_state_unchanged": True,
            }

            initial_state:
                演示开始前的重复检测状态快照。

            steps:
                两次非法动作提议对应的处理结果摘要。
                只记录参数类型说明，不记录 object 实例。

            final_state:
                第二次非法动作处理后的最终控制状态。

            stopped_on_second_invalid_action:
                是否恰好处理两个非法动作，
                并因 consecutive_invalid_actions 停止。

            initial_state_unchanged:
                最初创建的状态对象内容是否仍然
                等于演示开始前的快照。
    """
    initial_state = build_repetition_detection_state(
        max_consecutive_same_actions=3,
        max_consecutive_invalid_actions=2,
        repeated_pattern_length=3,
    )
    initial_state_snapshot = copy.deepcopy(initial_state)

    invalid_payload = object()
    invalid_action_name = "call_tool"
    invalid_action_arguments = {
        "tool_name": "dog_knowledge_search",
        "payload": invalid_payload,
    }
    action_arguments_description = {
        "tool_name": "dog_knowledge_search",
        "payload_type": type(invalid_payload).__name__,
    }

    current_state = initial_state
    step_records: list[dict[str, Any]] = []

    for attempt_number in range(1, 3):
        step_result = run_repetition_detection_step(
            repetition_state=current_state,
            action_name=invalid_action_name,
            action_arguments=invalid_action_arguments,
        )
        current_state = step_result["updated_state"]

        step_records.append(
            {
                "attempt_number": attempt_number,
                "action_name": invalid_action_name,
                "action_arguments_description": copy.deepcopy(
                    action_arguments_description
                ),
                "action_processed": step_result[
                    "action_processed"
                ],
                "action_valid": step_result["action_valid"],
                "action_fingerprint": step_result[
                    "action_fingerprint"
                ],
                "action_error": step_result["action_error"],
                "repetition_decision": copy.deepcopy(
                    step_result["repetition_decision"]
                ),
                "state_after_step": copy.deepcopy(current_state),
            }
        )

        if current_state["status"] == "stopped":
            break

    stopped_on_second_invalid_action = (
        len(step_records) == 2
        and current_state["status"] == "stopped"
        and current_state["termination_reason"]
        == "consecutive_invalid_actions"
    )

    return {
        "initial_state": initial_state_snapshot,
        "steps": step_records,
        "final_state": copy.deepcopy(current_state),
        "stopped_on_second_invalid_action": (
            stopped_on_second_invalid_action
        ),
        "initial_state_unchanged": (
            initial_state == initial_state_snapshot
        ),
    }


def run_day18_no_progress_detection_demo() -> dict[str, Any]:
    """
    演示建立进展基线后连续两次无进展的停止流程。

    功能：
        创建无进展检测初始状态，
        将连续无进展停止阈值设置为 2。

        构造一份稳定的狗狗护理任务进展快照，
        连续三次将内容完全相同的快照交给
        run_no_progress_detection_step。

        三次观察的预期语义是：

        1. 第一次观察建立比较基线，
           progress_changed 为 None，
           连续无进展次数保持为 0；
        2. 第二次观察与基线相同，
           progress_changed 为 False，
           连续无进展次数增加到 1；
        3. 第三次观察仍然相同，
           progress_changed 为 False，
           连续无进展次数增加到 2，
           达到阈值后状态更新为 stopped。

        每一步都把上一步返回的 updated_state
        作为下一步 no_progress_state，
        保证进展基线、观察次数和无进展次数不会丢失。

        本演示假设进展快照已经由真实业务状态提取完成，
        不调用模型，也不执行真实工具。

    参数：
        无。

    返回值：
        dict[str, Any]:
            Day 18 无进展检测演示报告，格式如下：

            {
                "progress_snapshot": {...},
                "initial_state": {...},
                "steps": [...],
                "final_state": {...},
                "stopped_on_third_observation": True,
                "initial_state_unchanged": True,
            }

            progress_snapshot:
                三次观察共同使用的稳定业务进展快照。

            initial_state:
                演示开始前的无进展检测状态快照。

            steps:
                三次进展观察对应的处理结果摘要，
                包含是否发生进展、进展指纹、
                无进展判定和应用后的状态快照。

            final_state:
                第三次进展观察完成后的最终控制状态。

            stopped_on_third_observation:
                是否恰好处理三次观察，
                并因 consecutive_no_progress 停止。

            initial_state_unchanged:
                最初创建的状态对象内容是否仍然
                等于演示开始前的快照。
    """
    initial_state = build_no_progress_detection_state(
        max_consecutive_no_progress=2,
    )
    initial_state_snapshot = copy.deepcopy(initial_state)

    progress_snapshot = {
        "current_phase": "research",
        "completed_steps": [
            "parse_user_question",
        ],
        "collected_chunk_ids": [],
        "remaining_goals": [
            "retrieve_dog_documents",
            "generate_final_answer",
        ],
        "final_answer_ready": False,
    }

    current_state = initial_state
    step_records: list[dict[str, Any]] = []

    for observation_number in range(1, 4):
        step_result = run_no_progress_detection_step(
            no_progress_state=current_state,
            progress_snapshot=progress_snapshot,
        )
        current_state = step_result["updated_state"]

        step_records.append(
            {
                "observation_number": observation_number,
                "observation_processed": step_result[
                    "observation_processed"
                ],
                "progress_fingerprint": step_result[
                    "progress_fingerprint"
                ],
                "progress_changed": step_result[
                    "progress_changed"
                ],
                "no_progress_decision": copy.deepcopy(
                    step_result["no_progress_decision"]
                ),
                "state_after_observation": copy.deepcopy(
                    current_state
                ),
            }
        )

        if current_state["status"] == "stopped":
            break

    stopped_on_third_observation = (
        len(step_records) == 3
        and current_state["status"] == "stopped"
        and current_state["termination_reason"]
        == "consecutive_no_progress"
        and current_state["consecutive_no_progress_count"] == 2
        and current_state["progress_observation_count"] == 3
    )

    return {
        "progress_snapshot": copy.deepcopy(progress_snapshot),
        "initial_state": initial_state_snapshot,
        "steps": step_records,
        "final_state": copy.deepcopy(current_state),
        "stopped_on_third_observation": (
            stopped_on_third_observation
        ),
        "initial_state_unchanged": (
            initial_state == initial_state_snapshot
        ),
    }


def run_day18_progress_recovery_demo() -> dict[str, Any]:
    """
    演示出现新业务进展后连续无进展次数会被重置。

    功能：
        创建无进展检测初始状态，
        将连续无进展停止阈值设置为 2。

        按照 F1、F1、F2 的顺序记录三份进展快照：

        1. 第一次 F1 建立比较基线，
           连续无进展次数为 0；
        2. 第二次 F1 与基线相同，
           连续无进展次数增加到 1；
        3. 第三次 F2 包含新的召回文档和已完成步骤，
           产生真实业务进展，
           连续无进展次数重置为 0。

        因为无进展次数在达到阈值 2 之前被重置，
        最终状态应当继续保持 running。

        本演示不调用模型，也不执行真实工具。

    参数：
        无。

    返回值：
        dict[str, Any]:
            Day 18 进展恢复反向演示报告，格式如下：

            {
                "progress_sequence": ["F1", "F1", "F2"],
                "initial_state": {...},
                "steps": [...],
                "final_state": {...},
                "progress_reset_after_new_snapshot": True,
                "initial_state_unchanged": True,
            }

            progress_sequence:
                三次观察使用的人类可读进展标签。

            initial_state:
                演示开始前的无进展检测状态快照。

            steps:
                每次进展观察的标签、快照、
                检测结果和状态快照。

            final_state:
                第三次进展观察后的最终控制状态。

            progress_reset_after_new_snapshot:
                是否先观察到一次无进展，
                随后因 F2 出现真实进展而把计数重置为 0，
                并保持 running 状态。

            initial_state_unchanged:
                最初创建的状态对象内容是否仍然
                等于演示开始前的快照。
    """
    initial_state = build_no_progress_detection_state(
        max_consecutive_no_progress=2,
    )
    initial_state_snapshot = copy.deepcopy(initial_state)

    progress_f1 = {
        "current_phase": "research",
        "completed_steps": [
            "parse_user_question",
        ],
        "collected_chunk_ids": [],
        "remaining_goals": [
            "retrieve_dog_documents",
            "generate_final_answer",
        ],
        "final_answer_ready": False,
    }
    progress_f2 = {
        "current_phase": "context_building",
        "completed_steps": [
            "parse_user_question",
            "retrieve_dog_documents",
        ],
        "collected_chunk_ids": [
            "PDF-day10_text_sample-chunk-0",
        ],
        "remaining_goals": [
            "generate_final_answer",
        ],
        "final_answer_ready": True,
    }
    progress_observations = [
        ("F1", progress_f1),
        ("F1", progress_f1),
        ("F2", progress_f2),
    ]

    current_state = initial_state
    step_records: list[dict[str, Any]] = []

    for observation_number, (
        progress_label,
        progress_snapshot,
    ) in enumerate(progress_observations, start=1):
        step_result = run_no_progress_detection_step(
            no_progress_state=current_state,
            progress_snapshot=progress_snapshot,
        )
        current_state = step_result["updated_state"]

        step_records.append(
            {
                "observation_number": observation_number,
                "progress_label": progress_label,
                "progress_snapshot": copy.deepcopy(
                    progress_snapshot
                ),
                "progress_changed": step_result[
                    "progress_changed"
                ],
                "state_after_observation": copy.deepcopy(
                    current_state
                ),
            }
        )

        if current_state["status"] == "stopped":
            break

    progress_reset_after_new_snapshot = (
        len(step_records) == 3
        and step_records[0]["progress_changed"] is None
        and step_records[1]["progress_changed"] is False
        and step_records[1]["state_after_observation"][
            "consecutive_no_progress_count"
        ]
        == 1
        and step_records[2]["progress_changed"] is True
        and current_state["consecutive_no_progress_count"] == 0
        and current_state["progress_observation_count"] == 3
        and current_state["status"] == "running"
        and current_state["termination_reason"] is None
    )

    return {
        "progress_sequence": [
            progress_label
            for progress_label, _ in progress_observations
        ],
        "initial_state": initial_state_snapshot,
        "steps": step_records,
        "final_state": copy.deepcopy(current_state),
        "progress_reset_after_new_snapshot": (
            progress_reset_after_new_snapshot
        ),
        "initial_state_unchanged": (
            initial_state == initial_state_snapshot
        ),
    }


def run_day18_combined_stagnation_guard_demo() -> dict[str, Any]:
    """
    演示执行前重复检查与执行后无进展检查的组合流程。

    功能：
        创建一份组合停滞保护初始状态，配置：

        - 连续相同动作停止阈值为 3；
        - 连续非法动作停止阈值为 2；
        - 重复动作模式长度为 3；
        - 连续无进展停止阈值为 2。

        准备三个业务内容不同的合法动作提议，
        使三轮动作生成不同的动作指纹，
        不触发连续相同动作或循环模式检测。

        每一轮先调用
        run_stagnation_guard_before_action，
        如果 action_can_proceed 为 True，
        就把该动作标记为已经完成模拟执行。

        模拟执行后，三轮都使用同一份稳定业务进展快照，
        再调用 run_stagnation_guard_after_action。

        三轮执行后进展语义为：

        1. 第一轮建立进展比较基线；
        2. 第二轮产生第一次连续无进展；
        3. 第三轮产生第二次连续无进展，
           达到阈值并由 no_progress 子状态
           同步停止父级组合状态。

        本演示不会真实调用模型或工具。
        simulated_action_executed 只表示代码流程
        模拟通过了动作执行阶段。

    参数：
        无。

    返回值：
        dict[str, Any]:
            Day 18 组合停滞保护演示报告，格式如下：

            {
                "initial_state": {...},
                "rounds": [...],
                "final_state": {...},
                "stopped_after_third_action_by_no_progress": True,
                "initial_state_unchanged": True,
            }

            initial_state:
                演示开始前的组合停滞保护状态快照。

            rounds:
                每轮动作执行前检查、模拟执行标记、
                执行后检查及其状态快照。

            final_state:
                第三轮执行后得到的最终父级组合状态。

            stopped_after_third_action_by_no_progress:
                是否正好完成三轮模拟动作，
                并由 no_progress 来源以
                consecutive_no_progress 原因停止。

            initial_state_unchanged:
                最初创建的组合状态内容是否仍然
                等于演示开始前的快照。
    """
    initial_state = build_agent_stagnation_guard_state(
        max_consecutive_same_actions=3,
        max_consecutive_invalid_actions=2,
        repeated_pattern_length=3,
        max_consecutive_no_progress=2,
    )
    initial_state_snapshot = copy.deepcopy(initial_state)

    action_proposals = [
        {
            "label": "A",
            "action_name": "reason",
            "action_arguments": {
                "topic": "analyze_dog_care_question",
            },
        },
        {
            "label": "B",
            "action_name": "call_tool",
            "action_arguments": {
                "tool_name": "dog_knowledge_search",
                "query": "dog daily care",
            },
        },
        {
            "label": "C",
            "action_name": "call_tool",
            "action_arguments": {
                "tool_name": "dog_health_search",
                "query": "dog health care",
            },
        },
    ]
    unchanged_progress_snapshot = {
        "current_phase": "research",
        "completed_steps": [
            "parse_user_question",
        ],
        "collected_chunk_ids": [],
        "remaining_goals": [
            "retrieve_dog_documents",
            "generate_final_answer",
        ],
        "final_answer_ready": False,
    }

    current_state = initial_state
    round_records: list[dict[str, Any]] = []

    for round_number, action_proposal in enumerate(
        action_proposals,
        start=1,
    ):
        before_result = run_stagnation_guard_before_action(
            stagnation_guard_state=current_state,
            action_name=action_proposal["action_name"],
            action_arguments=action_proposal["action_arguments"],
        )
        current_state = before_result["updated_state"]

        round_record: dict[str, Any] = {
            "round_number": round_number,
            "action_label": action_proposal["label"],
            "action_name": action_proposal["action_name"],
            "action_arguments": copy.deepcopy(
                action_proposal["action_arguments"]
            ),
            "before_action_result": copy.deepcopy(before_result),
            "simulated_action_executed": False,
            "after_action_result": None,
        }

        if not before_result["action_can_proceed"]:
            round_records.append(round_record)
            break

        round_record["simulated_action_executed"] = True

        after_result = run_stagnation_guard_after_action(
            stagnation_guard_state=current_state,
            progress_snapshot=unchanged_progress_snapshot,
        )
        current_state = after_result["updated_state"]
        round_record["after_action_result"] = copy.deepcopy(
            after_result
        )
        round_records.append(round_record)

        if not after_result["loop_can_continue"]:
            break

    stopped_after_third_action_by_no_progress = (
        len(round_records) == 3
        and all(
            round_record["simulated_action_executed"]
            for round_record in round_records
        )
        and current_state["status"] == "stopped"
        and current_state["termination_source"] == "no_progress"
        and current_state["termination_reason"]
        == "consecutive_no_progress"
    )

    return {
        "initial_state": initial_state_snapshot,
        "rounds": round_records,
        "final_state": copy.deepcopy(current_state),
        "stopped_after_third_action_by_no_progress": (
            stopped_after_third_action_by_no_progress
        ),
        "initial_state_unchanged": (
            initial_state == initial_state_snapshot
        ),
    }


if __name__ == "__main__":
    day18_demo_report = {
        # "repeated_action_pattern_demo": (
        #     run_day18_repetition_detection_demo()
        # ),
        # "invalid_action_demo": (
        #     run_day18_invalid_action_detection_demo()
        # ),
        # "no_progress_demo": (
        #     run_day18_no_progress_detection_demo()
        # ),
        "progress_recovery_demo": (
            run_day18_progress_recovery_demo()
        ),
        "combined_stagnation_guard_demo": (
            run_day18_combined_stagnation_guard_demo()
        ),
    }
    print(
        json.dumps(
            day18_demo_report,
            ensure_ascii=False,
            indent=2,
        )
    )
