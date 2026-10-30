import json
import math
from typing import Any


def build_agent_loop_budget_state(
    *,
    max_iterations: int,
    max_tool_calls: int,
    timeout_seconds: float,
    started_at_monotonic: float,
) -> dict[str, Any]:
    """
    创建一次 Agent Loop 运行使用的统一预算状态。

    功能：
        检查 max_iterations 必须是真正的整数，
        不能使用布尔值冒充整数，
        并且必须大于 0。

        检查 max_tool_calls 必须是真正的整数，
        不能使用布尔值冒充整数，
        并且必须大于或等于 0。

        检查 timeout_seconds 必须是整数或浮点数，
        不能是布尔值，
        并且必须大于 0。

        检查 started_at_monotonic 必须是整数或浮点数，
        不能是布尔值，
        并且必须大于或等于 0。

        根据 started_at_monotonic 和 timeout_seconds
        计算 deadline_monotonic。

        创建统一的 Agent Loop 预算状态，
        保存循环计数、工具尝试计数、次数预算、
        时间预算、生命周期状态和终止原因。

        本函数不读取真实系统时间，
        不调用 time.monotonic，
        由调用方显式传入起始时间，
        以保证函数行为确定并且便于测试。

        当前时间字段只用于一次进程内运行，
        不作为跨进程 Checkpoint 时间恢复方案。

    参数：
        max_iterations:
            本次 Agent Loop 允许的最大循环次数。

            必须是真正的整数并且大于 0。

        max_tool_calls:
            本次 Agent Loop 允许的最大工具执行尝试次数。

            必须是真正的整数并且大于或等于 0。
            0 表示本次运行禁止工具执行。

        timeout_seconds:
            本次 Agent Loop 允许持续运行的最大秒数。

            可以是整数或浮点数，
            但不能是布尔值，并且必须大于 0。

        started_at_monotonic:
            本次 Agent Loop 开始时的单调时间值。

            当前学习版由调用方传入，
            必须是整数或浮点数，
            但不能是布尔值，并且不能小于 0。

    返回值：
        dict[str, Any]:
            新创建的统一预算状态，格式如下：

            {
                "status": "running",
                "current_iteration": 0,
                "tool_call_count": 0,
                "max_iterations": 5,
                "max_tool_calls": 3,
                "timeout_seconds": 30.0,
                "started_at_monotonic": 100.0,
                "deadline_monotonic": 130.0,
                "termination_reason": None,
            }

            status:
                当前 Agent Loop 生命周期状态。
                初始值固定为 running。

            current_iteration:
                已经获准并进入执行阶段的非终止动作次数。
                初始值为 0。

            tool_call_count:
                已经获准并预占预算的工具执行尝试次数。
                初始值为 0，不代表工具成功次数。

            max_iterations:
                最大循环次数预算。

            max_tool_calls:
                最大工具执行尝试次数预算。

            timeout_seconds:
                最大运行时长，统一保存为 float。

            started_at_monotonic:
                运行开始时的单调时间，统一保存为 float。

            deadline_monotonic:
                单调时间截止点，等于
                started_at_monotonic + timeout_seconds。

            termination_reason:
                Agent Loop 的终止原因。
                初始运行状态下固定为 None。

    异常：
        TypeError:
            当次数预算不是真正的整数，
            或时间参数不是有效数字类型时抛出。

        ValueError:
            当 max_iterations 小于等于 0，
            max_tool_calls 小于 0，
            timeout_seconds 小于等于 0，
            或 started_at_monotonic 小于 0 时抛出。
    """

    if type(max_iterations) is not int:
        raise TypeError("max_iterations 必须是真正的整数")
    if max_iterations <= 0:
        raise ValueError("max_iterations 必须大于 0")

    if type(max_tool_calls) is not int:
        raise TypeError("max_tool_calls 必须是真正的整数")
    if max_tool_calls < 0:
        raise ValueError("max_tool_calls 必须大于或等于 0")

    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds,
        (int, float),
    ):
        raise TypeError("timeout_seconds 必须是整数或浮点数")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds 必须大于 0")

    if isinstance(started_at_monotonic, bool) or not isinstance(
        started_at_monotonic,
        (int, float),
    ):
        raise TypeError("started_at_monotonic 必须是整数或浮点数")
    if started_at_monotonic < 0:
        raise ValueError("started_at_monotonic 不能小于 0")

    normalized_timeout_seconds = float(timeout_seconds)
    normalized_started_at_monotonic = float(started_at_monotonic)
    deadline_monotonic = (
        normalized_started_at_monotonic + normalized_timeout_seconds
    )

    budget_state = {
        "status": "running",
        "current_iteration": 0,
        "tool_call_count": 0,
        "max_iterations": max_iterations,
        "max_tool_calls": max_tool_calls,
        "timeout_seconds": normalized_timeout_seconds,
        "started_at_monotonic": normalized_started_at_monotonic,
        "deadline_monotonic": deadline_monotonic,
        "termination_reason": None,
    }
    return validate_agent_loop_budget_state(budget_state)


def validate_agent_loop_budget_state(
    budget_state: dict[str, Any],
) -> dict[str, Any]:
    """
    校验 Agent Loop 统一预算状态并返回规范化副本。

    功能：
        检查预算状态是否包含生命周期、次数预算、
        时间预算和终止原因所需的全部字段。

        检查 status 必须是字符串，
        并且只能是 running、finished 或 stopped。

        检查 current_iteration、tool_call_count、
        max_iterations 和 max_tool_calls
        必须是真正的整数，不能使用布尔值冒充整数。

        检查当前计数和最大预算的取值范围，
        并确保当前计数没有超过对应预算。

        检查 timeout_seconds、started_at_monotonic
        和 deadline_monotonic 必须是有限数字，
        拒绝布尔值、NaN 和正负无穷。

        检查 timeout_seconds 必须大于 0，
        started_at_monotonic 必须大于或等于 0。

        检查 deadline_monotonic 必须与
        started_at_monotonic + timeout_seconds 一致。

        检查生命周期状态和终止原因是否一致：
        running 状态的 termination_reason 必须为 None；
        finished 或 stopped 状态必须具有非空字符串原因。

        校验通过后返回状态副本，
        并将三个时间字段统一规范化为 float。

        本函数不修改传入的 budget_state。

    参数：
        budget_state:
            需要校验的 Agent Loop 统一预算状态。

            必须包含：
            - status
            - current_iteration
            - tool_call_count
            - max_iterations
            - max_tool_calls
            - timeout_seconds
            - started_at_monotonic
            - deadline_monotonic
            - termination_reason

    返回值：
        dict[str, Any]:
            校验通过后的预算状态副本。

            返回副本中的 timeout_seconds、
            started_at_monotonic 和 deadline_monotonic
            均为 float。

    异常：
        TypeError:
            当状态、次数或时间字段类型错误时抛出。

        ValueError:
            当缺少必需字段、数值范围非法、
            计数超过预算、时间不是有限值、
            截止时刻计算不一致，
            或生命周期与终止原因不一致时抛出。
    """
    required_fields = {
        "status",
        "current_iteration",
        "tool_call_count",
        "max_iterations",
        "max_tool_calls",
        "timeout_seconds",
        "started_at_monotonic",
        "deadline_monotonic",
        "termination_reason",
    }
    missing_fields = required_fields - set(budget_state)
    if missing_fields:
        raise ValueError(f"预算状态缺少字段：{sorted(missing_fields)}")

    status = budget_state["status"]
    if not isinstance(status, str):
        raise TypeError("status 必须是字符串")
    if status not in {"running", "finished", "stopped"}:
        raise ValueError("status 必须是 running、finished 或 stopped")

    integer_field_names = (
        "current_iteration",
        "tool_call_count",
        "max_iterations",
        "max_tool_calls",
    )
    for field_name in integer_field_names:
        if type(budget_state[field_name]) is not int:
            raise TypeError(f"{field_name} 必须是真正的整数")

    current_iteration = budget_state["current_iteration"]
    tool_call_count = budget_state["tool_call_count"]
    max_iterations = budget_state["max_iterations"]
    max_tool_calls = budget_state["max_tool_calls"]

    if current_iteration < 0:
        raise ValueError("current_iteration 不能小于 0")
    if tool_call_count < 0:
        raise ValueError("tool_call_count 不能小于 0")
    if max_iterations <= 0:
        raise ValueError("max_iterations 必须大于 0")
    if max_tool_calls < 0:
        raise ValueError("max_tool_calls 必须大于或等于 0")
    if current_iteration > max_iterations:
        raise ValueError("current_iteration 不能超过 max_iterations")
    if tool_call_count > max_tool_calls:
        raise ValueError("tool_call_count 不能超过 max_tool_calls")

    time_field_names = (
        "timeout_seconds",
        "started_at_monotonic",
        "deadline_monotonic",
    )
    normalized_time_values: dict[str, float] = {}
    for field_name in time_field_names:
        field_value = budget_state[field_name]
        if isinstance(field_value, bool) or not isinstance(
            field_value,
            (int, float),
        ):
            raise TypeError(f"{field_name} 必须是整数或浮点数")

        normalized_value = float(field_value)
        if not math.isfinite(normalized_value):
            raise ValueError(f"{field_name} 必须是有限数字")
        normalized_time_values[field_name] = normalized_value

    timeout_seconds = normalized_time_values["timeout_seconds"]
    started_at_monotonic = normalized_time_values[
        "started_at_monotonic"
    ]
    deadline_monotonic = normalized_time_values["deadline_monotonic"]

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds 必须大于 0")
    if started_at_monotonic < 0:
        raise ValueError("started_at_monotonic 不能小于 0")

    expected_deadline_monotonic = (
        started_at_monotonic + timeout_seconds
    )
    if not math.isclose(
        deadline_monotonic,
        expected_deadline_monotonic,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "deadline_monotonic 必须等于 "
            "started_at_monotonic + timeout_seconds"
        )

    termination_reason = budget_state["termination_reason"]
    if status == "running":
        if termination_reason is not None:
            raise ValueError(
                "running 状态的 termination_reason 必须为 None"
            )
    else:
        if not isinstance(termination_reason, str):
            raise TypeError(
                "finished 或 stopped 状态的 "
                "termination_reason 必须是字符串"
            )
        if not termination_reason.strip():
            raise ValueError(
                "finished 或 stopped 状态必须具有非空终止原因"
            )

    validated_budget_state = dict(budget_state)
    validated_budget_state.update(normalized_time_values)
    return validated_budget_state


def evaluate_agent_loop_budget_action(
    *,
    budget_state: dict[str, Any],
    requested_action: str,
    now_monotonic: float,
) -> dict[str, Any]:
    """
    根据统一预算状态和当前单调时间评估动作是否允许执行。

    功能：
        首先调用 validate_agent_loop_budget_state，
        校验并取得预算状态副本。

        检查 requested_action 必须是字符串，
        清理首尾空格并转换为小写后，
        必须属于 reason、call_tool 或 finish。

        检查 now_monotonic 必须是有限的整数或浮点数，
        不能是布尔值、NaN 或正负无穷。

        检查 now_monotonic 不能早于
        started_at_monotonic，避免混用不同单调时钟来源。

        计算当前剩余时间 remaining_seconds。
        剩余时间最小为 0，不返回负数。

        如果状态已经不是 running，
        拒绝所有新动作并返回原有终止原因。

        如果状态仍为 running，并且请求 finish，
        优先允许正常结束。
        finish 不产生新的推理或工具副作用，
        因此不受剩余预算限制。

        如果当前时间已经达到或超过截止时刻，
        拒绝非 finish 动作，终止原因为
        time_budget_exhausted。

        如果当前循环次数已经达到最大循环次数，
        拒绝非 finish 动作，终止原因为
        iteration_budget_exhausted。

        如果请求 call_tool，并且工具尝试次数
        已经达到最大工具次数，
        拒绝动作，终止原因为
        tool_budget_exhausted。

        其他情况下允许 requested_action 继续执行。

        本函数只做预算准入判断，
        不增加计数、不修改状态、不调用模型，
        也不执行真实工具。

    参数：
        budget_state:
            当前 Agent Loop 统一预算状态。

        requested_action:
            模型或上游决策器提出的动作。

            当前支持：
            - reason
            - call_tool
            - finish

        now_monotonic:
            评估动作时的当前单调时间。

            当前学习版由调用方传入，
            便于确定性演示和测试。

    返回值：
        dict[str, Any]:
            预算动作准入结果，格式如下：

            {
                "requested_action": "call_tool",
                "allowed": True,
                "effective_action": "call_tool",
                "termination_reason": None,
                "now_monotonic": 105.0,
                "remaining_seconds": 25.0,
            }

            requested_action:
                清理并规范化后的原请求动作。

            allowed:
                Harness 是否允许本次动作。

            effective_action:
                实际应该执行的动作。
                被预算拒绝时固定为 stop。

            termination_reason:
                被拒绝或已经终止时的稳定终止原因。
                动作获准时为 None。

            now_monotonic:
                规范化为 float 的本次评估时间。

            remaining_seconds:
                距离 Deadline 的剩余秒数。
                最小值为 0.0。

    异常：
        TypeError:
            当动作或当前时间类型错误时抛出。

        ValueError:
            当动作名称不受支持、当前时间不是有限值、
            当前时间早于开始时间，
            或预算状态内容不合法时抛出。
    """
    validated_budget_state = validate_agent_loop_budget_state(
        budget_state
    )

    if not isinstance(requested_action, str):
        raise TypeError("requested_action 必须是字符串")
    normalized_action = requested_action.strip().lower()
    supported_actions = {"reason", "call_tool", "finish"}
    if normalized_action not in supported_actions:
        raise ValueError(
            "requested_action 必须是 reason、call_tool 或 finish"
        )

    if isinstance(now_monotonic, bool) or not isinstance(
        now_monotonic,
        (int, float),
    ):
        raise TypeError("now_monotonic 必须是整数或浮点数")
    normalized_now_monotonic = float(now_monotonic)
    if not math.isfinite(normalized_now_monotonic):
        raise ValueError("now_monotonic 必须是有限数字")

    started_at_monotonic = validated_budget_state[
        "started_at_monotonic"
    ]
    if normalized_now_monotonic < started_at_monotonic:
        raise ValueError(
            "now_monotonic 不能早于 started_at_monotonic"
        )

    deadline_monotonic = validated_budget_state[
        "deadline_monotonic"
    ]
    remaining_seconds = max(
        0.0,
        deadline_monotonic - normalized_now_monotonic,
    )

    common_result = {
        "requested_action": normalized_action,
        "now_monotonic": normalized_now_monotonic,
        "remaining_seconds": remaining_seconds,
    }

    if validated_budget_state["status"] != "running":
        return {
            **common_result,
            "allowed": False,
            "effective_action": "stop",
            "termination_reason": validated_budget_state[
                "termination_reason"
            ],
        }

    if normalized_action == "finish":
        return {
            **common_result,
            "allowed": True,
            "effective_action": "finish",
            "termination_reason": None,
        }

    if normalized_now_monotonic >= deadline_monotonic:
        return {
            **common_result,
            "allowed": False,
            "effective_action": "stop",
            "termination_reason": "time_budget_exhausted",
        }

    if (
        validated_budget_state["current_iteration"]
        >= validated_budget_state["max_iterations"]
    ):
        return {
            **common_result,
            "allowed": False,
            "effective_action": "stop",
            "termination_reason": "iteration_budget_exhausted",
        }

    if (
        normalized_action == "call_tool"
        and validated_budget_state["tool_call_count"]
        >= validated_budget_state["max_tool_calls"]
    ):
        return {
            **common_result,
            "allowed": False,
            "effective_action": "stop",
            "termination_reason": "tool_budget_exhausted",
        }

    return {
        **common_result,
        "allowed": True,
        "effective_action": normalized_action,
        "termination_reason": None,
    }


def apply_agent_loop_budget_decision(
    *,
    budget_state: dict[str, Any],
    action_decision: dict[str, Any],
) -> dict[str, Any]:
    """
    根据预算动作准入结果生成更新后的 Agent Loop 预算状态。

    功能：
        首先调用 validate_agent_loop_budget_state，
        校验并取得预算状态副本。

        检查 action_decision 必须是字典，
        并且包含 requested_action、allowed、
        effective_action、termination_reason、
        now_monotonic 和 remaining_seconds。

        使用 action_decision 中的 requested_action
        和 now_monotonic，再次调用
        evaluate_agent_loop_budget_action，
        重新计算当前预算状态下的权威准入结果。

        检查传入的 action_decision 与重新计算的
        权威结果是否一致，拒绝伪造
        或与当前预算状态不匹配的动作决定。

        如果预算状态已经是 finished 或 stopped，
        保持原终态、终止原因和计数不变。

        如果 running 状态下的动作被预算拒绝，
        将状态更新为 stopped，
        保存稳定的预算耗尽原因，
        不增加循环次数或工具尝试次数。

        如果 finish 获准，
        将状态更新为 finished，
        将终止原因更新为 completed，
        不增加任何计数。

        如果 reason 获准，
        将 current_iteration 增加 1。

        如果 call_tool 获准，
        将 current_iteration 和 tool_call_count
        分别增加 1。

        call_tool 获准后增加的 tool_call_count
        表示预占了一次工具执行尝试预算，
        不表示工具已经成功执行。

        最后再次校验并返回更新后的预算状态副本。

        本函数不修改传入的 budget_state，
        不调用模型，也不执行真实工具。

    参数：
        budget_state:
            当前 Agent Loop 统一预算状态。

        action_decision:
            evaluate_agent_loop_budget_action
            返回的预算动作准入结果。

            必须包含：
            - requested_action
            - allowed
            - effective_action
            - termination_reason
            - now_monotonic
            - remaining_seconds

    返回值：
        dict[str, Any]:
            应用动作决定后的统一预算状态副本。

            reason 获准时，循环次数增加 1。

            call_tool 获准时，循环次数和
            工具执行尝试次数分别增加 1。

            finish 获准时进入 finished，
            终止原因变成 completed。

            预算拒绝时进入 stopped，
            计数保持不变并保存预算耗尽原因。

    异常：
        TypeError:
            当 action_decision 不是字典，
            或重新评估时发现动作、时间字段类型错误时抛出。

        ValueError:
            当 action_decision 缺少字段，
            与重新计算的权威决定不一致，
            或预算状态内容不合法时抛出。
    """
    validated_budget_state = validate_agent_loop_budget_state(
        budget_state
    )

    if not isinstance(action_decision, dict):
        raise TypeError("action_decision 必须是字典")

    required_decision_fields = {
        "requested_action",
        "allowed",
        "effective_action",
        "termination_reason",
        "now_monotonic",
        "remaining_seconds",
    }
    missing_decision_fields = (
        required_decision_fields - set(action_decision)
    )
    if missing_decision_fields:
        raise ValueError(
            "action_decision 缺少字段："
            f"{sorted(missing_decision_fields)}"
        )

    if type(action_decision["allowed"]) is not bool:
        raise TypeError("action_decision.allowed 必须是真正的布尔值")
    if not isinstance(action_decision["effective_action"], str):
        raise TypeError(
            "action_decision.effective_action 必须是字符串"
        )

    decision_remaining_seconds = action_decision[
        "remaining_seconds"
    ]
    if isinstance(decision_remaining_seconds, bool) or not isinstance(
        decision_remaining_seconds,
        (int, float),
    ):
        raise TypeError(
            "action_decision.remaining_seconds "
            "必须是整数或浮点数"
        )
    if not math.isfinite(float(decision_remaining_seconds)):
        raise ValueError(
            "action_decision.remaining_seconds 必须是有限数字"
        )

    expected_decision = evaluate_agent_loop_budget_action(
        budget_state=validated_budget_state,
        requested_action=action_decision["requested_action"],
        now_monotonic=action_decision["now_monotonic"],
    )
    actual_decision = {
        field_name: action_decision[field_name]
        for field_name in required_decision_fields
    }
    if actual_decision != expected_decision:
        raise ValueError(
            "action_decision 与当前预算状态重新计算的结果不一致"
        )

    if validated_budget_state["status"] != "running":
        return validated_budget_state

    updated_budget_state = dict(validated_budget_state)
    if not expected_decision["allowed"]:
        updated_budget_state["status"] = "stopped"
        updated_budget_state["termination_reason"] = (
            expected_decision["termination_reason"]
        )
    elif expected_decision["effective_action"] == "finish":
        updated_budget_state["status"] = "finished"
        updated_budget_state["termination_reason"] = "completed"
    elif expected_decision["effective_action"] == "reason":
        updated_budget_state["current_iteration"] += 1
    elif expected_decision["effective_action"] == "call_tool":
        updated_budget_state["current_iteration"] += 1
        updated_budget_state["tool_call_count"] += 1

    return validate_agent_loop_budget_state(updated_budget_state)


def run_agent_loop_budget_step(
    *,
    budget_state: dict[str, Any],
    requested_action: str,
    now_monotonic: float,
) -> dict[str, Any]:
    """
    执行一次 Agent Loop 的预算评估与状态迁移步骤。

    功能：
        首先调用 validate_agent_loop_budget_state，
        校验并取得当前预算状态副本。

        使用调用方提供的同一个 now_monotonic，
        调用 evaluate_agent_loop_budget_action，
        生成本次动作的预算准入结果。

        随后立即调用 apply_agent_loop_budget_decision，
        将准入结果应用到同一份已校验预算状态，
        生成更新后的预算状态。

        评估与应用之间不调用模型、不执行工具、
        不读取新的时间，也不进行异步等待，
        避免同一个控制步骤使用两个不同时间样本。

        返回动作准入结果和更新后的预算状态。

        本函数采用原子风格编排，
        但它只是当前进程中的顺序内存操作，
        不等于数据库事务，也不能单独解决
        多线程或多 Worker 并发更新问题。

        本函数不修改传入的 budget_state。

    参数：
        budget_state:
            当前 Agent Loop 统一预算状态。

        requested_action:
            模型或上游决策器提出的动作。

            当前支持：
            - reason
            - call_tool
            - finish

        now_monotonic:
            本次预算控制步骤统一使用的
            当前单调时间样本。

            当前学习版由调用方传入，
            本函数不会再次读取真实时钟。

    返回值：
        dict[str, Any]:
            本次预算控制步骤的结果，格式如下：

            {
                "action_decision": {
                    "requested_action": "call_tool",
                    "allowed": True,
                    "effective_action": "call_tool",
                    "termination_reason": None,
                    "now_monotonic": 105.0,
                    "remaining_seconds": 25.0,
                },
                "updated_budget_state": {
                    "status": "running",
                    "current_iteration": 1,
                    "tool_call_count": 1,
                    "max_iterations": 5,
                    "max_tool_calls": 3,
                    "timeout_seconds": 30.0,
                    "started_at_monotonic": 100.0,
                    "deadline_monotonic": 130.0,
                    "termination_reason": None,
                },
            }

            action_decision:
                当前时间和预算状态下的动作准入结果。

            updated_budget_state:
                应用准入结果后的预算状态副本。

    异常：
        TypeError:
            当预算状态、动作或当前时间字段
            类型错误时，由内部函数抛出。

        ValueError:
            当预算状态、动作、当前时间
            或动作决定不合法时，由内部函数抛出。
    """
    validated_budget_state = validate_agent_loop_budget_state(
        budget_state
    )
    action_decision = evaluate_agent_loop_budget_action(
        budget_state=validated_budget_state,
        requested_action=requested_action,
        now_monotonic=now_monotonic,
    )
    updated_budget_state = apply_agent_loop_budget_decision(
        budget_state=validated_budget_state,
        action_decision=action_decision,
    )
    return {
        "action_decision": action_decision,
        "updated_budget_state": updated_budget_state,
    }


def run_day17_agent_loop_budget_demo() -> dict[str, Any]:
    """
    演示 Agent Loop 次数预算、时间预算和稳定终止优先级。

    功能：
        使用手动传入的单调时间构造三个独立场景，
        不读取真实时钟，也不等待真实时间流逝。

        场景一演示正常控制链路：
        从单调时间 100.0 开始，超时时长为 30 秒，
        依次在 105.0 请求 call_tool、
        在 110.0 请求 reason、
        在 130.0 请求 finish。

        finish 恰好发生在 Deadline，
        由于当前策略优先允许正常结束，
        最终状态应为 finished，终止原因为 completed，
        current_iteration 为 2，tool_call_count 为 1。

        场景二演示时间预算耗尽：
        从单调时间 200.0 开始，超时时长为 10 秒，
        在 Deadline 210.0 请求 reason。

        非 finish 动作在当前时间达到 Deadline 时
        应被拒绝，最终状态应为 stopped，
        终止原因为 time_budget_exhausted，
        被拒绝的动作不增加任何计数。

        场景三演示稳定终止优先级：
        使用 max_iterations=1、max_tool_calls=0，
        从单调时间 300.0 开始，超时时长为 10 秒。

        首先在 301.0 请求一次 reason，
        使循环次数达到最大值 1。

        随后在 Deadline 310.0 请求 call_tool。
        此时循环预算、工具预算和时间预算同时耗尽。

        按照时间预算、循环预算、工具预算的顺序，
        最终终止原因必须稳定为 time_budget_exhausted。

        三个场景分别使用独立状态，
        避免一个场景进入终态后干扰其他场景。

        本函数只演示预算控制逻辑，
        不调用模型、不执行真实工具，
        也不修改正式 Agent 状态。

    参数：
        无。

    返回值：
        dict[str, Any]:
            三个预算场景的演示结果，格式如下：

            {
                "finish_at_deadline_scenario": {
                    "action_decisions": [
                        call_tool 的决定,
                        reason 的决定,
                        finish 的决定,
                    ],
                    "final_budget_state": {...},
                },
                "time_budget_scenario": {
                    "action_decision": {...},
                    "final_budget_state": {...},
                },
                "priority_scenario": {
                    "preparation_action_decision": {...},
                    "blocked_action_decision": {...},
                    "final_budget_state": {...},
                    "time_reason_won": True,
                },
            }

            finish_at_deadline_scenario:
                验证 finish 在恰好达到 Deadline 时
                仍然可以正常结束。

            time_budget_scenario:
                验证非 finish 动作在达到 Deadline 时
                被时间预算拒绝。

            priority_scenario:
                验证多个预算同时耗尽时，
                time_budget_exhausted 具有稳定优先级。
    """
    finish_budget_state = build_agent_loop_budget_state(
        max_iterations=3,
        max_tool_calls=1,
        timeout_seconds=30.0,
        started_at_monotonic=100.0,
    )
    finish_tool_step = run_agent_loop_budget_step(
        budget_state=finish_budget_state,
        requested_action="call_tool",
        now_monotonic=105.0,
    )
    finish_reason_step = run_agent_loop_budget_step(
        budget_state=finish_tool_step["updated_budget_state"],
        requested_action="reason",
        now_monotonic=110.0,
    )
    finish_step = run_agent_loop_budget_step(
        budget_state=finish_reason_step["updated_budget_state"],
        requested_action="finish",
        now_monotonic=130.0,
    )

    time_budget_state = build_agent_loop_budget_state(
        max_iterations=5,
        max_tool_calls=2,
        timeout_seconds=10.0,
        started_at_monotonic=200.0,
    )
    time_blocked_step = run_agent_loop_budget_step(
        budget_state=time_budget_state,
        requested_action="reason",
        now_monotonic=210.0,
    )

    priority_budget_state = build_agent_loop_budget_state(
        max_iterations=1,
        max_tool_calls=0,
        timeout_seconds=10.0,
        started_at_monotonic=300.0,
    )
    priority_preparation_step = run_agent_loop_budget_step(
        budget_state=priority_budget_state,
        requested_action="reason",
        now_monotonic=301.0,
    )
    priority_blocked_step = run_agent_loop_budget_step(
        budget_state=priority_preparation_step[
            "updated_budget_state"
        ],
        requested_action="call_tool",
        now_monotonic=310.0,
    )
    priority_blocked_decision = priority_blocked_step[
        "action_decision"
    ]

    return {
        "finish_at_deadline_scenario": {
            "action_decisions": [
                finish_tool_step["action_decision"],
                finish_reason_step["action_decision"],
                finish_step["action_decision"],
            ],
            "final_budget_state": finish_step[
                "updated_budget_state"
            ],
        },
        "time_budget_scenario": {
            "action_decision": time_blocked_step[
                "action_decision"
            ],
            "final_budget_state": time_blocked_step[
                "updated_budget_state"
            ],
        },
        "priority_scenario": {
            "preparation_action_decision": (
                priority_preparation_step["action_decision"]
            ),
            "blocked_action_decision": priority_blocked_decision,
            "final_budget_state": priority_blocked_step[
                "updated_budget_state"
            ],
            "time_reason_won": (
                priority_blocked_decision["termination_reason"]
                == "time_budget_exhausted"
            ),
        },
    }


def plan_action_execution_timeout(
    *,
    budget_state: dict[str, Any],
    now_monotonic: float,
    configured_action_timeout_seconds: float,
) -> dict[str, Any]:
    """
    根据 Agent Loop 剩余时间规划一次动作的实际执行超时。

    功能：
        首先调用 validate_agent_loop_budget_state，
        校验并取得预算状态副本。

        检查 now_monotonic 必须是有限的整数或浮点数，
        不能是布尔值、NaN 或正负无穷，
        并且不能早于 started_at_monotonic。

        检查 configured_action_timeout_seconds
        必须是有限的整数或浮点数，
        不能是布尔值，并且必须大于 0。

        根据 deadline_monotonic 和 now_monotonic
        计算整个 Agent Loop 的剩余时间。

        如果预算状态已经不是 running，
        本次动作不能开始，返回已有终止原因。

        如果 Agent Loop 剩余时间已经为 0，
        本次动作不能开始，返回
        time_budget_exhausted。

        如果仍有剩余时间，
        使用配置动作超时和 Loop 剩余时间中的较小值，
        作为本次动作真正应该使用的执行超时。

        这样可以保证单次 LLM 或工具执行
        不会被允许越过整个 Agent Loop 的 Deadline。

        本函数只生成超时计划，
        不调用 LLM、不执行工具，
        也不真正取消正在运行的动作。

    参数：
        budget_state:
            当前 Agent Loop 统一预算状态。

        now_monotonic:
            准备启动本次动作时重新采样的
            当前单调时间。

            该时间可能晚于预算准入时使用的时间，
            因此必须在真正分发动作之前重新采样。

        configured_action_timeout_seconds:
            当前 LLM 或工具自身配置的最大执行秒数。

            例如工具配置允许最多执行 10 秒，
            但 Agent Loop 只剩 3 秒，
            本次有效执行超时只能是 3 秒。

    返回值：
        dict[str, Any]:
            本次动作的执行超时计划，格式如下：

            {
                "can_start": True,
                "configured_action_timeout_seconds": 10.0,
                "remaining_loop_seconds": 3.0,
                "effective_timeout_seconds": 3.0,
                "limited_by_loop_deadline": True,
                "termination_reason": None,
            }

            can_start:
                当前状态和剩余时间是否允许启动动作。

            configured_action_timeout_seconds:
                规范化后的动作自身配置超时。

            remaining_loop_seconds:
                当前 Agent Loop 距离 Deadline
                剩余的秒数，最小为 0.0。

            effective_timeout_seconds:
                本次动作真正应该使用的执行超时。

                动作不能开始时为 0.0；
                可以开始时等于配置动作超时
                与 Loop 剩余时间中的较小值。

            limited_by_loop_deadline:
                本次执行超时是否被整个 Loop Deadline
                缩短。

            termination_reason:
                不能启动时的原因。
                可以启动时为 None。

    异常：
        TypeError:
            当当前时间或动作超时配置类型错误时抛出。

        ValueError:
            当时间不是有限值、动作超时小于等于 0、
            当前时间早于开始时间，
            或预算状态不合法时抛出。
    """
    validated_budget_state = validate_agent_loop_budget_state(
        budget_state
    )

    if isinstance(now_monotonic, bool) or not isinstance(
        now_monotonic,
        (int, float),
    ):
        raise TypeError("now_monotonic 必须是整数或浮点数")
    normalized_now_monotonic = float(now_monotonic)
    if not math.isfinite(normalized_now_monotonic):
        raise ValueError("now_monotonic 必须是有限数字")

    started_at_monotonic = validated_budget_state[
        "started_at_monotonic"
    ]
    if normalized_now_monotonic < started_at_monotonic:
        raise ValueError(
            "now_monotonic 不能早于 started_at_monotonic"
        )

    if isinstance(configured_action_timeout_seconds, bool) or not (
        isinstance(configured_action_timeout_seconds, (int, float))
    ):
        raise TypeError(
            "configured_action_timeout_seconds "
            "必须是整数或浮点数"
        )
    normalized_action_timeout_seconds = float(
        configured_action_timeout_seconds
    )
    if not math.isfinite(normalized_action_timeout_seconds):
        raise ValueError(
            "configured_action_timeout_seconds 必须是有限数字"
        )
    if normalized_action_timeout_seconds <= 0:
        raise ValueError(
            "configured_action_timeout_seconds 必须大于 0"
        )

    deadline_monotonic = validated_budget_state[
        "deadline_monotonic"
    ]
    remaining_loop_seconds = max(
        0.0,
        deadline_monotonic - normalized_now_monotonic,
    )
    common_plan = {
        "configured_action_timeout_seconds": (
            normalized_action_timeout_seconds
        ),
        "remaining_loop_seconds": remaining_loop_seconds,
    }

    if validated_budget_state["status"] != "running":
        return {
            **common_plan,
            "can_start": False,
            "effective_timeout_seconds": 0.0,
            "limited_by_loop_deadline": False,
            "termination_reason": validated_budget_state[
                "termination_reason"
            ],
        }

    if normalized_now_monotonic >= deadline_monotonic:
        return {
            **common_plan,
            "can_start": False,
            "effective_timeout_seconds": 0.0,
            "limited_by_loop_deadline": True,
            "termination_reason": "time_budget_exhausted",
        }

    effective_timeout_seconds = min(
        normalized_action_timeout_seconds,
        remaining_loop_seconds,
    )
    return {
        **common_plan,
        "can_start": True,
        "effective_timeout_seconds": effective_timeout_seconds,
        "limited_by_loop_deadline": (
            remaining_loop_seconds
            < normalized_action_timeout_seconds
        ),
        "termination_reason": None,
    }


def run_day17_action_execution_timeout_demo() -> dict[str, Any]:
    """
    演示动作自身超时与 Agent Loop Deadline 的组合规则。

    功能：
        创建一个从单调时间 100.0 开始、
        超时时长为 30 秒的 Agent Loop 预算状态。

        该状态的 deadline_monotonic 为 130.0。

        场景一在单调时间 105.0 规划动作执行超时。
        此时 Loop 剩余 25 秒，动作自身配置超时为 10 秒。
        动作可以开始，有效超时应为 10 秒，
        不需要由 Loop Deadline 缩短。

        场景二在单调时间 127.0 规划动作执行超时。
        此时 Loop 只剩 3 秒，动作自身配置超时仍为 10 秒。
        动作可以开始，但有效超时必须缩短为 3 秒，
        limited_by_loop_deadline 应为 True。

        场景三在单调时间 130.0 规划动作执行超时。
        此时已经达到 Deadline，动作不能启动，
        有效超时为 0，终止原因为
        time_budget_exhausted。

        三个场景复用同一份预算状态，
        因为 plan_action_execution_timeout
        只读取状态并返回计划，不修改计数或生命周期。

        本函数只演示超时计划计算，
        不执行真实 LLM 或工具，
        也不真正等待或取消任务。

    参数：
        无。

    返回值：
        dict[str, Any]:
            三种动作执行超时计划，格式如下：

            {
                "action_timeout_limited_scenario": {
                    "can_start": True,
                    "configured_action_timeout_seconds": 10.0,
                    "remaining_loop_seconds": 25.0,
                    "effective_timeout_seconds": 10.0,
                    "limited_by_loop_deadline": False,
                    "termination_reason": None,
                },
                "loop_deadline_limited_scenario": {
                    "can_start": True,
                    "configured_action_timeout_seconds": 10.0,
                    "remaining_loop_seconds": 3.0,
                    "effective_timeout_seconds": 3.0,
                    "limited_by_loop_deadline": True,
                    "termination_reason": None,
                },
                "deadline_exhausted_scenario": {
                    "can_start": False,
                    "configured_action_timeout_seconds": 10.0,
                    "remaining_loop_seconds": 0.0,
                    "effective_timeout_seconds": 0.0,
                    "limited_by_loop_deadline": True,
                    "termination_reason": "time_budget_exhausted",
                },
            }
    """
    budget_state = build_agent_loop_budget_state(
        max_iterations=5,
        max_tool_calls=3,
        timeout_seconds=30.0,
        started_at_monotonic=100.0,
    )

    action_timeout_limited_scenario = plan_action_execution_timeout(
        budget_state=budget_state,
        now_monotonic=105.0,
        configured_action_timeout_seconds=10.0,
    )
    loop_deadline_limited_scenario = plan_action_execution_timeout(
        budget_state=budget_state,
        now_monotonic=127.0,
        configured_action_timeout_seconds=10.0,
    )
    deadline_exhausted_scenario = plan_action_execution_timeout(
        budget_state=budget_state,
        now_monotonic=130.0,
        configured_action_timeout_seconds=10.0,
    )

    return {
        "action_timeout_limited_scenario": (
            action_timeout_limited_scenario
        ),
        "loop_deadline_limited_scenario": (
            loop_deadline_limited_scenario
        ),
        "deadline_exhausted_scenario": deadline_exhausted_scenario,
    }


if __name__ == "__main__":
    demo_output = {
        "agent_loop_budget_demo": run_day17_agent_loop_budget_demo(),
        "action_execution_timeout_demo": (
            run_day17_action_execution_timeout_demo()
        ),
    }
    print(
        json.dumps(
            demo_output,
            ensure_ascii=False,
            indent=2,
        )
    )
