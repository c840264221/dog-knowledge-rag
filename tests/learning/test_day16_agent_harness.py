import copy
from typing import Any

def build_agent_harness_control_state(
    *,
    max_iterations: int,
    max_tool_calls: int,
) -> dict[str, Any]:
    """
    创建一次 Agent Harness 运行所需的初始控制状态。

    功能：
        检查最大循环次数 max_iterations 必须大于 0。

        检查最大工具调用次数 max_tool_calls
        必须大于或等于 0。
        其中 0 表示当前运行禁止调用任何工具。

        创建 Harness 专用的控制状态，
        保存运行状态、循环计数、工具调用计数、
        循环预算、工具预算和终止原因。

        本函数不保存用户输入、工具结果或最终答案，
        因为这些业务数据应由 DogState 管理。

        本函数不保存 Trace、Timeline 或 Metrics，
        因为这些可观测数据应由项目现有 RuntimeContext 管理。

        本函数不创建 LLM、Retriever 或工具执行器，
        因为这些依赖应由 RuntimeContainer 和 Provider 提供。

        当前函数只用于 Day16 独立学习，
        不修改正式 DogState 或 RuntimeContext。

    参数：
        max_iterations:
            本次 Agent 运行允许的最大循环次数。
            必须大于 0。

        max_tool_calls:
            本次 Agent 运行允许的最大工具调用次数。
            必须大于或等于 0。

            0 表示本次运行不允许执行工具，
            但仍然可以执行不依赖工具的推理。

    返回值：
        dict[str, Any]:
            新创建的 Harness 控制状态，格式如下：

            {
                "status": "running",
                "current_iteration": 0,
                "tool_call_count": 0,
                "max_iterations": 5,
                "max_tool_calls": 3,
                "termination_reason": None,
            }

            status:
                当前 Harness 生命周期状态。
                初始值固定为 running。

            current_iteration:
                当前已经完成的循环次数。
                初始值为 0。

            tool_call_count:
                当前已经执行的工具调用次数。
                初始值为 0。

            max_iterations:
                最大循环次数预算。

            max_tool_calls:
                最大工具调用次数预算。

            termination_reason:
                Harness 终止原因。
                初始运行阶段尚未终止，因此为 None。

    异常：
        ValueError:
            当 max_iterations 小于等于 0，
            或 max_tool_calls 小于 0 时抛出。
    """
    if max_iterations <= 0:
        raise ValueError("max_iterations必须大于0")

    if max_tool_calls < 0:
        raise ValueError("max_tool_calls必须大于等于0")

    control_state ={
                "status": "running",
                "current_iteration": 0,
                "tool_call_count": 0,
                "max_iterations": max_iterations,
                "max_tool_calls": max_tool_calls,
                "termination_reason": None,
            }
    return control_state

def attach_harness_control_to_graph_state(
    *,
    graph_state: dict[str, Any],
    control_state: dict[str, Any],
) -> dict[str, Any]:
    """
    将新任务的 Harness 控制状态附加到图状态副本中。

    功能：
        检查 graph_state 中不能已经存在
        agent_harness_control 字段。

        如果该字段已经存在，说明当前状态可能来自
        Checkpoint 恢复，不能使用新的初始控制状态覆盖，
        否则可能重置循环次数和工具调用次数。

        检查 control_state 是否包含 Harness 控制状态
        所需的全部字段。

        创建新的图状态字典，
        将 control_state 的副本保存到
        agent_harness_control 字段中。

        本函数不修改传入的 graph_state，
        也不修改传入的 control_state。

        当前函数只操作普通字典，
        用于模拟未来 DogState 与 Checkpoint 的接线，
        不修改正式 DogState 定义。

    参数：
        graph_state:
            新任务的初始图状态。

            可以包含 question、tool_calls、
            tool_results、final_answer 等业务字段。

            当前函数要求其中尚不存在
            agent_harness_control。

        control_state:
            build_agent_harness_control_state
            创建的 Harness 初始控制状态。

            必须包含：
            - status
            - current_iteration
            - tool_call_count
            - max_iterations
            - max_tool_calls
            - termination_reason

    返回值：
        dict[str, Any]:
            附加 Harness 控制状态后的新图状态。

            示例：

            {
                "question": "查询北京天气",
                "tool_calls": [],
                "tool_results": [],
                "final_answer": "",
                "agent_harness_control": {
                    "status": "running",
                    "current_iteration": 0,
                    "tool_call_count": 0,
                    "max_iterations": 5,
                    "max_tool_calls": 3,
                    "termination_reason": None,
                },
            }

    异常：
        ValueError:
            当 graph_state 已经包含 agent_harness_control，
            或 control_state 缺少必需字段时抛出。
    """
    if "agent_harness_control" in graph_state:
        raise ValueError(
            "graph_state 已存在 agent_harness_control，"
            "拒绝使用新的初始控制状态覆盖可能从 Checkpoint 恢复的状态"
        )

    required_control_fields = {
        "status",
        "current_iteration",
        "tool_call_count",
        "max_iterations",
        "max_tool_calls",
        "termination_reason",
    }

    missing_control_fields = (
            required_control_fields - set(control_state)
    )

    if missing_control_fields:
        raise ValueError(f"缺失字段：{sorted(missing_control_fields)}")

    updated_graph_state = dict(graph_state)
    updated_graph_state["agent_harness_control"] = dict(
        control_state
    )
    return updated_graph_state


def validate_harness_control_state(
    control_state: dict[str, Any],
) -> dict[str, Any]:
    """
    校验从内存或 Checkpoint 读取的 Harness 控制状态。

    功能：
        检查控制状态是否包含全部必需字段。

        检查 status 必须是以下状态之一：
        running、finished、stopped。

        检查 current_iteration、tool_call_count、
        max_iterations 和 max_tool_calls 都必须是真正的整数，
        不能使用布尔值冒充整数。

        检查当前循环次数和工具调用次数不能小于 0。

        检查 max_iterations 必须大于 0，
        max_tool_calls 必须大于或等于 0。

        检查当前计数不能超过对应最大预算。

        检查运行状态与终止原因是否一致：
        running 状态的 termination_reason 必须为 None；
        finished 或 stopped 状态必须具有非空终止原因。

        校验通过后返回控制状态的副本，
        不修改传入的原始字典。

        当前函数只校验学习版普通字典，
        不读取真实 Checkpoint，
        也不修改正式 DogState。

    参数：
        control_state:
            需要校验的 Harness 控制状态。

            必须包含：
            - status
            - current_iteration
            - tool_call_count
            - max_iterations
            - max_tool_calls
            - termination_reason

    返回值：
        dict[str, Any]:
            校验通过后的控制状态副本。

            返回副本的目的是避免调用方误以为
            可以安全修改传入的原始 Checkpoint 数据。

    异常：
        ValueError:
            当缺少必需字段；
            status 不在允许范围；
            计数或预算超出合法范围；
            当前计数超过最大预算；
            或 status 与 termination_reason 不一致时抛出。

        TypeError:
            当 status 不是字符串；
            计数或预算不是真正的整数；
            或非运行状态的 termination_reason
            不是字符串时抛出。
    """
    required_control_fields = {
        "status",
        "current_iteration",
        "tool_call_count",
        "max_iterations",
        "max_tool_calls",
        "termination_reason",
    }

    missing_control_fields = (
            required_control_fields - set(control_state)
    )

    if missing_control_fields:
        raise ValueError(f"缺失字段：{sorted(missing_control_fields)}")

    status = control_state["status"]
    if not isinstance(status, str):
        raise TypeError("status必须是字符串")

    if status not in ["running", "finished", "stopped"]:
        raise ValueError("status必须是running或finished或stopped")

    integer_field_names = (
        "current_iteration",
        "tool_call_count",
        "max_iterations",
        "max_tool_calls",
    )

    for field_name in integer_field_names:
        if type(control_state[field_name]) is not int:
            raise TypeError(f"{field_name}的值必须是int类型")

    current_iteration = control_state["current_iteration"]
    if current_iteration < 0:
        raise ValueError("current_iteration不能小于0")
    tool_call_count = control_state["tool_call_count"]
    if tool_call_count < 0:
        raise ValueError("tool_call_count不能小于0")

    max_iterations = control_state["max_iterations"]
    if max_iterations <= 0:
        raise ValueError("max_iterations必须大于0")

    max_tool_calls = control_state["max_tool_calls"]
    if max_tool_calls < 0:
        raise ValueError("max_tool_calls必须大于等于0")

    if current_iteration > max_iterations:
        raise ValueError("当前循环次数必须小于等于最大循环次数")

    if tool_call_count > max_tool_calls:
        raise ValueError("调用工具次数必须小于等于最大调用工具次数")

    termination_reason = control_state["termination_reason"]
    if status == "running":
        if termination_reason is not None:
            raise ValueError("状态为running时termination_reason必须是None")
    else:
        if not isinstance(termination_reason, str):
            raise TypeError("termination_reason必须是字符串类型")
        if not termination_reason.strip():
            raise ValueError("termination_reason不能为空")

    validated_control_state = dict(control_state)
    return validated_control_state

def resolve_harness_control_state(
    *,
    graph_state: dict[str, Any],
    max_iterations: int,
    max_tool_calls: int,
) -> dict[str, Any]:
    """
    根据图状态解析新任务或恢复任务应使用的 Harness 控制状态。

    功能：
        检查 graph_state 中是否存在
        agent_harness_control 字段。

        如果该字段不存在，说明当前按照新任务处理，
        调用 build_agent_harness_control_state，
        使用传入的最大循环次数和最大工具调用次数
        创建新的控制状态。

        如果该字段已经存在，说明当前按照恢复任务处理。

        恢复任务中的 agent_harness_control 必须是字典，
        随后调用 validate_harness_control_state
        校验并返回已有控制状态的副本。

        恢复任务不会使用传入的新预算覆盖旧预算，
        避免任务暂停期间配置变化导致预算被重置或放大。

        本函数只返回应该使用的控制状态，
        不负责把新控制状态写回 graph_state，
        也不修改传入的 graph_state。

        当前函数只操作学习版普通字典，
        不读取真实 Checkpoint，
        也不修改正式 DogState。

    参数：
        graph_state:
            当前图状态。

            新任务状态中不包含 agent_harness_control。

            恢复任务状态中应包含合法的
            agent_harness_control 字典。

        max_iterations:
            新任务允许的最大循环次数。

            只在 graph_state 不包含
            agent_harness_control 时使用。

            恢复任务使用已有控制状态中的预算，
            不使用此参数覆盖。

        max_tool_calls:
            新任务允许的最大工具调用次数。

            只在 graph_state 不包含
            agent_harness_control 时使用。

            恢复任务使用已有控制状态中的预算，
            不使用此参数覆盖。

    返回值：
        dict[str, Any]:
            当前任务应该使用的 Harness 控制状态副本。

            新任务返回新创建的初始控制状态。

            恢复任务返回经过校验的已有控制状态副本。

    异常：
        TypeError:
            当 graph_state 已包含 agent_harness_control，
            但该字段不是字典时抛出。

            恢复控制状态内部字段类型错误时，
            由 validate_harness_control_state 抛出。

        ValueError:
            新任务预算不合法时，
            由 build_agent_harness_control_state 抛出。

            恢复控制状态缺少字段、数值非法
            或状态关系不一致时，
            由 validate_harness_control_state 抛出。
    """
    resolved_control_state: dict[str, Any] = {}
    if "agent_harness_control" not in graph_state:
        resolved_control_state = build_agent_harness_control_state(
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
        )

    else:
        raw_control_state = graph_state["agent_harness_control"]

        if not isinstance(raw_control_state, dict):
            raise TypeError("agent_harness_control的值必须是字典")

        resolved_control_state = validate_harness_control_state(
            raw_control_state,
        )
    return resolved_control_state

def evaluate_harness_action_request(
    *,
    control_state: dict[str, Any],
    requested_action: str,
) -> dict[str, Any]:
    """
    根据 Harness 控制状态评估一个动作请求是否允许执行。

    功能：
        首先调用 validate_harness_control_state，
        校验并取得控制状态副本。

        检查 requested_action 必须是字符串，
        清理首尾空格后必须属于：
        reason、call_tool、finish。

        如果当前控制状态已经不是 running，
        拒绝继续执行任何新动作，
        返回已有的 termination_reason。

        如果当前状态仍在运行，并且请求动作为 finish，
        允许进入正常结束流程。
        finish 不消耗新的循环或工具预算。

        如果请求动作不是 finish，
        检查当前循环次数是否已经达到最大循环次数。
        达到上限时拒绝动作，并返回
        iteration_budget_exhausted。

        如果请求动作为 call_tool，
        继续检查工具调用次数是否已经达到工具预算。
        达到上限时拒绝动作，并返回
        tool_budget_exhausted。

        预算允许时，批准原动作继续执行。

        本函数只做动作准入判断，
        不调用模型、不执行工具，
        不增加计数，也不修改控制状态。

    参数：
        control_state:
            当前 Harness 控制状态。

            可以是新创建的状态，
            也可以是从 Checkpoint 恢复并校验的状态。

        requested_action:
            模型或上游决策器提出的动作。

            当前只支持：
            - reason
            - call_tool
            - finish

    返回值：
        dict[str, Any]:
            动作准入判断结果，格式如下：

            允许执行时：

            {
                "requested_action": "call_tool",
                "allowed": True,
                "effective_action": "call_tool",
                "termination_reason": None,
            }

            循环预算耗尽时：

            {
                "requested_action": "reason",
                "allowed": False,
                "effective_action": "stop",
                "termination_reason": "iteration_budget_exhausted",
            }

            已经终止时：

            {
                "requested_action": "reason",
                "allowed": False,
                "effective_action": "stop",
                "termination_reason": 原控制状态中的终止原因,
            }

    异常：
        TypeError:
            当 requested_action 不是字符串，
            或控制状态字段类型错误时抛出。

        ValueError:
            当 requested_action 清理后不属于支持的动作，
            或控制状态内容不合法时抛出。
    """
    validated_control_state = validate_harness_control_state(
        control_state
    )

    if not isinstance(requested_action, str):
        raise TypeError("requested_action必须是字符串类型")

    normalized_action = requested_action.strip().lower()

    if normalized_action not in ["reason","call_tool", "finish"]:
        raise ValueError("requested_action必须是reason或call_tool或finish")

    status = validated_control_state["status"]

    if status != "running":
        return {
            "requested_action": normalized_action,
            "allowed": False,
            "effective_action": "stop",
            "termination_reason": validated_control_state[
                "termination_reason"
            ],
        }
    if normalized_action == "finish":
        return {
            "requested_action": normalized_action,
            "allowed": True,
            "effective_action": "finish",
            "termination_reason": None,
        }

    current_iteration = validated_control_state["current_iteration"]
    max_iterations = validated_control_state["max_iterations"]

    if current_iteration >= max_iterations:
        return {
                "requested_action": normalized_action,
                "allowed": False,
                "effective_action": "stop",
                "termination_reason": "iteration_budget_exhausted",
            }
    if normalized_action == "call_tool":
        tool_call_count = validated_control_state["tool_call_count"]
        max_tool_calls = validated_control_state["max_tool_calls"]
        if tool_call_count >= max_tool_calls:
            return {
                "requested_action": normalized_action,
                "allowed": False,
                "effective_action": "stop",
                "termination_reason": "tool_budget_exhausted",
            }
        else:
            return {
                "requested_action": "call_tool",
                "allowed": True,
                "effective_action": "call_tool",
                "termination_reason": None,
            }
    else:
        return {
                "requested_action": normalized_action,
                "allowed": True,
                "effective_action": normalized_action,
                "termination_reason": None,
            }


def apply_harness_action_decision(
    *,
    control_state: dict[str, Any],
    action_decision: dict[str, Any],
) -> dict[str, Any]:
    """
    根据动作准入结果生成更新后的 Harness 控制状态。

    功能：
        首先调用 validate_harness_control_state，
        校验并取得控制状态副本。

        检查 action_decision 是否包含：
        requested_action、allowed、effective_action
        和 termination_reason 四个必需字段。

        检查 allowed 必须是真正的布尔值。

        如果动作未被允许，并且当前状态仍为 running，
        将 Harness 状态更新为 stopped，
        保存动作准入结果提供的终止原因，
        不增加循环次数和工具调用次数。

        如果动作未被允许，并且当前状态已经是
        finished 或 stopped，
        检查动作决定中的终止原因必须与已有终止原因一致，
        随后保持原终止状态、终止原因和计数不变。
        终止原因不一致时抛出 ValueError。

        如果动作被允许：
        termination_reason 必须为 None，
        effective_action 必须属于：
        reason、call_tool、finish。

        如果 effective_action 为 finish：
        将 Harness 状态更新为 finished，
        将 termination_reason 更新为 completed，
        不增加循环次数和工具调用次数。

        如果 effective_action 为 reason：
        将 current_iteration 增加 1，
        其他计数保持不变。

        如果 effective_action 为 call_tool：
        将 current_iteration 增加 1，
        同时将 tool_call_count 增加 1。

        最后再次调用 validate_harness_control_state，
        校验并返回更新后的控制状态副本。

        本函数不修改传入的 control_state，
        不调用模型，也不执行真实工具。

    参数：
        control_state:
            当前 Harness 控制状态。

            必须是能够通过
            validate_harness_control_state 校验的字典。

        action_decision:
            evaluate_harness_action_request
            返回的动作准入结果。

            必须包含：
            - requested_action
            - allowed
            - effective_action
            - termination_reason

    返回值：
        dict[str, Any]:
            应用动作准入结果之后的新控制状态。

            允许 reason 时示例：

            {
                "status": "running",
                "current_iteration": 1,
                "tool_call_count": 0,
                "max_iterations": 5,
                "max_tool_calls": 3,
                "termination_reason": None,
            }

            允许 call_tool 时示例：

            {
                "status": "running",
                "current_iteration": 1,
                "tool_call_count": 1,
                "max_iterations": 5,
                "max_tool_calls": 3,
                "termination_reason": None,
            }

            允许 finish 时示例：

            {
                "status": "finished",
                "current_iteration": 0,
                "tool_call_count": 0,
                "max_iterations": 5,
                "max_tool_calls": 3,
                "termination_reason": "completed",
            }

    异常：
        TypeError:
            当 allowed 不是真正的布尔值，
            或动作准入结果中的字段类型错误时抛出。

        ValueError:
            当 action_decision 缺少必需字段，
            动作准入结果内部关系不一致，
            或更新后的控制状态不合法时抛出。
    """
    validated_control_state = validate_harness_control_state(control_state)
    required_decision_fields = {
        "requested_action",
        "allowed",
        "effective_action",
        "termination_reason",
    }
    missing_decision_fields = (
            required_decision_fields  - set(action_decision)
    )
    if len(missing_decision_fields) > 0:
        raise ValueError("缺少关键字段")

    allowed = action_decision["allowed"]
    if not isinstance(allowed, bool):
        raise TypeError("allowed 必须是布尔类型")

    requested_action = action_decision["requested_action"]
    if not isinstance(requested_action, str):
        raise TypeError("requested_action 必须是字符串")
    supported_requested_actions = {
        "reason",
        "call_tool",
        "finish",
    }
    if requested_action not in supported_requested_actions:
        raise ValueError("requested_action 必须属于支持的动作集合")


    effective_action = action_decision["effective_action"]
    if not isinstance(effective_action, str):
        raise TypeError("effective_action 必须是字符串")

    decision_termination_reason = action_decision["termination_reason"]

    current_status = validated_control_state["status"]

    current_termination_reason = validated_control_state["termination_reason"]

    updated_control_state = dict(validated_control_state)


    if not allowed:
        if effective_action != "stop":
            raise ValueError("allowed is False 时，effective_action 必须是 stop")
        if not isinstance(decision_termination_reason, str):
            raise TypeError("allowed is False 时，终止原因必须是字符串类型。")
        else:
            if not decision_termination_reason.strip():
                raise ValueError("终止原因必须是非空字符串")
        if current_status != "running":
            if current_termination_reason != decision_termination_reason:
                raise ValueError("current_termination_reason 和decision_termination_reason 不一致")
            else:
                return  validate_harness_control_state(updated_control_state)

        updated_control_state["status"] = "stopped"
        updated_control_state["termination_reason"] = decision_termination_reason
    else:
        if requested_action != effective_action:
            raise ValueError("requested_action和effective_action必须一致")

        if effective_action not in ["reason", "call_tool", "finish"]:
            raise ValueError("allowed is True 时，effective_action 必须属于 reason/call_tool/finish")

        if decision_termination_reason is not None:
            raise ValueError("allowed is True 时，准入结果中的终止原因必须为 None。")

        if effective_action == "finish":
            updated_control_state["status"] = "finished"
            updated_control_state["termination_reason"] = "completed"

        elif effective_action == "reason":
            updated_control_state["current_iteration"] = validated_control_state["current_iteration"] + 1
        elif effective_action == "call_tool":
            updated_control_state["current_iteration"] += 1
            updated_control_state["tool_call_count"] += 1
            updated_control_state["termination_reason"] = None

    return validate_harness_control_state(updated_control_state)

def run_harness_control_step(
    *,
    graph_state: dict[str, Any],
    requested_action: str,
    max_iterations: int,
    max_tool_calls: int,
) -> dict[str, Any]:
    """
    执行一次 Agent Harness 控制步骤。

    功能：
        根据 graph_state 解析当前应该使用的 Harness 控制状态。

        如果 graph_state 不包含 agent_harness_control，
        使用 max_iterations 和 max_tool_calls
        创建新任务的初始控制状态。

        如果 graph_state 已包含 agent_harness_control，
        校验并恢复已有控制状态，
        不使用传入的新预算覆盖恢复状态中的原预算。

        调用 evaluate_harness_action_request，
        判断 requested_action 是否允许执行。

        调用 apply_harness_action_decision，
        根据动作准入结果生成新的 Harness 控制状态。

        创建 graph_state 的浅拷贝，
        将新的控制状态写入
        agent_harness_control 字段。

        返回动作准入结果和更新后的图状态。

        本函数只编排 Harness 控制流程，
        不调用模型、不执行工具，也不生成最终答案。

        本函数不修改传入的 graph_state，
        也不直接修改其中已有的控制状态。

    参数：
        graph_state:
            当前 Agent 图状态。

            新任务可以不包含 agent_harness_control。

            从 Checkpoint 恢复的任务可以包含
            已有的 agent_harness_control。

        requested_action:
            模型或上游决策器提出的动作。

            当前支持：
            - reason
            - call_tool
            - finish

        max_iterations:
            新任务允许的最大循环次数。

            仅当 graph_state 尚未包含
            agent_harness_control 时使用。

        max_tool_calls:
            新任务允许的最大工具调用次数。

            仅当 graph_state 尚未包含
            agent_harness_control 时使用。

    返回值：
        dict[str, Any]:
            本次 Harness 控制步骤的结果，格式如下：

            {
                "action_decision": {
                    "requested_action": "call_tool",
                    "allowed": True,
                    "effective_action": "call_tool",
                    "termination_reason": None,
                },
                "updated_graph_state": {
                    "question": "查询北京天气",
                    "agent_harness_control": {
                        "status": "running",
                        "current_iteration": 1,
                        "tool_call_count": 1,
                        "max_iterations": 5,
                        "max_tool_calls": 3,
                        "termination_reason": None,
                    },
                },
            }

            action_decision:
                Harness 对本次动作的准入判断。

                外层执行器应根据其中的
                effective_action 决定下一步操作。

            updated_graph_state:
                写入最新 Harness 控制状态后的
                图状态副本。

    异常：
        TypeError:
            当控制状态或动作字段类型错误时，
            由内部校验函数抛出。

        ValueError:
            当预算、控制状态、请求动作
            或动作决定不合法时，
            由内部函数抛出。
    """

    resolved_control_state = resolve_harness_control_state(
        graph_state=graph_state,
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
    )
    action_decision = evaluate_harness_action_request(
        requested_action=requested_action,
        control_state=resolved_control_state,
    )
    updated_control_state = apply_harness_action_decision(
        control_state=resolved_control_state,
        action_decision=action_decision,
    )
    updated_graph_state = copy.deepcopy(graph_state)
    updated_graph_state["agent_harness_control"] = updated_control_state
    return {
        "action_decision": action_decision,
        "updated_graph_state": updated_graph_state,
    }

def run_day16_agent_harness_demo() -> dict[str, Any]:
    """
    演示一次由 Agent Harness 控制的多步骤动作序列。

    功能：
        创建一个不包含 agent_harness_control 的初始图状态，
        并保存它的深拷贝，用于检查原始状态是否被修改。

        第一步请求 call_tool，
        使用最大循环次数 3 和最大工具调用次数 1
        创建并更新 Harness 控制状态。

        第二步以上一步返回的 updated_graph_state
        作为输入，请求 reason。

        第三步以上一步返回的 updated_graph_state
        作为输入，请求 finish。

        第四步在任务已经 finished 后再次请求 reason，
        验证新动作被拒绝，并且 Harness 控制状态
        仍然保持 finished，不会倒退为 stopped。

        返回四次动作准入结果、正常结束时的控制状态、
        终态后再次请求动作得到的控制状态，
        以及原始状态和终态是否保持不变的检查结果。

        本函数只演示 Harness 控制流程，
        不调用真实模型，不执行真实工具，
        也不修改正式 DogState。

    参数：
        无。

    返回值：
        dict[str, Any]:
            演示结果，格式如下：

            {
                "action_decisions": [
                    第一次 call_tool 的准入结果,
                    第二次 reason 的准入结果,
                    第三次 finish 的准入结果,
                    终态后再次 reason 的准入结果,
                ],
                "finished_control_state": {
                    "status": "finished",
                    "current_iteration": 2,
                    "tool_call_count": 1,
                    "max_iterations": 3,
                    "max_tool_calls": 1,
                    "termination_reason": "completed",
                },
                "post_finish_control_state": {
                    "status": "finished",
                    "current_iteration": 2,
                    "tool_call_count": 1,
                    "max_iterations": 3,
                    "max_tool_calls": 1,
                    "termination_reason": "completed",
                },
                "initial_graph_state_unchanged": True,
                "terminal_state_preserved": True,
            }

            initial_graph_state_unchanged:
                初始图状态是否仍然等于执行前保存的副本。

            terminal_state_preserved:
                终态后再次请求动作，
                控制状态是否仍然等于正常结束时的控制状态。
    """
    initial_graph_state = {
        "question": "查询北京天气",
        "tool_results": [],
        "final_answer": "",
    }
    original_graph_state = copy.deepcopy(initial_graph_state)

    call_tool_step_result = run_harness_control_step(
        graph_state=initial_graph_state,
        requested_action="call_tool",
        max_iterations=3,
        max_tool_calls=1,
    )

    reason_step_result = run_harness_control_step(
        graph_state=call_tool_step_result["updated_graph_state"],
        requested_action="reason",
        max_iterations=3,
        max_tool_calls=1,
    )

    finish_step_result = run_harness_control_step(
        graph_state=reason_step_result["updated_graph_state"],
        requested_action="finish",
        max_iterations=3,
        max_tool_calls=1,
    )
    finished_graph_state = finish_step_result["updated_graph_state"]
    finished_control_state = finished_graph_state[
        "agent_harness_control"
    ]

    post_finish_step_result = run_harness_control_step(
        graph_state=finished_graph_state,
        requested_action="reason",
        max_iterations=3,
        max_tool_calls=1,
    )
    post_finish_graph_state = post_finish_step_result[
        "updated_graph_state"
    ]
    post_finish_control_state = post_finish_graph_state[
        "agent_harness_control"
    ]

    return {
        "action_decisions": [
            call_tool_step_result["action_decision"],
            reason_step_result["action_decision"],
            finish_step_result["action_decision"],
            post_finish_step_result["action_decision"],
        ],
        "finished_control_state": finished_control_state,
        "post_finish_control_state": post_finish_control_state,
        "initial_graph_state_unchanged": (
            initial_graph_state == original_graph_state
        ),
        "terminal_state_preserved": (
            post_finish_control_state == finished_control_state
        ),
    }


def run_day16_harness_budget_demo() -> dict[str, Any]:
    """
    演示 Agent Harness 对工具预算和循环预算的独立限制。

    功能：
        创建两个互不关联的图状态，分别演示工具调用预算耗尽
        和循环次数预算耗尽。

        工具预算场景使用 max_iterations=3、max_tool_calls=1。
        第一次 call_tool 被允许并消耗一次循环和工具预算；
        第二次 call_tool 因工具预算耗尽而被拒绝，
        但不会为被拒绝的动作增加任何计数。

        循环预算场景使用 max_iterations=1、max_tool_calls=2。
        第一次 reason 被允许并消耗一次循环预算；
        第二次 reason 因循环预算耗尽而被拒绝，
        但不会为被拒绝的动作增加任何计数。

        本函数只演示 Harness 预算控制，
        不调用真实模型，也不执行真实工具。

    参数：
        无。

    返回值：
        dict[str, Any]:
            两种预算耗尽场景的演示结果。

            tool_budget_scenario 包含第一次工具动作的准入结果、
            被工具预算拦截的准入结果，以及最终控制状态。

            iteration_budget_scenario 包含第一次推理动作的准入结果、
            被循环预算拦截的准入结果，以及最终控制状态。

            返回格式如下：

            {
                "tool_budget_scenario": {
                    "first_action_decision": {...},
                    "blocked_action_decision": {...},
                    "final_control_state": {...},
                },
                "iteration_budget_scenario": {
                    "first_action_decision": {...},
                    "blocked_action_decision": {...},
                    "final_control_state": {...},
                },
            }
    """
    tool_budget_initial_state = {
        "question": "连续调用天气工具",
        "tool_results": [],
        "final_answer": "",
    }
    first_tool_step_result = run_harness_control_step(
        graph_state=tool_budget_initial_state,
        requested_action="call_tool",
        max_iterations=3,
        max_tool_calls=1,
    )
    blocked_tool_step_result = run_harness_control_step(
        graph_state=first_tool_step_result["updated_graph_state"],
        requested_action="call_tool",
        max_iterations=3,
        max_tool_calls=1,
    )
    tool_budget_final_state = blocked_tool_step_result[
        "updated_graph_state"
    ]["agent_harness_control"]

    iteration_budget_initial_state = {
        "question": "持续进行下一步推理",
        "tool_results": [],
        "final_answer": "",
    }
    first_reason_step_result = run_harness_control_step(
        graph_state=iteration_budget_initial_state,
        requested_action="reason",
        max_iterations=1,
        max_tool_calls=2,
    )
    blocked_reason_step_result = run_harness_control_step(
        graph_state=first_reason_step_result["updated_graph_state"],
        requested_action="reason",
        max_iterations=1,
        max_tool_calls=2,
    )
    iteration_budget_final_state = blocked_reason_step_result[
        "updated_graph_state"
    ]["agent_harness_control"]

    return {
        "tool_budget_scenario": {
            "first_action_decision": first_tool_step_result[
                "action_decision"
            ],
            "blocked_action_decision": blocked_tool_step_result[
                "action_decision"
            ],
            "final_control_state": tool_budget_final_state,
        },
        "iteration_budget_scenario": {
            "first_action_decision": first_reason_step_result[
                "action_decision"
            ],
            "blocked_action_decision": blocked_reason_step_result[
                "action_decision"
            ],
            "final_control_state": iteration_budget_final_state,
        },
    }


def run_day16_harness_checkpoint_resume_demo() -> dict[str, Any]:
    """
    演示 Agent Harness 从 Checkpoint 恢复控制状态的行为。

    功能：
        创建一个模拟从 Checkpoint 恢复的图状态。

        恢复状态中的 Harness 控制数据为：
        current_iteration=1、
        tool_call_count=1、
        max_iterations=3、
        max_tool_calls=2。

        保存恢复图状态和恢复控制状态的深拷贝，
        用于检查后续控制步骤是否修改了原始输入。

        调用 run_harness_control_step，
        请求执行一次 reason。

        本次调用故意传入新的配置预算：
        max_iterations=99、
        max_tool_calls=99。

        由于 graph_state 已经包含
        agent_harness_control，
        Harness 必须把该任务视为恢复任务。

        恢复任务不得使用新的 99/99 预算
        覆盖 Checkpoint 中已有的 3/2 预算。

        reason 获准后，
        current_iteration 应从 1 增加到 2，
        tool_call_count 应继续保持为 1。

        返回动作准入结果、恢复前后的控制状态，
        以及预算、进度和原始输入是否符合预期的检查结果。

        本函数只演示 Checkpoint 恢复语义，
        不读取真实 Checkpoint，
        不调用模型，也不执行工具。

    参数：
        无。

    返回值：
        dict[str, Any]:
            Checkpoint 恢复演示结果，格式如下：

            {
                "action_decision": {
                    "requested_action": "reason",
                    "allowed": True,
                    "effective_action": "reason",
                    "termination_reason": None,
                },
                "restored_control_state": {
                    "status": "running",
                    "current_iteration": 1,
                    "tool_call_count": 1,
                    "max_iterations": 3,
                    "max_tool_calls": 2,
                    "termination_reason": None,
                },
                "updated_control_state": {
                    "status": "running",
                    "current_iteration": 2,
                    "tool_call_count": 1,
                    "max_iterations": 3,
                    "max_tool_calls": 2,
                    "termination_reason": None,
                },
                "restored_budget_preserved": True,
                "progress_continued": True,
                "input_graph_state_unchanged": True,
            }

            restored_budget_preserved:
                更新后的最大循环预算和最大工具预算，
                是否仍然等于 Checkpoint 中保存的 3 和 2。

            progress_continued:
                current_iteration 是否从 1 继续增加到 2，
                并且 tool_call_count 是否保持为 1。

            input_graph_state_unchanged:
                传入的恢复图状态是否仍然等于
                执行前保存的深拷贝。
    """
    restored_graph_state = {
        "question": "继续处理之前暂停的任务",
        "tool_results": [],
        "final_answer": "",
        "agent_harness_control": {
            "status": "running",
            "current_iteration": 1,
            "tool_call_count": 1,
            "max_iterations": 3,
            "max_tool_calls": 2,
            "termination_reason": None,
        },
    }
    original_restored_graph_state = copy.deepcopy(restored_graph_state)
    restored_control_state = copy.deepcopy(
        restored_graph_state["agent_harness_control"]
    )

    resume_step_result = run_harness_control_step(
        graph_state=restored_graph_state,
        requested_action="reason",
        max_iterations=99,
        max_tool_calls=99,
    )
    updated_graph_state = resume_step_result["updated_graph_state"]
    updated_control_state = updated_graph_state[
        "agent_harness_control"
    ]

    restored_budget_preserved = (
        updated_control_state["max_iterations"]
        == restored_control_state["max_iterations"]
        and updated_control_state["max_tool_calls"]
        == restored_control_state["max_tool_calls"]
    )
    progress_continued = (
        updated_control_state["current_iteration"]
        == restored_control_state["current_iteration"] + 1
        and updated_control_state["tool_call_count"]
        == restored_control_state["tool_call_count"]
    )

    return {
        "action_decision": resume_step_result["action_decision"],
        "restored_control_state": restored_control_state,
        "updated_control_state": updated_control_state,
        "restored_budget_preserved": restored_budget_preserved,
        "progress_continued": progress_continued,
        "input_graph_state_unchanged": (
            restored_graph_state == original_restored_graph_state
        ),
    }
