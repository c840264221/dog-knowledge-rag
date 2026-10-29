import copy
from typing import Any
import json

def decide_agent_loop_next_step(
    *,
    has_final_answer: bool,
    pending_tool_name: str | None,
    current_iteration: int,
    max_iterations: int,
) -> str:
    """
    根据当前智能体循环状态，决定下一步应该执行什么操作。

    功能：
        检查智能体是否已经得到最终答案。

        如果已经得到最终答案，返回 finish，
        表示循环正常结束。

        如果还没有最终答案，但当前循环次数已经达到最大次数，
        返回 stop_budget_exhausted，
        表示因为循环预算耗尽而停止。

        如果循环仍可继续，并且存在待调用的工具名称，
        返回 call_tool，表示下一步应该执行工具。

        如果循环仍可继续，但没有最终答案，也没有待调用工具，
        返回 reason，表示下一步继续进行一次模型推理。

        本函数只负责决定下一步操作，
        不调用大语言模型、不执行工具，也不修改任何外部状态。

    参数：
        has_final_answer:
            当前是否已经得到可以返回给用户的最终答案。
            True 表示已经得到最终答案；
            False 表示仍需继续处理。

        pending_tool_name:
            当前等待调用的工具名称。
            可以是具体工具名称，也可以是 None 或空白字符串。

            None 或清理后为空，表示当前没有等待调用的工具。

        current_iteration:
            当前已经完成的循环次数。
            必须大于或等于 0。

        max_iterations:
            允许执行的最大循环次数。
            必须大于 0。

    返回值：
        str:
            返回以下四种操作之一：

            finish:
                已经得到最终答案，正常结束循环。

            stop_budget_exhausted:
                尚未得到最终答案，但循环次数已经达到上限，
                因预算耗尽而停止。

            call_tool:
                存在待调用工具，下一步应执行工具。

            reason:
                没有最终答案，也没有待调用工具，
                下一步应继续进行模型推理。

    异常：
        ValueError:
            当 current_iteration 小于 0，
            或 max_iterations 小于等于 0 时抛出。
    """
    if current_iteration < 0:
        raise ValueError("当前循环次数不能小于0")
    if max_iterations <= 0:
        raise ValueError("最大循环次数不能小于等于0")

    if has_final_answer:
        return "finish"

    if current_iteration >= max_iterations:
        return "stop_budget_exhausted"

    normalized_tool_name = (pending_tool_name or "").strip()

    if normalized_tool_name:
        return "call_tool"
    else:
        return "reason"

def record_tool_observation(
    *,
    state: dict[str, Any],
    tool_name: str,
    tool_succeeded: bool,
    tool_output: Any = None,
    tool_error: str | None = None,
) -> dict[str, Any]:
    """
    将一次工具执行结果记录为 Observation，并返回更新后的智能体状态。

    功能：
        检查工具名称不能为空。

        检查当前状态中的 pending_tool_name，
        必须与实际完成执行的 tool_name 一致，
        避免把某个工具的结果错误记录到另一个工具名下。

        根据工具是否执行成功，检查结果字段是否合理：
        成功时不能同时携带非空错误信息；
        失败时必须提供非空错误信息。

        创建一条结构化 Observation，记录工具名称、
        是否成功、工具输出和错误信息。

        将 Observation 追加到状态的 observations 列表中，
        并把 pending_tool_name 清空，
        表示这次等待执行的工具已经处理完成。

        本函数返回新的状态字典，
        不修改传入的原始 state，
        也不负责真正调用工具或控制下一轮循环。

    参数：
        state:
            当前智能体状态字典。

            状态中应包含 pending_tool_name，
            表示本轮等待执行的工具。

            observations 可以不存在；
            不存在时按照空列表处理。
            如果存在，则必须是列表。

        tool_name:
            实际完成执行的工具名称。
            清理首尾空格后不能为空，
            并且必须与 state 中的 pending_tool_name 一致。

        tool_succeeded:
            工具是否执行成功。

            True 表示工具正常完成；
            False 表示工具执行失败。

        tool_output:
            工具成功时产生的原始输出。
            可以是字符串、数字、字典、列表或 None。

        tool_error:
            工具失败时产生的错误信息。

            工具成功时应当是 None 或空白字符串；
            工具失败时清理首尾空格后不能为空。

    返回值：
        dict[str, Any]:
            更新后的新状态字典。

            新状态会保留原状态的其他字段，
            将新的 Observation 追加到 observations，
            并把 pending_tool_name 设置为 None。

            Observation 格式如下：

            {
                "tool_name": "weather",
                "succeeded": True,
                "output": {
                    "temperature": 18,
                },
                "error": None,
            }

    异常：
        ValueError:
            当工具名称为空；
            当前待执行工具与实际工具不一致；
            工具成功却携带错误信息；
            或工具失败但没有错误信息时抛出。

        TypeError:
            当 state 中已有的 observations 不是列表时抛出。
    """
    normalized_tool_name = tool_name.strip()
    if not normalized_tool_name:
        raise ValueError("工具名称不能为空")

    pending_tool_name = state["pending_tool_name"] or ""

    normalized_pending_tool_name = pending_tool_name.strip()

    if normalized_tool_name != normalized_pending_tool_name:
        raise ValueError("实际调用工具和状态中的工具不一致")

    normalized_tool_error = (tool_error or "").strip()
    if tool_succeeded and normalized_tool_error:
        raise ValueError("工具成功时不能有错误信息")

    if not tool_succeeded and not normalized_tool_error:
        raise ValueError("工具失败时不能没有错误信息")

    existing_observations = state.get("observations")

    if existing_observations is None:
        existing_observations = []
    elif not isinstance(existing_observations, list):
        raise TypeError("observations 必须是列表")

    observation = {
        "tool_name": normalized_tool_name,
        "succeeded": tool_succeeded,
        "output": tool_output,
        "error": normalized_tool_error or None,
    }
    updated_observations = copy.deepcopy(existing_observations)

    updated_observations.append(observation)
    updated_state = dict(state)
    updated_state["pending_tool_name"] = None
    updated_state["observations"] = updated_observations
    return updated_state

def decide_after_tool_observation(
    *,
    observation: dict[str, Any],
    failure_is_retryable: bool,
    current_retry_count: int,
    max_retry_count: int,
) -> str:
    """
    根据工具 Observation 和重试状态，决定下一步操作。

    功能：
        读取 Observation 中的 succeeded 字段，
        判断上一轮工具执行是否成功。

        如果工具执行成功，返回 reason，
        表示下一步应当让模型读取工具结果并继续推理，
        而不是直接把工具原始输出当作最终答案。

        如果工具执行失败且错误不可重试，
        返回 stop_tool_failed，
        表示停止工具执行流程。

        如果工具执行失败、错误允许重试，
        并且当前重试次数还没有达到上限，
        返回 retry_tool。

        如果错误允许重试，但当前重试次数已经达到上限，
        返回 stop_retry_exhausted。

        本函数只返回下一步决策，
        不执行工具、不增加重试次数，
        也不修改 Observation 或其他外部状态。

    参数：
        observation:
            record_tool_observation 创建的单条工具观察结果。

            其中必须包含 succeeded 字段，
            且 succeeded 必须是布尔值。

        failure_is_retryable:
            当前工具错误是否允许重试。

            True 表示临时错误，可以在预算范围内重试；
            False 表示永久错误，不应再次调用工具。

            当工具成功时，此参数不会影响返回结果。

        current_retry_count:
            当前已经完成的重试次数，不包含最初的工具调用。
            必须大于或等于 0。

            例如：
            首次工具调用失败后，此值为 0；
            已经额外重试一次后，此值为 1。

        max_retry_count:
            最多允许执行的额外重试次数。
            必须大于或等于 0。

            0 表示不允许重试，
            工具首次失败后立即停止。

    返回值：
        str:
            返回以下四种操作之一：

            reason:
                工具执行成功，下一步根据工具结果继续推理。

            retry_tool:
                工具执行失败，但允许继续重试。

            stop_tool_failed:
                工具执行失败，并且错误不可重试。

            stop_retry_exhausted:
                工具错误允许重试，但重试次数已经达到上限。

    异常：
        ValueError:
            当 current_retry_count 或 max_retry_count 小于 0，
            或 Observation 中不存在 succeeded 字段时抛出。

        TypeError:
            当 Observation 中的 succeeded 不是布尔值时抛出。
    """
    if current_retry_count < 0:
        raise ValueError("当前重试次数不能小于0")
    if max_retry_count < 0:
        raise ValueError("最大重试次数不能小于0")
    if "succeeded" not in observation:
        raise ValueError("observation中没有succeeded字段")

    tool_succeeded = observation["succeeded"]

    if not isinstance(tool_succeeded, bool):
        raise TypeError("tool_succeeded必须是布尔值")

    if tool_succeeded:
        return "reason"

    if not failure_is_retryable:
        return "stop_tool_failed"

    if current_retry_count < max_retry_count:
        return "retry_tool"
    else:
        return "stop_retry_exhausted"

def select_agent_loop_patterns(
    *,
    requires_upfront_plan: bool,
    requires_dynamic_tool_feedback: bool,
    requires_result_review: bool,
) -> list[str]:
    """
    根据任务特征选择适合的 Agent Loop 流程模式。

    功能：
        当任务需要先拆分多个步骤时，
        选择 plan_and_execute。

        当后续动作需要依赖工具返回结果时，
        选择 react。

        如果任务既不需要提前规划，
        也不需要动态工具反馈，
        则选择 single_pass 作为基础执行方式。

        当最终结果需要额外检查时，
        在基础执行模式之后追加 reflection。

        多种模式可以同时存在。
        返回顺序表示推荐的流程组合顺序：
        先规划，再进行动态执行，最后检查结果。

        本函数只选择流程模式，
        不生成计划、不调用模型、不执行工具，
        也不真正运行 Agent Loop。

    参数：
        requires_upfront_plan:
            是否需要在执行前先拆分任务步骤。

            True 表示选择 plan_and_execute；
            False 表示不需要提前规划。

        requires_dynamic_tool_feedback:
            后续动作是否依赖工具的实时返回结果。

            True 表示选择 react；
            False 表示不需要动态工具循环。

        requires_result_review:
            最终结果是否需要额外检查和修正。

            True 表示在执行流程最后追加 reflection；
            False 表示不进行额外反思检查。

    返回值：
        list[str]:
            按推荐执行顺序排列的模式名称列表。

            可能包含：
            - single_pass
            - plan_and_execute
            - react
            - reflection

            示例：

            简单一次性任务：
            ["single_pass"]

            动态工具任务：
            ["react"]

            多步骤动态工具任务：
            ["plan_and_execute", "react"]

            多步骤、动态工具并需要最终检查：
            ["plan_and_execute", "react", "reflection"]
    """
    selected_patterns: list[str] = []

    if requires_upfront_plan:
        selected_patterns.append("plan_and_execute")

    if requires_dynamic_tool_feedback:
        selected_patterns.append("react")

    if len(selected_patterns) == 0:
        selected_patterns.append("single_pass")

    if requires_result_review:
        selected_patterns.append("reflection")
    return selected_patterns


def run_day15_agent_loop_principles_demo() -> dict[str, Any]:
    """
    使用固定天气数据演示一次最小的成功 ReAct 流程。

    功能：
        创建一个等待调用 weather 工具的初始状态。

        调用 decide_agent_loop_next_step，
        得到第一次动作 call_tool。

        使用固定的天气数据模拟工具成功，
        并调用 record_tool_observation，
        将工具结果记录到新状态。

        读取最新 Observation，
        调用 decide_after_tool_observation，
        得到工具成功后的动作 reason。

        模拟模型已经根据工具结果生成最终答案，
        再次调用 decide_agent_loop_next_step，
        得到最终动作 finish。

        同时调用 select_agent_loop_patterns，
        确认当前动态工具任务选择 react 模式。

        本函数只进行内存演示，
        不调用真实大语言模型、工具或外部服务，
        也不修改正式项目主链路。

    参数：
        无。

    返回值：
        dict[str, Any]:
            返回最小流程演示结果，包含：

            patterns:
                当前任务选择的流程模式，
                预期为 ["react"]。

            actions:
                本次流程依次产生的动作，
                预期为 ["call_tool", "reason", "finish"]。

            initial_state:
                工具执行前的原始状态。

            updated_state:
                记录工具 Observation 后的新状态。

            initial_state_unchanged:
                原始状态是否保持不变，
                预期为 True。
    """

    initial_state = {
        "pending_tool_name": "weather",
        "observations": [],
    }
    initial_state_snapshot = copy.deepcopy(initial_state)

    first_action = decide_agent_loop_next_step(
        has_final_answer=False,
        pending_tool_name="weather",
        current_iteration=0,
        max_iterations=3
    )
    updated_state = record_tool_observation(
        state=initial_state,
        tool_name="weather",
        tool_succeeded=True,
        tool_output={
            "temperature": 18,
            "condition": "sunny",
        },
        tool_error=None
    )
    latest_observation = updated_state["observations"][-1]

    action_after_observation = decide_after_tool_observation(
        observation=latest_observation,
        failure_is_retryable=False,
        current_retry_count=0,
        max_retry_count=3
    )
    final_action = decide_agent_loop_next_step(
        has_final_answer=True,
        pending_tool_name=None,
        current_iteration=1,
        max_iterations=3,
    )
    selected_patterns = select_agent_loop_patterns(
        requires_upfront_plan=False,
        requires_dynamic_tool_feedback=True,
        requires_result_review=False,
    )

    initial_state_unchanged = (
            initial_state == initial_state_snapshot
    )

    return {
        "patterns": selected_patterns,
        "actions": [
            first_action,
            action_after_observation,
            final_action,
        ],
        "initial_state": initial_state,
        "updated_state": updated_state,
        "initial_state_unchanged": initial_state_unchanged,
    }


if __name__ == "__main__":
    demo_result = run_day15_agent_loop_principles_demo()

    print(
        json.dumps(
            demo_result,
            ensure_ascii=False,
            indent=2,
        )
    )