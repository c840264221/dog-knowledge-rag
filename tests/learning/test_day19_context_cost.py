import json
from decimal import Decimal
from typing import Any


def build_agent_context_budget_state(
    *,
    max_context_tokens: int,
    max_task_tokens: int,
    reserved_output_tokens_per_call: int,
) -> dict[str, Any]:
    """
    创建第 19 天学习版 Agent 上下文与 Token 预算初始状态。

    功能：
        校验三个配置值都是真正的整数，不能用布尔值冒充。

        max_context_tokens 是模型单次调用能够使用的
        输入与输出 Token 总容量。

        max_task_tokens 是整个 Agent 任务跨多次 LLM 调用
        允许累计消耗的输入与输出 Token 上限。

        reserved_output_tokens_per_call 是每次模型调用前
        为输出预留的 Token 数量，不是已经产生的消耗。

        创建运行中的预算状态，并将实际输入消耗、
        实际输出消耗和 LLM 逻辑调用次数初始化为 0。

        当前函数只建立预算账本，不估算 Prompt Token，
        不裁剪上下文，不调用模型，也不计算货币费用。

    参数：
        max_context_tokens:
            单次模型调用的上下文容量，必须大于 1。

            必须为预留输出之外的输入留下至少 1 个 Token。

        max_task_tokens:
            整个任务的累计 Token 上限，必须大于 1。

            输入和输出的实际消耗以后都会计入这个上限。

        reserved_output_tokens_per_call:
            每次调用前预留的输出 Token 数量。

            必须大于 0，并且分别小于单次上下文容量
            和整个任务的 Token 上限。

    返回值：
        dict[str, Any]:
            新创建的预算状态，格式如下：

            {
                "status": "running",
                "max_context_tokens": 8192,
                "max_task_tokens": 20000,
                "reserved_output_tokens_per_call": 1024,
                "consumed_input_tokens": 0,
                "consumed_output_tokens": 0,
                "llm_call_count": 0,
                "termination_reason": None,
            }

            status:
                预算控制的生命周期状态，初始为 running。

            max_context_tokens:
                单次调用输入和预留输出共同受到的容量限制。

            max_task_tokens:
                整个任务累计实际 Token 消耗上限。

            reserved_output_tokens_per_call:
                每次调用前预留的输出空间，不计入已消耗量。

            consumed_input_tokens:
                已完成 LLM 调用的累计实际输入 Token 数。

            consumed_output_tokens:
                已完成 LLM 调用的累计实际输出 Token 数。

            llm_call_count:
                已记录用量的 LLM 逻辑调用次数。

            termination_reason:
                预算控制导致的终止原因，初始为 None。

    异常：
        TypeError:
            任一配置值不是真正的整数时抛出。

        ValueError:
            容量或任务上限不大于 1、输出预留不大于 0，
            或预留输出没有小于两个上限时抛出。
    """
    integer_parameters = {
        "max_context_tokens": max_context_tokens,
        "max_task_tokens": max_task_tokens,
        "reserved_output_tokens_per_call": (
            reserved_output_tokens_per_call
        ),
    }
    for parameter_name, parameter_value in (
        integer_parameters.items()
    ):
        if type(parameter_value) is not int:
            raise TypeError(
                f"{parameter_name} 必须是真正的整数"
            )

    if max_context_tokens <= 1:
        raise ValueError("max_context_tokens 必须大于 1")
    if max_task_tokens <= 1:
        raise ValueError("max_task_tokens 必须大于 1")
    if reserved_output_tokens_per_call <= 0:
        raise ValueError(
            "reserved_output_tokens_per_call 必须大于 0"
        )
    if reserved_output_tokens_per_call >= max_context_tokens:
        raise ValueError(
            "reserved_output_tokens_per_call 必须小于 "
            "max_context_tokens"
        )
    if reserved_output_tokens_per_call >= max_task_tokens:
        raise ValueError(
            "reserved_output_tokens_per_call 必须小于 "
            "max_task_tokens"
        )

    return {
        "status": "running",
        "max_context_tokens": max_context_tokens,
        "max_task_tokens": max_task_tokens,
        "reserved_output_tokens_per_call": (
            reserved_output_tokens_per_call
        ),
        "consumed_input_tokens": 0,
        "consumed_output_tokens": 0,
        "llm_call_count": 0,
        "termination_reason": None,
    }


def validate_context_budget_state(
    *,
    context_budget_state: dict[str, Any],
) -> None:
    """
    统一校验第 19 天学习版上下文预算状态的数据约束。

    功能：
        检查必需字段、字段类型、数值下限、
        输出预留与两个预算上限的关系，
        以及状态和终止原因是否一致。

        只校验状态结构与不变量，不判断某次模型请求
        能否通过预算，也不修改传入的状态。

    参数：
        context_budget_state:
            待校验的预算状态，通常由
            build_agent_context_budget_state 创建，
            或由执行后记账函数返回。

    返回值：
        None：校验通过时不返回业务数据。

    异常：
        TypeError：状态或字段类型不符合约定。
        ValueError：缺少字段或字段值不符合约定。
    """
    if not isinstance(context_budget_state, dict):
        raise TypeError("context_budget_state 必须是字典")

    required_state_fields = {
        "status",
        "max_context_tokens",
        "max_task_tokens",
        "reserved_output_tokens_per_call",
        "consumed_input_tokens",
        "consumed_output_tokens",
        "llm_call_count",
        "termination_reason",
    }
    missing_state_fields = required_state_fields.difference(
        context_budget_state
    )
    if missing_state_fields:
        raise ValueError(
            "context_budget_state 缺少字段："
            f"{', '.join(sorted(missing_state_fields))}"
        )

    status = context_budget_state["status"]
    if not isinstance(status, str):
        raise TypeError("status 必须是字符串")
    if status not in {"running", "stopped"}:
        raise ValueError("status 必须是 running 或 stopped")

    termination_reason = context_budget_state[
        "termination_reason"
    ]
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
            "stopped 状态必须包含非空 termination_reason"
        )

    integer_field_minimums = {
        "max_context_tokens": 2,
        "max_task_tokens": 2,
        "reserved_output_tokens_per_call": 1,
        "consumed_input_tokens": 0,
        "consumed_output_tokens": 0,
        "llm_call_count": 0,
    }
    for field_name, minimum_value in (
        integer_field_minimums.items()
    ):
        field_value = context_budget_state[field_name]
        if type(field_value) is not int:
            raise TypeError(f"{field_name} 必须是真正的整数")
        if field_value < minimum_value:
            raise ValueError(
                f"{field_name} 不能小于 {minimum_value}"
            )

    max_context_tokens = context_budget_state["max_context_tokens"]
    max_task_tokens = context_budget_state["max_task_tokens"]
    reserved_output_tokens = context_budget_state[
        "reserved_output_tokens_per_call"
    ]
    if reserved_output_tokens >= max_context_tokens:
        raise ValueError(
            "reserved_output_tokens_per_call 必须小于 "
            "max_context_tokens"
        )
    if reserved_output_tokens >= max_task_tokens:
        raise ValueError(
            "reserved_output_tokens_per_call 必须小于 "
            "max_task_tokens"
        )


def evaluate_context_budget_request(
    *,
    context_budget_state: dict[str, Any],
    estimated_input_tokens: int,
) -> dict[str, Any]:
    """
    在调用 LLM 前评估预计输入与预留输出是否符合 Token 预算。

    功能：
        校验上下文预算状态中的必需字段、类型、数值关系
        及生命周期与终止原因的一致性。

        计算整个任务已经实际消耗的 Token、
        当前剩余任务 Token，以及本次预计需要的
        输入 Token 加预留输出 Token。

        使用固定优先级返回准入决定：

        1. 状态已经 stopped：拒绝新调用，继续保持停止；
        2. 剩余任务预算不足以容纳至少 1 个输入 Token
           加预留输出：要求停止；
        3. 本次预计总量超过单次上下文容量：
           要求调用方裁剪上下文后重新评估；
        4. 本次预计总量超过任务剩余预算：
           要求调用方裁剪上下文后重新评估；
        5. 两种预算均满足：允许进入后续调用流程。

        trim_context 只表示可以尝试缩小输入并重新评估，
        不允许直接跳过准入检查去调用模型。

        预计输入应当包括最终请求中所有输入内容的估算，
        例如系统指令、用户需求、消息格式开销、
        必要依赖结果、工具定义和检索上下文。

        本函数只返回决定，不裁剪消息、
        不修改 context_budget_state，
        不预占或增加实际消耗，
        也不调用模型。

    参数：
        context_budget_state:
            build_agent_context_budget_state 创建的预算状态，
            或遵循相同结构的后续状态。

        estimated_input_tokens:
            本次完整模型输入的预计 Token 数。

            必须是真正的整数，并且大于 0。

    返回值：
        dict[str, Any]:
            调用前准入决定，格式如下：

            {
                "allowed": True,
                "effective_action": "call_llm",
                "rejection_reason": None,
                "estimated_input_tokens": 700,
                "reserved_output_tokens": 400,
                "estimated_total_tokens": 1100,
                "remaining_task_tokens": 1200,
                "max_input_tokens_for_context": 7792,
            }

            allowed:
                本次预计请求是否通过两个预算检查。

            effective_action:
                下一步建议动作：call_llm、trim_context 或 stop。

            rejection_reason:
                拒绝时的稳定原因；获准时为 None。

            estimated_input_tokens:
                调用方传入的本次预计输入 Token 数。

            reserved_output_tokens:
                本次从配置中读取的输出预留量，
                不是已经消耗的 Token。

            estimated_total_tokens:
                预计输入与预留输出之和。

            remaining_task_tokens:
                任务上限减去已经记录的实际输入和输出消耗。

            max_input_tokens_for_context:
                单次上下文容量减去输出预留后，
                理论上可分配给输入的最大 Token 数。

    异常：
        TypeError:
            状态或预计输入的类型不正确时抛出。

        ValueError:
            状态缺少字段、数值关系不合法、
            生命周期与原因不一致，
            或预计输入不大于 0 时抛出。
    """
    validate_context_budget_state(
        context_budget_state=context_budget_state,
    )
    if type(estimated_input_tokens) is not int:
        raise TypeError("estimated_input_tokens 必须是真正的整数")
    if estimated_input_tokens <= 0:
        raise ValueError("estimated_input_tokens 必须大于 0")

    status = context_budget_state["status"]
    termination_reason = context_budget_state["termination_reason"]
    max_context_tokens = context_budget_state["max_context_tokens"]
    max_task_tokens = context_budget_state["max_task_tokens"]
    reserved_output_tokens = context_budget_state[
        "reserved_output_tokens_per_call"
    ]

    consumed_total_tokens = (
        context_budget_state["consumed_input_tokens"]
        + context_budget_state["consumed_output_tokens"]
    )
    remaining_task_tokens = max(
        0,
        max_task_tokens - consumed_total_tokens,
    )
    estimated_total_tokens = (
        estimated_input_tokens + reserved_output_tokens
    )
    max_input_tokens_for_context = (
        max_context_tokens - reserved_output_tokens
    )

    common_result = {
        "estimated_input_tokens": estimated_input_tokens,
        "reserved_output_tokens": reserved_output_tokens,
        "estimated_total_tokens": estimated_total_tokens,
        "remaining_task_tokens": remaining_task_tokens,
        "max_input_tokens_for_context": (
            max_input_tokens_for_context
        ),
    }

    if status == "stopped":
        return {
            **common_result,
            "allowed": False,
            "effective_action": "stop",
            "rejection_reason": termination_reason,
        }

    if remaining_task_tokens <= reserved_output_tokens:
        return {
            **common_result,
            "allowed": False,
            "effective_action": "stop",
            "rejection_reason": "task_token_budget_exhausted",
        }

    if estimated_total_tokens > max_context_tokens:
        return {
            **common_result,
            "allowed": False,
            "effective_action": "trim_context",
            "rejection_reason": "context_window_exceeded",
        }

    if estimated_total_tokens > remaining_task_tokens:
        return {
            **common_result,
            "allowed": False,
            "effective_action": "trim_context",
            "rejection_reason": "task_token_budget_insufficient",
        }

    return {
        **common_result,
        "allowed": True,
        "effective_action": "call_llm",
        "rejection_reason": None,
    }


def plan_context_trim(
    *,
    context_budget_state: dict[str, Any],
    context_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    在不丢弃必需内容的前提下，为本次模型请求规划上下文裁剪。

    功能：
        将输入看作有顺序的内容块。每个块包含 block_id、
        content、estimated_tokens、required 和 drop_priority。

        required=True 的块必须保留，例如原始用户目标、
        系统约束或回答所必需的依赖结果。其他块可按
        drop_priority 从小到大移除；优先级相同时，
        先移除原列表中更靠前的块。

        每次移除后都重新调用 evaluate_context_budget_request。
        一旦预算允许，就停止裁剪并保留剩余块的原始顺序。
        如果必需块本身仍然放不下，就返回 stop，
        不擅自删改或压缩必需内容。

        本函数只规划整块移除，不生成滚动摘要，
        不精确分词，不修改输入状态和原始内容块，
        也不调用模型。调用方必须把所有输入开销都表示为块，
        包括系统指令、工具定义和消息格式开销。

    参数：
        context_budget_state:
            当前上下文与任务 Token 预算状态。

        context_blocks:
            按最终消息顺序排列的内容块列表。每个块格式为：

            {
                "block_id": "original_user_goal",
                "content": "用户最初要求……",
                "estimated_tokens": 120,
                "required": True,
                "drop_priority": 0,
            }

            block_id 必须是唯一的非空字符串；content 是字符串；
            estimated_tokens 是大于 0 的真正整数；
            required 是布尔值；drop_priority 是非负整数，
            数字越小表示可选块越先被移除。

            列表至少包含一个 required=True 的块。

    返回值：
        dict[str, Any]:
            裁剪规划结果，包含：

            selected_blocks:
                保留的内容块，顺序与原列表一致，
                返回的是字典浅拷贝。

            dropped_block_ids:
                按实际移除顺序记录的块标识。

            budget_decision:
                对最终保留内容重新评估后的预算决定。

            effective_action:
                call_llm 或 stop。stop 也可能来自预算状态
                已终止，或必需内容无法容纳。

            failure_reason:
                可以调用时为 None；否则为稳定原因字符串。

    异常：
        TypeError:
            内容块列表或字段类型不符合约定。

        ValueError:
            缺字段、标识重复、数值不合法，
            或没有必须保留的内容块。
    """
    if not isinstance(context_blocks, list):
        raise TypeError("context_blocks 必须是列表")
    if not context_blocks:
        raise ValueError("context_blocks 不能为空")

    required_block_fields = {
        "block_id",
        "content",
        "estimated_tokens",
        "required",
        "drop_priority",
    }
    seen_block_ids: set[str] = set()
    normalized_blocks: list[dict[str, Any]] = []
    for index, block in enumerate(context_blocks):
        if not isinstance(block, dict):
            raise TypeError(f"第 {index} 个内容块必须是字典")
        missing_fields = required_block_fields.difference(block)
        if missing_fields:
            raise ValueError(
                f"第 {index} 个内容块缺少字段："
                f"{', '.join(sorted(missing_fields))}"
            )

        block_id = block["block_id"]
        if not isinstance(block_id, str):
            raise TypeError("block_id 必须是字符串")
        if not block_id.strip():
            raise ValueError("block_id 不能为空")
        if block_id in seen_block_ids:
            raise ValueError(f"block_id 重复：{block_id}")
        seen_block_ids.add(block_id)

        if not isinstance(block["content"], str):
            raise TypeError("content 必须是字符串")
        if type(block["estimated_tokens"]) is not int:
            raise TypeError("estimated_tokens 必须是真正的整数")
        if block["estimated_tokens"] <= 0:
            raise ValueError("estimated_tokens 必须大于 0")
        if type(block["required"]) is not bool:
            raise TypeError("required 必须是布尔值")
        if type(block["drop_priority"]) is not int:
            raise TypeError("drop_priority 必须是真正的整数")
        if block["drop_priority"] < 0:
            raise ValueError("drop_priority 不能小于 0")

        normalized_blocks.append(dict(block))

    if not any(block["required"] for block in normalized_blocks):
        raise ValueError("至少需要一个必须保留的内容块")

    selected_blocks = list(normalized_blocks)
    dropped_block_ids: list[str] = []

    def evaluate_selected_blocks() -> dict[str, Any]:
        """评估当前保留块；无参数，返回调用前预算决定。"""
        estimated_input_tokens = sum(
            block["estimated_tokens"] for block in selected_blocks
        )
        return evaluate_context_budget_request(
            context_budget_state=context_budget_state,
            estimated_input_tokens=estimated_input_tokens,
        )

    budget_decision = evaluate_selected_blocks()
    if budget_decision["effective_action"] == "trim_context":
        removable_blocks = sorted(
            (
                (index, block)
                for index, block in enumerate(normalized_blocks)
                if not block["required"]
            ),
            key=lambda item: (item[1]["drop_priority"], item[0]),
        )
        for _, block in removable_blocks:
            selected_blocks = [
                selected
                for selected in selected_blocks
                if selected["block_id"] != block["block_id"]
            ]
            dropped_block_ids.append(block["block_id"])
            budget_decision = evaluate_selected_blocks()
            if budget_decision["allowed"]:
                break

    if budget_decision["allowed"]:
        effective_action = "call_llm"
        failure_reason = None
    elif budget_decision["effective_action"] == "stop":
        effective_action = "stop"
        failure_reason = budget_decision["rejection_reason"]
    else:
        effective_action = "stop"
        failure_reason = "required_context_exceeds_budget"

    return {
        "selected_blocks": selected_blocks,
        "dropped_block_ids": dropped_block_ids,
        "budget_decision": budget_decision,
        "effective_action": effective_action,
        "failure_reason": failure_reason,
    }


def plan_rolling_summary_update(
    *,
    previous_summary: str | None,
    conversation_messages: list[dict[str, Any]],
    recent_message_count: int,
) -> dict[str, Any]:
    """
    规划滚动摘要的输入，不在本函数中生成或替换摘要。

    功能：
        将上一版摘要与较早的可摘要消息交给后续摘要步骤。
        最近 recent_message_count 条消息原样保留；
        更早但标记 must_keep=True 的消息也原样保留。
        只有既不在最近窗口内、又不是必需的消息，
        才能成为本轮摘要候选。

        这样可以在后续将“旧摘要 + 新增旧消息”合并为
        一份新摘要，但不能把用户原始任务目标或回答所需的
        关键工具结果，仅凭时间较早就自动摘要或删除。

        本函数不调用模型，不改变原消息，不产生摘要文本，
        也不承诺后续摘要一定能通过 Token 预算检查。
        真正生成新摘要后，仍需重新估算输入并执行预算准入。

    参数：
        previous_summary:
            上一版滚动摘要；首次摘要时传入 None。

        conversation_messages:
            按时间先后排列的对话消息。每条消息至少包含：

            {
                "message_id": "turn-1-user",
                "role": "user",
                "content": "用户说的话",
                "must_keep": True,
            }

            message_id 必须是唯一的非空字符串；role 和
            content 必须是字符串；must_keep 必须是布尔值。

        recent_message_count:
            要原样保留的最近消息条数，必须是非负整数。
            这里按消息条数而不是对话轮数计算。

    返回值：
        dict[str, Any]:
            摘要输入规划，包含：

            previous_summary:
                原样传递的上一版摘要。

            messages_to_summarize:
                本轮允许交给摘要器的较早消息，保持原顺序。

            retained_messages:
                必须保留及最近消息，保持原顺序。

            summary_needed:
                是否存在本轮待摘要消息。为 False 时，
                无须为这轮请求调用摘要器。

            两个消息列表中的字典是原消息的浅拷贝。

    异常：
        TypeError:
            参数或消息字段类型不符合约定。

        ValueError:
            摘要为空字符串、消息缺字段、标识重复、
            消息标识为空或最近消息数量小于 0。
    """
    if previous_summary is not None:
        if not isinstance(previous_summary, str):
            raise TypeError("previous_summary 必须是字符串或 None")
        if not previous_summary.strip():
            raise ValueError("previous_summary 不能是空字符串")
    if not isinstance(conversation_messages, list):
        raise TypeError("conversation_messages 必须是列表")
    if type(recent_message_count) is not int:
        raise TypeError("recent_message_count 必须是真正的整数")
    if recent_message_count < 0:
        raise ValueError("recent_message_count 不能小于 0")

    required_message_fields = {
        "message_id",
        "role",
        "content",
        "must_keep",
    }
    seen_message_ids: set[str] = set()
    normalized_messages: list[dict[str, Any]] = []
    for index, message in enumerate(conversation_messages):
        if not isinstance(message, dict):
            raise TypeError(f"第 {index} 条消息必须是字典")
        missing_fields = required_message_fields.difference(message)
        if missing_fields:
            raise ValueError(
                f"第 {index} 条消息缺少字段："
                f"{', '.join(sorted(missing_fields))}"
            )

        message_id = message["message_id"]
        if not isinstance(message_id, str):
            raise TypeError("message_id 必须是字符串")
        if not message_id.strip():
            raise ValueError("message_id 不能为空")
        if message_id in seen_message_ids:
            raise ValueError(f"message_id 重复：{message_id}")
        seen_message_ids.add(message_id)

        if not isinstance(message["role"], str):
            raise TypeError("role 必须是字符串")
        if not isinstance(message["content"], str):
            raise TypeError("content 必须是字符串")
        if type(message["must_keep"]) is not bool:
            raise TypeError("must_keep 必须是布尔值")

        normalized_messages.append(dict(message))

    recent_start_index = max(
        0,
        len(normalized_messages) - recent_message_count,
    )
    messages_to_summarize: list[dict[str, Any]] = []
    retained_messages: list[dict[str, Any]] = []
    for index, message in enumerate(normalized_messages):
        if index >= recent_start_index or message["must_keep"]:
            retained_messages.append(message)
        else:
            messages_to_summarize.append(message)

    return {
        "previous_summary": previous_summary,
        "messages_to_summarize": messages_to_summarize,
        "retained_messages": retained_messages,
        "summary_needed": bool(messages_to_summarize),
    }


def apply_rolling_summary_update(
    *,
    previous_summary: str | None,
    conversation_messages: list[dict[str, Any]],
    recent_message_count: int,
    generated_summary: str | None,
) -> dict[str, Any]:
    """
    接收外部已生成的摘要，产出新摘要与保留消息的副本。

    功能：
        重新执行滚动摘要规划，避免将需要保留的消息
        误当作摘要候选。若本轮确有可摘要消息，
        要求调用方提供非空的 generated_summary，
        用它替换 previous_summary；否则不允许提交新摘要，
        直接沿用 previous_summary。

        新摘要应当由外部摘要步骤根据上一版摘要与
        messages_to_summarize 一起生成。本函数只检查
        交接结构，不验证摘要是否忠实覆盖原意，
        不估算新摘要的 Token，不调用模型，
        也不修改传入的消息或预算状态。

        生成最终模型输入前，仍应将新摘要、保留消息、
        必要系统指令等重新组装并做 Token 预算准入。

    参数：
        previous_summary:
            上一版滚动摘要，首次更新时传入 None。

        conversation_messages:
            按时间顺序排列的原始消息列表，
            字段规则与 plan_rolling_summary_update 相同。

        recent_message_count:
            要原样保留的最近消息条数，必须是非负整数。

        generated_summary:
            外部摘要步骤生成的新摘要。
            有待摘要消息时必须是非空字符串；
            无待摘要消息时必须是 None。

    返回值：
        dict[str, Any]:
            新摘要交接结果，包含：

            updated_summary:
                本轮新摘要，或没有更新时沿用的旧摘要。

            retained_messages:
                必须保留及最近消息的浅拷贝列表。

            summarized_message_ids:
                本轮进入摘要的原消息标识，保持原顺序，
                便于追踪摘要的来源。

            summary_updated:
                本轮是否提交了新摘要。

            该结果不包含语义正确性或预算通过的保证。

    异常：
        TypeError:
            消息规划参数或 generated_summary 类型不正确。

        ValueError:
            有摘要候选却没有有效新摘要，或没有候选
            却提交新摘要；规划参数不合法时也会抛出。
    """
    summary_plan = plan_rolling_summary_update(
        previous_summary=previous_summary,
        conversation_messages=conversation_messages,
        recent_message_count=recent_message_count,
    )

    if generated_summary is not None and not isinstance(
        generated_summary,
        str,
    ):
        raise TypeError("generated_summary 必须是字符串或 None")

    if summary_plan["summary_needed"]:
        if generated_summary is None or not generated_summary.strip():
            raise ValueError("有待摘要消息时必须提供非空新摘要")
        updated_summary = generated_summary
    else:
        if generated_summary is not None:
            raise ValueError("没有待摘要消息时不能提交新摘要")
        updated_summary = summary_plan["previous_summary"]

    return {
        "updated_summary": updated_summary,
        "retained_messages": summary_plan["retained_messages"],
        "summarized_message_ids": [
            message["message_id"]
            for message in summary_plan["messages_to_summarize"]
        ],
        "summary_updated": summary_plan["summary_needed"],
    }


def record_context_budget_usage(
    *,
    context_budget_state: dict[str, Any],
    actual_input_tokens: int,
    actual_output_tokens: int,
) -> dict[str, Any]:
    """
    在一次模型调用完成后，将实际 Token 用量记入预算状态。

    功能：
        先复用独立的预算状态校验函数，
        再要求当前状态仍为 running。

        用模型服务返回的实际输入、输出 Token 数更新账本，
        并将已记录用量的 LLM 逻辑调用次数增加 1。
        输出预留量仅用于调用前检查，不能当作实际消耗。

        如果记账后剩余任务预算已经不足以容纳
        至少 1 个输入 Token 加下一次预留输出，
        就将新状态标为 stopped，并记录稳定终止原因。

        这是执行后记账函数：它不调用模型，
        不重新判断之前是否批准过这次调用，
        也不修改传入的原始状态。

        本学习版尚未实现 call_id 幂等保护。
        同一次实际调用只允许交给本函数记账一次；
        企业落地时应以调用标识防止重试导致重复计费。

    参数：
        context_budget_state:
            调用前的预算状态，必须处于 running。

        actual_input_tokens:
            本次模型调用产生的实际输入 Token 数，
            必须是真正的整数且大于 0。

        actual_output_tokens:
            本次模型调用产生的实际输出 Token 数，
            必须是真正的整数且大于等于 0。

    返回值：
        dict[str, Any]:
            记账结果，格式如下：

            {
                "updated_state": { ... },
                "usage_record": {
                    "actual_input_tokens": 700,
                    "actual_output_tokens": 250,
                    "actual_total_tokens": 950,
                    "remaining_task_tokens": 19050,
                },
            }

            updated_state:
                更新后的预算状态副本；包含累计用量、
                调用次数，以及可能改变的状态和终止原因。

            usage_record:
                本次实际用量及记账后的任务剩余额度。

    异常：
        TypeError:
            状态结构或实际用量类型不符合约定。

        ValueError:
            状态结构或用量数值不合法，
            或预算状态已经 stopped。
    """
    validate_context_budget_state(
        context_budget_state=context_budget_state,
    )
    if context_budget_state["status"] != "running":
        raise ValueError("已停止的预算状态不能记录新调用")

    if type(actual_input_tokens) is not int:
        raise TypeError("actual_input_tokens 必须是真正的整数")
    if type(actual_output_tokens) is not int:
        raise TypeError("actual_output_tokens 必须是真正的整数")
    if actual_input_tokens <= 0:
        raise ValueError("actual_input_tokens 必须大于 0")
    if actual_output_tokens < 0:
        raise ValueError("actual_output_tokens 不能小于 0")

    updated_state = dict(context_budget_state)
    updated_state["consumed_input_tokens"] += actual_input_tokens
    updated_state["consumed_output_tokens"] += actual_output_tokens
    updated_state["llm_call_count"] += 1

    consumed_total_tokens = (
        updated_state["consumed_input_tokens"]
        + updated_state["consumed_output_tokens"]
    )
    remaining_task_tokens = max(
        0,
        updated_state["max_task_tokens"] - consumed_total_tokens,
    )
    if (
        remaining_task_tokens
        <= updated_state["reserved_output_tokens_per_call"]
    ):
        updated_state["status"] = "stopped"
        updated_state["termination_reason"] = (
            "task_token_budget_exhausted"
        )

    return {
        "updated_state": updated_state,
        "usage_record": {
            "actual_input_tokens": actual_input_tokens,
            "actual_output_tokens": actual_output_tokens,
            "actual_total_tokens": (
                actual_input_tokens + actual_output_tokens
            ),
            "remaining_task_tokens": remaining_task_tokens,
        },
    }


def estimate_model_call_cost(
    *,
    actual_input_tokens: int,
    actual_output_tokens: int,
    input_price_per_million_tokens: Decimal,
    output_price_per_million_tokens: Decimal,
    currency: str,
) -> dict[str, Any]:
    """
    根据实际 Token 用量和外部提供的单价估算本次模型费用。

    功能：
        分别计算输入和输出费用，再求合计。
        使用 Decimal 进行十进制计算，避免普通浮点数
        在金额计算中产生难以解释的二进制舍入差异。

        本函数不查询实时模型价格，不处理缓存输入、
        分档计价、最低收费、税费或账单舍入规则。
        因此结果只是按所给价格估算的费用，
        不保证等于模型服务商的最终账单。

    参数：
        actual_input_tokens:
            本次实际输入 Token 数，必须是非负整数。

        actual_output_tokens:
            本次实际输出 Token 数，必须是非负整数。

        input_price_per_million_tokens:
            每一百万输入 Token 的单价，必须是
            非负、有限的 Decimal 数值。

        output_price_per_million_tokens:
            每一百万输出 Token 的单价，必须是
            非负、有限的 Decimal 数值。

        currency:
            单价对应的货币代码，例如 USD，
            必须是非空字符串。

    返回值：
        dict[str, Any]:
            包含货币、实际用量、输入和输出单价、
            input_cost、output_cost、total_cost。

            金额与单价均以十进制字符串返回，
            便于直接写入 JSON，且不引入浮点误差。

    异常：
        TypeError：实际用量、单价或货币类型不正确。
        ValueError：用量或单价为负、单价非有限，
            或货币代码为空。
    """
    token_fields = {
        "actual_input_tokens": actual_input_tokens,
        "actual_output_tokens": actual_output_tokens,
    }
    for field_name, token_count in token_fields.items():
        if type(token_count) is not int:
            raise TypeError(f"{field_name} 必须是真正的整数")
        if token_count < 0:
            raise ValueError(f"{field_name} 不能小于 0")

    price_fields = {
        "input_price_per_million_tokens": (
            input_price_per_million_tokens
        ),
        "output_price_per_million_tokens": (
            output_price_per_million_tokens
        ),
    }
    for field_name, price in price_fields.items():
        if not isinstance(price, Decimal):
            raise TypeError(f"{field_name} 必须是 Decimal")
        if not price.is_finite() or price < 0:
            raise ValueError(f"{field_name} 必须是非负有限数")

    if not isinstance(currency, str):
        raise TypeError("currency 必须是字符串")
    if not currency.strip():
        raise ValueError("currency 不能为空")

    tokens_per_million = Decimal("1000000")
    input_cost = (
        Decimal(actual_input_tokens)
        * input_price_per_million_tokens
        / tokens_per_million
    )
    output_cost = (
        Decimal(actual_output_tokens)
        * output_price_per_million_tokens
        / tokens_per_million
    )

    return {
        "currency": currency,
        "actual_input_tokens": actual_input_tokens,
        "actual_output_tokens": actual_output_tokens,
        "input_price_per_million_tokens": format(
            input_price_per_million_tokens,
            "f",
        ),
        "output_price_per_million_tokens": format(
            output_price_per_million_tokens,
            "f",
        ),
        "input_cost": format(input_cost, "f"),
        "output_cost": format(output_cost, "f"),
        "total_cost": format(input_cost + output_cost, "f"),
    }


def demo_day19_context_budget_flow() -> dict[str, Any]:
    """
    用固定数据演示第 19 天上下文整理与 Token 记账链路。

    功能：
        创建预算状态；规划较早消息的滚动摘要；
        接收一段事先写好的模拟摘要；组装完整输入块；
        检查原始输入是否超限；在必要时裁剪可选块；
        最后用模拟的模型实际用量完成一次记账，
        并使用教学用的假设单价估算费用。

        此函数没有接入真实摘要器、分词器或 LLM。
        所有 estimated_tokens、generated_summary、
        actual_input_tokens、actual_output_tokens 和单价
        都是为了讲解流程而固定提供的数据。

    参数：
        无。

    返回值：
        dict[str, Any]:
            分阶段演示结果，包含初始状态、摘要规划、
            摘要交接、裁剪前决定、裁剪结果、
            执行后记账结果、费用估算结果，
            及初始状态是否保持不变。

            这些字典用于观察学习链路，
            不是生产环境的模型响应格式。

    异常：
        RuntimeError：固定演示数据未能通过预期的裁剪准入。
    """
    context_budget_state = build_agent_context_budget_state(
        max_context_tokens=1200,
        max_task_tokens=3000,
        reserved_output_tokens_per_call=200,
    )
    initial_state_snapshot = dict(context_budget_state)

    previous_summary = "用户正在学习 Agent 上下文与成本控制。"
    conversation_messages = [
        {
            "message_id": "older-user",
            "role": "user",
            "content": "旧问题 C：怎样整理较早的消息？",
            "must_keep": False,
        },
        {
            "message_id": "older-assistant",
            "role": "assistant",
            "content": "旧回答 D：可以先规划滚动摘要。",
            "must_keep": False,
        },
        {
            "message_id": "latest-user",
            "role": "user",
            "content": "新问题 B：摘要之后怎样检查预算？",
            "must_keep": True,
        },
    ]
    summary_plan = plan_rolling_summary_update(
        previous_summary=previous_summary,
        conversation_messages=conversation_messages,
        recent_message_count=1,
    )
    generated_summary = (
        "用户正在学习 Agent 上下文与成本控制；"
        "较早对话讨论了滚动摘要的规划方法。"
    )
    summary_result = apply_rolling_summary_update(
        previous_summary=previous_summary,
        conversation_messages=conversation_messages,
        recent_message_count=1,
        generated_summary=generated_summary,
    )

    context_blocks = [
        {
            "block_id": "system_instructions",
            "content": "系统规则与消息格式开销",
            "estimated_tokens": 100,
            "required": True,
            "drop_priority": 0,
        },
        {
            "block_id": "original_user_goal",
            "content": "用户原始目标：理解上下文预算与滚动摘要",
            "estimated_tokens": 120,
            "required": True,
            "drop_priority": 0,
        },
        {
            "block_id": "rolling_summary",
            "content": summary_result["updated_summary"],
            "estimated_tokens": 180,
            "required": True,
            "drop_priority": 0,
        },
        {
            "block_id": "required_tool_evidence",
            "content": "回答所必需的工具结果摘录",
            "estimated_tokens": 230,
            "required": True,
            "drop_priority": 0,
        },
        {
            "block_id": "optional_retrieval",
            "content": "可按需重新检索的补充资料",
            "estimated_tokens": 200,
            "required": False,
            "drop_priority": 0,
        },
        {
            "block_id": "optional_example",
            "content": "额外的讲解例子",
            "estimated_tokens": 100,
            "required": False,
            "drop_priority": 1,
        },
        {
            "block_id": "latest_user_message",
            "content": summary_result["retained_messages"][-1]["content"],
            "estimated_tokens": 90,
            "required": True,
            "drop_priority": 0,
        },
    ]
    estimated_input_tokens = sum(
        block["estimated_tokens"] for block in context_blocks
    )
    initial_budget_decision = evaluate_context_budget_request(
        context_budget_state=context_budget_state,
        estimated_input_tokens=estimated_input_tokens,
    )
    trim_result = plan_context_trim(
        context_budget_state=context_budget_state,
        context_blocks=context_blocks,
    )
    if trim_result["effective_action"] != "call_llm":
        raise RuntimeError("固定演示数据应在裁剪后通过预算检查")

    usage_result = record_context_budget_usage(
        context_budget_state=context_budget_state,
        actual_input_tokens=830,
        actual_output_tokens=150,
    )
    estimated_cost_result = estimate_model_call_cost(
        actual_input_tokens=(
            usage_result["usage_record"]["actual_input_tokens"]
        ),
        actual_output_tokens=(
            usage_result["usage_record"]["actual_output_tokens"]
        ),
        input_price_per_million_tokens=Decimal("2.00"),
        output_price_per_million_tokens=Decimal("8.00"),
        currency="USD",
    )

    return {
        "initial_state": initial_state_snapshot,
        "summary_plan": summary_plan,
        "summary_result": summary_result,
        "initial_budget_decision": initial_budget_decision,
        "trim_result": trim_result,
        "usage_result": usage_result,
        "estimated_cost_result": estimated_cost_result,
        "initial_state_unchanged": (
            context_budget_state == initial_state_snapshot
        ),
    }


def demo_day19_context_cost_cases() -> dict[str, Any]:
    """
    汇总演示第 19 天上下文、摘要和 Token 预算的关键分支。

    功能：
        运行一条可裁剪后继续的完整固定数据链路，
        并补充必需内容仍然超限、任务预算耗尽、
        无需更新摘要三个边界场景。

        所有摘要和模型用量都是固定的模拟数据。
        本函数不调用真实模型、分词器或外部服务。

    参数：
        无。

    返回值：
        dict[str, Any]:
            按场景名称组织的结果字典，包含：

            successful_flow:
                摘要、裁剪、执行后记账的主流程。

            required_context_failure:
                只剩必需内容仍超限的裁剪结果。

            task_budget_exhaustion:
                本次调用获准、执行后余额不足，
                下一次调用被拒绝的结果。

            no_new_summary:
                没有待摘要旧消息时沿用旧摘要的结果。

    异常：
        RuntimeError：固定演示的首次调用没有通过预期准入。

        内部演示或校验函数若发现固定数据不合法，
        会按各自约定抛出异常。
    """
    successful_flow = demo_day19_context_budget_flow()

    narrow_context_state = build_agent_context_budget_state(
        max_context_tokens=400,
        max_task_tokens=2000,
        reserved_output_tokens_per_call=100,
    )
    required_context_failure = plan_context_trim(
        context_budget_state=narrow_context_state,
        context_blocks=[
            {
                "block_id": "original_goal_and_required_evidence",
                "content": "用户目标及不可丢失的必要依据",
                "estimated_tokens": 350,
                "required": True,
                "drop_priority": 0,
            },
            {
                "block_id": "optional_material",
                "content": "可丢弃的补充资料",
                "estimated_tokens": 100,
                "required": False,
                "drop_priority": 0,
            },
        ],
    )

    short_task_state = build_agent_context_budget_state(
        max_context_tokens=1000,
        max_task_tokens=1000,
        reserved_output_tokens_per_call=200,
    )
    preflight_decision = evaluate_context_budget_request(
        context_budget_state=short_task_state,
        estimated_input_tokens=700,
    )
    if not preflight_decision["allowed"]:
        raise RuntimeError("固定演示的首次调用应通过预算准入")
    usage_result = record_context_budget_usage(
        context_budget_state=short_task_state,
        actual_input_tokens=700,
        actual_output_tokens=150,
    )
    next_request_decision = evaluate_context_budget_request(
        context_budget_state=usage_result["updated_state"],
        estimated_input_tokens=100,
    )

    no_new_summary = apply_rolling_summary_update(
        previous_summary="已有的旧摘要 A",
        conversation_messages=[
            {
                "message_id": "latest-only",
                "role": "user",
                "content": "这一轮的新问题 B",
                "must_keep": True,
            },
        ],
        recent_message_count=1,
        generated_summary=None,
    )

    return {
        "successful_flow": successful_flow,
        "required_context_failure": required_context_failure,
        "task_budget_exhaustion": {
            "preflight_decision": preflight_decision,
            "usage_result": usage_result,
            "next_request_decision": next_request_decision,
        },
        "no_new_summary": no_new_summary,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            demo_day19_context_cost_cases(),
            ensure_ascii=False,
            indent=2,
        )
    )
