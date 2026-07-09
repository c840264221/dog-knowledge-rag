"""
ToolAgent 模块职责契约。

功能：
    定义 V1.8 之后 ToolAgent（工具智能体）应该逐步具备的职责分层。

当前阶段：
    该模块只描述 ToolAgent 的目标边界。
    V1.8 起主图中的 tool_agent 路由已经开始接入新版 ToolAgent 子图。

设计目标：
    1. 明确 ToolAgent 是工具调用链路的统一入口。
    2. 把工具意图识别、参数解析、权限确认、工具执行、结果格式化拆开。
    3. 允许继续复用当前 src.graph.tools 下的 ToolExecutor、ToolRegistry、
       middleware 等底层工具运行时能力。
    4. 主图工具请求进入新版 ToolAgent 子图，旧 general_qa_agent 工具链路暂时保留兼容。
    5. 为后续接入 ToolNode、LangChain Tool、MCP 和更多工具做准备。

专业名词：
    ToolAgent：工具智能体，专门负责任务中的工具调用流程。
    Contract：契约，表示模块之间约定好的职责、输入和输出。
    Tool Registry：工具注册表，统一保存工具名称、描述、参数和执行函数。
    Tool Runtime：工具运行时，负责超时、重试、日志、追踪等执行治理。
    Permission：权限确认，表示执行工具前是否需要用户确认。
    Tool Result：工具结果，表示工具执行后的结构化返回值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ToolAgentLayer = Literal[
    "entry",
    "intent",
    "argument",
    "permission",
    "execution",
    "result_formatter",
    "debug_report",
]


@dataclass(frozen=True)
class ToolAgentModuleContract:
    """
    ToolAgent 模块职责契约。

    功能：
        描述 ToolAgent 内部某一层的职责、输入、输出和企业级设计原因。

    参数：
        layer:
            模块层级名称，例如 entry、intent、execution。

        module_name:
            模块英文名称。

        chinese_name:
            模块中文名称。

        responsibility:
            模块职责说明。

        expected_input:
            该模块预期输入。

        expected_output:
            该模块预期输出。

        should_not_do:
            该模块不应该承担的职责。

        enterprise_reason:
            企业级项目中为什么需要该职责拆分。

    返回值：
        ToolAgentModuleContract:
            一个不可变的模块职责契约对象。
    """

    layer: ToolAgentLayer
    module_name: str
    chinese_name: str
    responsibility: str
    expected_input: str
    expected_output: str
    should_not_do: tuple[str, ...]
    enterprise_reason: str


TOOL_AGENT_MODULE_CONTRACTS: tuple[
    ToolAgentModuleContract,
    ...
] = (
    ToolAgentModuleContract(
        layer="entry",
        module_name="tool_agent_entry",
        chinese_name="工具智能体入口",
        responsibility=(
            "接收 RootAgent 路由过来的工具类请求，协调 ToolAgent 内部工具调用流程。"
        ),
        expected_input=(
            "DogState，至少包含 question、user_id、session_id、trace_id、"
            "route_decision、tool_calls、tool_results、tool_round 等字段。"
        ),
        expected_output=(
            "更新后的 DogState 字段，例如 need_tool、tool_calls、tool_results、"
            "tool_confirmed、tool_round、final_answer 和 tool debug metadata。"
        ),
        should_not_do=(
            "不直接写死某一个具体工具的业务逻辑。",
            "不直接绕过 Tool Registry 调用外部 API。",
            "不直接生成复杂自然语言最终答案。",
            "不修改 RootAgent 路由决策。",
            "不依赖 general_qa_agent 作为内部实现。",
        ),
        enterprise_reason=(
            "入口层只负责编排工具流程，避免工具选择、权限确认、执行和回答生成混在一起。"
        ),
    ),
    ToolAgentModuleContract(
        layer="intent",
        module_name="tool_intent_router",
        chinese_name="工具意图路由器",
        responsibility=(
            "判断用户问题是否需要工具，以及候选工具类型，例如 weather、date、time。"
        ),
        expected_input=(
            "用户问题、RootAgent route_decision、可选历史 tool_results。"
        ),
        expected_output=(
            "工具意图判断结果，例如 need_tool、candidate_tools、tool_reason。"
        ),
        should_not_do=(
            "不执行真实工具。",
            "不调用外部 API。",
            "不直接写入最终 tool_results。",
            "不处理工具权限确认。",
            "不拼接最终回答。",
        ),
        enterprise_reason=(
            "工具意图单独拆分后，可以独立升级为规则路由、LLM 路由或 ToolNode 条件路由。"
        ),
    ),
    ToolAgentModuleContract(
        layer="argument",
        module_name="tool_argument_builder",
        chinese_name="工具参数构建器",
        responsibility=(
            "根据用户问题和工具意图，构建标准 tool_calls，包括工具名称和参数。"
        ),
        expected_input=(
            "用户问题、candidate_tools、工具参数 schema、可选上下文。"
        ),
        expected_output=(
            "tool_calls，格式通常为 list[dict]，每项包含 name 和 args。"
        ),
        should_not_do=(
            "不执行工具。",
            "不保存工具结果。",
            "不处理重试和超时。",
            "不直接调用 ToolExecutor。",
            "不直接决定是否需要人工确认。",
        ),
        enterprise_reason=(
            "参数构建层独立后，可以单独测试城市、日期、时区等工具参数解析是否正确。"
        ),
    ),
    ToolAgentModuleContract(
        layer="permission",
        module_name="tool_permission_gate",
        chinese_name="工具权限确认门",
        responsibility=(
            "根据工具类型、参数和策略判断是否需要用户确认，必要时触发 interrupt。"
        ),
        expected_input=(
            "tool_calls、用户信息、工具权限策略、当前 session / checkpoint 信息。"
        ),
        expected_output=(
            "tool_confirmed、permission_status，或者 GraphInterruptResult。"
        ),
        should_not_do=(
            "不执行真实工具。",
            "不修改工具参数。",
            "不伪造用户确认结果。",
            "不生成最终答案。",
            "不吞掉用户拒绝工具调用的结果。",
        ),
        enterprise_reason=(
            "权限确认单独拆分后，可以统一处理危险工具、人机确认、审计日志和恢复流程。"
        ),
    ),
    ToolAgentModuleContract(
        layer="execution",
        module_name="tool_execution_runner",
        chinese_name="工具执行器节点",
        responsibility=(
            "调用 ToolExecutor 或未来 ToolNode 执行当前待处理工具，并收集结构化结果。"
        ),
        expected_input=(
            "已确认的 tool_calls、ToolExecutor、Tool Runtime middleware、RuntimeContext。"
        ),
        expected_output=(
            "tool_results、剩余 tool_calls、need_tool、tool_round 和工具执行 metadata。"
        ),
        should_not_do=(
            "不重新解析用户意图。",
            "不重新构建工具参数。",
            "不绕过 Tool Runtime middleware。",
            "不直接访问 RootAgent。",
            "不把工具结果直接当最终答案返回。",
        ),
        enterprise_reason=(
            "执行层独立后，工具超时、重试、追踪、统计和错误处理可以集中治理。"
        ),
    ),
    ToolAgentModuleContract(
        layer="result_formatter",
        module_name="tool_result_formatter",
        chinese_name="工具结果格式化器",
        responsibility=(
            "把结构化工具结果转换成可供回答节点或用户展示的文本和标准响应字段。"
        ),
        expected_input=(
            "tool_results、用户问题、工具意图、可选历史上下文。"
        ),
        expected_output=(
            "tool_answer_draft、final_answer 或 tool_response_contract。"
        ),
        should_not_do=(
            "不执行工具。",
            "不修改原始 tool_results。",
            "不再次请求用户确认。",
            "不修改 checkpoint 配置。",
            "不改写工具运行时指标。",
        ),
        enterprise_reason=(
            "结果格式化层独立后，同一个工具结果可以支持 UI 展示、日志、API 响应和最终回答生成。"
        ),
    ),
    ToolAgentModuleContract(
        layer="debug_report",
        module_name="tool_debug_report",
        chinese_name="工具调试报告",
        responsibility=(
            "整理工具意图、参数、权限、执行结果、耗时、错误和恢复信息。"
        ),
        expected_input=(
            "DogState 中的 tool_calls、tool_results、tool_round、metrics、timeline 等字段。"
        ),
        expected_output=(
            "tool_debug_report 或 Markdown debug report section。"
        ),
        should_not_do=(
            "不影响真实工具执行。",
            "不修改工具结果。",
            "不修改用户确认状态。",
            "不改变最终回答。",
            "不承担主业务决策。",
        ),
        enterprise_reason=(
            "工具调试报告独立后，可以复盘工具为什么被调用、是否确认、是否成功和耗时多少。"
        ),
    ),
)


def get_tool_agent_module_contracts() -> tuple[
    ToolAgentModuleContract,
    ...
]:
    """
    获取 ToolAgent 模块职责契约。

    功能：
        返回 V1.8 之后 ToolAgent 应该逐步具备的内部职责分层。

    参数：
        无。

    返回值：
        tuple[ToolAgentModuleContract, ...]:
            ToolAgent 内部模块职责契约列表。
    """

    return TOOL_AGENT_MODULE_CONTRACTS


def get_tool_agent_contract_by_layer(
    layer: ToolAgentLayer,
) -> ToolAgentModuleContract:
    """
    根据 layer 获取 ToolAgent 模块契约。

    功能：
        通过模块层级名称查找对应的 ToolAgent 职责契约。

    参数：
        layer:
            模块层级名称，例如 entry、intent、execution。

    返回值：
        ToolAgentModuleContract:
            匹配到的模块职责契约。

    异常：
        ValueError:
            如果没有找到对应 layer，则抛出异常。
    """

    for contract in TOOL_AGENT_MODULE_CONTRACTS:
        if contract.layer == layer:
            return contract

    raise ValueError(
        f"未知的 ToolAgent 模块层级: {layer}"
    )


def get_expected_tool_agent_layers() -> tuple[
    ToolAgentLayer,
    ...
]:
    """
    获取 ToolAgent 预期内部层级。

    功能：
        返回 V1.8 之后 ToolAgent 应该逐步具备的职责层列表。

    参数：
        无。

    返回值：
        tuple[ToolAgentLayer, ...]:
            预期内部职责层。
    """

    return tuple(
        contract.layer
        for contract in TOOL_AGENT_MODULE_CONTRACTS
    )


def render_tool_agent_contract_markdown() -> str:
    """
    渲染 ToolAgent 模块职责契约 Markdown。

    功能：
        将 ToolAgent 模块职责契约渲染成 Markdown 文本，
        方便写入项目文档或 Debug Report 附录。

    参数：
        无。

    返回值：
        str:
            Markdown 格式的 ToolAgent 模块职责说明。
    """

    lines = [
        "# ToolAgent 模块职责契约",
        "",
        "该文档由 module_contracts.py 中的结构契约生成。",
        "",
        "## 标准执行顺序",
        "",
        "entry -> intent -> argument -> permission -> execution -> result_formatter -> debug_report",
        "",
    ]

    for contract in TOOL_AGENT_MODULE_CONTRACTS:
        lines.extend(
            [
                f"## {contract.chinese_name} / {contract.module_name}",
                "",
                f"- 层级: `{contract.layer}`",
                f"- 职责: {contract.responsibility}",
                f"- 输入: {contract.expected_input}",
                f"- 输出: {contract.expected_output}",
                f"- 企业级设计原因: {contract.enterprise_reason}",
                "- 不应该做:",
            ]
        )

        for item in contract.should_not_do:
            lines.append(
                f"  - {item}"
            )

        lines.append(
            ""
        )

    return "\n".join(
        lines
    )
