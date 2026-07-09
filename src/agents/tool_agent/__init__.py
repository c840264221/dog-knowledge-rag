"""
ToolAgent 工具智能体模块。

功能：
    定义 V1.8 之后工具调用链路的独立 Agent 边界。

当前阶段：
    V1.8 正在拆分独立 ToolAgent。
    当前模块已经包含契约、适配器、节点、最小统一入口和 LangGraph 子图。
    主图中的 tool_agent 已开始接入新版 ToolAgent 子图。
"""

from src.agents.tool_agent.contracts.module_contracts import (
    TOOL_AGENT_MODULE_CONTRACTS,
    ToolAgentLayer,
    ToolAgentModuleContract,
    get_expected_tool_agent_layers,
    get_tool_agent_contract_by_layer,
    get_tool_agent_module_contracts,
    render_tool_agent_contract_markdown,
)
from src.agents.tool_agent.agent import (
    ToolAgentNode,
    build_tool_agent,
    merge_state_update,
)
from src.agents.tool_agent.graph import (
    build_tool_agent_graph,
    route_after_tool_confirm,
)
from src.agents.tool_agent.nodes.response_adapter_node import (
    ToolAgentResponseAdapterNode,
    build_tool_agent_response_adapter_node,
)
from src.agents.tool_agent.nodes.tool_parse_node import (
    ToolParseNode,
    build_tool_agent_tool_parse_node,
)
from src.agents.tool_agent.nodes.tool_confirm_node import (
    ToolConfirmNode,
    build_tool_agent_tool_confirm_node,
)
from src.agents.tool_agent.nodes.tool_answer_node import (
    ToolAnswerNode,
    build_tool_agent_tool_answer_node,
)
from src.agents.tool_agent.nodes.tool_execute_node import (
    ToolExecuteNode,
    build_tool_agent_tool_execute_node,
)
from src.agents.tool_agent.contracts.schemas import (
    ToolAgentExecutionRecord,
    ToolAgentIntent,
    ToolAgentPermissionDecision,
    ToolAgentPermissionStatus,
    ToolAgentPlannedCall,
    ToolAgentResponse,
    ToolAgentResponseStatus,
)
from src.agents.tool_agent.adapters.state_adapter import (
    TOOL_AGENT_RESPONSE_STATE_KEY,
    build_tool_agent_response_from_state,
    build_tool_agent_response_state_update,
    dump_tool_agent_response_for_state,
)
from src.agents.tool_agent.adapters.registry_adapter import (
    TOOL_AGENT_TOOL_CATALOG_STATE_KEY,
    build_tool_agent_tool_catalog,
    build_tool_agent_tool_catalog_state_update,
    dump_tool_metadata_for_agent,
    get_registered_tool_metadata,
    list_registered_tool_metadata,
    tool_requires_confirmation,
)
from src.agents.tool_agent.adapters.runtime_adapter import (
    TOOL_AGENT_RUNTIME_RECORDS_STATE_KEY,
    build_failed_tool_result,
    build_runtime_call_id,
    build_tool_agent_runtime_state_update,
    dump_tool_agent_execution_records_for_state,
    execute_tool_call_with_runtime,
    execute_tool_calls_with_runtime,
)

__all__ = [
    "TOOL_AGENT_MODULE_CONTRACTS",
    "TOOL_AGENT_RESPONSE_STATE_KEY",
    "TOOL_AGENT_RUNTIME_RECORDS_STATE_KEY",
    "TOOL_AGENT_TOOL_CATALOG_STATE_KEY",
    "ToolAgentExecutionRecord",
    "ToolAgentIntent",
    "ToolAgentLayer",
    "ToolAgentModuleContract",
    "ToolAgentNode",
    "ToolAgentResponseAdapterNode",
    "ToolAgentPermissionDecision",
    "ToolAgentPermissionStatus",
    "ToolAgentPlannedCall",
    "ToolAgentResponse",
    "ToolAgentResponseStatus",
    "ToolAnswerNode",
    "ToolConfirmNode",
    "ToolExecuteNode",
    "ToolParseNode",
    "build_failed_tool_result",
    "build_runtime_call_id",
    "build_tool_agent",
    "build_tool_agent_graph",
    "build_tool_agent_response_adapter_node",
    "build_tool_agent_response_from_state",
    "build_tool_agent_response_state_update",
    "build_tool_agent_runtime_state_update",
    "build_tool_agent_tool_answer_node",
    "build_tool_agent_tool_confirm_node",
    "build_tool_agent_tool_execute_node",
    "build_tool_agent_tool_parse_node",
    "build_tool_agent_tool_catalog",
    "build_tool_agent_tool_catalog_state_update",
    "dump_tool_agent_execution_records_for_state",
    "dump_tool_agent_response_for_state",
    "dump_tool_metadata_for_agent",
    "execute_tool_call_with_runtime",
    "execute_tool_calls_with_runtime",
    "get_expected_tool_agent_layers",
    "get_registered_tool_metadata",
    "get_tool_agent_contract_by_layer",
    "get_tool_agent_module_contracts",
    "list_registered_tool_metadata",
    "merge_state_update",
    "render_tool_agent_contract_markdown",
    "route_after_tool_confirm",
    "tool_requires_confirmation",
]
