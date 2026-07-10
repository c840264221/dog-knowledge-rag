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
from src.agents.tool_agent.nodes.tool_validate_node import (
    ToolValidateNode,
    build_tool_agent_tool_validate_node,
)
from src.agents.tool_agent.nodes.tool_confirm_node import (
    ToolConfirmNode,
    build_tool_agent_tool_confirm_node,
)
from src.agents.tool_agent.nodes.tool_catalog_node import (
    ToolCatalogNode,
    build_tool_agent_tool_catalog_node,
)
from src.agents.tool_agent.nodes.tool_answer_node import (
    ToolAnswerNode,
    build_tool_agent_tool_answer_node,
)
from src.agents.tool_agent.nodes.tool_execute_node import (
    ToolExecuteNode,
    build_tool_agent_tool_execute_node,
    read_tool_catalog_from_state,
    resolve_mcp_client,
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
from src.agents.tool_agent.contracts.tool_catalog_item_schema import (
    ToolCatalogItem,
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
    build_tool_agent_tool_catalog_state_update_with_mcp,
    build_tool_agent_tool_catalog_with_mcp,
    dump_tool_metadata_for_agent,
    get_registered_tool_metadata,
    list_registered_tool_metadata,
    merge_tool_catalog_items,
    merge_tool_metadata_items,
    tool_requires_confirmation,
)
from src.agents.tool_agent.adapters.tool_catalog_item_adapter import (
    MCP_TOOL_SOURCE,
    LOCAL_TOOL_SOURCE,
    build_tool_catalog_item_from_mcp_tool,
    build_tool_catalog_item_from_tool_metadata,
    build_tool_catalog_items_from_mcp_tools,
    build_tool_catalog_items_from_metadata,
    dump_tool_catalog_item_for_state,
    dump_tool_catalog_items_for_state,
)
from src.agents.tool_agent.adapters.tool_call_validation_adapter import (
    TOOL_CALL_VALIDATION_ERRORS_STATE_KEY,
    TOOL_CALL_VALIDATION_INVALID_CALLS_STATE_KEY,
    TOOL_CALL_VALIDATION_OK_STATE_KEY,
    TOOL_CALL_VALIDATION_SKIPPED_STATE_KEY,
    build_tool_call_validation_state_update,
    build_tool_catalog_by_name,
    build_validation_error,
    is_value_matching_json_schema_type,
    normalize_tool_call_for_validation,
    read_required_fields,
    read_schema_properties,
    validate_args_against_input_schema,
    validate_single_tool_call,
    validate_tool_calls_against_catalog,
    validate_tool_calls_from_state,
)
from src.agents.tool_agent.adapters.runtime_adapter import (
    TOOL_AGENT_RUNTIME_RECORDS_STATE_KEY,
    build_failed_tool_result,
    build_runtime_call_id,
    build_tool_agent_runtime_state_update,
    dump_tool_agent_execution_records_for_state,
    execute_tool_call_with_runtime,
    execute_tool_calls_with_runtime,
    resolve_tool_source,
)

__all__ = [
    "TOOL_AGENT_MODULE_CONTRACTS",
    "TOOL_AGENT_RESPONSE_STATE_KEY",
    "TOOL_AGENT_RUNTIME_RECORDS_STATE_KEY",
    "TOOL_AGENT_TOOL_CATALOG_STATE_KEY",
    "TOOL_CALL_VALIDATION_ERRORS_STATE_KEY",
    "TOOL_CALL_VALIDATION_INVALID_CALLS_STATE_KEY",
    "TOOL_CALL_VALIDATION_OK_STATE_KEY",
    "TOOL_CALL_VALIDATION_SKIPPED_STATE_KEY",
    "LOCAL_TOOL_SOURCE",
    "MCP_TOOL_SOURCE",
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
    "ToolCatalogItem",
    "ToolAnswerNode",
    "ToolCatalogNode",
    "ToolConfirmNode",
    "ToolExecuteNode",
    "ToolParseNode",
    "ToolValidateNode",
    "build_failed_tool_result",
    "build_runtime_call_id",
    "build_tool_call_validation_state_update",
    "build_tool_catalog_by_name",
    "build_tool_catalog_item_from_mcp_tool",
    "build_tool_catalog_item_from_tool_metadata",
    "build_tool_catalog_items_from_mcp_tools",
    "build_tool_catalog_items_from_metadata",
    "build_tool_agent",
    "build_tool_agent_graph",
    "build_tool_agent_response_adapter_node",
    "build_tool_agent_response_from_state",
    "build_tool_agent_response_state_update",
    "build_tool_agent_runtime_state_update",
    "build_tool_agent_tool_answer_node",
    "build_tool_agent_tool_catalog_node",
    "build_tool_agent_tool_confirm_node",
    "build_tool_agent_tool_execute_node",
    "build_tool_agent_tool_parse_node",
    "build_tool_agent_tool_validate_node",
    "build_tool_agent_tool_catalog",
    "build_tool_agent_tool_catalog_state_update",
    "build_tool_agent_tool_catalog_state_update_with_mcp",
    "build_tool_agent_tool_catalog_with_mcp",
    "build_validation_error",
    "dump_tool_agent_execution_records_for_state",
    "dump_tool_agent_response_for_state",
    "dump_tool_catalog_item_for_state",
    "dump_tool_catalog_items_for_state",
    "dump_tool_metadata_for_agent",
    "execute_tool_call_with_runtime",
    "execute_tool_calls_with_runtime",
    "get_expected_tool_agent_layers",
    "get_registered_tool_metadata",
    "get_tool_agent_contract_by_layer",
    "get_tool_agent_module_contracts",
    "is_value_matching_json_schema_type",
    "list_registered_tool_metadata",
    "merge_state_update",
    "merge_tool_catalog_items",
    "merge_tool_metadata_items",
    "normalize_tool_call_for_validation",
    "read_required_fields",
    "read_schema_properties",
    "read_tool_catalog_from_state",
    "render_tool_agent_contract_markdown",
    "resolve_mcp_client",
    "resolve_tool_source",
    "route_after_tool_confirm",
    "tool_requires_confirmation",
    "validate_args_against_input_schema",
    "validate_single_tool_call",
    "validate_tool_calls_against_catalog",
    "validate_tool_calls_from_state",
]
