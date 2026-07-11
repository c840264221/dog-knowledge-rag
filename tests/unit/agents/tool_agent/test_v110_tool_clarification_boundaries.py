"""V1.10 ToolAgent 多轮参数澄清边界审计测试。"""

from __future__ import annotations

import ast

from scripts.audit_v110_tool_clarification_boundaries import (
    TOOL_AGENT_GRAPH_PATH,
    audit_tool_agent_graph,
    run_audit,
)


def test_v110_tool_clarification_boundary_audit_should_pass_current_code() -> None:
    """测试当前真实 ToolAgent 子图和多轮澄清链路通过完整边界审计。"""

    assert run_audit() == []


def test_graph_audit_should_report_missing_catalog_route() -> None:
    """测试审计器能发现工具目录后缺少部分补参条件路由。"""

    syntax_tree = ast.parse(
        """
def build_tool_agent_graph():
    graph.add_edge("tool_catalog", "tool_parse")

def route_after_tool_validate(state):
    return "valid"
"""
    )

    findings = audit_tool_agent_graph(
        syntax_tree
    )

    assert any(
        finding.path == TOOL_AGENT_GRAPH_PATH
        and "route_after_tool_catalog" in finding.message
        for finding in findings
    )
