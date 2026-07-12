"""V1.11 Memory（记忆系统）边界审计测试。"""

import ast

from scripts.audit_v111_memory_boundaries import (
    DOG_KNOWLEDGE_AGENT_PATH,
    DOG_STATE_PATH,
    GRAPH_RUNTIME_SERVICE_PATH,
    MEMORY_EXTRACT_NODE_PATH,
    MEMORY_RECALL_SERVICE_PATH,
    audit_checkpoint_serialization,
    audit_container_boundary,
    audit_dog_knowledge_memory_route,
    audit_graph_runtime_injection,
    audit_state_contract,
    run_audit,
)


def test_v111_memory_boundary_audit_should_pass_current_code() -> None:
    """测试当前真实 V1.11 Memory 主链路通过完整边界审计。"""

    assert run_audit() == []


def test_container_boundary_should_detect_runtime_container_import() -> None:
    """测试审计器能识别节点直接导入 RuntimeContainer。"""

    syntax_tree = ast.parse(
        "from src.runtime.container.init import container\n"
    )

    findings = audit_container_boundary(
        syntax_tree,
        MEMORY_EXTRACT_NODE_PATH,
    )

    assert any(finding.code == "MEM001" for finding in findings)


def test_container_boundary_should_detect_container_get_call() -> None:
    """测试审计器能识别 container.get 隐式服务查询。"""

    syntax_tree = ast.parse(
        "def node():\n    return container.get('memory')\n"
    )

    findings = audit_container_boundary(
        syntax_tree,
        MEMORY_EXTRACT_NODE_PATH,
    )

    assert any(finding.code == "MEM002" for finding in findings)


def test_container_boundary_should_ignore_comments_and_strings() -> None:
    """测试注释和说明字符串不会被当成真实 Container 依赖。"""

    syntax_tree = ast.parse(
        "# 不要调用 container.get('memory')\n"
        "message = 'src.runtime.container.init'\n"
    )

    assert audit_container_boundary(
        syntax_tree,
        MEMORY_EXTRACT_NODE_PATH,
    ) == []


def test_graph_runtime_audit_should_detect_missing_injection_keyword() -> None:
    """测试主图记忆抽取节点缺少 MemoryProvider 注入时失败。"""

    syntax_tree = ast.parse(
        """
async def _build_graph(self):
    memory_extract_node = build_memory_extract_node(
        llm_provider=self.llm_provider,
        checkpoint_manager=self.checkpoint_provider.manager,
    )
    graph.add_node("memory_extract", memory_extract_node)
"""
    )

    findings = audit_graph_runtime_injection(syntax_tree)

    assert any(
        finding.path == GRAPH_RUNTIME_SERVICE_PATH
        and "memory_provider" in finding.message
        for finding in findings
    )


def test_dog_graph_audit_should_detect_missing_memory_generate_edge() -> None:
    """测试 DogKnowledgeAgent 缺少 memory_retrieve 到 generate 边时失败。"""

    syntax_tree = ast.parse(
        """
def build_dog_knowledge_agent():
    node = build_memory_retrieve_node()
    builder.add_node("memory_retrieve", node)
    generation_entry_node = "memory_retrieve"
    return generation_entry_node
"""
    )

    findings = audit_dog_knowledge_memory_route(syntax_tree)

    assert any(
        finding.path == DOG_KNOWLEDGE_AGENT_PATH
        and finding.code == "MEM021"
        for finding in findings
    )


def test_state_contract_should_detect_missing_memory_field() -> None:
    """测试 DogState 缺少 memory_recall_result 时审计失败。"""

    dog_state_tree = ast.parse(
        """
class DogState:
    memory_saved: bool
    memory_extract_result: dict
    memory_save_result: dict
    memory_context: str
"""
    )
    graph_run_tree = ast.parse(
        """
def create_initial_state():
    return {"memory_context": "", "memory_recall_result": {}}
"""
    )

    findings = audit_state_contract(
        dog_state_tree,
        graph_run_tree,
    )

    assert any(
        finding.path == DOG_STATE_PATH
        and "memory_recall_result" in finding.message
        for finding in findings
    )


def test_checkpoint_audit_should_require_model_dump() -> None:
    """测试 MemoryRecallResult 未 model_dump 时报告 checkpoint 序列化风险。"""

    recall_service_tree = ast.parse(
        """
def retrieve_with_details():
    return MemoryRecallResult(status="empty", reason="none")
"""
    )
    retrieve_node_tree = ast.parse(
        "def build_memory_retrieve_node():\n    return {}\n"
    )

    findings = audit_checkpoint_serialization(
        recall_service_tree,
        retrieve_node_tree,
    )

    assert any(
        finding.path == MEMORY_RECALL_SERVICE_PATH
        and finding.code == "MEM040"
        for finding in findings
    )
