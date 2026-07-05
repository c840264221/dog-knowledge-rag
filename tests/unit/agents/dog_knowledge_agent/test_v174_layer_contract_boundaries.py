from pathlib import Path

from scripts.audit_v174_layer_contract_boundaries import (
    audit_agent_graph,
    audit_forbidden_imports,
    audit_legacy_adapter_guards,
    run_audit,
)


def test_v174_layer_contract_boundary_audit_should_pass_current_code() -> None:
    """
    测试当前项目代码通过 V1.7.4 分层契约边界审计。

    功能：
        直接运行完整 audit，确认当前 DogKnowledgeAgent 没有旧依赖 import，
        并且主图已经接入分层契约节点。

    参数含义：
        无。

    返回值含义：
        None。
    """

    assert run_audit() == []


def test_forbidden_import_audit_should_detect_legacy_query_parse(
    tmp_path: Path,
) -> None:
    """
    测试审计脚本能发现旧 query parse import。

    功能：
        构造一个临时 Python 文件，里面 import 旧 query parse 模块，
        验证 audit_forbidden_imports 会返回错误。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    bad_file = tmp_path / "bad_import.py"
    bad_file.write_text(
        "from src.parser.query_parser import parse_query_with_llm\n",
        encoding="utf-8",
    )

    issues = audit_forbidden_imports(
        python_files=[bad_file],
        project_root=tmp_path,
    )

    assert len(issues) == 1
    assert issues[0].code == "forbidden_legacy_import"


def test_agent_graph_audit_should_detect_missing_layer_contract_edge(
    tmp_path: Path,
) -> None:
    """
    测试审计脚本能发现主图缺少分层契约连接。

    功能：
        构造一个只有节点、没有关键边连接的临时 agent.py，
        验证 audit_agent_graph 会返回分层契约接线错误。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    agent_file = tmp_path / "agent.py"
    agent_file.write_text(
        "\n".join(
            [
                '"query_layer_output"',
                '"retrieval_layer_output"',
                '"generation_layer_output"',
                '"fallback_layer_output"',
                '"legacy_state_to_layer_outputs"',
                '"aggregate_layer_outputs"',
                '"finalize_answer"',
            ]
        ),
        encoding="utf-8",
    )

    issues = audit_agent_graph(
        agent_file=agent_file,
        project_root=tmp_path,
    )

    assert any(
        issue.code == "missing_layer_contract_edge"
        for issue in issues
    )


def test_legacy_adapter_audit_should_detect_missing_existing_guards(
    tmp_path: Path,
) -> None:
    """
    测试审计脚本能发现 legacy adapter 缺少覆盖保护。

    功能：
        构造一个没有 existing_* 保护逻辑的临时适配器文件，
        验证 audit_legacy_adapter_guards 会返回错误。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    adapter_file = tmp_path / "legacy_state_to_layer_outputs_node.py"
    adapter_file.write_text(
        "def adapter(state):\n"
        "    return {'dog_query_result': {}}\n",
        encoding="utf-8",
    )

    issues = audit_legacy_adapter_guards(
        adapter_file=adapter_file,
        project_root=tmp_path,
    )

    assert any(
        issue.code == "missing_legacy_adapter_guard"
        for issue in issues
    )
