"""
ToolAgent boundary audit 契约测试。

这些测试用于保证 V1.8 之后 tool_agent 不会直接依赖 general_qa_agent，
也不会直接 import 当前仍位于旧位置的工具图节点实现。
"""

from __future__ import annotations

from pathlib import Path

from scripts import audit_v180_tool_agent_boundaries as boundary_audit


def write_python_file(
    directory: Path,
    relative_path: str,
    content: str,
) -> Path:
    """
    写入测试用 Python 文件。

    功能：
        在 pytest 临时目录中创建一个 Python 文件，供边界审计脚本扫描。

    参数：
        directory:
            pytest 提供的临时目录。

        relative_path:
            相对临时目录的文件路径。

        content:
            要写入文件的 Python 源码文本。

    返回值：
        Path:
            写入后的文件绝对路径。
    """

    file_path = directory / relative_path
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    file_path.write_text(
        data=content,
        encoding="utf-8",
    )
    return file_path


def test_tool_agent_boundary_audit_should_pass_current_code() -> None:
    """
    测试当前 tool_agent 真实代码是否通过边界审计。

    功能：
        防止后续提交把 general_qa_agent 或旧位置工具图节点直接引入 tool_agent。

    参数：
        无。

    返回值：
        None。
    """

    findings = boundary_audit.run_audit()

    assert findings == []


def test_boundary_audit_should_fail_when_general_qa_agent_import_exists(
    tmp_path: Path,
) -> None:
    """
    测试直接 import general_qa_agent 时审计会失败。

    功能：
        构造一个临时文件，模拟 ToolAgent 反向依赖 general_qa_agent。

    参数：
        tmp_path:
            pytest 提供的临时目录 fixture（测试夹具）。

    返回值：
        None。
    """

    write_python_file(
        directory=tmp_path,
        relative_path="bad_general_agent_import.py",
        content=(
            "from src.agents.general_qa_agent.graph import build_general_qa_graph\n"
            "import src.agents.general_qa_agent.supervisor\n"
        ),
    )

    findings = boundary_audit.run_audit(
        target=tmp_path,
        project_root=tmp_path,
    )

    assert len(findings) == 2
    assert all(
        finding.category == "forbidden_agent_import"
        for finding in findings
    )


def test_boundary_audit_should_fail_when_importing_general_agent_from_src_agents(
    tmp_path: Path,
) -> None:
    """
    测试 from src.agents import general_qa_agent 时审计会失败。

    功能：
        覆盖另一种常见 import 写法，避免绕过边界审计。

    参数：
        tmp_path:
            pytest 提供的临时目录 fixture（测试夹具）。

    返回值：
        None。
    """

    write_python_file(
        directory=tmp_path,
        relative_path="bad_from_src_agents.py",
        content="from src.agents import general_qa_agent\n",
    )

    findings = boundary_audit.run_audit(
        target=tmp_path,
        project_root=tmp_path,
    )

    assert len(findings) == 1
    assert findings[0].pattern == "src.agents.general_qa_agent"


def test_boundary_audit_should_fail_when_old_tool_graph_node_import_exists(
    tmp_path: Path,
) -> None:
    """
    测试直接 import 旧位置工具图节点时审计会失败。

    功能：
        当前旧工具节点可以继续存在，但 ToolAgent 新模块不能直接拿它们当内部实现。

    参数：
        tmp_path:
            pytest 提供的临时目录 fixture（测试夹具）。

    返回值：
        None。
    """

    write_python_file(
        directory=tmp_path,
        relative_path="bad_graph_node_import.py",
        content=(
            "from src.graph.nodes.tool_parse_node import build_tool_parse_node\n"
            "from src.graph.nodes.execute_tool_node import build_execute_tool_node\n"
        ),
    )

    findings = boundary_audit.run_audit(
        target=tmp_path,
        project_root=tmp_path,
    )

    assert len(findings) == 2
    assert all(
        finding.category == "forbidden_graph_node_import"
        for finding in findings
    )


def test_boundary_audit_should_allow_tool_runtime_imports(
    tmp_path: Path,
) -> None:
    """
    测试允许 ToolAgent 依赖工具运行时底座。

    功能：
        ToolAgent 可以复用 ToolExecutor、Tool Registry 和工具 schema，
        因为这些属于工具底层能力，不是 general_qa_agent 业务实现。

    参数：
        tmp_path:
            pytest 提供的临时目录 fixture（测试夹具）。

    返回值：
        None。
    """

    write_python_file(
        directory=tmp_path,
        relative_path="allowed_tool_runtime_import.py",
        content=(
            "from src.graph.tools.runtime.tool_executor import ToolExecutor\n"
            "from src.graph.tools.registry.tool_registry import ToolRegistry\n"
            "from src.graph.tools.schemas.tool_call_schema import ToolCall\n"
        ),
    )

    findings = boundary_audit.run_audit(
        target=tmp_path,
        project_root=tmp_path,
    )

    assert findings == []


def test_boundary_audit_should_not_fail_on_words_in_comments_or_strings(
    tmp_path: Path,
) -> None:
    """
    测试注释或普通字符串中的旧模块名不会触发审计。

    功能：
        ToolAgent 文档可以说明 general_qa_agent 迁移背景，
        但不允许真实 import 它。

    参数：
        tmp_path:
            pytest 提供的临时目录 fixture（测试夹具）。

    返回值：
        None。
    """

    write_python_file(
        directory=tmp_path,
        relative_path="comment_only.py",
        content=(
            "# general_qa_agent currently owns old tool nodes.\n"
            "note = 'src.graph.nodes.tool_parse_node is mentioned in docs only'\n"
        ),
    )

    findings = boundary_audit.run_audit(
        target=tmp_path,
        project_root=tmp_path,
    )

    assert findings == []


def test_boundary_audit_should_fail_when_tool_agent_node_file_name_invalid(
    tmp_path: Path,
) -> None:
    """
    测试 ToolAgent 节点文件命名不符合约定时审计会失败。

    功能：
        ToolAgent nodes 目录下的业务节点文件必须以 _node.py 结尾，
        这样后续查看目录时可以直接区分节点文件和普通工具文件。

    参数：
        tmp_path:
            pytest 提供的临时目录 fixture（测试夹具）。

    返回值：
        None。
    """

    write_python_file(
        directory=tmp_path,
        relative_path="tool_confirm.py",
        content="def node(state):\n    return {}\n",
    )

    findings = boundary_audit.audit_tool_agent_node_file_names(
        nodes_target=tmp_path,
        project_root=tmp_path,
    )

    assert len(findings) == 1
    assert findings[0].category == "invalid_tool_agent_node_filename"


def test_boundary_audit_should_pass_when_tool_agent_node_file_name_valid(
    tmp_path: Path,
) -> None:
    """
    测试 ToolAgent 节点文件命名符合约定时审计通过。

    功能：
        确认 _node.py 文件和 __init__.py 不会触发命名审计错误。

    参数：
        tmp_path:
            pytest 提供的临时目录 fixture（测试夹具）。

    返回值：
        None。
    """

    write_python_file(
        directory=tmp_path,
        relative_path="tool_confirm_node.py",
        content="def node(state):\n    return {}\n",
    )
    write_python_file(
        directory=tmp_path,
        relative_path="__init__.py",
        content="",
    )

    findings = boundary_audit.audit_tool_agent_node_file_names(
        nodes_target=tmp_path,
        project_root=tmp_path,
    )

    assert findings == []


def test_boundary_audit_should_fail_when_main_graph_missing_tool_agent_builder(
    tmp_path: Path,
) -> None:
    """
    测试主图没有构建 ToolAgent 子图时审计会失败。

    功能：
        构造一个只注册普通节点、不调用 build_tool_agent_graph 的 GraphRuntimeService，
        验证审计能发现主图接入缺失。

    参数：
        tmp_path:
            pytest 提供的临时目录 fixture（测试夹具）。

    返回值：
        None。
    """

    graph_file = write_python_file(
        directory=tmp_path,
        relative_path="graph_runtime_service.py",
        content=(
            "def build_graph(graph):\n"
            "    graph.add_node('general', object())\n"
        ),
    )

    findings = boundary_audit.audit_main_graph_tool_agent_integration(
        graph_runtime_service_path=graph_file,
        project_root=tmp_path,
    )

    categories = {
        finding.category
        for finding in findings
    }

    assert "missing_tool_agent_graph_builder" in categories
    assert "missing_tool_agent_graph_node" in categories


def test_boundary_audit_should_pass_when_main_graph_registers_tool_agent(
    tmp_path: Path,
) -> None:
    """
    测试主图构建并注册 ToolAgent 时审计通过。

    功能：
        构造一个最小 GraphRuntimeService 片段，
        验证 build_tool_agent_graph 调用和 graph.add_node('tool_agent', ...)
        会被审计识别为有效接入。

    参数：
        tmp_path:
            pytest 提供的临时目录 fixture（测试夹具）。

    返回值：
        None。
    """

    graph_file = write_python_file(
        directory=tmp_path,
        relative_path="graph_runtime_service.py",
        content=(
            "def build_graph(graph):\n"
            "    tool_agent = build_tool_agent_graph()\n"
            "    graph.add_node('tool_agent', tool_agent)\n"
        ),
    )

    findings = boundary_audit.audit_main_graph_tool_agent_integration(
        graph_runtime_service_path=graph_file,
        project_root=tmp_path,
    )

    assert findings == []


def test_boundary_audit_should_fail_when_tool_agent_route_maps_to_general(
    tmp_path: Path,
) -> None:
    """
    测试 tool_agent 路由被映射回 general 时审计会失败。

    功能：
        V1.8 之后 tool_agent 应该进入独立 ToolAgent 子图，
        不允许重新回退到 general/general_agent。

    参数：
        tmp_path:
            pytest 提供的临时目录 fixture（测试夹具）。

    返回值：
        None。
    """

    route_file = write_python_file(
        directory=tmp_path,
        relative_path="routes.py",
        content=(
            "TOOL_AGENT_NODE = 'general'\n"
            "def build_map():\n"
            "    return {'tool_agent': 'general_agent'}\n"
        ),
    )

    findings = boundary_audit.audit_tool_agent_route_targets(
        route_file_path=route_file,
        project_root=tmp_path,
    )

    assert len(findings) == 2
    assert all(
        finding.category == "tool_agent_route_mapped_to_general"
        for finding in findings
    )


def test_boundary_audit_should_pass_when_tool_agent_route_maps_to_tool_agent(
    tmp_path: Path,
) -> None:
    """
    测试 tool_agent 路由指向 tool_agent 时审计通过。

    功能：
        覆盖 V1.8 正确主图路由配置，确认不会误报。

    参数：
        tmp_path:
            pytest 提供的临时目录 fixture（测试夹具）。

    返回值：
        None。
    """

    route_file = write_python_file(
        directory=tmp_path,
        relative_path="routes.py",
        content=(
            "TOOL_AGENT_NODE = 'tool_agent'\n"
            "def build_map():\n"
            "    return {'tool_agent': TOOL_AGENT_NODE}\n"
        ),
    )

    findings = boundary_audit.audit_tool_agent_route_targets(
        route_file_path=route_file,
        project_root=tmp_path,
    )

    assert findings == []
