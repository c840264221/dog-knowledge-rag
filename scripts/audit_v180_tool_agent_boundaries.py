"""
V1.8.0 ToolAgent boundary audit.

该脚本用于审计 ToolAgent 的模块边界，防止后续拆分工具链路时，
让 ToolAgent 反向依赖 general_qa_agent 或直接绑定旧主图节点实现。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_AGENT_TARGET = Path("src/agents/tool_agent")
TOOL_AGENT_NODES_TARGET = Path("src/agents/tool_agent/nodes")
GRAPH_RUNTIME_SERVICE_TARGET = Path("src/runtime/services/graph_runtime_service.py")
ROOT_AGENT_ROUTES_TARGET = Path("src/agents/root_agent/routes.py")
MAIN_ROUTE_ALIAS_TARGET = Path("src/graph/routes/main_route_alias.py")

FORBIDDEN_MODULE_PREFIXES = (
    "src.agents.general_qa_agent",
)

FORBIDDEN_IMPORT_NAMES_FROM_SRC_AGENTS = {
    "general_qa_agent",
}

FORBIDDEN_GRAPH_NODE_IMPORTS = {
    "src.graph.nodes.tool_parse_node",
    "src.graph.nodes.ask_confirm_tool_node",
    "src.graph.nodes.execute_tool_node",
}

FORBIDDEN_TOOL_AGENT_ROUTE_TARGETS = {
    "general",
    "general_agent",
}


@dataclass(frozen=True)
class ToolAgentBoundaryFinding:
    """
    表示一次 ToolAgent boundary audit（边界审计）的命中结果。

    功能：
        保存违规依赖出现的位置、类型和文本内容。

    参数：
        category:
            违规类别，例如 forbidden_agent_import 或 forbidden_graph_node_import。

        path:
            命中文件相对项目根目录的路径。

        line_number:
            命中的行号，从 1 开始。

        pattern:
            命中的禁止模式。

        line_text:
            命中行的原始文本。

    返回值：
        ToolAgentBoundaryFinding:
            dataclass 数据对象本身，无额外计算逻辑。
    """

    category: str
    path: Path
    line_number: int
    pattern: str
    line_text: str


def iter_python_files(
    target: Path,
) -> list[Path]:
    """
    收集需要审计的 Python 文件。

    功能：
        如果 target 是目录，则递归收集其中所有 .py 文件；
        如果 target 是文件，则只返回该文件；
        如果 target 不存在，则返回空列表。

    参数：
        target:
            需要扫描的文件或目录路径，可以是绝对路径或相对路径。

    返回值：
        list[Path]:
            按路径排序后的 Python 文件列表。
    """

    if target.is_dir():
        return sorted(
            target.rglob("*.py")
        )

    if target.is_file() and target.suffix == ".py":
        return [
            target
        ]

    return []


def get_relative_path(
    file_path: Path,
    project_root: Path,
) -> Path:
    """
    获取用于报告展示的相对路径。

    功能：
        优先返回 file_path 相对 project_root 的路径；
        如果文件不在 project_root 下，则返回文件名，避免测试临时目录时报错。

    参数：
        file_path:
            当前被审计的文件路径。

        project_root:
            项目根目录路径。

    返回值：
        Path:
            用于打印到审计报告中的可读路径。
    """

    try:
        return file_path.relative_to(
            project_root
        )
    except ValueError:
        return Path(
            file_path.name
        )


def is_forbidden_agent_module(
    module_name: str | None,
) -> bool:
    """
    判断 import module 是否指向禁止依赖的 Agent 模块。

    功能：
        检查 import 语句中的 module 名称是否以 general_qa_agent 路径开头。

    参数：
        module_name:
            AST（抽象语法树）中解析出的模块名，可能为 None。

    返回值：
        bool:
            True 表示该 module 是 ToolAgent 禁止直接依赖的 Agent 模块。
    """

    if not module_name:
        return False

    return any(
        module_name == prefix or module_name.startswith(
            f"{prefix}."
        )
        for prefix in FORBIDDEN_MODULE_PREFIXES
    )


def is_forbidden_graph_node_module(
    module_name: str | None,
) -> bool:
    """
    判断 import module 是否指向旧位置工具图节点。

    功能：
        V1.8 Step 1 阶段允许旧工具节点继续存在，
        但 ToolAgent 新模块不应该直接 import 旧位置节点作为内部实现。

    参数：
        module_name:
            AST 中解析出的模块名，可能为 None。

    返回值：
        bool:
            True 表示该 module 是 ToolAgent 禁止直接 import 的旧位置工具节点。
    """

    if not module_name:
        return False

    return module_name in FORBIDDEN_GRAPH_NODE_IMPORTS


def audit_imports(
    file_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> list[ToolAgentBoundaryFinding]:
    """
    审计 ToolAgent import 边界。

    功能：
        使用 AST（抽象语法树）只检查 import 语句，禁止 ToolAgent：
        1. 直接 import general_qa_agent。
        2. 直接 import 旧位置工具图节点。

        注释、普通字符串、文档说明不会被误判。

    参数：
        file_path:
            待扫描 Python 文件路径。

        project_root:
            项目根目录，用于生成相对路径报告。

    返回值：
        list[ToolAgentBoundaryFinding]:
            包含所有 import 边界违规命中结果；没有命中时返回空列表。
    """

    source_text = file_path.read_text(
        encoding="utf-8"
    )
    syntax_tree = ast.parse(
        source=source_text,
        filename=str(file_path),
    )
    source_lines = source_text.splitlines()
    relative_path = get_relative_path(
        file_path=file_path,
        project_root=project_root,
    )
    findings: list[ToolAgentBoundaryFinding] = []

    for node in ast.walk(
        syntax_tree
    ):
        if isinstance(
            node,
            ast.Import,
        ):
            for alias in node.names:
                findings.extend(
                    build_findings_for_module(
                        module_name=alias.name,
                        node=node,
                        source_lines=source_lines,
                        relative_path=relative_path,
                    )
                )

        if isinstance(
            node,
            ast.ImportFrom,
        ):
            module_name = node.module or ""

            findings.extend(
                build_findings_for_module(
                    module_name=module_name,
                    node=node,
                    source_lines=source_lines,
                    relative_path=relative_path,
                )
            )

            if module_name == "src.agents":
                for alias in node.names:
                    if alias.name in FORBIDDEN_IMPORT_NAMES_FROM_SRC_AGENTS:
                        findings.append(
                            ToolAgentBoundaryFinding(
                                category="forbidden_agent_import",
                                path=relative_path,
                                line_number=node.lineno,
                                pattern=f"{module_name}.{alias.name}",
                                line_text=source_lines[node.lineno - 1].strip(),
                            )
                        )

    return findings


def build_findings_for_module(
    module_name: str,
    node: ast.AST,
    source_lines: list[str],
    relative_path: Path,
) -> list[ToolAgentBoundaryFinding]:
    """
    根据单个 module 名称构建边界命中结果。

    功能：
        将 import 的 module 名称同时放入 Agent 边界规则和旧工具节点规则中检查。

    参数：
        module_name:
            import 语句中的模块名。

        node:
            当前 AST 节点，用于获取行号。

        source_lines:
            源文件按行拆分后的内容。

        relative_path:
            当前文件相对项目根目录的路径。

    返回值：
        list[ToolAgentBoundaryFinding]:
            当前 module 命中的所有边界问题。
    """

    findings: list[ToolAgentBoundaryFinding] = []
    line_number = getattr(
        node,
        "lineno",
        1,
    )
    line_text = source_lines[line_number - 1].strip()

    if is_forbidden_agent_module(
        module_name
    ):
        findings.append(
            ToolAgentBoundaryFinding(
                category="forbidden_agent_import",
                path=relative_path,
                line_number=line_number,
                pattern=module_name,
                line_text=line_text,
            )
        )

    if is_forbidden_graph_node_module(
        module_name
    ):
        findings.append(
            ToolAgentBoundaryFinding(
                category="forbidden_graph_node_import",
                path=relative_path,
                line_number=line_number,
                pattern=module_name,
                line_text=line_text,
            )
        )

    return findings


def audit_file(
    file_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> list[ToolAgentBoundaryFinding]:
    """
    审计单个 ToolAgent 文件。

    功能：
        对一个 Python 文件执行 ToolAgent import 边界审计。

    参数：
        file_path:
            待扫描 Python 文件路径。

        project_root:
            项目根目录，用于生成相对路径报告。

    返回值：
        list[ToolAgentBoundaryFinding]:
            该文件中的所有边界违规命中结果。
    """

    return audit_imports(
        file_path=file_path,
        project_root=project_root,
    )


def read_python_syntax_tree(
    file_path: Path,
) -> ast.AST:
    """
    读取 Python 文件并解析 AST。

    功能：
        将 Python 源码解析成 AST（抽象语法树），让审计规则只检查真实代码结构，
        避免注释、普通字符串或文档说明造成误报。

    参数：
        file_path:
            待解析的 Python 文件路径。

    返回值：
        ast.AST:
            Python 源码对应的抽象语法树。
    """

    source_text = file_path.read_text(
        encoding="utf-8"
    )

    return ast.parse(
        source=source_text,
        filename=str(file_path),
    )


def audit_tool_agent_node_file_names(
    nodes_target: Path,
    project_root: Path = PROJECT_ROOT,
) -> list[ToolAgentBoundaryFinding]:
    """
    审计 ToolAgent 节点文件命名。

    功能：
        检查 src/agents/tool_agent/nodes 下的节点文件是否以 _node.py 结尾。
        __init__.py 不是业务节点文件，因此跳过。

    参数：
        nodes_target:
            ToolAgent nodes 目录路径。

        project_root:
            项目根目录，用于生成相对路径报告。

    返回值：
        list[ToolAgentBoundaryFinding]:
            命名不符合约定的文件列表；全部符合时返回空列表。
    """

    findings: list[ToolAgentBoundaryFinding] = []

    if not nodes_target.exists():
        return [
            ToolAgentBoundaryFinding(
                category="missing_tool_agent_nodes_dir",
                path=get_relative_path(
                    file_path=nodes_target,
                    project_root=project_root,
                ),
                line_number=1,
                pattern=str(nodes_target),
                line_text="ToolAgent nodes 目录不存在。",
            )
        ]

    for file_path in iter_python_files(
        target=nodes_target,
    ):
        if file_path.name == "__init__.py":
            continue

        if not file_path.name.endswith(
            "_node.py"
        ):
            findings.append(
                ToolAgentBoundaryFinding(
                    category="invalid_tool_agent_node_filename",
                    path=get_relative_path(
                        file_path=file_path,
                        project_root=project_root,
                    ),
                    line_number=1,
                    pattern="*_node.py",
                    line_text=f"ToolAgent 节点文件必须以 _node.py 结尾: {file_path.name}",
                )
            )

    return findings


def has_call_to_name(
    syntax_tree: ast.AST,
    function_name: str,
) -> bool:
    """
    判断 AST 中是否调用了指定函数名。

    功能：
        遍历 AST 的所有 Call 节点，检查是否存在形如 build_tool_agent_graph(...)
        的函数调用。

    参数：
        syntax_tree:
            Python 抽象语法树。

        function_name:
            需要查找的函数名称。

    返回值：
        bool:
            True 表示找到调用；False 表示没有找到。
    """

    for node in ast.walk(
        syntax_tree
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if isinstance(
            node.func,
            ast.Name,
        ) and node.func.id == function_name:
            return True

    return False


def has_graph_add_node_call(
    syntax_tree: ast.AST,
    node_name: str,
) -> bool:
    """
    判断 AST 中是否注册了指定 LangGraph 节点。

    功能：
        查找形如 graph.add_node("tool_agent", tool_agent) 的调用，
        用于确认真实主图已经注册 ToolAgent 节点。

    参数：
        syntax_tree:
            Python 抽象语法树。

        node_name:
            需要检查的主图节点名称。

    返回值：
        bool:
            True 表示找到 add_node 调用；False 表示没有找到。
    """

    for node in ast.walk(
        syntax_tree
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if not isinstance(
            node.func,
            ast.Attribute,
        ):
            continue

        if node.func.attr != "add_node":
            continue

        if not node.args:
            continue

        first_arg = node.args[0]

        if isinstance(
            first_arg,
            ast.Constant,
        ) and first_arg.value == node_name:
            return True

    return False


def audit_main_graph_tool_agent_integration(
    graph_runtime_service_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> list[ToolAgentBoundaryFinding]:
    """
    审计主图是否接入新版 ToolAgent。

    功能：
        检查 GraphRuntimeService 中是否：
        1. 调用了 build_tool_agent_graph。
        2. 注册了 graph.add_node("tool_agent", ...)。

    参数：
        graph_runtime_service_path:
            GraphRuntimeService 文件路径。

        project_root:
            项目根目录，用于生成相对路径报告。

    返回值：
        list[ToolAgentBoundaryFinding]:
            主图接入缺失问题列表；没有问题时返回空列表。
    """

    relative_path = get_relative_path(
        file_path=graph_runtime_service_path,
        project_root=project_root,
    )

    if not graph_runtime_service_path.exists():
        return [
            ToolAgentBoundaryFinding(
                category="missing_graph_runtime_service",
                path=relative_path,
                line_number=1,
                pattern=str(graph_runtime_service_path),
                line_text="GraphRuntimeService 文件不存在。",
            )
        ]

    syntax_tree = read_python_syntax_tree(
        file_path=graph_runtime_service_path,
    )
    findings: list[ToolAgentBoundaryFinding] = []

    if not has_call_to_name(
        syntax_tree=syntax_tree,
        function_name="build_tool_agent_graph",
    ):
        findings.append(
            ToolAgentBoundaryFinding(
                category="missing_tool_agent_graph_builder",
                path=relative_path,
                line_number=1,
                pattern="build_tool_agent_graph",
                line_text="主图没有调用 build_tool_agent_graph。",
            )
        )

    if not has_graph_add_node_call(
        syntax_tree=syntax_tree,
        node_name="tool_agent",
    ):
        findings.append(
            ToolAgentBoundaryFinding(
                category="missing_tool_agent_graph_node",
                path=relative_path,
                line_number=1,
                pattern='graph.add_node("tool_agent", ...)',
                line_text="主图没有注册 tool_agent 节点。",
            )
        )

    return findings


def get_string_assignments(
    syntax_tree: ast.AST,
) -> dict[str, str]:
    """
    提取模块级字符串赋值。

    功能：
        从 AST 中读取形如 TOOL_AGENT_NODE = "tool_agent" 的赋值，
        供路由审计判断常量是否被错误改回 general。

    参数：
        syntax_tree:
            Python 抽象语法树。

    返回值：
        dict[str, str]:
            变量名到字符串值的映射。
    """

    assignments: dict[str, str] = {}

    for node in ast.walk(
        syntax_tree
    ):
        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        if not isinstance(
            node.value,
            ast.Constant,
        ) or not isinstance(
            node.value.value,
            str,
        ):
            continue

        for target in node.targets:
            if isinstance(
                target,
                ast.Name,
            ):
                assignments[target.id] = node.value.value

    return assignments


def resolve_static_string(
    node: ast.AST,
    assignments: dict[str, str],
) -> str | None:
    """
    将 AST 节点解析成静态字符串。

    功能：
        支持直接字符串常量和模块级字符串常量引用，
        例如 "tool_agent" 或 TOOL_AGENT_NODE。

    参数：
        node:
            待解析的 AST 节点。

        assignments:
            模块级字符串赋值表。

    返回值：
        str | None:
            解析成功返回字符串；无法静态解析时返回 None。
    """

    if isinstance(
        node,
        ast.Constant,
    ) and isinstance(
        node.value,
        str,
    ):
        return node.value

    if isinstance(
        node,
        ast.Name,
    ):
        return assignments.get(
            node.id
        )

    return None


def audit_tool_agent_route_targets(
    route_file_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> list[ToolAgentBoundaryFinding]:
    """
    审计 tool_agent 路由是否被映射回 general。

    功能：
        检查路由文件中的 dict 映射和 TOOL_AGENT_NODE 常量，
        防止 V1.8 已接入的 tool_agent 又回退到 general/general_agent。

    参数：
        route_file_path:
            待检查的路由文件路径。

        project_root:
            项目根目录，用于生成相对路径报告。

    返回值：
        list[ToolAgentBoundaryFinding]:
            路由回退问题列表；没有问题时返回空列表。
    """

    if not route_file_path.exists():
        return []

    syntax_tree = read_python_syntax_tree(
        file_path=route_file_path,
    )
    assignments = get_string_assignments(
        syntax_tree=syntax_tree,
    )
    source_lines = route_file_path.read_text(
        encoding="utf-8"
    ).splitlines()
    relative_path = get_relative_path(
        file_path=route_file_path,
        project_root=project_root,
    )
    findings: list[ToolAgentBoundaryFinding] = []

    if assignments.get(
        "TOOL_AGENT_NODE"
    ) in FORBIDDEN_TOOL_AGENT_ROUTE_TARGETS:
        findings.append(
            ToolAgentBoundaryFinding(
                category="tool_agent_route_mapped_to_general",
                path=relative_path,
                line_number=1,
                pattern="TOOL_AGENT_NODE",
                line_text="TOOL_AGENT_NODE 不允许指向 general/general_agent。",
            )
        )

    for node in ast.walk(
        syntax_tree
    ):
        if not isinstance(
            node,
            ast.Dict,
        ):
            continue

        for key_node, value_node in zip(
            node.keys,
            node.values,
        ):
            if key_node is None:
                continue

            key = resolve_static_string(
                node=key_node,
                assignments=assignments,
            )
            value = resolve_static_string(
                node=value_node,
                assignments=assignments,
            )

            if key == "tool_agent" and value in FORBIDDEN_TOOL_AGENT_ROUTE_TARGETS:
                line_number = getattr(
                    node,
                    "lineno",
                    1,
                )
                findings.append(
                    ToolAgentBoundaryFinding(
                        category="tool_agent_route_mapped_to_general",
                        path=relative_path,
                        line_number=line_number,
                        pattern=f"{key}->{value}",
                        line_text=source_lines[line_number - 1].strip(),
                    )
                )

    return findings


def run_audit(
    target: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> list[ToolAgentBoundaryFinding]:
    """
    执行 ToolAgent boundary audit（边界审计）。

    功能：
        扫描 ToolAgent 目录和主图接入点，确认：
        1. ToolAgent 没有直接依赖 general_qa_agent。
        2. ToolAgent 没有直接 import 当前仍位于旧位置的工具图节点。
        3. ToolAgent 节点文件命名符合 _node.py 约定。
        4. 主图真实注册 tool_agent 节点。
        5. tool_agent 路由没有回退到 general/general_agent。

    参数：
        target:
            可选扫描目标。默认扫描项目中的 src/agents/tool_agent。

        project_root:
            项目根目录，用于生成相对路径报告。

    返回值：
        list[ToolAgentBoundaryFinding]:
            全部违规命中结果；没有违规时返回空列表。
    """

    resolved_target = target or project_root / TOOL_AGENT_TARGET
    findings: list[ToolAgentBoundaryFinding] = []

    for file_path in iter_python_files(
        target=resolved_target,
    ):
        findings.extend(
            audit_file(
                file_path=file_path,
                project_root=project_root,
            )
        )

    if target is None:
        findings.extend(
            audit_tool_agent_node_file_names(
                nodes_target=project_root / TOOL_AGENT_NODES_TARGET,
                project_root=project_root,
            )
        )
        findings.extend(
            audit_main_graph_tool_agent_integration(
                graph_runtime_service_path=project_root / GRAPH_RUNTIME_SERVICE_TARGET,
                project_root=project_root,
            )
        )

        for route_file_path in (
            project_root / ROOT_AGENT_ROUTES_TARGET,
            project_root / MAIN_ROUTE_ALIAS_TARGET,
        ):
            findings.extend(
                audit_tool_agent_route_targets(
                    route_file_path=route_file_path,
                    project_root=project_root,
                )
            )

    return findings


def main() -> int:
    """
    命令行入口函数。

    功能：
        执行 V1.8.0 ToolAgent 边界审计，并根据结果返回进程退出码。

    参数：
        无。

    返回值：
        int:
            0 表示审计通过，1 表示发现边界违规依赖。
    """

    findings = run_audit()

    if findings:
        print("V1.8.0 ToolAgent 边界审计失败.")
        print("找到ToolAgent禁止的字段 位置和内容在：")

        for finding in findings:
            print(
                f"- {finding.path}:{finding.line_number} "
                f"[{finding.category}:{finding.pattern}] {finding.line_text}"
            )

        return 1

    print("V1.8.0 ToolAgent 边界审计通过.")
    print("没有发现ToolAgent禁止的字段.")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
