"""
V1.10 ToolAgent 多轮参数澄清边界审计脚本。

功能：
    使用 AST（抽象语法树）检查真实 ToolAgent 子图、主图运行入口、
    澄清恢复适配器和参数校验适配器是否保持 V1.10 契约边界。

运行方式：
    python -m scripts.audit_v110_tool_clarification_boundaries
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOOL_AGENT_GRAPH_PATH = Path("src/agents/tool_agent/graph.py")
CLARIFICATION_ADAPTER_PATH = Path(
    "src/agents/tool_agent/adapters/clarification_resume_adapter.py"
)
VALIDATION_ADAPTER_PATH = Path(
    "src/agents/tool_agent/adapters/tool_call_validation_adapter.py"
)
ROUTER_NODE_PATH = Path("src/graph/nodes/router_node.py")
GRAPH_RUN_PATH = Path("src/graph/graph_run.py")
GRAPH_RUNTIME_SERVICE_PATH = Path(
    "src/runtime/services/graph_runtime_service.py"
)

TARGET_PATHS = (
    TOOL_AGENT_GRAPH_PATH,
    CLARIFICATION_ADAPTER_PATH,
    VALIDATION_ADAPTER_PATH,
    ROUTER_NODE_PATH,
    GRAPH_RUN_PATH,
    GRAPH_RUNTIME_SERVICE_PATH,
)

CLARIFICATION_CHECKPOINT_KEYS = {
    "tool_agent_clarification_request",
    "tool_agent_pending_tool_call",
    "tool_agent_pending_original_question",
    "tool_agent_pending_created_at",
}


@dataclass(frozen=True)
class V110BoundaryFinding:
    """
    表示一条 V1.10 边界审计违规记录。

    功能：
        保存规则编号、文件路径、问题说明和可选行号。

    参数：
        code:
            稳定的审计规则编号。
        path:
            相对项目根目录的文件路径。
        message:
            中文违规说明。
        line_number:
            违规代码行号；文件级问题可以为 None。

    返回值：
        V110BoundaryFinding:
            不可变的 dataclass（数据类）记录对象。
    """

    code: str
    path: Path
    message: str
    line_number: int | None = None


def parse_python_file(
    relative_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> ast.Module:
    """
    将 Python 文件解析成 AST 模块对象。

    功能：
        读取 UTF-8 Python 源码并调用 ast.parse，避免注释和普通文本
        被当成真实 import、函数调用或状态字段。

    参数：
        relative_path:
            相对项目根目录的 Python 文件路径。
        project_root:
            项目根目录，测试时可以替换为临时目录。

    返回值：
        ast.Module:
            Python 文件对应的 AST 根节点。
    """

    file_path = project_root / relative_path
    source = file_path.read_text(
        encoding="utf-8-sig"
    )
    return ast.parse(
        source=source,
        filename=str(file_path),
    )


def find_function(
    syntax_tree: ast.Module,
    function_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """
    在 AST 中查找指定函数定义。

    功能：
        遍历模块级和嵌套函数，返回名称匹配的第一个函数节点。

    参数：
        syntax_tree:
            ast.parse 返回的 AST 模块对象。
        function_name:
            要查找的函数名称。

    返回值：
        ast.FunctionDef | ast.AsyncFunctionDef | None:
            找到时返回函数 AST 节点，否则返回 None。
    """

    for node in ast.walk(
        syntax_tree
    ):
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ) and node.name == function_name:
            return node

    return None


def collect_string_constants(
    node: ast.AST,
) -> set[str]:
    """
    收集 AST 节点中的字符串常量。

    功能：
        遍历指定节点，只收集 Python 语法中的真实字符串常量；
        注释不会进入 AST，同时显式排除模块、类和函数 docstring，
        防止只在说明文字中写出契约字段也通过审计。

    参数：
        node:
            要遍历的 AST 节点。

    返回值：
        set[str]:
            去重后的字符串常量集合。
    """

    docstring_constant_ids: set[int] = set()
    for parent in ast.walk(
        node
    ):
        if not isinstance(
            parent,
            (
                ast.Module,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ) or not parent.body:
            continue

        first_statement = parent.body[0]
        if isinstance(
            first_statement,
            ast.Expr,
        ) and isinstance(
            first_statement.value,
            ast.Constant,
        ) and isinstance(
            first_statement.value.value,
            str,
        ):
            docstring_constant_ids.add(
                id(first_statement.value)
            )

    return {
        child.value
        for child in ast.walk(
            node
        )
        if isinstance(
            child,
            ast.Constant,
        ) and isinstance(
            child.value,
            str,
        )
        and id(child) not in docstring_constant_ids
    }


def collect_referenced_names(
    node: ast.AST,
) -> set[str]:
    """
    收集 AST 节点中真实引用的变量或函数名称。

    功能：
        读取 ast.Name 节点的 id，例如条件边传入的
        route_after_tool_catalog 函数引用。

    参数：
        node:
            要遍历的 AST 节点。

    返回值：
        set[str]:
            节点中引用过的名称集合。
    """

    return {
        child.id
        for child in ast.walk(
            node
        )
        if isinstance(
            child,
            ast.Name,
        )
    }


def collect_called_names(
    node: ast.AST,
) -> list[tuple[str, int]]:
    """
    收集 AST 节点中的函数调用名称及行号。

    功能：
        同时识别 func() 和 object.method() 两种调用形式，
        用于检查真实调用关系及调用先后顺序。

    参数：
        node:
            要遍历的 AST 节点。

    返回值：
        list[tuple[str, int]]:
            每项为“调用名称、代码行号”。
    """

    calls: list[tuple[str, int]] = []
    for child in ast.walk(
        node
    ):
        if not isinstance(
            child,
            ast.Call,
        ):
            continue

        if isinstance(
            child.func,
            ast.Name,
        ):
            calls.append(
                (
                    child.func.id,
                    child.lineno,
                )
            )
        elif isinstance(
            child.func,
            ast.Attribute,
        ):
            calls.append(
                (
                    child.func.attr,
                    child.lineno,
                )
            )

    return calls


def build_missing_symbol_finding(
    path: Path,
    symbol: str,
    reason: str,
) -> V110BoundaryFinding:
    """
    构建缺少必需符号的审计记录。

    参数：
        path:
            违规文件路径。
        symbol:
            缺少的函数名、字段或关键状态值。
        reason:
            该符号必须存在的原因。

    返回值：
        V110BoundaryFinding:
            文件级缺失符号记录。
    """

    return V110BoundaryFinding(
        code="missing_required_contract",
        path=path,
        message=f"缺少 {symbol}（{reason}）。",
    )


def audit_tool_agent_graph(
    syntax_tree: ast.Module,
) -> list[V110BoundaryFinding]:
    """
    审计真实 ToolAgent 子图的参数澄清路由。

    功能：
        检查工具目录后条件路由函数是否存在，部分补参是否可以直接进入
        工具澄清节点，以及参数校验后的澄清路由是否仍然保留。

    参数：
        syntax_tree:
            ToolAgent graph.py 的 AST 模块对象。

    返回值：
        list[V110BoundaryFinding]:
            子图边界违规列表。
    """

    findings: list[V110BoundaryFinding] = []
    builder = find_function(
        syntax_tree,
        "build_tool_agent_graph",
    )
    route_after_catalog = find_function(
        syntax_tree,
        "route_after_tool_catalog",
    )
    route_after_validate = find_function(
        syntax_tree,
        "route_after_tool_validate",
    )

    for function_name, function_node in (
        ("build_tool_agent_graph", builder),
        ("route_after_tool_catalog", route_after_catalog),
        ("route_after_tool_validate", route_after_validate),
    ):
        if function_node is None:
            findings.append(
                build_missing_symbol_finding(
                    path=TOOL_AGENT_GRAPH_PATH,
                    symbol=function_name,
                    reason="真实 ToolAgent 子图必须保留 V1.10 参数澄清边界",
                )
            )

    if builder is not None:
        names = collect_referenced_names(
            builder
        )
        strings = collect_string_constants(
            builder
        )
        for symbol in (
            "route_after_tool_catalog",
            "route_after_tool_validate",
        ):
            if symbol not in names:
                findings.append(
                    build_missing_symbol_finding(
                        path=TOOL_AGENT_GRAPH_PATH,
                        symbol=symbol,
                        reason="真实子图必须通过条件边调用该路由函数",
                    )
                )

        for state_value in (
            "tool_catalog",
            "tool_parse",
            "tool_validate",
            "tool_clarification",
            "response_adapter",
        ):
            if state_value not in strings:
                findings.append(
                    build_missing_symbol_finding(
                        path=TOOL_AGENT_GRAPH_PATH,
                        symbol=state_value,
                        reason="真实子图缺少必要节点或条件边目标",
                    )
                )

    if route_after_catalog is not None:
        strings = collect_string_constants(
            route_after_catalog
        )
        for state_value in (
            "tool_agent_clarification_resolution",
            "action",
            "partial",
            "clarification",
            "parse",
        ):
            if state_value not in strings:
                findings.append(
                    build_missing_symbol_finding(
                        path=TOOL_AGENT_GRAPH_PATH,
                        symbol=state_value,
                        reason="工具目录后路由必须识别部分补参并区分澄清与解析",
                    )
                )

    if route_after_validate is not None and (
        "TOOL_AGENT_CLARIFICATION_REQUEST_STATE_KEY"
        not in collect_referenced_names(
            route_after_validate
        )
    ):
        findings.append(
            build_missing_symbol_finding(
                path=TOOL_AGENT_GRAPH_PATH,
                symbol="TOOL_AGENT_CLARIFICATION_REQUEST_STATE_KEY",
                reason="首次缺参必须在参数校验后路由到工具澄清节点",
            )
        )

    return findings


def audit_clarification_adapter(
    syntax_tree: ast.Module,
) -> list[V110BoundaryFinding]:
    """
    审计参数澄清恢复适配器契约。

    参数：
        syntax_tree:
            clarification_resume_adapter.py 的 AST 模块对象。

    返回值：
        list[V110BoundaryFinding]:
            缺失恢复动作或核心函数的违规列表。
    """

    findings: list[V110BoundaryFinding] = []
    module_strings = collect_string_constants(
        syntax_tree
    )
    for action in (
        "partial",
        "resumed",
        "cancelled",
        "new_question",
    ):
        if action not in module_strings:
            findings.append(
                build_missing_symbol_finding(
                    path=CLARIFICATION_ADAPTER_PATH,
                    symbol=action,
                    reason="多轮参数澄清必须区分该恢复动作",
                )
            )

    for function_name in (
        "resolve_tool_clarification_input",
        "match_clarification_candidate",
        "build_clarification_cleanup_update",
    ):
        if find_function(
            syntax_tree,
            function_name,
        ) is None:
            findings.append(
                build_missing_symbol_finding(
                    path=CLARIFICATION_ADAPTER_PATH,
                    symbol=function_name,
                    reason="参数补全、候选匹配和状态清理必须职责分离",
                )
            )

    return findings


def audit_validation_adapter(
    syntax_tree: ast.Module,
) -> list[V110BoundaryFinding]:
    """
    审计工具参数显式输入和澄清恢复校验契约。

    参数：
        syntax_tree:
            tool_call_validation_adapter.py 的 AST 模块对象。

    返回值：
        list[V110BoundaryFinding]:
            缺失显式参数校验或恢复信任标记的违规列表。
    """

    module_strings = collect_string_constants(
        syntax_tree
    )
    findings: list[V110BoundaryFinding] = []
    for symbol in (
        "x-requires-explicit-user-input",
        "implicit_required_arg",
        "tool_agent_clarification_resolution",
        "resumed",
    ):
        if symbol not in module_strings:
            findings.append(
                build_missing_symbol_finding(
                    path=VALIDATION_ADAPTER_PATH,
                    symbol=symbol,
                    reason="必须区分 LLM 猜测参数与用户跨轮澄清确认参数",
                )
            )

    return findings


def audit_main_router(
    syntax_tree: ast.Module,
) -> list[V110BoundaryFinding]:
    """
    审计主图语义路由适配节点的澄清处理顺序。

    参数：
        syntax_tree:
            router_node.py 的 AST 模块对象。

    返回值：
        list[V110BoundaryFinding]:
            澄清恢复缺失或执行顺序错误的违规列表。
    """

    function_node = find_function(
        syntax_tree,
        "semantic_router_node",
    )
    if function_node is None:
        return [
            build_missing_symbol_finding(
                path=ROUTER_NODE_PATH,
                symbol="semantic_router_node",
                reason="主图必须保留 RootAgent 兼容路由入口",
            )
        ]

    calls = collect_called_names(
        function_node
    )
    line_by_name = {
        name: line_number
        for name, line_number in calls
        if name in {
            "resolve_tool_clarification_input",
            "root_supervisor_node",
        }
    }
    if "resolve_tool_clarification_input" not in line_by_name:
        return [
            build_missing_symbol_finding(
                path=ROUTER_NODE_PATH,
                symbol="resolve_tool_clarification_input",
                reason="RootAgent 路由前必须先判断本轮输入是否为参数补充",
            )
        ]
    if "root_supervisor_node" not in line_by_name:
        return [
            build_missing_symbol_finding(
                path=ROUTER_NODE_PATH,
                symbol="root_supervisor_node",
                reason="澄清处理后必须继续调用 RootAgent 做粗路由",
            )
        ]
    if (
        line_by_name["resolve_tool_clarification_input"]
        > line_by_name["root_supervisor_node"]
    ):
        return [
            V110BoundaryFinding(
                code="invalid_clarification_order",
                path=ROUTER_NODE_PATH,
                message="参数澄清恢复必须发生在 RootAgent 路由之前。",
                line_number=line_by_name[
                    "resolve_tool_clarification_input"
                ],
            )
        ]

    return []


def read_checkpoint_whitelist(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """
    读取检查点恢复函数中的 clarification_keys 白名单。

    功能：
        查找 clarification_keys 变量赋值，并从 tuple/list/set 字面量中
        提取字符串字段，避免只做全文关键词匹配。

    参数：
        function_node:
            restore_pending_tool_clarification_state 函数 AST 节点。

    返回值：
        set[str]:
            实际配置的检查点恢复字段集合；未找到时返回空集合。
    """

    for node in ast.walk(
        function_node
    ):
        if not isinstance(
            node,
            ast.Assign,
        ):
            continue
        if not any(
            isinstance(target, ast.Name)
            and target.id == "clarification_keys"
            for target in node.targets
        ):
            continue
        if not isinstance(
            node.value,
            (
                ast.Tuple,
                ast.List,
                ast.Set,
            ),
        ):
            return set()

        return {
            element.value
            for element in node.value.elts
            if isinstance(
                element,
                ast.Constant,
            ) and isinstance(
                element.value,
                str,
            )
        }

    return set()


def audit_graph_run(
    syntax_tree: ast.Module,
) -> list[V110BoundaryFinding]:
    """
    审计主图入口的澄清 Checkpoint 白名单恢复。

    参数：
        syntax_tree:
            graph_run.py 的 AST 模块对象。

    返回值：
        list[V110BoundaryFinding]:
            恢复函数、调用关系或白名单不符合要求时的违规列表。
    """

    findings: list[V110BoundaryFinding] = []
    restore_function = find_function(
        syntax_tree,
        "restore_pending_tool_clarification_state",
    )
    runner_function = find_function(
        syntax_tree,
        "run_main_graph_with_result",
    )
    if restore_function is None:
        return [
            build_missing_symbol_finding(
                path=GRAPH_RUN_PATH,
                symbol="restore_pending_tool_clarification_state",
                reason="相同 thread_id 的新一轮请求必须恢复待补参数",
            )
        ]

    actual_keys = read_checkpoint_whitelist(
        restore_function
    )
    if actual_keys != CLARIFICATION_CHECKPOINT_KEYS:
        findings.append(
            V110BoundaryFinding(
                code="invalid_checkpoint_restore_whitelist",
                path=GRAPH_RUN_PATH,
                message=(
                    "参数澄清 Checkpoint 恢复字段必须严格等于白名单；"
                    f"expected={sorted(CLARIFICATION_CHECKPOINT_KEYS)}, "
                    f"actual={sorted(actual_keys)}。"
                ),
                line_number=restore_function.lineno,
            )
        )

    if runner_function is None or (
        "restore_pending_tool_clarification_state"
        not in {
            name
            for name, _ in collect_called_names(
                runner_function
            )
        }
    ):
        findings.append(
            build_missing_symbol_finding(
                path=GRAPH_RUN_PATH,
                symbol="restore_pending_tool_clarification_state 调用",
                reason="主图运行入口必须在新一轮执行前恢复澄清白名单字段",
            )
        )

    return findings


def audit_graph_runtime_service(
    syntax_tree: ast.Module,
) -> list[V110BoundaryFinding]:
    """
    审计主图是否接入真实 ToolAgent 子图。

    参数：
        syntax_tree:
            graph_runtime_service.py 的 AST 模块对象。

    返回值：
        list[V110BoundaryFinding]:
            未调用真实子图或误调用手动链路时的违规列表。
    """

    builder = find_function(
        syntax_tree,
        "_build_tool_agent_node",
    )
    if builder is None:
        return [
            build_missing_symbol_finding(
                path=GRAPH_RUNTIME_SERVICE_PATH,
                symbol="_build_tool_agent_node",
                reason="主图运行时必须集中构建 ToolAgent 子图节点",
            )
        ]

    calls = collect_called_names(
        builder
    )
    call_names = {
        name
        for name, _ in calls
    }
    findings: list[V110BoundaryFinding] = []
    if "build_tool_agent_graph" not in call_names:
        findings.append(
            build_missing_symbol_finding(
                path=GRAPH_RUNTIME_SERVICE_PATH,
                symbol="build_tool_agent_graph",
                reason="真实主图必须接入 LangGraph ToolAgent 子图",
            )
        )
    if "build_tool_agent" in call_names:
        line_number = next(
            line
            for name, line in calls
            if name == "build_tool_agent"
        )
        findings.append(
            V110BoundaryFinding(
                code="manual_tool_agent_in_main_graph",
                path=GRAPH_RUNTIME_SERVICE_PATH,
                message=(
                    "主图不允许接入 build_tool_agent（手动 ToolAgent 链路）；"
                    "真实运行必须以 build_tool_agent_graph 为准。"
                ),
                line_number=line_number,
            )
        )

    return findings


def run_audit(
    project_root: Path = PROJECT_ROOT,
) -> list[V110BoundaryFinding]:
    """
    执行完整 V1.10 ToolAgent 参数澄清边界审计。

    参数：
        project_root:
            项目根目录，测试时可以替换为临时目录。

    返回值：
        list[V110BoundaryFinding]:
            全部违规记录；空列表表示审计通过。
    """

    findings: list[V110BoundaryFinding] = []
    syntax_trees: dict[Path, ast.Module] = {}
    for relative_path in TARGET_PATHS:
        file_path = project_root / relative_path
        if not file_path.is_file():
            findings.append(
                V110BoundaryFinding(
                    code="missing_required_file",
                    path=relative_path,
                    message="V1.10 边界审计所需文件不存在。",
                )
            )
            continue

        try:
            syntax_trees[relative_path] = parse_python_file(
                relative_path=relative_path,
                project_root=project_root,
            )
        except SyntaxError as exc:
            findings.append(
                V110BoundaryFinding(
                    code="syntax_error",
                    path=relative_path,
                    message=f"文件存在语法错误，无法完成 AST 审计：{exc.msg}。",
                    line_number=exc.lineno,
                )
            )

    auditors: tuple[tuple[Path, Any], ...] = (
        (TOOL_AGENT_GRAPH_PATH, audit_tool_agent_graph),
        (CLARIFICATION_ADAPTER_PATH, audit_clarification_adapter),
        (VALIDATION_ADAPTER_PATH, audit_validation_adapter),
        (ROUTER_NODE_PATH, audit_main_router),
        (GRAPH_RUN_PATH, audit_graph_run),
        (GRAPH_RUNTIME_SERVICE_PATH, audit_graph_runtime_service),
    )
    for relative_path, auditor in auditors:
        syntax_tree = syntax_trees.get(
            relative_path
        )
        if syntax_tree is not None:
            findings.extend(
                auditor(
                    syntax_tree
                )
            )

    return findings


def render_findings(
    findings: list[V110BoundaryFinding],
) -> str:
    """
    将审计结果渲染为终端可读文本。

    参数：
        findings:
            审计违规记录列表。

    返回值：
        str:
            PASS 或包含逐条违规信息的 FAIL 报告。
    """

    if not findings:
        return (
            "V1.10 ToolAgent clarification boundary audit: PASS\n"
            "未发现真实子图、参数澄清、Checkpoint 或参数校验边界回退。"
        )

    lines = [
        "V1.10 ToolAgent clarification boundary audit: FAIL",
        f"Findings: {len(findings)}",
        "",
    ]
    for finding in findings:
        location = finding.path.as_posix()
        if finding.line_number is not None:
            location += f":{finding.line_number}"
        lines.extend(
            [
                f"[{finding.code}] {location}",
                f"  {finding.message}",
            ]
        )

    return "\n".join(
        lines
    )


def main() -> int:
    """
    运行命令行审计入口。

    参数：
        无。

    返回值：
        int:
            0 表示审计通过，1 表示发现违规。
    """

    findings = run_audit()
    print(
        render_findings(
            findings
        )
    )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
