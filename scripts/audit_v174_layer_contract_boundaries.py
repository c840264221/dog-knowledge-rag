from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOG_KNOWLEDGE_AGENT_DIR = Path("src/agents/dog_knowledge_agent")
DOG_KNOWLEDGE_AGENT_FILE = Path("src/agents/dog_knowledge_agent/agent.py")
LEGACY_ADAPTER_FILE = Path(
    "src/agents/dog_knowledge_agent/nodes/legacy_state_to_layer_outputs_node.py"
)

FORBIDDEN_IMPORT_MODULES = {
    "src.parser.query_parser": "DogKnowledgeAgent 新主链路不允许 import 旧 query parse 模块。",
    "src.parser.schema": "DogKnowledgeAgent 新主链路不允许 import 旧 query parse schema。",
    "src.agents.exact_search_agent": "DogKnowledgeAgent 新主链路不允许直接 import 旧精确查询 Agent。",
    "src.agents.recommendation_agent": "DogKnowledgeAgent 新主链路不允许直接 import 旧推荐 Agent。",
}

REQUIRED_GRAPH_NODE_NAMES = (
    "query_layer_output",
    "retrieval_layer_output",
    "generation_layer_output",
    "fallback_layer_output",
    "legacy_state_to_layer_outputs",
    "aggregate_layer_outputs",
    "finalize_answer",
)

REQUIRED_GRAPH_FLOW_FRAGMENTS = {
    "retrieve_to_query_layer_output": '"retrieve","query_layer_output"',
    "query_layer_output_to_evaluate": '"query_layer_output","evaluate"',
    "evaluate_rerank_to_retrieval_layer_output": '"rerank":"retrieval_layer_output"',
    "retrieval_layer_output_to_rerank": '"retrieval_layer_output","rerank"',
    "generate_to_generation_layer_output": '"generate","generation_layer_output"',
    "generation_layer_output_to_fallback_layer_output": (
        '"generation_layer_output","fallback_layer_output"'
    ),
    "fallback_layer_output_to_legacy_adapter": (
        '"fallback_layer_output","legacy_state_to_layer_outputs"'
    ),
    "legacy_adapter_to_aggregate": (
        '"legacy_state_to_layer_outputs","aggregate_layer_outputs"'
    ),
    "aggregate_to_finalize": '"aggregate_layer_outputs","finalize_answer"',
}

REQUIRED_LEGACY_ADAPTER_FRAGMENTS = {
    "existing_query_result": "existing_query_result",
    "existing_retrieval_result": "existing_retrieval_result",
    "existing_generation_result": "existing_generation_result",
    "existing_fallback_result": "existing_fallback_result",
    "guard_query_result": "ifnotexisting_query_result:",
    "guard_retrieval_result": "ifnotexisting_retrieval_result:",
    "guard_generation_result": "andnotexisting_generation_result",
    "guard_fallback_result": "andnotexisting_fallback_result",
}


@dataclass(frozen=True)
class AuditIssue:
    """
    V1.7.4 分层契约边界审计问题。

    功能：
        表示审计脚本发现的一条问题，包括问题等级、规则编号、文件路径、
        行号和中文说明。

    参数含义：
        level:
            问题等级。当前使用 error 表示必须修复的问题。
        code:
            规则编号，方便定位是哪类审计失败。
        path:
            出现问题的文件路径。
        message:
            问题的中文解释。
        line:
            出现问题的行号。如果是文件级问题，可以为 None。

    返回值含义：
        dataclass 实例本身用于在 run_audit 中收集和输出审计结果。
    """

    level: str
    code: str
    path: str
    message: str
    line: int | None = None


def collect_python_files(root: Path) -> list[Path]:
    """
    收集指定目录下的 Python 文件。

    功能：
        递归遍历 root 目录，返回所有 .py 文件，用于后续 AST import 审计。

    参数含义：
        root:
            要扫描的目录。

    返回值含义：
        list[Path]:
            Python 文件路径列表。如果目录不存在，返回空列表。
    """

    if not root.exists():
        return []

    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def extract_imports(path: Path) -> list[tuple[str, int]]:
    """
    提取 Python 文件中的 import 路径。

    功能：
        使用 AST（Abstract Syntax Tree，抽象语法树）解析源码，只检查真实
        import 语句，避免把注释、docstring 或字符串返回值误判成违规依赖。

    参数含义：
        path:
            要解析的 Python 文件路径。

    返回值含义：
        list[tuple[str, int]]:
            每一项是 import 路径和对应行号。
    """

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                (alias.name, node.lineno)
                for alias in node.names
            )

        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            imports.extend(
                (f"{module_name}.{alias.name}", node.lineno)
                for alias in node.names
                if module_name
            )

    return imports


def audit_forbidden_imports(
    python_files: list[Path],
    project_root: Path,
) -> list[AuditIssue]:
    """
    审计 DogKnowledgeAgent 是否 import 了禁止的旧链路模块。

    功能：
        检查所有 Python 文件的真实 import 语句，禁止新主链路依赖旧 query parse、
        旧精确查询 Agent 和旧推荐 Agent。

    参数含义：
        python_files:
            要扫描的 Python 文件列表。
        project_root:
            项目根目录，用于生成相对路径。

    返回值含义：
        list[AuditIssue]:
            审计问题列表。没有问题时返回空列表。
    """

    issues: list[AuditIssue] = []

    for path in python_files:
        try:
            imports = extract_imports(path)
        except SyntaxError as exc:
            issues.append(
                AuditIssue(
                    level="error",
                    code="syntax_error",
                    path=to_relative_path(path, project_root),
                    line=exc.lineno,
                    message="文件存在语法错误，无法完成 AST import 审计。",
                )
            )
            continue

        for import_name, line in imports:
            for forbidden_module, message in FORBIDDEN_IMPORT_MODULES.items():
                if (
                    import_name == forbidden_module
                    or import_name.startswith(f"{forbidden_module}.")
                ):
                    issues.append(
                        AuditIssue(
                            level="error",
                            code="forbidden_legacy_import",
                            path=to_relative_path(path, project_root),
                            line=line,
                            message=message,
                        )
                    )

    return issues


def audit_agent_graph(
    agent_file: Path,
    project_root: Path,
) -> list[AuditIssue]:
    """
    审计 DogKnowledgeAgent 主图是否接入 V1.7.4 分层契约节点。

    功能：
        检查 agent.py 中是否包含必要的 layer output 节点和关键边连接，
        确保主图不会绕开 Layer Contract First（分层契约优先）链路。

    参数含义：
        agent_file:
            DogKnowledgeAgent 主图构建文件。
        project_root:
            项目根目录，用于生成相对路径。

    返回值含义：
        list[AuditIssue]:
            审计问题列表。没有问题时返回空列表。
    """

    if not agent_file.exists():
        return [
            AuditIssue(
                level="error",
                code="missing_agent_file",
                path=to_relative_path(agent_file, project_root),
                message="缺少 DogKnowledgeAgent 主图文件，无法审计分层契约接线。",
            )
        ]

    source = agent_file.read_text(encoding="utf-8")
    normalized_source = normalize_source(source)
    issues: list[AuditIssue] = []

    for node_name in REQUIRED_GRAPH_NODE_NAMES:
        if node_name not in source:
            issues.append(
                AuditIssue(
                    level="error",
                    code="missing_layer_contract_node",
                    path=to_relative_path(agent_file, project_root),
                    message=f"主图缺少 V1.7.4 分层契约节点：{node_name}。",
                )
            )

    for rule_name, fragment in REQUIRED_GRAPH_FLOW_FRAGMENTS.items():
        if fragment not in normalized_source:
            issues.append(
                AuditIssue(
                    level="error",
                    code="missing_layer_contract_edge",
                    path=to_relative_path(agent_file, project_root),
                    message=f"主图缺少 V1.7.4 分层契约关键连接：{rule_name}。",
                )
            )

    return issues


def audit_legacy_adapter_guards(
    adapter_file: Path,
    project_root: Path,
) -> list[AuditIssue]:
    """
    审计 legacy adapter 是否保护已生成的标准分层结果。

    功能：
        检查 legacy_state_to_layer_outputs_node.py 中是否存在 existing_* 判断。
        这些判断用于保证真实分层节点已经生成的结果不会被旧 state 适配器覆盖。

    参数含义：
        adapter_file:
            legacy state 到 layer output 的适配节点文件。
        project_root:
            项目根目录，用于生成相对路径。

    返回值含义：
        list[AuditIssue]:
            审计问题列表。没有问题时返回空列表。
    """

    if not adapter_file.exists():
        return [
            AuditIssue(
                level="error",
                code="missing_legacy_adapter_file",
                path=to_relative_path(adapter_file, project_root),
                message="缺少 legacy state 到 layer output 的适配节点文件。",
            )
        ]

    source = adapter_file.read_text(encoding="utf-8")
    normalized_source = normalize_source(source)
    issues: list[AuditIssue] = []

    for rule_name, fragment in REQUIRED_LEGACY_ADAPTER_FRAGMENTS.items():
        if fragment not in normalized_source:
            issues.append(
                AuditIssue(
                    level="error",
                    code="missing_legacy_adapter_guard",
                    path=to_relative_path(adapter_file, project_root),
                    message=f"legacy adapter 缺少保护标准分层结果的逻辑：{rule_name}。",
                )
            )

    return issues


def normalize_source(source: str) -> str:
    """
    归一化源码文本。

    功能：
        删除所有空白字符，让审计脚本可以稳定匹配跨行的 builder.add_edge
        和条件路由字典。

    参数含义：
        source:
            原始源码文本。

    返回值含义：
        str:
            删除空白后的源码文本。
    """

    return re.sub(r"\s+", "", source)


def to_relative_path(path: Path, project_root: Path) -> str:
    """
    将路径转成相对项目根目录的展示格式。

    功能：
        审计报告中使用相对路径，便于阅读和复制定位。

    参数含义：
        path:
            要展示的文件路径。
        project_root:
            项目根目录。

    返回值含义：
        str:
            POSIX 风格的相对路径。
    """

    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def run_audit(project_root: Path | None = None) -> list[AuditIssue]:
    """
    运行 V1.7.4 DogKnowledgeAgent 分层契约边界审计。

    功能：
        统一执行旧依赖 import 审计、主图分层契约接线审计、
        legacy adapter 覆盖保护审计。

    参数含义：
        project_root:
            项目根目录。测试时可以传入临时目录；命令行运行时默认使用当前项目。

    返回值含义：
        list[AuditIssue]:
            审计问题列表。空列表表示审计通过。
    """

    root = project_root or PROJECT_ROOT
    agent_root = root / DOG_KNOWLEDGE_AGENT_DIR
    agent_file = root / DOG_KNOWLEDGE_AGENT_FILE
    adapter_file = root / LEGACY_ADAPTER_FILE

    issues: list[AuditIssue] = []

    if not agent_root.exists():
        issues.append(
            AuditIssue(
                level="error",
                code="missing_dog_knowledge_agent_dir",
                path=to_relative_path(agent_root, root),
                message="缺少 DogKnowledgeAgent 目录，无法执行边界审计。",
            )
        )
        return issues

    issues.extend(
        audit_forbidden_imports(
            python_files=collect_python_files(agent_root),
            project_root=root,
        )
    )
    issues.extend(
        audit_agent_graph(
            agent_file=agent_file,
            project_root=root,
        )
    )
    issues.extend(
        audit_legacy_adapter_guards(
            adapter_file=adapter_file,
            project_root=root,
        )
    )

    return issues


def print_report(issues: list[AuditIssue]) -> None:
    """
    打印审计报告。

    功能：
        将审计结果输出到终端。通过时打印通过信息；失败时逐条打印错误。

    参数含义：
        issues:
            run_audit 返回的审计问题列表。

    返回值含义：
        None:
            该函数只负责输出，不返回业务数据。
    """

    if not issues:
        print("V1.7.4 DogKnowledgeAgent layer contract boundary audit passed.")
        return

    print("V1.7.4 DogKnowledgeAgent layer contract boundary audit failed.")
    print(f"Errors: {len(issues)}")
    print("-" * 100)

    for issue in issues:
        print(f"[{issue.level.upper()}] {issue.code}")
        print(f"  path   : {issue.path}")
        if issue.line is not None:
            print(f"  line   : {issue.line}")
        print(f"  message: {issue.message}")
        print("-" * 100)


def main() -> int:
    """
    命令行入口函数。

    功能：
        执行 V1.7.4 分层契约边界审计，并根据是否存在问题返回退出码。

    参数含义：
        无。

    返回值含义：
        int:
            0 表示审计通过；1 表示审计失败。
    """

    issues = run_audit()
    print_report(issues)
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
