"""
V1.11 Memory（记忆系统）边界审计脚本。

功能：
    使用 AST（Abstract Syntax Tree，抽象语法树）检查记忆节点依赖注入、
    DogKnowledgeAgent 记忆召回路由、DogState 契约和 checkpoint（检查点）序列化边界。

运行方式：
    python -m scripts.audit_v111_memory_boundaries
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MEMORY_EXTRACT_NODE_PATH = Path("src/graph/nodes/memory_extract_node.py")
MEMORY_RETRIEVE_NODE_PATH = Path("src/graph/nodes/memory_retrieve_node.py")
GENERATE_NODE_PATH = Path("src/graph/nodes/generate_node.py")
DOG_STATE_PATH = Path("src/graph/states/dog_state.py")
GRAPH_RUN_PATH = Path("src/graph/graph_run.py")
DOG_KNOWLEDGE_AGENT_PATH = Path("src/agents/dog_knowledge_agent/agent.py")
GRAPH_RUNTIME_SERVICE_PATH = Path("src/runtime/services/graph_runtime_service.py")
MEMORY_RECALL_SERVICE_PATH = Path("src/memory/memory_semantic_recall.py")

NODE_BOUNDARY_PATHS = (
    MEMORY_EXTRACT_NODE_PATH,
    MEMORY_RETRIEVE_NODE_PATH,
    GENERATE_NODE_PATH,
)

REQUIRED_MEMORY_STATE_FIELDS = {
    "memory_saved",
    "memory_extract_result",
    "memory_save_result",
    "memory_context",
    "memory_recall_result",
}


@dataclass(frozen=True)
class V111MemoryBoundaryFinding:
    """
    V1.11 记忆边界违规记录。

    参数：
        code：稳定的审计规则编号。
        path：问题所在的项目相对路径。
        message：中文问题说明。
        line_number：可选的违规代码行号。

    返回值：
        V111MemoryBoundaryFinding：不可变的边界审计结果。
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
    读取 Python 文件并返回 AST 根节点。

    参数：
        relative_path：相对项目根目录的 Python 文件路径。
        project_root：项目根目录，测试时可替换为临时目录。

    返回值：
        ast.Module：ast.parse 产生的模块级抽象语法树。
    """

    file_path = project_root / relative_path
    return ast.parse(
        file_path.read_text(encoding="utf-8-sig"),
        filename=str(file_path),
    )


def find_function(
        syntax_tree: ast.AST,
        function_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """
    在 AST 中查找指定名称的函数。

    参数：
        syntax_tree：需要遍历的 AST 节点。
        function_name：需要查找的函数名称。

    返回值：
        ast.FunctionDef | ast.AsyncFunctionDef | None：找到的函数节点或 None。
    """

    for node in ast.walk(syntax_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == function_name
        ):
            return node
    return None


def get_call_name(call_node: ast.Call) -> str:
    """
    提取函数调用的最后一段名称。

    参数：
        call_node：AST 中的 ast.Call 函数调用节点。

    返回值：
        str：func() 返回 func，object.method() 返回 method。
    """

    if isinstance(call_node.func, ast.Name):
        return call_node.func.id
    if isinstance(call_node.func, ast.Attribute):
        return call_node.func.attr
    return ""


def get_constant_string(node: ast.AST) -> str | None:
    """
    读取 AST 字符串常量。

    参数：
        node：待检查的 AST 节点。

    返回值：
        str | None：字符串常量或 None。
    """

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def has_call_with_positional_strings(
        syntax_tree: ast.AST,
        call_name: str,
        expected_args: tuple[str, ...],
) -> bool:
    """
    检查是否存在指定字符串位置参数的函数调用。

    参数：
        syntax_tree：需要遍历的 AST。
        call_name：目标函数或方法名。
        expected_args：期望的字符串位置参数。

    返回值：
        bool：找到完全匹配的调用时返回 True。
    """

    for node in ast.walk(syntax_tree):
        if not isinstance(node, ast.Call) or get_call_name(node) != call_name:
            continue
        actual_args = tuple(
            get_constant_string(argument)
            for argument in node.args[:len(expected_args)]
        )
        if actual_args == expected_args:
            return True
    return False


def audit_container_boundary(
        syntax_tree: ast.Module,
        relative_path: Path,
) -> list[V111MemoryBoundaryFinding]:
    """
    审计记忆节点是否直接导入或调用全局 Container。

    参数：
        syntax_tree：节点文件的 AST。
        relative_path：当前文件的项目相对路径。

    返回值：
        list[V111MemoryBoundaryFinding]：发现的容器边界违规。
    """

    findings: list[V111MemoryBoundaryFinding] = []
    for node in ast.walk(syntax_tree):
        imported_modules: list[str] = []
        if isinstance(node, ast.Import):
            imported_modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported_modules = [node.module or ""]

        if any(
            module == "src.runtime.container"
            or module.startswith("src.runtime.container.")
            for module in imported_modules
        ):
            findings.append(
                V111MemoryBoundaryFinding(
                    code="MEM001",
                    path=relative_path,
                    message="Memory 节点不允许直接导入 RuntimeContainer。",
                    line_number=node.lineno,
                )
            )

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "container":
                    findings.append(
                        V111MemoryBoundaryFinding(
                            code="MEM002",
                            path=relative_path,
                            message="Memory 节点不允许调用 container.get。",
                            line_number=node.lineno,
                        )
                    )
    return findings


def audit_graph_runtime_injection(
        syntax_tree: ast.Module,
) -> list[V111MemoryBoundaryFinding]:
    """
    审计 GraphRuntimeService 是否在构图时注入记忆抽取节点依赖。

    参数：
        syntax_tree：graph_runtime_service.py 的 AST。

    返回值：
        list[V111MemoryBoundaryFinding]：缺失构建器、参数或节点注册时的问题。
    """

    findings: list[V111MemoryBoundaryFinding] = []
    build_graph = find_function(syntax_tree, "_build_graph")
    if build_graph is None:
        return [V111MemoryBoundaryFinding(
            code="MEM010",
            path=GRAPH_RUNTIME_SERVICE_PATH,
            message="GraphRuntimeService 缺少 _build_graph 构图函数。",
        )]

    builder_calls = [
        node
        for node in ast.walk(build_graph)
        if isinstance(node, ast.Call)
        and get_call_name(node) == "build_memory_extract_node"
    ]
    if not builder_calls:
        findings.append(V111MemoryBoundaryFinding(
            code="MEM011",
            path=GRAPH_RUNTIME_SERVICE_PATH,
            message="主图未通过 build_memory_extract_node 构建记忆抽取节点。",
        ))
    else:
        keyword_names = {
            keyword.arg
            for keyword in builder_calls[0].keywords
            if keyword.arg is not None
        }
        for required_keyword in (
            "llm_provider",
            "memory_provider",
            "checkpoint_manager",
        ):
            if required_keyword not in keyword_names:
                findings.append(V111MemoryBoundaryFinding(
                    code="MEM012",
                    path=GRAPH_RUNTIME_SERVICE_PATH,
                    message=f"记忆抽取节点缺少注入参数：{required_keyword}。",
                    line_number=builder_calls[0].lineno,
                ))

    if not has_call_with_positional_strings(
        build_graph,
        "add_node",
        ("memory_extract",),
    ):
        findings.append(V111MemoryBoundaryFinding(
            code="MEM013",
            path=GRAPH_RUNTIME_SERVICE_PATH,
            message="主图未注册 memory_extract 节点。",
        ))
    return findings


def audit_dog_knowledge_memory_route(
        syntax_tree: ast.Module,
) -> list[V111MemoryBoundaryFinding]:
    """
    审计 DogKnowledgeAgent 生成前的记忆召回路由。

    参数：
        syntax_tree：dog_knowledge_agent/agent.py 的 AST。

    返回值：
        list[V111MemoryBoundaryFinding]：缺失必需节点或边的问题。
    """

    findings: list[V111MemoryBoundaryFinding] = []
    build_agent = find_function(syntax_tree, "build_dog_knowledge_agent")
    if build_agent is None:
        return [V111MemoryBoundaryFinding(
            code="MEM020",
            path=DOG_KNOWLEDGE_AGENT_PATH,
            message="缺少 build_dog_knowledge_agent 构图函数。",
        )]

    required_edges = (
        ("memory_retrieve", "generate"),
    )
    for start, end in required_edges:
        if not has_call_with_positional_strings(
            build_agent,
            "add_edge",
            (start, end),
        ):
            findings.append(V111MemoryBoundaryFinding(
                code="MEM021",
                path=DOG_KNOWLEDGE_AGENT_PATH,
                message=f"DogKnowledgeAgent 缺少记忆边：{start} -> {end}。",
            ))

    names = {
        node.id
        for node in ast.walk(build_agent)
        if isinstance(node, ast.Name)
    }
    strings = {
        node.value
        for node in ast.walk(build_agent)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    if "build_memory_retrieve_node" not in names or "memory_retrieve" not in strings:
        findings.append(V111MemoryBoundaryFinding(
            code="MEM022",
            path=DOG_KNOWLEDGE_AGENT_PATH,
            message="DogKnowledgeAgent 未构建并注册 memory_retrieve 节点。",
        ))

    generation_entry_references = sum(
        1
        for node in ast.walk(build_agent)
        if isinstance(node, ast.Name) and node.id == "generation_entry_node"
    )
    if generation_entry_references < 3:
        findings.append(V111MemoryBoundaryFinding(
            code="MEM023",
            path=DOG_KNOWLEDGE_AGENT_PATH,
            message="evaluate、ask_user 或 rerank 存在绕过统一记忆生成入口的风险。",
        ))
    return findings


def collect_annotated_fields(syntax_tree: ast.Module) -> set[str]:
    """
    收集模块中的注解赋值字段名。

    参数：
        syntax_tree：需要遍历的 AST。

    返回值：
        set[str]：AnnAssign（注解赋值）左侧的变量名集合。
    """

    return {
        node.target.id
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }


def audit_state_contract(
        dog_state_tree: ast.Module,
        graph_run_tree: ast.Module,
) -> list[V111MemoryBoundaryFinding]:
    """
    审计 DogState 记忆字段和新请求初始值。

    参数：
        dog_state_tree：dog_state.py 的 AST。
        graph_run_tree：graph_run.py 的 AST。

    返回值：
        list[V111MemoryBoundaryFinding]：缺少契约字段或初始化时的问题。
    """

    findings: list[V111MemoryBoundaryFinding] = []
    state_fields = collect_annotated_fields(dog_state_tree)
    for field_name in sorted(REQUIRED_MEMORY_STATE_FIELDS - state_fields):
        findings.append(V111MemoryBoundaryFinding(
            code="MEM030",
            path=DOG_STATE_PATH,
            message=f"DogState 缺少记忆字段：{field_name}。",
        ))

    create_initial_state = find_function(graph_run_tree, "create_initial_state")
    initial_strings = {
        node.value
        for node in ast.walk(create_initial_state or graph_run_tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for field_name in sorted(REQUIRED_MEMORY_STATE_FIELDS):
        if field_name not in initial_strings:
            findings.append(V111MemoryBoundaryFinding(
                code="MEM031",
                path=GRAPH_RUN_PATH,
                message=f"初始 state 缺少记忆字段：{field_name}。",
            ))
    return findings


def audit_checkpoint_serialization(
        recall_service_tree: ast.Module,
        retrieve_node_tree: ast.Module,
) -> list[V111MemoryBoundaryFinding]:
    """
    审计 MemoryRecallResult 写入 state 前是否转换为普通字典。

    参数：
        recall_service_tree：memory_semantic_recall.py 的 AST。
        retrieve_node_tree：memory_retrieve_node.py 的 AST。

    返回值：
        list[V111MemoryBoundaryFinding]：自定义对象直接进入 checkpoint 的风险。
    """

    findings: list[V111MemoryBoundaryFinding] = []
    has_model_dump = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "model_dump"
        and isinstance(node.func.value, ast.Call)
        and get_call_name(node.func.value) == "MemoryRecallResult"
        for node in ast.walk(recall_service_tree)
    )
    if not has_model_dump:
        findings.append(V111MemoryBoundaryFinding(
            code="MEM040",
            path=MEMORY_RECALL_SERVICE_PATH,
            message="MemoryRecallResult 写入图状态前必须调用 model_dump 转为普通字典。",
        ))

    for node in ast.walk(retrieve_node_tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "MemoryRecallResult" for alias in node.names):
                findings.append(V111MemoryBoundaryFinding(
                    code="MEM041",
                    path=MEMORY_RETRIEVE_NODE_PATH,
                    message="图节点不应直接构建 MemoryRecallResult 自定义对象。",
                    line_number=node.lineno,
                ))
    return findings


def audit_generate_memory_reuse(
        syntax_tree: ast.Module,
) -> list[V111MemoryBoundaryFinding]:
    """
    审计 generate_node 是否优先复用 state 中的 memory_context。

    参数：
        syntax_tree：generate_node.py 的 AST。

    返回值：
        list[V111MemoryBoundaryFinding]：缺少记忆复用入口时的问题。
    """

    resolver = find_function(syntax_tree, "resolve_memory_text")
    if resolver is None:
        return [V111MemoryBoundaryFinding(
            code="MEM050",
            path=GENERATE_NODE_PATH,
            message="generate_node 缺少 resolve_memory_text 记忆解析入口。",
        )]

    argument_names = {
        argument.arg
        for argument in resolver.args.args
    }
    has_context_return = any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Name)
        and node.value.id == "resolved_memory_context"
        for node in ast.walk(resolver)
    )
    findings: list[V111MemoryBoundaryFinding] = []
    if "memory_context" not in argument_names or not has_context_return:
        findings.append(V111MemoryBoundaryFinding(
            code="MEM051",
            path=GENERATE_NODE_PATH,
            message="generate_node 必须优先返回已召回的 memory_context，避免重复召回。",
        ))
    return findings


def run_audit(
        project_root: Path = PROJECT_ROOT,
) -> list[V111MemoryBoundaryFinding]:
    """
    执行完整 V1.11 记忆边界审计。

    参数：
        project_root：项目根目录，测试时可注入临时代码库。

    返回值：
        list[V111MemoryBoundaryFinding]：全部边界违规，空列表表示通过。
    """

    memory_layer_paths = tuple(
        file_path.relative_to(project_root)
        for file_path in (project_root / "src/memory").rglob("*.py")
    )
    all_paths = tuple(dict.fromkeys((
        *memory_layer_paths,
        *NODE_BOUNDARY_PATHS,
        DOG_STATE_PATH,
        GRAPH_RUN_PATH,
        DOG_KNOWLEDGE_AGENT_PATH,
        GRAPH_RUNTIME_SERVICE_PATH,
        MEMORY_RECALL_SERVICE_PATH,
    )))
    trees = {
        path: parse_python_file(path, project_root)
        for path in all_paths
    }
    findings: list[V111MemoryBoundaryFinding] = []
    for path in (*memory_layer_paths, *NODE_BOUNDARY_PATHS):
        findings.extend(audit_container_boundary(trees[path], path))
    findings.extend(audit_graph_runtime_injection(trees[GRAPH_RUNTIME_SERVICE_PATH]))
    findings.extend(audit_dog_knowledge_memory_route(trees[DOG_KNOWLEDGE_AGENT_PATH]))
    findings.extend(audit_state_contract(trees[DOG_STATE_PATH], trees[GRAPH_RUN_PATH]))
    findings.extend(audit_checkpoint_serialization(
        trees[MEMORY_RECALL_SERVICE_PATH],
        trees[MEMORY_RETRIEVE_NODE_PATH],
    ))
    findings.extend(audit_generate_memory_reuse(trees[GENERATE_NODE_PATH]))
    return findings


def main() -> int:
    """
    运行审计并输出中文报告。

    参数：
        无。

    返回值：
        int：0 表示审计通过，1 表示发现违规。
    """

    findings = run_audit()
    if not findings:
        print("V1.11 Memory boundary audit: PASS")
        return 0

    print("V1.11 Memory boundary audit: FAIL")
    for finding in findings:
        location = str(finding.path)
        if finding.line_number is not None:
            location += f":{finding.line_number}"
        print(f"[{finding.code}] {location} - {finding.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
