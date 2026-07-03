"""
V1.7.2 DogKnowledgeAgent boundary audit.

该脚本用于审计 dog_knowledge_agent 的模块边界，防止后续内部结构收拢时，
重新依赖旧版 query_parse 链路或直接 import 旧版精确查询 / 推荐 Agent。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOG_KNOWLEDGE_AGENT_TARGET = Path("src/agents/dog_knowledge_agent")

FORBIDDEN_QUERY_PARSE_PATTERNS = [
    "src.parser.query_parser",
    "parse_query_with_llm",
    "src.parser.schema",
    "QueryParseResult",
]

FORBIDDEN_LEGACY_AGENT_MODULE_PREFIXES = (
    "src.agents.exact_search_agent",
    "src.agents.recommendation_agent",
)

FORBIDDEN_LEGACY_AGENT_IMPORT_NAMES = {
    "exact_search_agent",
    "recommendation_agent",
}


@dataclass(frozen=True)
class BoundaryFinding:
    """
    表示一次 DogKnowledgeAgent boundary audit（边界审计）的命中结果。

    功能：
        保存违规依赖出现的位置、类型和文本内容。

    参数含义：
        category: 违规类别，例如 query_parse 或 legacy_agent_import。
        path: 命中文件相对项目根目录的路径。
        line_number: 命中的行号，从 1 开始。
        pattern: 命中的禁止模式。
        line_text: 命中行的原始文本。

    返回值含义：
        dataclass 数据对象本身，无额外计算逻辑。
    """

    category: str
    path: Path
    line_number: int
    pattern: str
    line_text: str


def iter_python_files(target: Path) -> list[Path]:
    """
    收集需要审计的 Python 文件。

    功能：
        如果 target 是目录，则递归收集其中所有 .py 文件；
        如果 target 是文件，则只返回该文件；
        如果 target 不存在，则返回空列表。

    参数含义：
        target: 需要扫描的文件或目录路径，可以是绝对路径或相对路径。

    返回值含义：
        list[Path]，按路径排序后的 Python 文件列表。
    """

    if target.is_dir():
        return sorted(target.rglob("*.py"))

    if target.is_file() and target.suffix == ".py":
        return [target]

    return []


def get_relative_path(
        file_path: Path,
        project_root: Path,
) -> Path:
    """
    获取用于报告展示的相对路径。

    功能：
        优先返回 file_path 相对 project_root 的路径；
        如果文件不在 project_root 下，则返回文件名，避免审计测试使用临时目录时报错。

    参数含义：
        file_path: 当前被审计的文件路径。
        project_root: 项目根目录路径。

    返回值含义：
        Path，用于打印到审计报告中的可读路径。
    """

    try:
        return file_path.relative_to(project_root)
    except ValueError:
        return Path(file_path.name)


def audit_query_parse_patterns(
        file_path: Path,
        project_root: Path = PROJECT_ROOT,
) -> list[BoundaryFinding]:
    """
    审计旧版 query_parse 依赖。

    功能：
        按文本扫描 dog_knowledge_agent 文件，禁止出现旧版 query_parse 相关依赖。
        这里使用 raw text audit（原始文本审计），因为这些依赖无论出现在 import、
        类型注解还是直接函数调用中，都代表边界风险。

    参数含义：
        file_path: 待扫描 Python 文件路径。
        project_root: 项目根目录，用于生成相对路径报告。

    返回值含义：
        list[BoundaryFinding]，包含所有 query_parse 命中结果；没有命中时返回空列表。
    """

    findings: list[BoundaryFinding] = []
    relative_path = get_relative_path(
        file_path=file_path,
        project_root=project_root,
    )

    for line_number, line_text in enumerate(
        file_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        for pattern in FORBIDDEN_QUERY_PARSE_PATTERNS:
            if pattern in line_text:
                findings.append(
                    BoundaryFinding(
                        category="query_parse",
                        path=relative_path,
                        line_number=line_number,
                        pattern=pattern,
                        line_text=line_text.strip(),
                    )
                )

    return findings


def is_forbidden_legacy_agent_module(module_name: str | None) -> bool:
    """
    判断 import module 是否指向旧版 Agent。

    功能：
        检查 import 语句中的 module 名称是否以旧版 exact_search_agent
        或 recommendation_agent 路径开头。

    参数含义：
        module_name: AST（抽象语法树）中解析出的模块名，可能为 None。

    返回值含义：
        bool，True 表示该 module 是禁止直接依赖的旧版 Agent 模块。
    """

    if not module_name:
        return False

    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in FORBIDDEN_LEGACY_AGENT_MODULE_PREFIXES
    )


def audit_legacy_agent_imports(
        file_path: Path,
        project_root: Path = PROJECT_ROOT,
) -> list[BoundaryFinding]:
    """
    审计旧版 Agent 直接 import。

    功能：
        使用 AST（抽象语法树）只检查 import 语句，禁止 dog_knowledge_agent
        直接 import 旧版 exact_search_agent / recommendation_agent。
        注释、普通字符串、状态字段名不会被误判。

    参数含义：
        file_path: 待扫描 Python 文件路径。
        project_root: 项目根目录，用于生成相对路径报告。

    返回值含义：
        list[BoundaryFinding]，包含所有旧版 Agent import 命中结果；没有命中时返回空列表。
    """

    source_text = file_path.read_text(encoding="utf-8")
    syntax_tree = ast.parse(
        source=source_text,
        filename=str(file_path),
    )
    source_lines = source_text.splitlines()
    relative_path = get_relative_path(
        file_path=file_path,
        project_root=project_root,
    )
    findings: list[BoundaryFinding] = []

    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if is_forbidden_legacy_agent_module(alias.name):
                    findings.append(
                        BoundaryFinding(
                            category="legacy_agent_import",
                            path=relative_path,
                            line_number=node.lineno,
                            pattern=alias.name,
                            line_text=source_lines[node.lineno - 1].strip(),
                        )
                    )

        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""

            if is_forbidden_legacy_agent_module(module_name):
                findings.append(
                    BoundaryFinding(
                        category="legacy_agent_import",
                        path=relative_path,
                        line_number=node.lineno,
                        pattern=module_name,
                        line_text=source_lines[node.lineno - 1].strip(),
                    )
                )
                continue

            if module_name == "src.agents":
                for alias in node.names:
                    if alias.name in FORBIDDEN_LEGACY_AGENT_IMPORT_NAMES:
                        findings.append(
                            BoundaryFinding(
                                category="legacy_agent_import",
                                path=relative_path,
                                line_number=node.lineno,
                                pattern=f"{module_name}.{alias.name}",
                                line_text=source_lines[node.lineno - 1].strip(),
                            )
                        )

    return findings


def audit_file(
        file_path: Path,
        project_root: Path = PROJECT_ROOT,
) -> list[BoundaryFinding]:
    """
    审计单个 dog_knowledge_agent 文件。

    功能：
        同时执行 query_parse 文本审计和旧版 Agent import 审计。

    参数含义：
        file_path: 待扫描 Python 文件路径。
        project_root: 项目根目录，用于生成相对路径报告。

    返回值含义：
        list[BoundaryFinding]，该文件中的所有边界违规命中结果。
    """

    findings: list[BoundaryFinding] = []
    findings.extend(
        audit_query_parse_patterns(
            file_path=file_path,
            project_root=project_root,
        )
    )
    findings.extend(
        audit_legacy_agent_imports(
            file_path=file_path,
            project_root=project_root,
        )
    )
    return findings


def run_audit(
        target: Path | None = None,
        project_root: Path = PROJECT_ROOT,
) -> list[BoundaryFinding]:
    """
    执行 DogKnowledgeAgent boundary audit（边界审计）。

    功能：
        扫描 dog_knowledge_agent 目录，确认它没有依赖旧版 query_parse，
        也没有直接 import 旧版 exact_search_agent / recommendation_agent。

    参数含义：
        target: 可选扫描目标。默认扫描项目中的 src/agents/dog_knowledge_agent。
        project_root: 项目根目录，用于生成相对路径报告。

    返回值含义：
        list[BoundaryFinding]，全部违规命中结果；没有违规时返回空列表。
    """

    resolved_target = target or project_root / DOG_KNOWLEDGE_AGENT_TARGET
    findings: list[BoundaryFinding] = []

    for file_path in iter_python_files(
        target=resolved_target,
    ):
        findings.extend(
            audit_file(
                file_path=file_path,
                project_root=project_root,
            )
        )

    return findings


def main() -> int:
    """
    命令行入口函数。

    功能：
        执行 V1.7.2 DogKnowledgeAgent 边界审计，并根据结果返回进程退出码。

    参数含义：
        无参数。

    返回值含义：
        int，0 表示审计通过，1 表示发现边界违规依赖。
    """

    findings = run_audit()

    if findings:
        print("V1.7.2 DogKnowledgeAgent boundary audit failed.")
        print("Forbidden boundary dependencies were found:")

        for finding in findings:
            print(
                f"- {finding.path}:{finding.line_number} "
                f"[{finding.category}:{finding.pattern}] {finding.line_text}"
            )

        return 1

    print("V1.7.2 DogKnowledgeAgent boundary audit passed.")
    print("No forbidden query_parse or legacy Agent imports were found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
