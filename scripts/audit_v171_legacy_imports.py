"""
Audit V1.7.1 main-chain legacy query-parse imports.

该脚本用于检查新的主路由链路是否误用了旧版 query_parse 相关依赖。
如果命中任何禁止模式，脚本会返回非 0 退出码，方便放进 CI（持续集成）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCAN_TARGETS = [
    Path("src/agents/root_agent"),
    Path("src/agents/dog_knowledge_agent"),
    Path("src/agents/collaboration"),
    Path("src/graph/nodes/router_node.py"),
    Path("src/graph/routes/route_after_semantic.py"),
    Path("src/runtime/services/graph_runtime_service.py"),
]

FORBIDDEN_PATTERNS = [
    "src.parser.query_parser",
    "parse_query_with_llm",
    "src.parser.schema",
    "QueryParseResult",
]


@dataclass(frozen=True)
class AuditFinding:
    """
    表示一次 legacy import audit（遗留导入审计）的命中结果。

    功能：
        保存被禁止模式出现的位置和文本内容。

    参数含义：
        path: 命中文件相对项目根目录的路径。
        line_number: 命中的行号，从 1 开始。
        pattern: 命中的禁止字符串。
        line_text: 命中行的原始文本。

    返回值含义：
        dataclass 数据对象本身，无额外计算逻辑。
    """

    path: Path
    line_number: int
    pattern: str
    line_text: str


def iter_python_files(target: Path) -> list[Path]:
    """
    收集需要扫描的 Python 文件。

    功能：
        如果 target 是目录，则递归收集其中的 .py 文件；
        如果 target 是文件，则直接返回该文件；
        如果 target 不存在，则返回空列表。

    参数含义：
        target: 相对项目根目录的扫描目标路径。

    返回值含义：
        list[Path]，需要执行文本审计的 Python 文件绝对路径列表。
    """

    absolute_target = PROJECT_ROOT / target

    if absolute_target.is_dir():
        return sorted(absolute_target.rglob("*.py"))

    if absolute_target.is_file():
        return [absolute_target]

    return []


def audit_file(file_path: Path) -> list[AuditFinding]:
    """
    审计单个文件中的禁止模式。

    功能：
        按行读取文件内容，检查是否包含 V1.7.1 禁止出现在新主链路中的旧版依赖字符串。

    参数含义：
        file_path: 待扫描文件的绝对路径。

    返回值含义：
        list[AuditFinding]，包含该文件中所有命中结果；没有命中时返回空列表。
    """

    findings: list[AuditFinding] = []
    relative_path = file_path.relative_to(PROJECT_ROOT)

    for line_number, line_text in enumerate(
        file_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in line_text:
                findings.append(
                    AuditFinding(
                        path=relative_path,
                        line_number=line_number,
                        pattern=pattern,
                        line_text=line_text.strip(),
                    )
                )

    return findings


def run_audit() -> list[AuditFinding]:
    """
    执行 V1.7.1 legacy import audit（遗留导入审计）。

    功能：
        扫描 RootAgent、DogKnowledgeAgent、多 Agent 协作模块、语义路由
        adapter、路由分发函数和运行时服务，确认新的主链路没有重新引入
        旧版 query_parse 依赖。

    参数含义：
        无参数。

    返回值含义：
        list[AuditFinding]，所有扫描目标中的命中结果；没有命中时返回空列表。
    """

    findings: list[AuditFinding] = []

    for target in SCAN_TARGETS:
        for file_path in iter_python_files(target):
            findings.extend(audit_file(file_path))

    return findings


def main() -> int:
    """
    命令行入口函数。

    功能：
        运行审计并打印结果。若发现旧版依赖残留，则返回 1；否则返回 0。

    参数含义：
        无参数。

    返回值含义：
        int，进程退出码。0 表示通过，1 表示发现 legacy import（遗留导入）残留。
    """

    findings = run_audit()

    if findings:
        print("V1.7.1 legacy import audit failed.")
        print("Forbidden legacy patterns were found:")

        for finding in findings:
            print(
                f"- {finding.path}:{finding.line_number} "
                f"[{finding.pattern}] {finding.line_text}"
            )

        return 1

    print("V1.7.1 legacy import audit passed.")
    print("No legacy query_parse imports remain in the new main chain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
