from __future__ import annotations

import io
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_LEGACY_PROTOCOL_FILES = {
    Path("src/runtime/resume/legacy_protocol.py"),
}

AUDIT_TARGETS = (
    Path("src/graph/graph_run.py"),
    Path("scripts/ui_run.py"),
    Path("src/runtime/resume"),
    Path("src/runtime/hooks"),
)

FORBIDDEN_LEGACY_PROTOCOL_LITERALS = (
    "__INTERRUPT__:",
    "RESUME:",
)

FORBIDDEN_LEGACY_RUNTIME_CONTEXT_NAMES = (
    "runtime_data",
)


@dataclass(frozen=True)
class AuditIssue:
    """
    V1.7.5 checkpoint resume 边界审计问题。

    功能：
        保存一次审计命中的问题信息，包括规则编号、文件路径、行号和命中内容。

    参数含义：
        code:
            审计规则编号。
        path:
            命中文件相对项目根目录的路径。
        line:
            命中行号，从 1 开始。
        message:
            问题中文说明。
        snippet:
            命中的代码片段。

    返回值含义：
        AuditIssue:
            dataclass 数据对象，用于 run_audit 收集结果和 print_report 输出报告。
    """

    code: str
    path: str
    line: int
    message: str
    snippet: str


def collect_python_files(
    target: Path,
) -> list[Path]:
    """
    收集审计目标中的 Python 文件。

    功能：
        如果 target 是文件，则返回该文件；
        如果 target 是目录，则递归返回目录下所有 .py 文件；
        如果 target 不存在，则返回空列表。

    参数含义：
        target:
            要扫描的文件或目录路径。

    返回值含义：
        list[Path]:
            按路径排序后的 Python 文件列表。
    """

    if target.is_file() and target.suffix == ".py":
        return [
            target,
        ]

    if target.is_dir():
        return sorted(
            path
            for path in target.rglob("*.py")
            if "__pycache__" not in path.parts
        )

    return []


def collect_audit_files(
    project_root: Path = PROJECT_ROOT,
) -> list[Path]:
    """
    收集 V1.7.5 checkpoint resume 边界审计文件。

    功能：
        根据 AUDIT_TARGETS 中定义的目标，收集需要审计的 Python 文件。

    参数含义：
        project_root:
            项目根目录。测试时可以传入临时目录。

    返回值含义：
        list[Path]:
            去重并排序后的 Python 文件列表。
    """

    files: set[Path] = set()

    for relative_target in AUDIT_TARGETS:
        target = project_root / relative_target
        files.update(
            collect_python_files(target)
        )

    return sorted(files)


def to_relative_path(
    path: Path,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """
    将文件路径转换成相对项目根目录的路径。

    功能：
        审计报告使用相对路径，方便用户定位文件。
        如果 path 不在 project_root 下，则返回原路径。

    参数含义：
        path:
            文件路径。
        project_root:
            项目根目录。

    返回值含义：
        Path:
            相对路径或原始路径。
    """

    try:
        return path.relative_to(project_root)
    except ValueError:
        return path


def is_allowed_legacy_protocol_file(
    path: Path,
    project_root: Path = PROJECT_ROOT,
) -> bool:
    """
    判断文件是否允许直接出现旧协议字面量。

    功能：
        旧协议前缀只能集中保存在 legacy_protocol.py 中。
        测试目录不在本审计目标内，因此这里不需要额外放行测试文件。

    参数含义：
        path:
            当前被审计文件。
        project_root:
            项目根目录。

    返回值含义：
        bool:
            True 表示该文件允许出现旧协议字面量；
            False 表示不允许。
    """

    relative_path = to_relative_path(
        path=path,
        project_root=project_root,
    )

    return relative_path in ALLOWED_LEGACY_PROTOCOL_FILES


def iter_string_tokens(
    source: str,
) -> list[tuple[int, str]]:
    """
    提取 Python 源码中的字符串 token。

    功能：
        使用 tokenize（词法切分器）扫描源码，只提取真实字符串字面量。
        注释不会被扫描，从而避免注释里提到旧协议时误报。

    参数含义：
        source:
            Python 源码文本。

    返回值含义：
        list[tuple[int, str]]:
            每一项包含字符串所在行号和字符串 token 原文。
    """

    tokens: list[tuple[int, str]] = []
    token_stream = tokenize.generate_tokens(
        io.StringIO(source).readline
    )

    for token_info in token_stream:
        if token_info.type == tokenize.STRING:
            tokens.append(
                (
                    token_info.start[0],
                    token_info.string,
                )
            )

    return tokens


def iter_name_tokens(
    source: str,
) -> list[tuple[int, str]]:
    """
    提取 Python 源码中的名称 token。

    功能：
        使用 tokenize（词法切分器）扫描源码，只提取真实代码中的名称。
        注释不会被扫描，字符串字面量也不会被当成名称处理。
        主要用于检查 runtime_data 这类旧 RuntimeContext 字段是否回流。

    参数含义：
        source:
            Python 源码文本。

    返回值含义：
        list[tuple[int, str]]:
            每一项包含名称所在行号和名称文本。
    """

    tokens: list[tuple[int, str]] = []
    token_stream = tokenize.generate_tokens(
        io.StringIO(source).readline
    )

    for token_info in token_stream:
        if token_info.type == tokenize.NAME:
            tokens.append(
                (
                    token_info.start[0],
                    token_info.string,
                )
            )

    return tokens


def audit_file(
    path: Path,
    project_root: Path = PROJECT_ROOT,
) -> list[AuditIssue]:
    """
    审计单个文件是否绕过 legacy_protocol 直接使用旧协议字符串。

    功能：
        检查 Python 字符串字面量中是否出现 __INTERRUPT__: 或 RESUME:。
        如果命中文件不是允许文件，则记录为审计错误。

    参数含义：
        path:
            要审计的 Python 文件。
        project_root:
            项目根目录。

    返回值含义：
        list[AuditIssue]:
            当前文件中的审计问题列表。没有问题时返回空列表。
    """

    if is_allowed_legacy_protocol_file(
        path=path,
        project_root=project_root,
    ):
        return []

    issues: list[AuditIssue] = []
    source = path.read_text(encoding="utf-8")
    relative_path = to_relative_path(
        path=path,
        project_root=project_root,
    ).as_posix()

    for line_number, token_text in iter_string_tokens(source):
        for literal in FORBIDDEN_LEGACY_PROTOCOL_LITERALS:
            if literal in token_text:
                issues.append(
                    AuditIssue(
                        code="forbidden_inline_legacy_resume_protocol",
                        path=relative_path,
                        line=line_number,
                        message=(
                            "业务代码不允许直接手写旧版 interrupt/resume 协议前缀；"
                            "请通过 src.runtime.resume.legacy_protocol 中的解析和编码函数处理。"
                        ),
                        snippet=token_text,
                    )
                )

    for line_number, token_text in iter_name_tokens(source):
        for name in FORBIDDEN_LEGACY_RUNTIME_CONTEXT_NAMES:
            if token_text == name:
                issues.append(
                    AuditIssue(
                        code="forbidden_legacy_runtime_context_field",
                        path=relative_path,
                        line=line_number,
                        message=(
                            "V1.7.5 新版 RuntimeContext 不允许继续依赖旧 runtime_data 字段；"
                            "请使用 MetricsScope、StateScope 或 metadata 等新版作用域字段。"
                        ),
                        snippet=token_text,
                    )
                )

    return issues


def run_audit(
    project_root: Path = PROJECT_ROOT,
) -> list[AuditIssue]:
    """
    执行 V1.7.5 checkpoint resume 边界审计。

    功能：
        扫描 graph_run.py、ui_run.py、src/runtime/resume 和 src/runtime/hooks，
        确认旧协议字符串只集中在 legacy_protocol.py，
        并确认 hook / resume 主链路不再依赖旧 RuntimeContext.runtime_data 字段。

    参数含义：
        project_root:
            项目根目录。测试时可以传入临时目录。

    返回值含义：
        list[AuditIssue]:
            所有审计问题。空列表表示审计通过。
    """

    issues: list[AuditIssue] = []

    for path in collect_audit_files(project_root):
        issues.extend(
            audit_file(
                path=path,
                project_root=project_root,
            )
        )

    return issues


def print_report(
    issues: list[AuditIssue],
) -> None:
    """
    打印 V1.7.5 checkpoint resume 边界审计报告。

    功能：
        如果没有问题，打印通过信息；
        如果存在问题，逐条打印文件、行号、说明和命中代码。

    参数含义：
        issues:
            run_audit 返回的审计问题列表。

    返回值含义：
        None:
            该函数只负责终端输出。
    """

    if not issues:
        print("V1.7.5 checkpoint resume boundary audit passed.")
        return

    print("V1.7.5 checkpoint resume boundary audit failed.")
    print(f"Errors: {len(issues)}")
    print("-" * 100)

    for issue in issues:
        print(f"[ERROR] {issue.code}")
        print(f"  path   : {issue.path}")
        print(f"  line   : {issue.line}")
        print(f"  message: {issue.message}")
        print(f"  code   : {issue.snippet}")
        print("-" * 100)


def main() -> int:
    """
    命令行入口函数。

    功能：
        执行 V1.7.5 checkpoint resume 边界审计，并根据结果返回退出码。

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
