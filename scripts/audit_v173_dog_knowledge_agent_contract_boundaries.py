import argparse
import io
import json
import re
import tokenize
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


AuditSeverity = Literal[
    "error",
    "warning",
]


@dataclass(frozen=True)
class AuditViolation:
    """
    DogKnowledgeAgent 边界审计违规项。

    功能：
        用来描述一次审计发现的问题，包括规则编号、严重等级、文件路径、行号和说明。

    字段说明：
        rule_id:
            审计规则 ID。
            例如 forbidden_legacy_import、missing_required_file。

        severity:
            严重等级。
            error 表示必须修复；
            warning 表示建议关注。

        path:
            发生问题的文件路径。

        message:
            问题说明。

        line_number:
            行号。
            如果是文件级问题，可以为 None。

        line:
            触发问题的原始代码行。
            如果是文件级问题，可以为 None。
    """

    rule_id: str
    severity: AuditSeverity
    path: str
    message: str
    line_number: int | None = None
    line: str | None = None


@dataclass(frozen=True)
class RequiredSymbolRule:
    """
    必须存在的符号审计规则。

    功能：
        用来检查指定文件中是否存在指定 class、function、变量名或关键字符串。

    字段说明：
        relative_path:
            相对项目根目录的文件路径。

        symbols:
            该文件中必须出现的符号列表。

        reason:
            为什么需要这些符号。
    """

    relative_path: str
    symbols: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ForbiddenPatternRule:
    """
    禁止出现的旧链路模式规则。

    功能：
        用来禁止 DogKnowledgeAgent 新主链路重新依赖旧 query_parse、
        exact_search_agent、recommendation_agent 等模块。

    字段说明：
        pattern:
            正则表达式字符串。

        message:
            命中该规则时的错误说明。
    """

    pattern: str
    message: str


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOG_KNOWLEDGE_AGENT_DIR = PROJECT_ROOT / "src" / "agents" / "dog_knowledge_agent"

SKIP_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "legacy",
    "deprecated",
    "archive",
    "archives",
    "backup",
    "backups",
}

REQUIRED_FILES = (
    "src/agents/dog_knowledge_agent/schemas.py",
    "src/agents/dog_knowledge_agent/answer_formatter.py",
    "src/agents/dog_knowledge_agent/response_adapter.py",
    "src/agents/dog_knowledge_agent/nodes/finalize_answer_node.py",
)

REQUIRED_SYMBOL_RULES = (
    RequiredSymbolRule(
        relative_path="src/agents/dog_knowledge_agent/schemas.py",
        symbols=(
            "DogKnowledgeAnswer",
            "DogKnowledgeEvidence",
            "DogKnowledgeRecommendationItem",
            "build_fallback",
            "to_public_dict",
        ),
        reason="v1.7.3 要求 DogKnowledgeAgent 必须有统一 Response Contract 输出模型。",
    ),
    RequiredSymbolRule(
        relative_path="src/agents/dog_knowledge_agent/answer_formatter.py",
        symbols=(
            "DogKnowledgeAnswerFormatter",
            "format_dog_knowledge_answer",
            "recommended_breeds",
            "evidences",
            "confidence",
        ),
        reason="v1.7.3 要求内部 pipeline_result 必须能统一格式化为 DogKnowledgeAnswer。",
    ),
    RequiredSymbolRule(
        relative_path="src/agents/dog_knowledge_agent/response_adapter.py",
        symbols=(
            "DogKnowledgeAgentResponseAdapter",
            "finalize_dog_knowledge_response",
            "finalize_dog_knowledge_state",
            "dog_knowledge_answer",
            "dog_knowledge_answer_public",
            "final_answer",
        ),
        reason="v1.7.3 要求最终出口必须统一生成对象结构、public dict 和旧字段兼容。",
    ),
    RequiredSymbolRule(
        relative_path="src/agents/dog_knowledge_agent/nodes/finalize_answer_node.py",
        symbols=(
            "build_finalize_dog_knowledge_answer_node",
            "finalize_dog_knowledge_answer_node",
            "DogKnowledgeAgentResponseAdapter",
        ),
        reason="v1.7.3 要求 LangGraph 最后一环必须接入 finalize answer node。",
    ),
)

FORBIDDEN_PATTERN_RULES = (
    ForbiddenPatternRule(
        pattern=r"\bquery_parse\b",
        message="DogKnowledgeAgent 新主链路不允许回流旧 query_parse。",
    ),
    ForbiddenPatternRule(
        pattern=r"\bQueryParseResult\b",
        message="DogKnowledgeAgent 新主链路不允许重新依赖旧 QueryParseResult。",
    ),
    ForbiddenPatternRule(
        pattern=r"\bparse_query_with_llm\b",
        message="DogKnowledgeAgent 新主链路不允许重新调用旧 parse_query_with_llm。",
    ),
    ForbiddenPatternRule(
        pattern=r"\bexact_search_agent\b",
        message="DogKnowledgeAgent 新主链路不允许依赖旧 exact_search_agent。",
    ),
    ForbiddenPatternRule(
        pattern=r"\brecommendation_agent\b",
        message="DogKnowledgeAgent 新主链路不允许依赖旧 recommendation_agent。",
    ),
    ForbiddenPatternRule(
        pattern=r"from\s+src\.agents\.exact_search_agent\b",
        message="DogKnowledgeAgent 新主链路不允许 import 旧 exact_search_agent。",
    ),
    ForbiddenPatternRule(
        pattern=r"from\s+src\.agents\.recommendation_agent\b",
        message="DogKnowledgeAgent 新主链路不允许 import 旧 recommendation_agent。",
    ),
)

GRAPH_ENTRY_CANDIDATE_FILES = (
    "src/agents/dog_knowledge_agent/agent.py",
    "src/agents/dog_knowledge_agent/graph.py",
    "src/agents/dog_knowledge_agent/builder.py",
    "src/agents/dog_knowledge_agent/pipeline.py",
)

GRAPH_REQUIRED_ANY_SYMBOLS = (
    "finalize_answer",
    "finalize_dog_knowledge_answer_node",
    "build_finalize_dog_knowledge_answer_node",
)

SMOKE_SCRIPT_PATH = (
    "scripts/smoke_v173_dog_knowledge_answer_contract.py"
)


def to_relative_path(path: Path) -> str:
    """
    将绝对路径转换成相对项目根目录的路径。

    参数：
        path:
            文件路径。

    返回值：
        str:
            相对项目根目录的 POSIX 风格路径。
    """

    return path.relative_to(PROJECT_ROOT).as_posix()


def read_text(path: Path) -> str:
    """
    安全读取文本文件。

    参数：
        path:
            要读取的文件路径。

    返回值：
        str:
            文件内容。
    """

    return path.read_text(encoding="utf-8")


def build_auditable_code_lines(
    content: str,
) -> list[tuple[int, str]]:
    """
    构建去除注释和字符串字面量后的可审计代码行。

    功能：
        使用 Python tokenize 词法切分器读取源码，
        跳过 COMMENT 注释 token 和 STRING 字符串 token，
        只保留 import、变量名、函数名、属性名等真实代码 token，
        避免审计脚本把说明文案、状态字符串、metadata 字符串误判成旧链路依赖。

    参数：
        content:
            Python 源码文本。

    返回值：
        list[tuple[int, str]]:
            可审计代码行列表。
            tuple 第一个值是原始行号；
            第二个值是去除注释和字符串字面量后的代码片段。
    """

    auditable_lines: dict[int, list[str]] = {}
    try:
        token_stream = tokenize.generate_tokens(
            io.StringIO(content).readline
        )

        for token_info in token_stream:
            token_type = token_info.type
            token_string = token_info.string
            start_line = token_info.start[0]

            if token_type in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.COMMENT,
                tokenize.STRING,
            }:
                continue

            if token_type in {
                tokenize.NL,
                tokenize.NEWLINE,
            }:
                continue

            if token_string.strip():
                auditable_lines.setdefault(
                    start_line,
                    [],
                ).append(token_string)

    except tokenize.TokenError:
        return [
            (
                line_number,
                line.split("#", 1)[0].strip(),
            )
            for line_number, line in enumerate(
                content.splitlines(),
                start=1,
            )
            if line.split("#", 1)[0].strip()
        ]

    return [
        (
            line_number,
            " ".join(parts).strip(),
        )
        for line_number, parts in sorted(auditable_lines.items())
        if " ".join(parts).strip()
    ]


def iter_dog_knowledge_python_files() -> list[Path]:
    """
    遍历 DogKnowledgeAgent 目录下需要审计的 Python 文件。

    功能：
        默认跳过 __pycache__、legacy、deprecated、archive、backup 等目录。
        这些目录如果保存旧代码，不应该影响新主链路审计。

    参数：
        无。

    返回值：
        list[Path]:
            Python 文件路径列表。
    """

    if not DOG_KNOWLEDGE_AGENT_DIR.exists():
        return []

    python_files: list[Path] = []

    for path in DOG_KNOWLEDGE_AGENT_DIR.rglob("*.py"):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue

        python_files.append(path)

    return sorted(python_files)


def audit_required_files() -> list[AuditViolation]:
    """
    审计 v1.7.3 必需文件是否存在。

    参数：
        无。

    返回值：
        list[AuditViolation]:
            审计违规列表。
    """

    violations: list[AuditViolation] = []

    for relative_path in REQUIRED_FILES:
        path = PROJECT_ROOT / relative_path

        if not path.exists():
            violations.append(
                AuditViolation(
                    rule_id="missing_required_file",
                    severity="error",
                    path=relative_path,
                    message=f"缺少 v1.7.3 必需文件：{relative_path}",
                )
            )

    return violations


def audit_required_symbols() -> list[AuditViolation]:
    """
    审计 v1.7.3 必需符号是否存在。

    参数：
        无。

    返回值：
        list[AuditViolation]:
            审计违规列表。
    """

    violations: list[AuditViolation] = []

    for rule in REQUIRED_SYMBOL_RULES:
        path = PROJECT_ROOT / rule.relative_path

        if not path.exists():
            continue

        content = read_text(path)

        for symbol in rule.symbols:
            if symbol not in content:
                violations.append(
                    AuditViolation(
                        rule_id="missing_required_symbol",
                        severity="error",
                        path=rule.relative_path,
                        message=(
                            f"文件缺少必需符号：{symbol}。"
                            f"原因：{rule.reason}"
                        ),
                    )
                )

    return violations


def audit_forbidden_legacy_patterns() -> list[AuditViolation]:
    """
    审计 DogKnowledgeAgent 新主链路是否出现旧链路关键词或 import。

    参数：
        无。

    返回值：
        list[AuditViolation]:
            审计违规列表。
    """

    violations: list[AuditViolation] = []

    for path in iter_dog_knowledge_python_files():
        relative_path = to_relative_path(path)
        lines = build_auditable_code_lines(read_text(path))

        for line_number, line in lines:
            for rule in FORBIDDEN_PATTERN_RULES:
                if re.search(rule.pattern, line):
                    violations.append(
                        AuditViolation(
                            rule_id="forbidden_legacy_pattern",
                            severity="error",
                            path=relative_path,
                            message=rule.message,
                            line_number=line_number,
                            line=line.strip(),
                        )
                    )

    return violations


def audit_graph_finalize_integration() -> list[AuditViolation]:
    """
    审计 DogKnowledgeAgent graph 是否接入 finalize answer 节点。

    功能：
        在候选 graph 构建文件中查找 finalize_answer 相关符号。
        只要任一候选文件命中，就认为 graph 已经接入最终输出节点。

    参数：
        无。

    返回值：
        list[AuditViolation]:
            审计违规列表。
    """

    existing_candidate_files = [
        PROJECT_ROOT / relative_path
        for relative_path in GRAPH_ENTRY_CANDIDATE_FILES
        if (PROJECT_ROOT / relative_path).exists()
    ]

    if not existing_candidate_files:
        return [
            AuditViolation(
                rule_id="missing_graph_entry_file",
                severity="error",
                path="src/agents/dog_knowledge_agent/",
                message=(
                    "没有找到 DogKnowledgeAgent graph 构建入口文件。"
                    f"候选文件：{', '.join(GRAPH_ENTRY_CANDIDATE_FILES)}"
                ),
            )
        ]

    for path in existing_candidate_files:
        content = read_text(path)

        if any(symbol in content for symbol in GRAPH_REQUIRED_ANY_SYMBOLS):
            return []

    return [
        AuditViolation(
            rule_id="missing_finalize_integration",
            severity="error",
            path="src/agents/dog_knowledge_agent/",
            message=(
                "DogKnowledgeAgent graph 构建入口中没有发现 finalize_answer 接入痕迹。"
                "请确认最终出口统一经过 build_finalize_dog_knowledge_answer_node "
                "或 finalize_dog_knowledge_answer_node。"
            ),
        )
    ]


def audit_smoke_script_exists() -> list[AuditViolation]:
    """
    审计 v1.7.3 smoke 脚本是否存在。

    参数：
        无。

    返回值：
        list[AuditViolation]:
            审计违规列表。
    """

    path = PROJECT_ROOT / SMOKE_SCRIPT_PATH

    if path.exists():
        return []

    return [
        AuditViolation(
            rule_id="missing_smoke_script",
            severity="warning",
            path=SMOKE_SCRIPT_PATH,
            message=(
                "建议保留 v1.7.3 smoke 脚本，用于发版前验证 DogKnowledgeAgent 输出协议。"
            ),
        )
    ]


def run_audit() -> list[AuditViolation]:
    """
    执行 DogKnowledgeAgent v1.7.3 边界审计。

    参数：
        无。

    返回值：
        list[AuditViolation]:
            所有审计违规项。
    """

    violations: list[AuditViolation] = []

    violations.extend(audit_required_files())
    violations.extend(audit_required_symbols())
    violations.extend(audit_forbidden_legacy_patterns())
    violations.extend(audit_graph_finalize_integration())
    violations.extend(audit_smoke_script_exists())

    return violations


def split_violations(
    violations: list[AuditViolation],
) -> tuple[list[AuditViolation], list[AuditViolation]]:
    """
    按严重等级拆分审计违规项。

    参数：
        violations:
            审计违规项列表。

    返回值：
        tuple[list[AuditViolation], list[AuditViolation]]:
            第一个列表是 error；
            第二个列表是 warning。
    """

    errors = [
        item
        for item in violations
        if item.severity == "error"
    ]

    warnings = [
        item
        for item in violations
        if item.severity == "warning"
    ]

    return errors, warnings


def print_violations(
    violations: list[AuditViolation],
) -> None:
    """
    打印审计违规项。

    参数：
        violations:
            审计违规项列表。

    返回值：
        None。
    """

    if not violations:
        print("[PASS] DogKnowledgeAgent v1.7.3 boundary audit passed.")
        return

    errors, warnings = split_violations(violations)

    print()
    print("=" * 100)
    print("DogKnowledgeAgent v1.7.3 Boundary Audit Report")
    print("=" * 100)
    print(f"Errors  : {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print("=" * 100)

    for item in violations:
        print(f"[{item.severity.upper()}] {item.rule_id}")
        print(f"  path   : {item.path}")

        if item.line_number is not None:
            print(f"  line   : {item.line_number}")

        print(f"  message: {item.message}")

        if item.line:
            print(f"  code   : {item.line}")

        print("-" * 100)


def save_json_report(
    violations: list[AuditViolation],
    output_path: Path,
) -> None:
    """
    保存审计结果为 JSON 文件。

    参数：
        violations:
            审计违规项列表。

        output_path:
            JSON 输出路径。

    返回值：
        None。
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "audit_name": "dog_knowledge_agent_v173_contract_boundaries",
        "error_count": len(
            [
                item
                for item in violations
                if item.severity == "error"
            ]
        ),
        "warning_count": len(
            [
                item
                for item in violations
                if item.severity == "warning"
            ]
        ),
        "violations": [
            asdict(item)
            for item in violations
        ],
    }

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    参数：
        无。

    返回值：
        argparse.Namespace:
            命令行参数对象。
    """

    parser = argparse.ArgumentParser(
        description="DogKnowledgeAgent v1.7.3 Response Contract 边界审计脚本。"
    )

    parser.add_argument(
        "--json-output",
        default=None,
        help="可选 JSON 报告输出路径。",
    )

    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="开启后 warning 也会导致脚本返回失败。",
    )

    return parser.parse_args()


def main() -> None:
    """
    审计脚本主入口。

    参数：
        无。

    返回值：
        None。
    """

    args = parse_args()

    violations = run_audit()

    print_violations(violations)

    if args.json_output:
        save_json_report(
            violations=violations,
            output_path=PROJECT_ROOT / args.json_output,
        )
        print(f"JSON audit report saved to: {args.json_output}")

    errors, warnings = split_violations(violations)

    should_fail = bool(errors)

    if args.fail_on_warning and warnings:
        should_fail = True

    if should_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
