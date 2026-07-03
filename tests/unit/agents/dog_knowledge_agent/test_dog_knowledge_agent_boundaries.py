"""
DogKnowledgeAgent boundary audit 契约测试。

这些测试用于保证 V1.7.2 之后 dog_knowledge_agent 不会重新依赖旧版
query_parse 链路，也不会直接 import 旧版 exact_search_agent / recommendation_agent。
"""

from __future__ import annotations

from pathlib import Path

from scripts import audit_v172_dog_knowledge_agent_boundaries as boundary_audit


def write_python_file(
        directory: Path,
        relative_path: str,
        content: str,
) -> Path:
    """
    写入测试用 Python 文件。

    功能：
        在 pytest 临时目录中创建一个 Python 文件，供边界审计脚本扫描。

    参数含义：
        directory: pytest 提供的临时目录。
        relative_path: 相对临时目录的文件路径。
        content: 要写入文件的 Python 源码文本。

    返回值含义：
        Path，写入后的文件绝对路径。
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


def test_dog_knowledge_agent_boundary_audit_should_pass_current_code():
    """
    测试当前 dog_knowledge_agent 真实代码是否通过边界审计。

    功能：
        这是主契约测试，防止后续提交把旧版 query_parse 或旧版 Agent
        直接依赖重新引入 dog_knowledge_agent。

    参数含义：
        无参数。

    返回值含义：
        None，pytest 会根据 assert 判断测试是否通过。
    """

    findings = boundary_audit.run_audit()

    assert findings == []


def test_boundary_audit_should_fail_when_query_parse_dependency_exists(tmp_path):
    """
    测试出现旧版 query_parse 依赖时审计会失败。

    功能：
        构造一个临时文件，模拟 dog_knowledge_agent 重新 import 旧版查询解析模块。

    参数含义：
        tmp_path: pytest 提供的临时目录 fixture（测试夹具）。

    返回值含义：
        None，pytest 会根据 assert 判断测试是否通过。
    """

    write_python_file(
        directory=tmp_path,
        relative_path="bad_query_parse.py",
        content="from src.parser.query_parser import parse_query_with_llm\n",
    )

    findings = boundary_audit.run_audit(
        target=tmp_path,
        project_root=tmp_path,
    )

    assert len(findings) == 2
    assert {
        finding.pattern
        for finding in findings
    } == {
        "src.parser.query_parser",
        "parse_query_with_llm",
    }


def test_boundary_audit_should_fail_when_legacy_agent_import_exists(tmp_path):
    """
    测试直接 import 旧版 Agent 时审计会失败。

    功能：
        构造一个临时文件，模拟 dog_knowledge_agent 直接依赖旧版
        exact_search_agent 和 recommendation_agent。

    参数含义：
        tmp_path: pytest 提供的临时目录 fixture（测试夹具）。

    返回值含义：
        None，pytest 会根据 assert 判断测试是否通过。
    """

    write_python_file(
        directory=tmp_path,
        relative_path="bad_legacy_agent_import.py",
        content=(
            "from src.agents.exact_search_agent.agent import build_exact_search_agent\n"
            "import src.agents.recommendation_agent.agent\n"
        ),
    )

    findings = boundary_audit.run_audit(
        target=tmp_path,
        project_root=tmp_path,
    )

    assert len(findings) == 2
    assert all(
        finding.category == "legacy_agent_import"
        for finding in findings
    )


def test_boundary_audit_should_fail_when_importing_legacy_agent_from_src_agents(tmp_path):
    """
    测试 from src.agents import 旧版 Agent 时审计会失败。

    功能：
        覆盖另一种常见 import 写法，避免绕过边界审计。

    参数含义：
        tmp_path: pytest 提供的临时目录 fixture（测试夹具）。

    返回值含义：
        None，pytest 会根据 assert 判断测试是否通过。
    """

    write_python_file(
        directory=tmp_path,
        relative_path="bad_from_src_agents.py",
        content="from src.agents import recommendation_agent\n",
    )

    findings = boundary_audit.run_audit(
        target=tmp_path,
        project_root=tmp_path,
    )

    assert len(findings) == 1
    assert findings[0].pattern == "src.agents.recommendation_agent"


def test_boundary_audit_should_not_fail_on_legacy_agent_words_in_comments(tmp_path):
    """
    测试注释中的旧 Agent 名称不会触发直接 import 审计。

    功能：
        dog_knowledge_agent 迁移期间允许注释说明旧 exact_search_agent /
        recommendation_agent 的兼容背景，但不允许真实 import 它们。

    参数含义：
        tmp_path: pytest 提供的临时目录 fixture（测试夹具）。

    返回值含义：
        None，pytest 会根据 assert 判断测试是否通过。
    """

    write_python_file(
        directory=tmp_path,
        relative_path="comment_only.py",
        content=(
            "# exact_search_agent and recommendation_agent are legacy routes.\n"
            "current_agent = 'recommendation_agent'\n"
        ),
    )

    findings = boundary_audit.run_audit(
        target=tmp_path,
        project_root=tmp_path,
    )

    assert findings == []
