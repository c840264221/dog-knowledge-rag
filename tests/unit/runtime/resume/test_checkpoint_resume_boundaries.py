from pathlib import Path

from scripts.audit_v175_checkpoint_resume_boundaries import (
    audit_file,
    collect_audit_files,
    run_audit,
)


def write_file(
    path: Path,
    content: str,
) -> None:
    """
    写入测试文件。

    功能：
        为审计脚本单元测试创建临时 Python 文件。

    参数含义：
        path:
            要写入的文件路径。
        content:
            文件内容。

    返回值含义：
        None。
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        content,
        encoding="utf-8",
    )


def test_collect_audit_files_should_include_configured_targets(
    tmp_path: Path,
) -> None:
    """
    测试审计脚本会收集配置好的目标文件。

    功能：
        验证 graph_run.py、ui_run.py、src/runtime/resume 和 src/runtime/hooks 下的 Python 文件
        会进入审计范围。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    graph_run = tmp_path / "src" / "graph" / "graph_run.py"
    ui_run = tmp_path / "scripts" / "ui_run.py"
    legacy_protocol = tmp_path / "src" / "runtime" / "resume" / "legacy_protocol.py"
    tool_counter_hook = tmp_path / "src" / "runtime" / "hooks" / "tool_counter_hook.py"

    write_file(graph_run, "value = 'ok'\n")
    write_file(ui_run, "value = 'ok'\n")
    write_file(legacy_protocol, "value = 'ok'\n")
    write_file(tool_counter_hook, "value = 'ok'\n")

    files = collect_audit_files(
        project_root=tmp_path,
    )

    assert graph_run in files
    assert ui_run in files
    assert legacy_protocol in files
    assert tool_counter_hook in files


def test_audit_file_should_allow_legacy_protocol_file(
    tmp_path: Path,
) -> None:
    """
    测试 legacy_protocol.py 允许保存旧协议前缀。

    功能：
        验证旧协议字符串只允许集中出现在兼容协议模块中。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    legacy_protocol = tmp_path / "src" / "runtime" / "resume" / "legacy_protocol.py"
    write_file(
        legacy_protocol,
        'LEGACY_INTERRUPT_PREFIX = "__INTERRUPT__:"\n'
        'LEGACY_RESUME_PREFIX = "RESUME:"\n',
    )

    issues = audit_file(
        path=legacy_protocol,
        project_root=tmp_path,
    )

    assert issues == []


def test_audit_file_should_reject_inline_legacy_protocol_in_graph_run(
    tmp_path: Path,
) -> None:
    """
    测试 graph_run.py 不允许直接手写旧协议前缀。

    功能：
        验证业务入口必须通过 legacy_protocol.py 的函数编码和解析旧协议。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    graph_run = tmp_path / "src" / "graph" / "graph_run.py"
    write_file(
        graph_run,
        'result = "__INTERRUPT__:请选择"\n',
    )

    issues = audit_file(
        path=graph_run,
        project_root=tmp_path,
    )

    assert len(issues) == 1
    assert issues[0].code == "forbidden_inline_legacy_resume_protocol"
    assert issues[0].path == "src/graph/graph_run.py"


def test_audit_file_should_ignore_comments(
    tmp_path: Path,
) -> None:
    """
    测试审计脚本不会误判注释。

    功能：
        验证注释中提到旧协议前缀不会触发违规。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    ui_run = tmp_path / "scripts" / "ui_run.py"
    write_file(
        ui_run,
        "# 这里说明旧版 __INTERRUPT__: 协议，但不是代码使用。\n"
        "value = 'ok'\n",
    )

    issues = audit_file(
        path=ui_run,
        project_root=tmp_path,
    )

    assert issues == []


def test_audit_file_should_reject_legacy_runtime_data_field(
    tmp_path: Path,
) -> None:
    """
    测试 hook 不允许继续依赖 RuntimeContext.runtime_data。

    功能：
        验证真实代码中出现 runtime_data 名称时会触发审计错误。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    hook_file = tmp_path / "src" / "runtime" / "hooks" / "tool_counter_hook.py"
    write_file(
        hook_file,
        "count = ctx.runtime_data.get('tool_count', 0)\n",
    )

    issues = audit_file(
        path=hook_file,
        project_root=tmp_path,
    )

    assert len(issues) == 1
    assert issues[0].code == "forbidden_legacy_runtime_context_field"
    assert issues[0].path == "src/runtime/hooks/tool_counter_hook.py"


def test_audit_file_should_ignore_runtime_data_in_comments_and_strings(
    tmp_path: Path,
) -> None:
    """
    测试审计脚本不会误判注释和字符串中的 runtime_data。

    功能：
        验证 runtime_data 只在真实代码名称中出现时才触发审计。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    hook_file = tmp_path / "src" / "runtime" / "hooks" / "tool_counter_hook.py"
    write_file(
        hook_file,
        "# runtime_data 是旧字段说明\n"
        "message = 'runtime_data 只在文档字符串里出现'\n"
        "value = 'ok'\n",
    )

    issues = audit_file(
        path=hook_file,
        project_root=tmp_path,
    )

    assert issues == []


def test_run_audit_should_pass_when_business_files_use_adapter(
    tmp_path: Path,
) -> None:
    """
    测试业务文件使用兼容函数时审计通过。

    功能：
        验证 graph_run.py 和 ui_run.py 不直接出现旧协议前缀时，
        run_audit 返回空问题列表。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    write_file(
        tmp_path / "src" / "graph" / "graph_run.py",
        "from src.runtime.resume.legacy_protocol import parse_legacy_resume_message\n",
    )
    write_file(
        tmp_path / "scripts" / "ui_run.py",
        "from src.runtime.resume.legacy_protocol import encode_legacy_resume_message\n",
    )
    write_file(
        tmp_path / "src" / "runtime" / "resume" / "legacy_protocol.py",
        'LEGACY_RESUME_PREFIX = "RESUME:"\n',
    )
    write_file(
        tmp_path / "src" / "runtime" / "hooks" / "tool_counter_hook.py",
        "metrics_scope.increment('tool_before_hook_count')\n",
    )

    issues = run_audit(
        project_root=tmp_path,
    )

    assert issues == []
