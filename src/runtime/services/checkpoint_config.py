from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_GRAPH_CHECKPOINT_NS = "main_graph"


def build_graph_checkpoint_config(
    thread_id: str,
    checkpoint_ns: str = DEFAULT_GRAPH_CHECKPOINT_NS,
    checkpoint_id: str | None = None,
    run_name: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    构建 LangGraph checkpoint config。

    功能：
        统一生成 LangGraph 执行时使用的 config，避免 thread_id、
        checkpoint_ns、checkpoint_id、run_name、tags、metadata 在不同脚本中
        手写散落。

    参数含义：
        thread_id:
            LangGraph 线程 ID。
            中文解释：用于标识一条可恢复的图执行线程，interrupt 后恢复必须使用同一个 thread_id。

        checkpoint_ns:
            checkpoint namespace，检查点命名空间。
            中文解释：用于区分不同图、不同业务域或不同 smoke test 的检查点空间。

        checkpoint_id:
            checkpoint ID，具体检查点 ID。
            中文解释：可选字段，用于精确恢复到某一个检查点；一般恢复最新状态时不需要传。

        run_name:
            运行名称。
            中文解释：常用于 LangSmith 或日志中展示本次运行的名称。

        tags:
            标签列表。
            中文解释：用于给本次图运行打标签，方便观测、过滤和排查。

        metadata:
            元数据字典。
            中文解释：用于携带 trace_id、user_id、session_id 等辅助信息。

    返回值含义：
        dict[str, Any]:
            LangGraph graph.invoke / graph.ainvoke / graph.stream / graph.astream 可使用的 config。

    输出格式：
        {
            "configurable": {
                "thread_id": "...",
                "checkpoint_ns": "...",
                "checkpoint_id": "..."
            },
            "run_name": "...",
            "tags": ["..."],
            "metadata": {...}
        }
    """

    normalized_thread_id = _require_non_empty_string(
        value=thread_id,
        field_name="thread_id",
    )
    normalized_checkpoint_ns = _require_non_empty_string(
        value=checkpoint_ns,
        field_name="checkpoint_ns",
    )

    configurable: dict[str, Any] = {
        "thread_id": normalized_thread_id,
        "checkpoint_ns": normalized_checkpoint_ns,
    }

    if checkpoint_id is not None:
        configurable["checkpoint_id"] = _require_non_empty_string(
            value=checkpoint_id,
            field_name="checkpoint_id",
        )

    config: dict[str, Any] = {
        "configurable": configurable,
    }

    if run_name is not None:
        config["run_name"] = _require_non_empty_string(
            value=run_name,
            field_name="run_name",
        )

    if tags is not None:
        config["tags"] = _normalize_tags(tags)

    if metadata is not None:
        config["metadata"] = dict(metadata)

    return config


def _require_non_empty_string(
    value: str,
    field_name: str,
) -> str:
    """
    校验并归一化非空字符串。

    功能：
        去除字符串前后空白，并在字段为空或不是字符串时抛出 ValueError。

    参数含义：
        value:
            要校验的字段值。
        field_name:
            字段名称，用于生成错误信息。

    返回值含义：
        str:
            去除前后空白后的非空字符串。
    """

    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是非空字符串。")

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(f"{field_name} 必须是非空字符串。")

    return normalized_value


def _normalize_tags(
    tags: Sequence[str],
) -> list[str]:
    """
    归一化 LangGraph config tags。

    功能：
        将 tags 转换为 list[str]，并校验每个 tag 都是非空字符串。

    参数含义：
        tags:
            标签序列。

    返回值含义：
        list[str]:
            归一化后的标签列表。
    """

    if isinstance(tags, str):
        raise ValueError("tags 必须是字符串序列，不能是单个字符串。")

    normalized_tags: list[str] = []

    for tag in tags:
        normalized_tags.append(
            _require_non_empty_string(
                value=tag,
                field_name="tags",
            )
        )

    return normalized_tags
