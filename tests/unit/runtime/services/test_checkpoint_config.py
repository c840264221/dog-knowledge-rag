import pytest

from src.runtime.services.checkpoint_config import (
    DEFAULT_GRAPH_CHECKPOINT_NS,
    build_graph_checkpoint_config,
)


def test_build_graph_checkpoint_config_should_include_required_configurable() -> None:
    """
    测试生成最小 LangGraph checkpoint config。

    功能：
        验证只传 thread_id 时，函数会生成 thread_id 和默认 checkpoint_ns。

    参数含义：
        无。

    返回值含义：
        None。
    """

    config = build_graph_checkpoint_config(
        thread_id="session_001",
    )

    assert config == {
        "configurable": {
            "thread_id": "session_001",
            "checkpoint_ns": DEFAULT_GRAPH_CHECKPOINT_NS,
        },
    }


def test_build_graph_checkpoint_config_should_include_optional_fields() -> None:
    """
    测试生成包含可选字段的 LangGraph checkpoint config。

    功能：
        验证 checkpoint_id、run_name、tags、metadata 会被写入输出 config。

    参数含义：
        无。

    返回值含义：
        None。
    """

    config = build_graph_checkpoint_config(
        thread_id="session_001",
        checkpoint_ns="dog_agent_main_graph",
        checkpoint_id="checkpoint_001",
        run_name="query_golden_retriever",
        tags=("dog_agent", "runtime"),
        metadata={
            "trace_id": "trace_001",
        },
    )

    assert config == {
        "configurable": {
            "thread_id": "session_001",
            "checkpoint_ns": "dog_agent_main_graph",
            "checkpoint_id": "checkpoint_001",
        },
        "run_name": "query_golden_retriever",
        "tags": [
            "dog_agent",
            "runtime",
        ],
        "metadata": {
            "trace_id": "trace_001",
        },
    }


def test_build_graph_checkpoint_config_should_strip_string_fields() -> None:
    """
    测试字符串字段会去除前后空白。

    功能：
        验证 thread_id、checkpoint_ns、checkpoint_id、run_name、tags 会被 strip。

    参数含义：
        无。

    返回值含义：
        None。
    """

    config = build_graph_checkpoint_config(
        thread_id=" session_001 ",
        checkpoint_ns=" main_graph ",
        checkpoint_id=" checkpoint_001 ",
        run_name=" query_name ",
        tags=(" dog_agent ",),
    )

    assert config["configurable"] == {
        "thread_id": "session_001",
        "checkpoint_ns": "main_graph",
        "checkpoint_id": "checkpoint_001",
    }
    assert config["run_name"] == "query_name"
    assert config["tags"] == ["dog_agent"]


def test_build_graph_checkpoint_config_should_copy_metadata() -> None:
    """
    测试 metadata 会复制为新字典。

    功能：
        验证外部 metadata 在函数返回后被修改，不会影响已经生成的 config。

    参数含义：
        无。

    返回值含义：
        None。
    """

    metadata = {
        "trace_id": "trace_001",
    }

    config = build_graph_checkpoint_config(
        thread_id="session_001",
        metadata=metadata,
    )

    metadata["trace_id"] = "changed"

    assert config["metadata"] == {
        "trace_id": "trace_001",
    }


@pytest.mark.parametrize(
    "thread_id",
    [
        "",
        "   ",
    ],
)
def test_build_graph_checkpoint_config_should_reject_empty_thread_id(
    thread_id: str,
) -> None:
    """
    测试 thread_id 不能为空。

    功能：
        验证空 thread_id 会抛出 ValueError。

    参数含义：
        thread_id:
            待测试的 thread_id。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match="thread_id"):
        build_graph_checkpoint_config(
            thread_id=thread_id,
        )


def test_build_graph_checkpoint_config_should_reject_empty_checkpoint_ns() -> None:
    """
    测试 checkpoint_ns 不能为空。

    功能：
        验证空 checkpoint_ns 会抛出 ValueError。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match="checkpoint_ns"):
        build_graph_checkpoint_config(
            thread_id="session_001",
            checkpoint_ns="",
        )


def test_build_graph_checkpoint_config_should_reject_empty_checkpoint_id() -> None:
    """
    测试 checkpoint_id 如果传入则不能为空。

    功能：
        验证空 checkpoint_id 会抛出 ValueError。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match="checkpoint_id"):
        build_graph_checkpoint_config(
            thread_id="session_001",
            checkpoint_id=" ",
        )


def test_build_graph_checkpoint_config_should_reject_single_string_tags() -> None:
    """
    测试 tags 不能传单个字符串。

    功能：
        防止把 tags="abc" 误拆成 ["a", "b", "c"]。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match="tags"):
        build_graph_checkpoint_config(
            thread_id="session_001",
            tags="dog_agent",
        )


def test_build_graph_checkpoint_config_should_reject_empty_tag() -> None:
    """
    测试 tags 中不能包含空字符串。

    功能：
        验证空 tag 会抛出 ValueError。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match="tags"):
        build_graph_checkpoint_config(
            thread_id="session_001",
            tags=["dog_agent", ""],
        )
