import pytest

from src.runtime.resume.contracts import (
    DEFAULT_RESUME_CHECKPOINT_NS,
    GraphFinalResult,
    GraphInterruptResult,
    GraphInterruptType,
    GraphResumeRequest,
    GraphRunStatus,
)


def test_graph_interrupt_result_should_convert_to_dict() -> None:
    """
    测试 GraphInterruptResult 可以转换为 dict。

    功能：
        验证中断结果会输出 status、prompt、thread_id、checkpoint_ns、trace_id、
        interrupt_type 和 metadata。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = GraphInterruptResult(
        prompt="是否执行工具？",
        thread_id="thread_001",
        checkpoint_ns="main_graph",
        trace_id="trace_001",
        interrupt_type=GraphInterruptType.TOOL_CONFIRMATION,
        metadata={
            "current_agent": "general_agent",
        },
    )

    assert result.status == GraphRunStatus.INTERRUPTED
    assert result.to_dict() == {
        "status": "interrupted",
        "prompt": "是否执行工具？",
        "thread_id": "thread_001",
        "checkpoint_ns": "main_graph",
        "trace_id": "trace_001",
        "interrupt_type": "tool_confirmation",
        "metadata": {
            "current_agent": "general_agent",
        },
    }


def test_graph_interrupt_result_should_use_default_interrupt_type() -> None:
    """
    测试 GraphInterruptResult 默认中断类型。

    功能：
        验证未传 interrupt_type 时，默认值是 unknown。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = GraphInterruptResult(
        prompt="请选择下一步",
        thread_id="thread_001",
    )

    assert result.interrupt_type == GraphInterruptType.UNKNOWN
    assert result.checkpoint_ns == DEFAULT_RESUME_CHECKPOINT_NS


def test_graph_final_result_should_convert_to_dict() -> None:
    """
    测试 GraphFinalResult 可以转换为 dict。

    功能：
        验证最终结果会输出 completed 状态和最终答案。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = GraphFinalResult(
        answer="这是最终答案",
        thread_id="thread_001",
        checkpoint_ns="main_graph",
        trace_id="trace_001",
    )

    assert result.status == GraphRunStatus.COMPLETED
    assert result.to_dict() == {
        "status": "completed",
        "answer": "这是最终答案",
        "thread_id": "thread_001",
        "checkpoint_ns": "main_graph",
        "trace_id": "trace_001",
        "metadata": {},
    }


def test_graph_resume_request_should_convert_to_dict() -> None:
    """
    测试 GraphResumeRequest 可以转换为 dict。

    功能：
        验证恢复请求会输出 resume_value、thread_id、checkpoint_ns、trace_id 和 metadata。

    参数含义：
        无。

    返回值含义：
        None。
    """

    request = GraphResumeRequest(
        resume_value="y",
        thread_id="thread_001",
        checkpoint_ns="main_graph",
        trace_id="trace_001",
        metadata={
            "source": "ui",
        },
    )

    assert request.to_dict() == {
        "resume_value": "y",
        "thread_id": "thread_001",
        "checkpoint_ns": "main_graph",
        "trace_id": "trace_001",
        "metadata": {
            "source": "ui",
        },
    }


def test_resume_contracts_should_strip_string_fields() -> None:
    """
    测试协议对象会去除字符串字段前后空白。

    功能：
        验证 prompt、answer、resume_value、thread_id、checkpoint_ns 会被 strip。

    参数含义：
        无。

    返回值含义：
        None。
    """

    interrupt_result = GraphInterruptResult(
        prompt=" prompt ",
        thread_id=" thread_001 ",
        checkpoint_ns=" main_graph ",
    )
    final_result = GraphFinalResult(
        answer=" answer ",
        thread_id=" thread_001 ",
        checkpoint_ns=" main_graph ",
    )
    resume_request = GraphResumeRequest(
        resume_value=" y ",
        thread_id=" thread_001 ",
        checkpoint_ns=" main_graph ",
    )

    assert interrupt_result.prompt == "prompt"
    assert interrupt_result.thread_id == "thread_001"
    assert interrupt_result.checkpoint_ns == "main_graph"
    assert final_result.answer == "answer"
    assert resume_request.resume_value == "y"


def test_resume_contracts_should_copy_metadata() -> None:
    """
    测试 metadata 会被复制。

    功能：
        验证对象创建后，外部 metadata 修改不会影响对象内部 metadata。

    参数含义：
        无。

    返回值含义：
        None。
    """

    metadata = {
        "trace_id": "trace_001",
    }

    result = GraphInterruptResult(
        prompt="请选择",
        thread_id="thread_001",
        metadata=metadata,
    )

    metadata["trace_id"] = "changed"

    assert result.metadata == {
        "trace_id": "trace_001",
    }
    assert result.to_dict()["metadata"] == {
        "trace_id": "trace_001",
    }


@pytest.mark.parametrize(
    "field_name, kwargs",
    [
        (
            "prompt",
            {
                "prompt": "",
                "thread_id": "thread_001",
            },
        ),
        (
            "thread_id",
            {
                "prompt": "请选择",
                "thread_id": "",
            },
        ),
        (
            "checkpoint_ns",
            {
                "prompt": "请选择",
                "thread_id": "thread_001",
                "checkpoint_ns": "",
            },
        ),
    ],
)
def test_graph_interrupt_result_should_reject_empty_required_fields(
    field_name: str,
    kwargs: dict,
) -> None:
    """
    测试 GraphInterruptResult 必填字段不能为空。

    功能：
        验证 prompt、thread_id、checkpoint_ns 为空时会抛出 ValueError。

    参数含义：
        field_name:
            预期报错字段名。
        kwargs:
            构造 GraphInterruptResult 使用的参数。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match=field_name):
        GraphInterruptResult(**kwargs)


def test_graph_final_result_should_reject_empty_answer() -> None:
    """
    测试 GraphFinalResult answer 不能为空。

    功能：
        验证最终结果 answer 为空时会抛出 ValueError。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match="answer"):
        GraphFinalResult(
            answer="",
            thread_id="thread_001",
        )


def test_graph_resume_request_should_reject_empty_resume_value() -> None:
    """
    测试 GraphResumeRequest resume_value 不能为空。

    功能：
        验证恢复请求 resume_value 为空时会抛出 ValueError。

    参数含义：
        无。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match="resume_value"):
        GraphResumeRequest(
            resume_value="",
            thread_id="thread_001",
        )
