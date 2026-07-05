import pytest

from src.runtime.resume.contracts import (
    GraphInterruptResult,
    GraphResumeRequest,
)
from src.runtime.resume.legacy_protocol import (
    LEGACY_INTERRUPT_PREFIX,
    LEGACY_RESUME_PREFIX,
    encode_legacy_interrupt_result,
    encode_legacy_resume_message,
    is_legacy_interrupt_message,
    is_legacy_resume_message,
    parse_legacy_interrupt_message,
    parse_legacy_resume_message,
)


def test_is_legacy_interrupt_message_should_detect_prefix() -> None:
    """
    测试识别旧版 interrupt 字符串。

    功能：
        验证以 __INTERRUPT__: 开头的消息会被识别为旧版中断消息。

    参数含义：
        无。

    返回值含义：
        None。
    """

    assert is_legacy_interrupt_message("__INTERRUPT__:请选择") is True
    assert is_legacy_interrupt_message("请选择") is False


def test_is_legacy_resume_message_should_detect_prefix() -> None:
    """
    测试识别旧版 resume 字符串。

    功能：
        验证以 RESUME: 开头的消息会被识别为旧版恢复消息。

    参数含义：
        无。

    返回值含义：
        None。
    """

    assert is_legacy_resume_message("RESUME:y") is True
    assert is_legacy_resume_message("y") is False


def test_encode_legacy_interrupt_result_should_return_prefixed_message() -> None:
    """
    测试结构化中断结果编码为旧版字符串。

    功能：
        验证 GraphInterruptResult 会被编码成 __INTERRUPT__:prompt 格式。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = GraphInterruptResult(
        prompt="是否执行工具？",
        thread_id="thread_001",
    )

    message = encode_legacy_interrupt_result(result)

    assert message == f"{LEGACY_INTERRUPT_PREFIX}是否执行工具？"


def test_parse_legacy_interrupt_message_should_return_structured_result() -> None:
    """
    测试旧版中断字符串解析为结构化结果。

    功能：
        验证 __INTERRUPT__:prompt 可以解析成 GraphInterruptResult。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = parse_legacy_interrupt_message(
        message="__INTERRUPT__: 是否执行工具？ ",
        thread_id="thread_001",
        checkpoint_ns="main_graph",
        trace_id="trace_001",
    )

    assert isinstance(result, GraphInterruptResult)
    assert result.prompt == "是否执行工具？"
    assert result.thread_id == "thread_001"
    assert result.checkpoint_ns == "main_graph"
    assert result.trace_id == "trace_001"


def test_parse_legacy_interrupt_message_should_return_none_for_normal_message() -> None:
    """
    测试普通消息不会被解析为中断结果。

    功能：
        验证非 __INTERRUPT__: 前缀的消息会返回 None。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = parse_legacy_interrupt_message(
        message="这是普通回答",
        thread_id="thread_001",
    )

    assert result is None


def test_encode_legacy_resume_message_should_return_prefixed_message() -> None:
    """
    测试结构化恢复请求编码为旧版字符串。

    功能：
        验证 GraphResumeRequest 会被编码成 RESUME:value 格式。

    参数含义：
        无。

    返回值含义：
        None。
    """

    request = GraphResumeRequest(
        resume_value="y",
        thread_id="thread_001",
    )

    message = encode_legacy_resume_message(request)

    assert message == f"{LEGACY_RESUME_PREFIX}y"


def test_parse_legacy_resume_message_should_return_structured_request() -> None:
    """
    测试旧版恢复字符串解析为结构化请求。

    功能：
        验证 RESUME:value 可以解析成 GraphResumeRequest。

    参数含义：
        无。

    返回值含义：
        None。
    """

    request = parse_legacy_resume_message(
        message="RESUME: y ",
        thread_id="thread_001",
        checkpoint_ns="main_graph",
        trace_id="trace_001",
    )

    assert isinstance(request, GraphResumeRequest)
    assert request.resume_value == "y"
    assert request.thread_id == "thread_001"
    assert request.checkpoint_ns == "main_graph"
    assert request.trace_id == "trace_001"


def test_parse_legacy_resume_message_should_return_none_for_normal_message() -> None:
    """
    测试普通消息不会被解析为恢复请求。

    功能：
        验证非 RESUME: 前缀的消息会返回 None。

    参数含义：
        无。

    返回值含义：
        None。
    """

    request = parse_legacy_resume_message(
        message="y",
        thread_id="thread_001",
    )

    assert request is None


@pytest.mark.parametrize(
    "message",
    [
        "__INTERRUPT__:",
        "__INTERRUPT__:   ",
    ],
)
def test_parse_legacy_interrupt_message_should_reject_empty_prompt(
    message: str,
) -> None:
    """
    测试旧版中断消息 prompt 不能为空。

    功能：
        验证只有前缀、没有 prompt 的旧版中断消息会抛出 ValueError。

    参数含义：
        message:
            待解析的旧版中断消息。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match="prompt"):
        parse_legacy_interrupt_message(
            message=message,
            thread_id="thread_001",
        )


@pytest.mark.parametrize(
    "message",
    [
        "RESUME:",
        "RESUME:   ",
    ],
)
def test_parse_legacy_resume_message_should_reject_empty_resume_value(
    message: str,
) -> None:
    """
    测试旧版恢复消息 resume_value 不能为空。

    功能：
        验证只有前缀、没有 resume_value 的旧版恢复消息会抛出 ValueError。

    参数含义：
        message:
            待解析的旧版恢复消息。

    返回值含义：
        None。
    """

    with pytest.raises(ValueError, match="resume_value"):
        parse_legacy_resume_message(
            message=message,
            thread_id="thread_001",
        )
