from __future__ import annotations

from src.runtime.resume.contracts import (
    DEFAULT_RESUME_CHECKPOINT_NS,
    GraphInterruptResult,
    GraphResumeRequest,
)


LEGACY_INTERRUPT_PREFIX = "__INTERRUPT__:"
LEGACY_RESUME_PREFIX = "RESUME:"


def is_legacy_interrupt_message(
    message: str,
) -> bool:
    """
    判断是否为旧版 interrupt 字符串消息。

    功能：
        检查 message 是否以旧版中断前缀 __INTERRUPT__: 开头。

    参数含义：
        message:
            待检查的字符串消息。

    返回值含义：
        bool:
            True 表示是旧版中断消息；False 表示不是。
    """

    return isinstance(message, str) and message.startswith(
        LEGACY_INTERRUPT_PREFIX
    )


def is_legacy_resume_message(
    message: str,
) -> bool:
    """
    判断是否为旧版 resume 字符串消息。

    功能：
        检查 message 是否以旧版恢复前缀 RESUME: 开头。

    参数含义：
        message:
            待检查的字符串消息。

    返回值含义：
        bool:
            True 表示是旧版恢复消息；False 表示不是。
    """

    return isinstance(message, str) and message.startswith(
        LEGACY_RESUME_PREFIX
    )


def encode_legacy_interrupt_result(
    result: GraphInterruptResult,
) -> str:
    """
    将结构化中断结果编码为旧版 interrupt 字符串。

    功能：
        把 GraphInterruptResult 转换成当前 UI 仍在使用的 __INTERRUPT__:prompt 格式。

    参数含义：
        result:
            结构化 Graph 中断结果。

    返回值含义：
        str:
            旧版中断字符串消息。
    """

    return f"{LEGACY_INTERRUPT_PREFIX}{result.prompt}"


def parse_legacy_interrupt_message(
    message: str,
    thread_id: str,
    checkpoint_ns: str = DEFAULT_RESUME_CHECKPOINT_NS,
    trace_id: str | None = None,
) -> GraphInterruptResult | None:
    """
    解析旧版 interrupt 字符串为结构化中断结果。

    功能：
        如果 message 是 __INTERRUPT__:prompt 格式，则返回 GraphInterruptResult；
        如果不是旧版中断消息，则返回 None。

    参数含义：
        message:
            待解析的旧版字符串消息。
        thread_id:
            LangGraph thread_id，用于后续恢复同一条图执行线程。
        checkpoint_ns:
            checkpoint namespace，检查点命名空间。
        trace_id:
            当前请求链路追踪 ID，可选。

    返回值含义：
        GraphInterruptResult | None:
            解析成功时返回结构化中断结果；非中断消息返回 None。
    """

    if not is_legacy_interrupt_message(message):
        return None

    prompt = message[len(LEGACY_INTERRUPT_PREFIX):].strip()

    return GraphInterruptResult(
        prompt=prompt,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        trace_id=trace_id,
    )


def encode_legacy_resume_message(
    request: GraphResumeRequest,
) -> str:
    """
    将结构化恢复请求编码为旧版 resume 字符串。

    功能：
        把 GraphResumeRequest 转换成当前 graph_run.py 仍在使用的 RESUME:value 格式。

    参数含义：
        request:
            结构化 Graph 恢复请求。

    返回值含义：
        str:
            旧版恢复字符串消息。
    """

    return f"{LEGACY_RESUME_PREFIX}{request.resume_value}"


def parse_legacy_resume_message(
    message: str,
    thread_id: str,
    checkpoint_ns: str = DEFAULT_RESUME_CHECKPOINT_NS,
    trace_id: str | None = None,
) -> GraphResumeRequest | None:
    """
    解析旧版 resume 字符串为结构化恢复请求。

    功能：
        如果 message 是 RESUME:value 格式，则返回 GraphResumeRequest；
        如果不是旧版恢复消息，则返回 None。

    参数含义：
        message:
            待解析的旧版字符串消息。
        thread_id:
            LangGraph thread_id，必须和中断时一致。
        checkpoint_ns:
            checkpoint namespace，检查点命名空间。
        trace_id:
            当前请求链路追踪 ID，可选。

    返回值含义：
        GraphResumeRequest | None:
            解析成功时返回结构化恢复请求；非恢复消息返回 None。
    """

    if not is_legacy_resume_message(message):
        return None

    resume_value = message[len(LEGACY_RESUME_PREFIX):].strip()

    return GraphResumeRequest(
        resume_value=resume_value,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        trace_id=trace_id,
    )
