from __future__ import annotations

from typing import Any

import pytest

from src.api.services import AgentApiService
from src.runtime.resume.contracts import (
    GraphFinalResult,
    GraphInterruptResult,
    GraphInterruptType,
)


class FakeGraphRuntime:
    """记录 API 服务转发的多 Agent 取消任务编号。"""

    def __init__(self, cancellation_result: bool = True) -> None:
        self.cancellation_result = cancellation_result
        self.cancelled_task_ids: list[str] = []

    def cancel_multi_agent_task(self, multi_agent_task_id: str) -> bool:
        """记录任务编号并返回预设取消结果。"""

        self.cancelled_task_ids.append(multi_agent_task_id)
        return self.cancellation_result


@pytest.mark.asyncio
async def test_service_should_convert_final_graph_result() -> None:
    """测试 API 服务把主图完成结果转换成统一 HTTP 契约。"""

    received_kwargs: dict[str, Any] = {}

    async def fake_graph_runner(
        question: str,
        **kwargs: Any,
    ) -> GraphFinalResult:
        """返回确定性的主图完成结果。"""

        received_kwargs["question"] = question
        received_kwargs.update(kwargs)
        return GraphFinalResult(
            answer="金毛通常很亲人。",
            thread_id=kwargs["thread_id"],
            trace_id=kwargs["trace_id"],
        )

    service = AgentApiService(
        graph_runtime=FakeGraphRuntime(),
        graph_runner=fake_graph_runner,
    )
    response = await service.chat(
        question="金毛性格怎么样？",
        session_id="session_001",
        trace_id="trace_001",
    )

    assert response.status == "completed"
    assert response.business_status == "completed"
    assert response.answer == "金毛通常很亲人。"
    assert response.multi_agent_task_id == "multi_agent_task_trace_001"
    assert received_kwargs["thread_id"] == "session_001"
    assert received_kwargs["resume_value"] is None
    task_status = service.get_task_status(
        "multi_agent_task_trace_001"
    )
    assert task_status is not None
    assert task_status.status == "completed"


@pytest.mark.asyncio
async def test_service_should_keep_business_failure_reason() -> None:
    """测试主图正常结束时 API 仍会保留业务失败和超时原因。"""

    async def fake_graph_runner(
        question: str,
        **kwargs: Any,
    ) -> GraphFinalResult:
        """返回携带多 Agent 步骤超时摘要的主图完成结果。"""

        return GraphFinalResult(
            answer="多 Agent 任务执行失败。",
            thread_id=kwargs["thread_id"],
            trace_id=kwargs["trace_id"],
            metadata={
                "business_status": "failed",
                "business_error": {
                    "code": "MULTI_AGENT_STEP_TIMEOUT",
                    "message": "多 Agent 步骤执行超时。",
                    "details": {
                        "timed_out_steps": [
                            {
                                "step_id": "step_health",
                                "timeout_seconds": 10,
                                "attempt_count": 2,
                            }
                        ]
                    },
                },
            },
        )

    service = AgentApiService(
        graph_runtime=FakeGraphRuntime(),
        graph_runner=fake_graph_runner,
    )
    response = await service.chat(
        question="生成健康方案",
        session_id="session_failed",
        trace_id="trace_failed",
    )

    assert response.status == "completed"
    assert response.business_status == "failed"
    assert response.business_error is not None
    assert response.business_error.code == "MULTI_AGENT_STEP_TIMEOUT"
    task_status = service.get_task_status(
        "multi_agent_task_trace_failed"
    )
    assert task_status is not None
    assert task_status.status == "completed"
    assert task_status.business_status == "failed"


@pytest.mark.asyncio
async def test_service_should_convert_interrupt_graph_result() -> None:
    """测试 API 服务把主图中断结果转换成等待输入契约。"""

    async def fake_graph_runner(
        question: str,
        **kwargs: Any,
    ) -> GraphInterruptResult:
        """返回确定性的主图中断结果。"""

        return GraphInterruptResult(
            prompt="请补充狗狗年龄。",
            thread_id=kwargs["thread_id"],
            trace_id=kwargs["trace_id"],
            interrupt_type=GraphInterruptType.USER_CLARIFICATION,
        )

    service = AgentApiService(
        graph_runtime=FakeGraphRuntime(),
        graph_runner=fake_graph_runner,
    )
    response = await service.chat(
        question="制定健康方案",
        session_id="session_002",
        trace_id="trace_002",
    )

    assert response.status == "interrupted"
    assert response.business_status == "awaiting_input"
    assert response.prompt == "请补充狗狗年龄。"
    assert response.interrupt_type == "user_clarification"


def test_service_should_forward_cancellation() -> None:
    """测试 API 服务把取消请求转交给 GraphRuntimeService。"""

    graph_runtime = FakeGraphRuntime(cancellation_result=False)
    service = AgentApiService(graph_runtime=graph_runtime)

    response = service.cancel("multi_agent_task_missing")

    assert response.cancellation_requested is False
    assert graph_runtime.cancelled_task_ids == [
        "multi_agent_task_missing"
    ]


@pytest.mark.asyncio
async def test_stream_service_should_emit_identity_heartbeat_and_result() -> None:
    """测试 SSE 服务会先返回身份，再发送心跳和最终结果。"""

    async def delayed_graph_runner(
        question: str,
        **kwargs: Any,
    ) -> GraphFinalResult:
        """稍作等待后返回主图完成结果。"""

        import asyncio

        await asyncio.sleep(0.03)
        return GraphFinalResult(
            answer=f"{question}完成",
            thread_id=kwargs["thread_id"],
            trace_id=kwargs["trace_id"],
        )

    service = AgentApiService(
        graph_runtime=FakeGraphRuntime(),
        graph_runner=delayed_graph_runner,
        heartbeat_seconds=0.005,
    )

    events = [
        event
        async for event in service.stream_chat(
            question="流式任务",
            session_id="session_stream",
            trace_id="trace_stream",
        )
    ]

    assert events[0]["event"] == "accepted"
    assert events[0]["data"]["multi_agent_task_id"] == (
        "multi_agent_task_trace_stream"
    )
    assert any(event["event"] == "heartbeat" for event in events)
    assert events[-1]["event"] == "completed"
    task_status = service.get_task_status(
        "multi_agent_task_trace_stream"
    )
    assert task_status is not None
    assert task_status.status == "completed"
    assert task_status.business_status == "completed"
