import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Request,
    status,
)
from fastapi.responses import StreamingResponse

from src.api.dependencies import (
    get_agent_api_service,
    require_api_key,
)
from src.api.schemas import (
    CancellationResponse,
    ChatRequest,
    GraphRunResponse,
    ResumeRequest,
    TaskStatusResponse,
)
from src.api.services import AgentApiService


router = APIRouter(
    prefix="/v1",
    tags=["agent"],
    dependencies=[
        Depends(require_api_key),
    ],
)


@router.post(
    "/chat",
    response_model=GraphRunResponse,
    summary="执行一次新的 Agent 对话",
)
async def chat(
    payload: ChatRequest,
    request: Request,
    service: Annotated[
        AgentApiService,
        Depends(get_agent_api_service),
    ],
) -> GraphRunResponse:
    """
    接收新问题并调用现有 Main Graph。

    参数含义：
        payload:
            已完成结构校验的对话请求。
        request:
            当前 HTTP 请求，用于统一解析 trace_id。
        service:
            FastAPI 通过依赖注入提供的 AgentApiService。

    返回值含义：
        GraphRunResponse:
            completed 或 interrupted 状态的统一主图结果。
    """

    return await service.chat(
        question=payload.question,
        session_id=payload.session_id,
        trace_id=_resolve_trace_id(request, payload.trace_id),
    )


@router.post(
    "/chat/stream",
    response_class=StreamingResponse,
    summary="以 SSE 生命周期事件执行一次 Agent 对话",
)
async def stream_chat(
    payload: ChatRequest,
    request: Request,
    service: Annotated[
        AgentApiService,
        Depends(get_agent_api_service),
    ],
) -> StreamingResponse:
    """
    建立 SSE 连接并持续返回请求生命周期事件。

    参数含义：
        payload:
            已完成结构校验的新对话请求。
        request:
            当前 HTTP 请求，用于统一解析 trace_id。
        service:
            FastAPI 通过依赖注入提供的 AgentApiService。

    返回值含义：
        StreamingResponse:
            text/event-stream 响应；当前输出 accepted、heartbeat 和最终事件。
    """

    async def event_stream() -> AsyncIterator[str]:
        """
        将服务层事件转换成 SSE 文本帧。

        参数含义：
            无。

        返回值含义：
            AsyncIterator[str]:
                每次产出一个符合 SSE 格式的事件文本。
        """

        async for event in service.stream_chat(
            question=payload.question,
            session_id=payload.session_id,
            trace_id=_resolve_trace_id(request, payload.trace_id),
        ):
            yield _encode_sse_event(event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/chat/resume",
    response_model=GraphRunResponse,
    summary="恢复一条等待用户输入的 Agent 对话",
)
async def resume(
    payload: ResumeRequest,
    request: Request,
    service: Annotated[
        AgentApiService,
        Depends(get_agent_api_service),
    ],
) -> GraphRunResponse:
    """
    使用用户补充信息恢复原 LangGraph 线程。

    参数含义：
        payload:
            包含原 session_id、trace_id 和补充内容的恢复请求。
        request:
            当前 HTTP 请求，用于让响应头继续使用原 trace_id。
        service:
            FastAPI 通过依赖注入提供的 AgentApiService。

    返回值含义：
        GraphRunResponse:
            恢复后完成或再次中断的统一结果。
    """

    return await service.resume(
        resume_value=payload.resume_value,
        session_id=payload.session_id,
        trace_id=_resolve_trace_id(request, payload.trace_id),
    )


@router.post(
    "/multi-agent/tasks/{multi_agent_task_id}/cancel",
    response_model=CancellationResponse,
    summary="取消一个运行中的多 Agent 任务",
)
async def cancel_multi_agent_task(
    multi_agent_task_id: Annotated[
        str,
        Path(min_length=1, max_length=300),
    ],
    service: Annotated[
        AgentApiService,
        Depends(get_agent_api_service),
    ],
) -> CancellationResponse:
    """
    向指定多 Agent 任务的共享取消令牌发送信号。

    参数含义：
        multi_agent_task_id:
            对话响应返回的多 Agent 任务编号。
        service:
            FastAPI 通过依赖注入提供的 AgentApiService。

    返回值含义：
        CancellationResponse:
            是否找到运行中任务并成功发送信号。
    """

    return service.cancel(multi_agent_task_id)


@router.get(
    "/tasks/{multi_agent_task_id}",
    response_model=TaskStatusResponse,
    summary="查询当前进程中的 Agent 请求状态",
)
async def get_task_status(
    multi_agent_task_id: Annotated[
        str,
        Path(min_length=1, max_length=300),
    ],
    service: Annotated[
        AgentApiService,
        Depends(get_agent_api_service),
    ],
) -> TaskStatusResponse:
    """
    查询一条已登记 API 请求的最新生命周期状态。

    参数含义：
        multi_agent_task_id:
            SSE accepted 事件或普通对话响应返回的任务编号。
        service:
            FastAPI 通过依赖注入提供的 AgentApiService。

    返回值含义：
        TaskStatusResponse:
            当前任务状态；找不到时返回 HTTP 404。
    """

    task_status = service.get_task_status(multi_agent_task_id)
    if task_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="没有找到对应的 API 任务",
        )
    return task_status


def _encode_sse_event(event: dict[str, Any]) -> str:
    """
    把结构化事件编码成 SSE 文本帧。

    参数含义：
        event:
            包含 event 名称和 data JSON 数据的事件字典。

    返回值含义：
        str:
            以空行结尾、可被 EventSource 客户端解析的 SSE 文本。
    """

    event_name = str(event["event"])
    data = json.dumps(
        event["data"],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event_name}\ndata: {data}\n\n"


def _resolve_trace_id(
    request: Request,
    payload_trace_id: str | None,
) -> str:
    """
    解析业务请求最终使用的 trace_id。

    功能：
        请求体显式提供 trace_id 时优先使用，并同步回 request.state，使日志
        和响应头保持一致；否则使用中间件从请求头读取或生成的编号。

    参数含义：
        request:
            当前 FastAPI 请求。
        payload_trace_id:
            ChatRequest 或 ResumeRequest 中的可选链路编号。

    返回值含义：
        str:
            当前业务执行、响应头和请求日志共同使用的 trace_id。
    """

    resolved_trace_id = str(
        payload_trace_id
        or getattr(request.state, "trace_id", "")
    ).strip()
    request.state.trace_id = resolved_trace_id
    return resolved_trace_id
