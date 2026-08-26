"""持久化长任务查询路由。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    status,
)
from fastapi.responses import StreamingResponse

from src.api.dependencies import (
    get_long_task_application_service,
    get_long_task_query_service,
    require_api_key,
)
from src.api.long_task_services import (
    LongTaskApiQueryService,
    build_long_task_status_response,
)
from src.api.schemas import (
    LongTaskInteractionCommandRequest,
    LongTaskStatusResponse,
)
from src.runtime.long_tasks.application_service import (
    LongTaskApplicationService,
)
from src.runtime.long_tasks.interaction_service import (
    InvalidLongTaskInteractionError,
)
from src.runtime.long_tasks.store import (
    LongTaskNotFoundError,
    LongTaskVersionConflictError,
)


router = APIRouter(
    prefix="/v1/long-tasks",
    tags=["long-tasks"],
    dependencies=[Depends(require_api_key)],
)

LONG_TASK_SSE_POLL_INTERVAL_SECONDS = 1.0
LONG_TASK_SSE_HEARTBEAT_INTERVAL_SECONDS = 15.0
LONG_TASK_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled"}
)


@router.get(
    "/{task_id}",
    response_model=LongTaskStatusResponse,
    summary="查询一条持久化长任务的最新状态",
)
async def get_long_task_status(
    task_id: Annotated[
        str,
        Path(min_length=1, max_length=300),
    ],
    user_id: Annotated[
        str,
        Query(min_length=1, max_length=200),
    ],
    service: Annotated[
        LongTaskApiQueryService,
        Depends(get_long_task_query_service),
    ],
) -> LongTaskStatusResponse:
    """
    根据任务编号和用户归属返回最新的持久化状态。

    参数含义：
        task_id:
            后台长任务唯一编号。
        user_id:
            当前调用方声明的用户编号；必须与任务归属一致。
        service:
            FastAPI 通过依赖注入提供的长任务查询服务。

    返回值含义：
        LongTaskStatusResponse:
            前端安全的任务、步骤、进度和待交互状态；任务不存在或不属于
            当前用户时统一返回 HTTP 404。
    """

    task_status = await service.get_status(
        task_id=task_id,
        user_id=user_id,
    )
    if task_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="没有找到对应的长任务",
        )
    return task_status


@router.get(
    "/{task_id}/events",
    response_class=StreamingResponse,
    summary="订阅一条持久化长任务的状态变化",
)
async def stream_long_task_status(
    task_id: Annotated[
        str,
        Path(min_length=1, max_length=300),
    ],
    user_id: Annotated[
        str,
        Query(min_length=1, max_length=200),
    ],
    request: Request,
    service: Annotated[
        LongTaskApiQueryService,
        Depends(get_long_task_query_service),
    ],
) -> StreamingResponse:
    """
    建立 SSE 连接，按 LongTask version 变化推送公开状态。

    参数含义：
        task_id:
            准备订阅的持久化长任务编号。
        user_id:
            当前调用方声明的用户编号，用于每次轮询时复核任务归属。
        request:
            当前 HTTP 请求，用于检测前端是否已经断开 SSE 连接。
        service:
            FastAPI 依赖注入提供的只读长任务查询服务。

    返回值含义：
        StreamingResponse:
            首先发送 snapshot；版本变化时发送 updated，终态时发送 terminal
            并关闭连接，长时间无变化时发送 heartbeat。
    """

    initial_status = await service.get_status(
        task_id=task_id,
        user_id=user_id,
    )
    if initial_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="没有找到对应的长任务",
        )

    async def event_stream() -> AsyncIterator[str]:
        """
        把结构化长任务状态事件编码成 SSE 文本帧。

        返回值含义：
            AsyncIterator[str]:
                每次产出一个可由浏览器流式解析的 SSE 事件文本。
        """

        async for event in _iterate_long_task_status_events(
            service=service,
            task_id=task_id,
            user_id=user_id,
            initial_status=initial_status,
            is_disconnected=request.is_disconnected,
        ):
            yield _encode_long_task_sse_event(event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/{task_id}/interactions/{interaction_id}/responses",
    response_model=LongTaskStatusResponse,
    summary="提交长任务缺少输入交互的回答或取消任务",
)
async def respond_to_long_task_interaction(
    task_id: Annotated[
        str,
        Path(min_length=1, max_length=300),
    ],
    interaction_id: Annotated[
        str,
        Path(min_length=1, max_length=300),
    ],
    payload: LongTaskInteractionCommandRequest,
    service: Annotated[
        LongTaskApplicationService,
        Depends(get_long_task_application_service),
    ],
) -> LongTaskStatusResponse:
    """
    消费一次缺少输入交互，并返回已经持久化的最新任务状态。

    参数含义：
        task_id:
            等待恢复或取消的持久化长任务编号。
        interaction_id:
            查询状态时获得的当前 pending_interaction 编号。
        payload:
            包含用户归属、动作和逐 Step 回答的结构化请求体。
        service:
            FastAPI 依赖注入提供的长任务命令应用服务。

    返回值含义：
        LongTaskStatusResponse:
            submit_input 时 Task 已恢复 running、Step 已回到 ready；cancel
            时 Task 已进入 cancelled。状态冲突返回 HTTP 409。
    """

    try:
        task = await service.respond_to_missing_input(
            task_id=task_id,
            user_id=payload.user_id,
            interaction_id=interaction_id,
            action=payload.action,
            answers=payload.answers,
        )
    except LongTaskNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="没有找到对应的长任务",
        ) from exc
    except (
        InvalidLongTaskInteractionError,
        LongTaskVersionConflictError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return build_long_task_status_response(task)


async def _iterate_long_task_status_events(
    *,
    service: LongTaskApiQueryService,
    task_id: str,
    user_id: str,
    initial_status: LongTaskStatusResponse,
    is_disconnected: Callable[[], Awaitable[bool]],
    poll_interval_seconds: float = LONG_TASK_SSE_POLL_INTERVAL_SECONDS,
    heartbeat_interval_seconds: float = (
        LONG_TASK_SSE_HEARTBEAT_INTERVAL_SECONDS
    ),
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AsyncIterator[dict[str, Any]]:
    """
    轮询权威 LongTask 状态，并只在版本变化或需要心跳时产生事件。

    参数含义：
        service:
            每次都从 LongTaskStore 读取最新公开状态的查询服务。
        task_id、user_id:
            当前订阅任务和归属用户编号。
        initial_status:
            建立 StreamingResponse 前已经完成归属校验的初始状态。
        is_disconnected:
            判断前端是否已经关闭连接的异步函数。
        poll_interval_seconds:
            两次 Redis 状态查询之间的等待秒数。
        heartbeat_interval_seconds:
            状态长期不变时发送心跳的间隔秒数。
        sleep:
            异步等待函数；生产使用 asyncio.sleep，测试可注入无等待替身。

    返回值含义：
        AsyncIterator[dict[str, Any]]:
            snapshot、updated、terminal、heartbeat 或 unavailable 事件。
    """

    if poll_interval_seconds <= 0:
        raise ValueError("SSE 轮询间隔必须大于 0")
    if heartbeat_interval_seconds < poll_interval_seconds:
        raise ValueError("SSE 心跳间隔不能小于轮询间隔")

    current_status = initial_status
    last_version = current_status.version
    unchanged_seconds = 0.0
    yield {
        "event": "snapshot",
        "data": current_status.model_dump(mode="json"),
    }
    if current_status.status in LONG_TASK_TERMINAL_STATUSES:
        return

    while not await is_disconnected():
        await sleep(poll_interval_seconds)
        if await is_disconnected():
            return

        current_status = await service.get_status(
            task_id=task_id,
            user_id=user_id,
        )
        if current_status is None:
            yield {
                "event": "unavailable",
                "data": {
                    "task_id": task_id,
                    "message": "长任务已不存在或当前用户无权继续订阅",
                },
            }
            return

        if current_status.version != last_version:
            last_version = current_status.version
            unchanged_seconds = 0.0
            is_terminal = (
                current_status.status in LONG_TASK_TERMINAL_STATUSES
            )
            yield {
                "event": "terminal" if is_terminal else "updated",
                "data": current_status.model_dump(mode="json"),
            }
            if is_terminal:
                return
            continue

        unchanged_seconds += poll_interval_seconds
        if unchanged_seconds >= heartbeat_interval_seconds:
            unchanged_seconds = 0.0
            yield {
                "event": "heartbeat",
                "data": {
                    "task_id": task_id,
                    "status": current_status.status,
                    "version": current_status.version,
                },
            }


def _encode_long_task_sse_event(event: dict[str, Any]) -> str:
    """
    把结构化长任务事件编码为一个以空行结束的 SSE 文本帧。

    参数含义：
        event:
            包含 event 名称和 JSON 兼容 data 的结构化事件。

    返回值含义：
        str:
            浏览器或 Fetch 流式客户端可以解析的 SSE 文本。
    """

    event_name = str(event["event"])
    data = json.dumps(
        event["data"],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event_name}\ndata: {data}\n\n"
