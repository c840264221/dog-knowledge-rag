from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from uuid import uuid4

from src.agents.collaboration.scheduler import build_multi_agent_task_id
from src.api.schemas import (
    CancellationResponse,
    GraphRunResponse,
    TaskStatusResponse,
)
from src.api.task_registry import ApiTaskRegistry
from src.graph.graph_run import run_main_graph_with_result
from src.runtime.context import RuntimeContext, runtime_ctx
from src.runtime.hooks.tool_counter_hook import ToolCounterHook
from src.runtime.resume.contracts import GraphFinalResult, GraphInterruptResult
from src.runtime.trace.init import trace_manager

GraphRunner = Callable[..., Awaitable[GraphFinalResult | GraphInterruptResult]]


class AgentApiService:
    """
    把 HTTP API 请求适配到现有 Main Graph 运行链路。

    功能：
        为每次请求创建 RuntimeContext 和 trace，调用真实主图入口，并在请求
        结束后释放请求级资源；同时把内部结果转换成稳定的 API 响应。

    参数含义：
        graph_runtime:
            容器中已经启动的 GraphRuntimeService，用于转发取消请求。
        graph_runner:
            主图执行函数。生产环境默认使用 run_main_graph_with_result，
            测试时可以注入确定性替身。

    返回值含义：
        AgentApiService:
            可以处理新对话、恢复执行和多 Agent 取消的 API 业务服务。
    """

    def __init__(
        self,
        *,
        graph_runtime: Any,
        graph_runner: GraphRunner = run_main_graph_with_result,
        task_registry: ApiTaskRegistry | None = None,
        heartbeat_seconds: float = 15.0,
    ) -> None:
        self.graph_runtime = graph_runtime
        self.graph_runner = graph_runner
        self.task_registry = task_registry or ApiTaskRegistry()
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds 必须大于 0")
        self.heartbeat_seconds = heartbeat_seconds

    async def chat(
        self,
        *,
        question: str,
        session_id: str,
        trace_id: str | None = None,
    ) -> GraphRunResponse:
        """
        执行一次新的 Agent 对话。

        参数含义：
            question:
                用户本轮问题。
            session_id:
                连续会话编号，同时作为 LangGraph thread_id。
            trace_id:
                可选链路编号；未提供时由服务端生成 UUID。

        返回值含义：
            GraphRunResponse:
                主图完成或等待用户输入时的统一 API 结果。
        """

        resolved_trace_id, task_id = self._start_task(
            trace_id=trace_id,
            session_id=session_id,
        )
        try:
            response = await self._run_graph(
                question=question,
                session_id=session_id,
                trace_id=resolved_trace_id,
                resume_value=None,
                component="api_chat_handler",
            )
        except Exception:
            self.task_registry.update(
                task_id,
                status="failed",
                error_message="Agent 请求执行失败",
            )
            raise
        self._finish_task(task_id, response)
        return response

    async def resume(
        self,
        *,
        resume_value: str,
        session_id: str,
        trace_id: str,
    ) -> GraphRunResponse:
        """
        使用用户补充信息恢复一条中断的主图。

        参数含义：
            resume_value:
                用户确认或补充的自然语言内容。
            session_id:
                中断时使用的会话编号。
            trace_id:
                中断时使用的链路追踪编号。

        返回值含义：
            GraphRunResponse:
                恢复后完成或再次中断的统一 API 结果。
        """

        resolved_trace_id, task_id = self._start_task(
            trace_id=trace_id,
            session_id=session_id,
        )
        try:
            response = await self._run_graph(
                question=resume_value,
                session_id=session_id,
                trace_id=resolved_trace_id,
                resume_value=resume_value,
                component="api_resume_handler",
            )
        except Exception:
            self.task_registry.update(
                task_id,
                status="failed",
                error_message="Agent 请求执行失败",
            )
            raise
        self._finish_task(task_id, response)
        return response

    async def stream_chat(
        self,
        *,
        question: str,
        session_id: str,
        trace_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        以 SSE 所需的生命周期事件执行一次新对话。

        功能：
            立即产出 accepted 事件，把 trace_id 和任务编号交给前端；主图
            执行期间定时产出 heartbeat，结束后产出 completed 或 interrupted。
            当前事件流不是 LLM 逐 Token 输出。

        参数含义：
            question:
                用户本轮问题。
            session_id:
                当前会话和 LangGraph 线程编号。
            trace_id:
                可选链路编号；未提供时由服务端生成。

        返回值含义：
            AsyncIterator[dict[str, Any]]:
                按发生顺序产出的 SSE 事件名称与 JSON 数据。
        """

        resolved_trace_id, task_id = self._start_task(
            trace_id=trace_id,
            session_id=session_id,
        )
        execution_task = asyncio.create_task(
            self._run_graph(
                question=question,
                session_id=session_id,
                trace_id=resolved_trace_id,
                resume_value=None,
                component="api_stream_handler",
            )
        )
        try:
            yield {
                "event": "accepted",
                "data": {
                    "status": "running",
                    "session_id": session_id,
                    "trace_id": resolved_trace_id,
                    "multi_agent_task_id": task_id,
                },
            }
            while True:
                try:
                    response = await asyncio.wait_for(
                        asyncio.shield(execution_task),
                        timeout=self.heartbeat_seconds,
                    )
                except TimeoutError:
                    yield {
                        "event": "heartbeat",
                        "data": {
                            "status": "running",
                            "multi_agent_task_id": task_id,
                        },
                    }
                    continue
                self._finish_task(task_id, response)
                yield {
                    "event": response.status,
                    "data": response.model_dump(mode="json"),
                }
                break
        except asyncio.CancelledError:
            self.task_registry.update(
                task_id,
                status="failed",
                error_message="SSE 客户端连接已断开",
            )
            raise
        except Exception:
            self.task_registry.update(
                task_id,
                status="failed",
                error_message="Agent 请求执行失败",
            )
            yield {
                "event": "error",
                "data": {
                    "status": "failed",
                    "multi_agent_task_id": task_id,
                    "message": "Agent 请求执行失败",
                },
            }
        finally:
            if not execution_task.done():
                execution_task.cancel()
            await asyncio.gather(
                execution_task,
                return_exceptions=True,
            )

    def cancel(self, multi_agent_task_id: str) -> CancellationResponse:
        """
        向指定运行中的多 Agent 任务发送取消信号。

        参数含义：
            multi_agent_task_id:
                需要取消的整次多 Agent 任务编号。

        返回值含义：
            CancellationResponse:
                是否找到任务并成功打开取消令牌的结构化结果。
        """

        requested = self.graph_runtime.cancel_multi_agent_task(
            multi_agent_task_id
        )
        if requested:
            self.task_registry.update(
                multi_agent_task_id,
                status="cancel_requested",
            )
        return CancellationResponse(
            multi_agent_task_id=multi_agent_task_id,
            cancellation_requested=requested,
            message=(
                "已发送取消请求，正在停止未完成步骤。"
                if requested
                else "没有找到对应的运行中多 Agent 任务。"
            ),
        )

    def get_task_status(
        self,
        multi_agent_task_id: str,
    ) -> TaskStatusResponse | None:
        """
        查询当前 API 进程中的任务状态。

        参数含义：
            multi_agent_task_id:
                需要查询的任务编号。

        返回值含义：
            TaskStatusResponse | None:
                找到时返回 HTTP 状态模型，否则返回 None。
        """

        snapshot = self.task_registry.get(multi_agent_task_id)
        return snapshot.to_response() if snapshot is not None else None

    def _start_task(
        self,
        *,
        trace_id: str | None,
        session_id: str,
    ) -> tuple[str, str]:
        """
        解析请求身份并登记 running 状态。

        参数含义：
            trace_id:
                调用方提供的可选链路编号。
            session_id:
                当前会话编号。

        返回值含义：
            tuple[str, str]:
                最终 trace_id 和由它构建的 multi_agent_task_id。
        """

        resolved_trace_id = str(trace_id or uuid4())
        task_id = build_multi_agent_task_id(resolved_trace_id)
        self.task_registry.start(
            multi_agent_task_id=task_id,
            trace_id=resolved_trace_id,
            session_id=session_id,
        )
        return resolved_trace_id, task_id

    def _finish_task(
        self,
        multi_agent_task_id: str,
        response: GraphRunResponse,
    ) -> None:
        """
        根据主图 API 响应更新任务终态。

        参数含义：
            multi_agent_task_id:
                当前任务编号。
            response:
                主图完成或中断后的 API 响应。

        返回值含义：
            None。
        """

        self.task_registry.update(
            multi_agent_task_id,
            status=(
                "completed"
                if response.status == "completed"
                else "interrupted"
            ),
            business_status=response.business_status,
        )

    async def _run_graph(
        self,
        *,
        question: str,
        session_id: str,
        trace_id: str,
        resume_value: str | None,
        component: str,
    ) -> GraphRunResponse:
        """
        在完整请求级运行时上下文中调用主图。

        参数含义：
            question:
                交给主图处理的用户文本。
            session_id:
                当前会话和 LangGraph 线程编号。
            trace_id:
                当前链路追踪编号。
            resume_value:
                新请求为 None；恢复请求为用户补充内容。
            component:
                写入 RuntimeContext 的 API 处理器名称。

        返回值含义：
            GraphRunResponse:
                已转换为 HTTP 层契约的主图运行结果。
        """

        trace_manager.ensure_trace(trace_id)
        context = RuntimeContext(
            trace_id=trace_id,
            session_id=session_id,
            user_id="unknown",
            component=component,
        )
        context.hooks().register(
            "tool.before",
            ToolCounterHook(),
        )
        await runtime_ctx.create(context)
        try:
            result = await self.graph_runner(
                question,
                thread_id=session_id,
                trace_id=trace_id,
                resume_value=resume_value,
            )
            return self._build_response(
                result=result,
                session_id=session_id,
                trace_id=trace_id,
            )
        finally:
            await runtime_ctx.destroy()

    @staticmethod
    def _build_response(
        *,
        result: GraphFinalResult | GraphInterruptResult,
        session_id: str,
        trace_id: str,
    ) -> GraphRunResponse:
        """
        把主图内部结果转换成 API 统一响应。

        参数含义：
            result:
                主图返回的完成结果或中断结果。
            session_id:
                当前 API 会话编号。
            trace_id:
                当前 API 链路编号。

        返回值含义：
            GraphRunResponse:
                不暴露内部 dataclass 的稳定 HTTP 响应。
        """

        common_fields = {
            "session_id": session_id,
            "thread_id": result.thread_id,
            "trace_id": trace_id,
            "multi_agent_task_id": build_multi_agent_task_id(trace_id),
            "checkpoint_ns": result.checkpoint_ns,
            "metadata": dict(result.metadata),
        }
        if isinstance(result, GraphFinalResult):
            business_status = str(
                result.metadata.get("business_status")
                or "completed"
            )
            return GraphRunResponse(
                status="completed",
                business_status=business_status,
                business_error=result.metadata.get("business_error"),
                answer=result.answer,
                **common_fields,
            )
        if isinstance(result, GraphInterruptResult):
            return GraphRunResponse(
                status="interrupted",
                business_status="awaiting_input",
                prompt=result.prompt,
                interrupt_type=result.interrupt_type.value,
                **common_fields,
            )
        raise TypeError("主图返回了不支持的结果类型")
