"""
Gradio 会话到运行中多 Agent 任务的轻量追踪器。

功能：
    在 UI 请求尚未返回、gr.State 还不能获得最新 trace_id 时，使用服务端
    session_id 找到本轮可取消的 multi_agent_task_id。
"""

from __future__ import annotations

from threading import Lock


class MultiAgentUiTaskTracker:
    """
    保存 Gradio 会话当前正在处理的多 Agent 任务编号。

    功能：
        请求开始时登记 session_id 与任务编号，请求结束时按原编号清理；
        取消按钮可以在回答返回前通过 session_id 查到任务编号。

    参数含义：
        无。

    返回值含义：
        MultiAgentUiTaskTracker:
            可以被同一 UI 进程中的提交与取消回调共享的线程安全追踪器。
    """

    def __init__(self) -> None:
        self._task_ids_by_session: dict[str, str] = {}
        self._lock = Lock()

    def register(self, session_id: str, multi_agent_task_id: str) -> None:
        """
        登记一个 UI 会话当前正在处理的任务编号。

        参数含义：
            session_id:
                Gradio 为当前浏览器会话提供的 session_hash。
            multi_agent_task_id:
                根据本轮 trace_id 构建的多 Agent 任务编号。

        返回值含义：
            None。
        """

        normalized_session_id = str(session_id or "").strip()
        normalized_task_id = str(multi_agent_task_id or "").strip()
        if not normalized_session_id:
            raise ValueError("UI session_id 不能为空")
        if not normalized_task_id:
            raise ValueError("multi_agent_task_id 不能为空")

        with self._lock:
            existing_task_id = self._task_ids_by_session.get(
                normalized_session_id
            )
            if existing_task_id is not None:
                raise ValueError(
                    "当前 UI 会话已经有正在运行的请求: "
                    f"{existing_task_id}"
                )
            self._task_ids_by_session[normalized_session_id] = (
                normalized_task_id
            )

    def get(self, session_id: str) -> str | None:
        """
        获取指定 UI 会话当前登记的任务编号。

        参数含义：
            session_id:
                Gradio 当前浏览器会话编号。

        返回值含义：
            str | None:
                找到时返回任务编号，没有运行中请求时返回 None。
        """

        normalized_session_id = str(session_id or "").strip()
        with self._lock:
            return self._task_ids_by_session.get(normalized_session_id)

    def unregister(
        self,
        session_id: str,
        multi_agent_task_id: str,
    ) -> bool:
        """
        清理已经结束的 UI 请求登记。

        参数含义：
            session_id:
                Gradio 当前浏览器会话编号。
            multi_agent_task_id:
                请求开始时登记的任务编号，用于避免误删后续新请求。

        返回值含义：
            bool:
                当前登记与传入编号一致并成功删除时返回 True，否则返回 False。
        """

        normalized_session_id = str(session_id or "").strip()
        normalized_task_id = str(multi_agent_task_id or "").strip()
        with self._lock:
            if self._task_ids_by_session.get(
                normalized_session_id
            ) != normalized_task_id:
                return False
            del self._task_ids_by_session[normalized_session_id]
        return True
