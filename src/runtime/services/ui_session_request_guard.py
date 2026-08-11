"""Gradio UI 同一会话请求防重服务。"""

from __future__ import annotations

from threading import Lock


class UiSessionRequestGuard:
    """
    防止同一个 UI 会话同时启动多次主图执行。

    功能：
        记录当前正在执行请求的 session_id（会话编号）。同一个会话的第一
        个请求可以开始；在它结束前到达的重复请求会被拒绝，避免多个请求
        共同恢复同一份 Checkpoint（检查点）并重复调用答案生成节点。

    参数含义：
        无。

    返回值含义：
        UiSessionRequestGuard:
            可以登记和释放 UI 会话执行权的防重对象。
    """

    def __init__(self) -> None:
        # 当前仍有主图请求正在执行的会话编号。
        self._active_session_ids: set[str] = set()
        self._lock = Lock()

    def try_start(self, session_id: str) -> bool:
        """
        尝试取得当前会话的主图执行权。

        功能：
            使用进程内互斥锁原子地完成“检查是否执行中”和“登记执行中”，
            防止两个并发请求同时通过检查。

        参数含义：
            session_id:
                Gradio 为当前浏览器会话生成的稳定编号。

        返回值含义：
            bool:
                True 表示本次请求可以执行；False 表示已有请求正在执行。
        """

        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id:
            return False

        with self._lock:
            if normalized_session_id in self._active_session_ids:
                return False
            self._active_session_ids.add(normalized_session_id)
            return True

    def finish(self, session_id: str) -> None:
        """
        释放当前会话的主图执行权。

        功能：
            请求正常完成或抛出异常后移除执行中标记，让该会话可以提交下一
            个请求。重复释放不会报错，便于放在 finally（最终清理块）中。

        参数含义：
            session_id:
                需要释放执行权的 Gradio 会话编号。

        返回值含义：
            None:
                本方法只更新内部状态，不返回业务数据。
        """

        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id:
            return

        with self._lock:
            self._active_session_ids.discard(normalized_session_id)


def should_resume_from_primary_input(state: object) -> bool:
    """
    判断普通输入框内容是否应该作为等待任务的恢复输入。

    功能：
        当 Gradio session state（会话状态）已经标记 pending（正在等待输入）
        时，让普通输入框与确认面板统一进入恢复链路，避免把同一份补充内容
        当成一条新的主图请求。

    参数含义：
        state:
            Gradio 保存的当前会话状态；非字典值按没有等待任务处理。

    返回值含义：
        bool:
            True 表示应该恢复旧任务；False 表示应该启动新任务。
    """

    return isinstance(state, dict) and state.get("pending") is True
