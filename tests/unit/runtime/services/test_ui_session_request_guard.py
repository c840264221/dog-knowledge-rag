"""Gradio UI 会话请求防重服务测试。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from src.runtime.services.ui_session_request_guard import (
    UiSessionRequestGuard,
    should_resume_from_primary_input,
)


def test_guard_should_reject_same_session_until_first_request_finishes():
    """同一会话的前一个请求未结束时，后一个请求不能再次开始。"""

    guard = UiSessionRequestGuard()

    assert guard.try_start("session-1") is True
    assert guard.try_start("session-1") is False

    guard.finish("session-1")

    assert guard.try_start("session-1") is True


def test_guard_should_allow_different_sessions_to_run_concurrently():
    """不同会话之间互不阻塞，可以同时执行主图。"""

    guard = UiSessionRequestGuard()

    assert guard.try_start("session-1") is True
    assert guard.try_start("session-2") is True


def test_guard_should_allow_only_one_concurrent_request_for_same_session():
    """两个并发请求争抢同一会话时，只能有一个取得执行权。"""

    guard = UiSessionRequestGuard()
    start_barrier = Barrier(2)

    def try_start_after_both_threads_are_ready() -> bool:
        """等待两个测试线程就绪后，同时尝试取得会话执行权。"""

        start_barrier.wait()
        return guard.try_start("session-1")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: try_start_after_both_threads_are_ready(),
                range(2),
            )
        )

    assert sorted(results) == [False, True]


def test_primary_input_should_resume_only_when_state_is_pending():
    """只有明确处于等待状态时，普通输入框才进入恢复链路。"""

    assert should_resume_from_primary_input({"pending": True}) is True
    assert should_resume_from_primary_input({"pending": False}) is False
    assert should_resume_from_primary_input({}) is False
    assert should_resume_from_primary_input(None) is False
