"""RAG 召回质量评估节点测试。"""

from __future__ import annotations

from typing import Any

from src.graph.nodes import evaluate_node


class _FakeRuntimeState:
    """接收测试节点写入的当前节点名称。"""

    def set_node(self, _node_name: str) -> None:
        """接收节点名称；本测试不需要保存。"""


class _FakeRuntimeTimeline:
    """接收测试节点写入的时间线事件。"""

    def add_event(self, **_event: Any) -> None:
        """接收时间线事件；本测试不需要保存。"""


class _FakeRuntime:
    """提供评估节点运行需要的最小 Runtime Context。"""

    def state(self) -> _FakeRuntimeState:
        """返回最小运行状态对象。"""

        return _FakeRuntimeState()

    def timeline(self) -> _FakeRuntimeTimeline:
        """返回最小时间线对象。"""

        return _FakeRuntimeTimeline()


class _RecordingLogger:
    """记录评估节点产生的日志文本。"""

    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.debug_messages: list[str] = []

    def info(self, message: str) -> None:
        """保存 INFO 日志。"""

        self.info_messages.append(message)

    def debug(self, message: str) -> None:
        """保存 DEBUG 日志。"""

        self.debug_messages.append(message)


def test_evaluate_node_should_log_clean_retrieval_question(
    monkeypatch: Any,
) -> None:
    """
    验证质量评估入口日志不会展示完整 Skill 执行说明。

    功能：
        构造同时包含完整 question 和干净 retrieval_question 的状态，确认日志
        与质量评估统一使用专用检索问题。

    参数含义：
        monkeypatch:
            pytest 提供的临时替换工具，用于隔离 Runtime Context 和日志器。

    返回值含义：
        None，断言失败时由 pytest 报错。
    """

    recording_logger = _RecordingLogger()
    monkeypatch.setattr(evaluate_node.runtime_ctx, "get", lambda: _FakeRuntime())
    monkeypatch.setattr(evaluate_node, "logger", recording_logger)

    evaluate_node.evaluate_retrieval_node(
        {
            "question": "制定训练计划\n\nSkill 执行说明",
            "retrieval_question": "6岁金毛等待与召回训练方法",
            "rag_context": {},
            "docs": [],
            "filters": {},
            "dog_name": "Golden Retriever",
        }
    )

    entry_message = recording_logger.info_messages[0]
    assert "retrieval_question=6岁金毛等待与召回训练方法" in entry_message
    assert "Skill 执行说明" not in entry_message
