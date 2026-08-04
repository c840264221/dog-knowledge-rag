"""RAG 查询问题来源测试。"""

from typing import Any

from src.rag.evaluators import retrieval_quality_evaluator
from src.rag.observation.diagnostics import build_retrieval_diagnostics
from src.rag.query_builders.rag_query_builder import (
    resolve_question_from_state,
)


def test_rag_query_should_prefer_clean_retrieval_question() -> None:
    """
    测试 RAG 优先使用不含 Skill 说明的专用检索问题。

    功能：
        当 question 包含执行控制说明而 retrieval_question 保存简洁业务目标
        时，确认查询构建器不会把整段执行说明交给向量检索。

    参数含义：
        无。

    返回值含义：
        None。
    """

    question = resolve_question_from_state(
        {
            "question": (
                "制定训练计划。\n\n"
                "以下是当前步骤必须遵守的 Skill 执行说明：很长的说明。"
            ),
            "retrieval_question": "6岁金毛等待与召回训练方法",
        }
    )

    assert question == "6岁金毛等待与召回训练方法"


def test_rag_query_should_keep_question_backward_compatibility() -> None:
    """
    测试没有专用检索字段时继续读取原有 question。

    功能：
        保证尚未接入 Skill 的旧调用方不需要立即增加 retrieval_question。

    参数含义：
        无。

    返回值含义：
        None。
    """

    question = resolve_question_from_state(
        {"question": "金毛有哪些性格特点？"}
    )

    assert question == "金毛有哪些性格特点？"


def test_retrieval_diagnostics_should_record_clean_question() -> None:
    """
    测试 RAG 诊断报告记录真正参与检索的干净问题。

    功能：
        防止 Debug Report 的“原始问题”继续展示 Skill 说明和 Worker 恢复信息。

    参数含义：
        无。

    返回值含义：
        None。
    """

    diagnostics = build_retrieval_diagnostics(
        state={
            "question": "业务问题\n\nSkill 执行说明",
            "retrieval_question": "6岁金毛训练方法",
        },
        stage="retrieve",
    )

    assert diagnostics["question"] == "6岁金毛训练方法"


def test_retrieval_quality_should_evaluate_clean_question(
    monkeypatch: Any,
) -> None:
    """
    测试召回质量评估使用与向量召回相同的干净问题。

    功能：
        截获质量评估器收到的 question，确认 Skill 说明不会参与主题匹配判断。

    参数含义：
        monkeypatch:
            pytest 提供的临时替换工具，用来记录核心评估函数收到的参数。

    返回值含义：
        None。
    """

    captured: dict[str, Any] = {}
    expected_result = object()

    def fake_evaluate_rag_context_quality(**kwargs: Any) -> object:
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr(
        retrieval_quality_evaluator,
        "evaluate_rag_context_quality",
        fake_evaluate_rag_context_quality,
    )

    result = retrieval_quality_evaluator.evaluate_retrieval_quality(
        {
            "question": "业务问题\n\nSkill 执行说明",
            "retrieval_question": "6岁金毛训练方法",
            "rag_context": {},
        }
    )

    assert result is expected_result
    assert captured["question"] == "6岁金毛训练方法"
