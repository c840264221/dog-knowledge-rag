"""RAG 查询问题来源测试。"""

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
