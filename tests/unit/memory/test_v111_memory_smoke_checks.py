from types import SimpleNamespace

from src.memory.v111_smoke_checks import (
    render_memory_pipeline_smoke_markdown,
    validate_memory_pipeline_smoke,
)


def build_valid_smoke_inputs() -> dict:
    """
    构建通过 V1.11 记忆管线检查的测试数据。

    功能：
        组装有效的保存结果、向量 metadata（向量元数据）、
        相关问题召回结果和不相关问题拦截结果。

    参数：
        无。

    返回值：
        dict：可传给 validate_memory_pipeline_smoke 的关键字参数。
    """

    return {
        "save_result": {
            "action": "created",
            "memory_id": 1,
            "content": "Golden Retriever",
        },
        "vector_documents": [
            SimpleNamespace(
                metadata={
                    "memory_id": "1",
                    "user_id": "user_001",
                    "memory_type": "favorite_dog",
                    "status": "active",
                    "source": "conversation",
                    "importance": 0.8,
                }
            )
        ],
        "related_state": {
            "memory_context": "- 用户喜欢的狗狗：Golden Retriever",
            "memory_recall_result": {
                "status": "applied",
                "selected_count": 1,
                "selected_memory_ids": [1],
            },
        },
        "unrelated_state": {
            "memory_context": "暂无用户记忆",
            "memory_recall_result": {
                "status": "empty",
                "selected_count": 0,
                "selected_memory_ids": [],
            },
        },
    }


def test_validate_memory_pipeline_smoke_should_pass_valid_pipeline() -> None:
    """
    测试完整有效的记忆管线数据通过检查。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_memory_pipeline_smoke(
        **build_valid_smoke_inputs()
    )

    assert result.passed is True
    assert result.errors == ()
    assert result.observations["related_status"] == "applied"
    assert result.observations["unrelated_status"] == "empty"


def test_validate_memory_pipeline_smoke_should_fail_missing_metadata() -> None:
    """
    测试向量 metadata 不完整时冒烟检查失败。

    参数：
        无。

    返回值：
        None。
    """

    inputs = build_valid_smoke_inputs()
    inputs["vector_documents"][0].metadata.pop("importance")

    result = validate_memory_pipeline_smoke(**inputs)

    assert result.passed is False
    assert any(
        "importance" in error
        for error in result.errors
    )


def test_validate_memory_pipeline_smoke_should_fail_memory_pollution() -> None:
    """
    测试不相关问题错误使用记忆时冒烟检查失败。

    参数：
        无。

    返回值：
        None。
    """

    inputs = build_valid_smoke_inputs()
    inputs["unrelated_state"] = {
        "memory_context": "用户喜欢金毛。",
        "memory_recall_result": {
            "status": "applied",
            "selected_count": 1,
        },
    }

    result = validate_memory_pipeline_smoke(**inputs)

    assert result.passed is False
    assert any(
        "语义门槛" in error
        for error in result.errors
    )


def test_render_memory_pipeline_smoke_markdown_should_show_real_status() -> None:
    """
    测试 Markdown 报告根据检查结果显示真实 PASS 状态。

    参数：
        无。

    返回值：
        None。
    """

    result = validate_memory_pipeline_smoke(
        **build_valid_smoke_inputs()
    )

    markdown = render_memory_pipeline_smoke_markdown(result)

    assert "V1.11 Memory Pipeline Smoke Report" in markdown
    assert "status: PASS" in markdown
    assert "related_status: applied" in markdown
