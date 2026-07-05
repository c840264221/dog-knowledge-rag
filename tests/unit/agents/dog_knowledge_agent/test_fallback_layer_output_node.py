from src.agents.dog_knowledge_agent.nodes.fallback_layer_output_node import (
    build_dog_fallback_layer_output_from_state,
    build_dog_knowledge_fallback_layer_output_node,
)


def test_fallback_layer_output_should_return_none_without_fallback_signal() -> None:
    """
    测试没有兜底迹象时不生成兜底层输出。

    参数：
        无。

    返回值：
        None。
    """

    result = build_dog_fallback_layer_output_from_state(
        {
            "retrieval_evaluated": True,
            "retrieval_ok": True,
            "retrieval_failure_type": "",
        }
    )

    assert result is None


def test_fallback_layer_output_should_build_result_from_retrieval_failure() -> None:
    """
    测试检索失败时可以生成兜底层输出。

    参数：
        无。

    返回值：
        None。
    """

    result = build_dog_fallback_layer_output_from_state(
        {
            "retrieval_evaluated": True,
            "retrieval_ok": False,
            "retrieval_failure_type": "no_relevant_context",
            "dog_generation_result": {
                "generated_answer": "我暂时无法基于当前犬种知识库可靠回答这个问题。",
            },
        }
    )

    assert result is not None
    assert result.is_fallback is True
    assert result.fallback_reason == "no_relevant_context"
    assert result.generated_answer == (
        "我暂时无法基于当前犬种知识库可靠回答这个问题。"
    )
    assert result.confidence == 0.1
    assert result.metadata["source"] == "fallback_layer_output_node"
    assert result.metadata["retrieval_ok"] is False


def test_fallback_layer_output_node_should_return_empty_update_without_signal() -> None:
    """
    测试节点在没有兜底迹象时返回空 update。

    参数：
        无。

    返回值：
        None。
    """

    node = build_dog_knowledge_fallback_layer_output_node()

    update = node(
        {
            "retrieval_evaluated": True,
            "retrieval_ok": True,
            "retrieval_failure_type": "",
        }
    )

    assert update == {}


def test_fallback_layer_output_node_should_return_state_update_when_error_exists() -> None:
    """
    测试存在 error 时节点可以返回兜底层 state update。

    参数：
        无。

    返回值：
        None。
    """

    node = build_dog_knowledge_fallback_layer_output_node()

    update = node(
        {
            "error": "问题超出当前犬种知识库边界。",
            "final_answer": "我暂时无法回答这个问题。",
        }
    )

    assert update["dog_fallback_result"]["is_fallback"] is True
    assert update["dog_fallback_result"]["fallback_reason"] == (
        "问题超出当前犬种知识库边界。"
    )
    assert update["dog_fallback_result"]["generated_answer"] == (
        "我暂时无法回答这个问题。"
    )
