from src.graph.states.dog_state import (
    DogState,
)


def test_dog_state_should_include_v174_layer_contract_fields() -> None:
    """
    测试 DogState 包含 v1.7.4 分层契约中间产物字段。

    功能：
        确认 LangGraph state schema 中声明了每层标准输出字段，
        避免后续节点返回 layer output 时缺少状态通道。

    参数：
        无。

    返回值：
        None。
    """

    expected_fields = {
        "dog_query_result",
        "dog_retrieval_result",
        "dog_recommendation_result",
        "dog_generation_result",
        "dog_fallback_result",
        "dog_knowledge_pipeline_result",
    }

    assert expected_fields <= set(DogState.__annotations__)
