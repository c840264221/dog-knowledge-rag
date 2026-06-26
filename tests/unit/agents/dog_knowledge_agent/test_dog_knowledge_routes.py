from src.agents.dog_knowledge_agent.routes import (
    route_dog_knowledge_model,
    route_after_dog_knowledge_evaluate,
)


def test_route_dog_knowledge_model_to_recommendation_by_intent():
    """
    测试 dog_knowledge_agent 可以根据 intent 路由到 recommendation_model。

    参数：
        无。

    返回值：
        无。
    """

    state = {
        "question": "新手适合养什么狗？",
        "intent": "recommend",
    }

    result = route_dog_knowledge_model(
        state=state,
    )

    assert result == "recommendation_model"


def test_route_dog_knowledge_model_to_recommendation_by_question():
    """
    测试 dog_knowledge_agent 可以根据推荐关键词路由到 recommendation_model。

    参数：
        无。

    返回值：
        无。
    """

    state = {
        "question": "我住公寓，帮我推荐一种狗",
    }

    result = route_dog_knowledge_model(
        state=state,
    )

    assert result == "recommendation_model"


def test_route_dog_knowledge_model_to_extract_by_exact_question():
    """
    测试具体犬种信息问题路由到 extract_model。

    参数：
        无。

    返回值：
        无。
    """

    state = {
        "question": "金毛寿命多长？",
    }

    result = route_dog_knowledge_model(
        state=state,
    )

    assert result == "extract_model"


def test_route_after_evaluate_to_rerank_by_route_decision():
    """
    测试 evaluate 后根据 route_decision 进入 rerank。

    参数：
        无。

    返回值：
        无。
    """

    state = {
        "route_decision": {
            "route": "rerank",
            "reason": "检索结果足够，进入重排序。",
        }
    }

    result = route_after_dog_knowledge_evaluate(
        state=state,
    )

    assert result == "rerank"


def test_route_after_evaluate_to_retry_by_route_decision():
    """
    测试 evaluate 后根据 route_decision 进入 retry。

    参数：
        无。

    返回值：
        无。
    """

    state = {
        "route_decision": {
            "route": "retry",
            "reason": "检索结果不足，需要重试。",
        }
    }

    result = route_after_dog_knowledge_evaluate(
        state=state,
    )

    assert result == "retry"


def test_route_after_evaluate_to_generate_after_retry_limit():
    """
    测试超过重试次数后进入 generate。

    参数：
        无。

    返回值：
        无。
    """

    state = {
        "retrieval_ok": False,
        "retry_count": 2,
    }

    result = route_after_dog_knowledge_evaluate(
        state=state,
    )

    assert result == "generate"