from src.agents.dog_knowledge_agent.v174_smoke_checks import (
    extract_v174_layer_contract_fields,
    render_dog_knowledge_layer_contract_smoke_markdown,
    validate_dog_knowledge_layer_contract_state,
)


def build_valid_v174_state() -> dict:
    """
    构建有效的 V1.7.4 分层契约测试 state。

    功能：
        生成一个包含必需分层契约字段的测试 state。

    参数含义：
        无。

    返回值含义：
        dict:
            有效测试 state。
    """

    return {
        "dog_query_result": {
            "question": "金毛适合新手养吗？",
            "query_type": "exact_lookup",
        },
        "dog_retrieval_result": {
            "query_type": "exact_lookup",
            "evidences": [],
            "retrieved_count": 0,
        },
        "dog_generation_result": {
            "generated_answer": "金毛适合新手，但需要运动和训练。",
        },
        "dog_knowledge_pipeline_result": {
            "question": "金毛适合新手养吗？",
            "query_type": "exact_lookup",
            "status": "answered",
            "answer": "金毛适合新手，但需要运动和训练。",
            "confidence": 0.8,
            "metadata": {},
        },
        "dog_knowledge_answer": {
            "answer": "金毛适合新手，但需要运动和训练。",
        },
        "dog_knowledge_answer_public": {
            "answer": "金毛适合新手，但需要运动和训练。",
        },
        "final_answer": "金毛适合新手，但需要运动和训练。",
    }


def test_extract_v174_layer_contract_fields() -> None:
    """
    测试提取 V1.7.4 分层契约字段。

    功能：
        验证工具函数可以从 state 中识别已经存在的分层契约字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    fields = extract_v174_layer_contract_fields(
        state=build_valid_v174_state(),
    )

    assert "dog_query_result" in fields
    assert "dog_knowledge_pipeline_result" in fields
    assert "final_answer" in fields


def test_validate_v174_layer_contract_state_passes() -> None:
    """
    测试有效 V1.7.4 state 可以通过检查。

    功能：
        验证包含必需字段的 state 会被判定为通过。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = validate_dog_knowledge_layer_contract_state(
        state=build_valid_v174_state(),
    )

    assert result.passed is True
    assert result.errors == ()


def test_validate_v174_layer_contract_state_fails_when_missing_fields() -> None:
    """
    测试缺少必需字段时检查失败。

    功能：
        验证空 state 会返回错误。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = validate_dog_knowledge_layer_contract_state(
        state={},
    )

    assert result.passed is False
    assert result.errors


def test_validate_v174_layer_contract_state_fails_when_final_answer_empty() -> None:
    """
    测试 final_answer 为空时检查失败。

    功能：
        验证旧兼容输出 final_answer 仍然是必需的非空字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    state = build_valid_v174_state()
    state["final_answer"] = ""

    result = validate_dog_knowledge_layer_contract_state(
        state=state,
    )

    assert result.passed is False
    assert any(
        "final_answer" in error
        for error in result.errors
    )


def test_render_v174_layer_contract_smoke_markdown() -> None:
    """
    测试渲染 V1.7.4 smoke markdown 报告。

    功能：
        验证检查结果可以转换成 Markdown 文本。

    参数含义：
        无。

    返回值含义：
        None。
    """

    result = validate_dog_knowledge_layer_contract_state(
        state=build_valid_v174_state(),
    )

    markdown = render_dog_knowledge_layer_contract_smoke_markdown(result)

    assert "V1.7.4 DogKnowledgeAgent Layer Contract Smoke Report" in markdown
    assert "dog_query_result" in markdown
