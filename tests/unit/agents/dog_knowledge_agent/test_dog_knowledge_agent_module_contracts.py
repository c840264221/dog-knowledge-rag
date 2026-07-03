"""
DogKnowledgeAgent 内部模块职责契约测试。

功能：
    测试 V1.7.2 DogKnowledgeAgent 内部职责地图是否完整、稳定。

测试重点：
    1. 所有预期职责层都存在。
    2. 每个职责层都有名称、职责、输入、输出。
    3. 每个职责层都声明了不应该做什么。
    4. 可以按 layer 查找模块契约。
    5. 可以渲染 Markdown 文档。
"""

from __future__ import annotations

import pytest

from src.agents.dog_knowledge_agent.module_contracts import (
    DogKnowledgeModuleContract,
    get_contract_by_layer,
    get_dog_knowledge_module_contracts,
    get_expected_dog_knowledge_layers,
    render_dog_knowledge_contract_markdown,
)


EXPECTED_LAYERS = (
    "entry",
    "query_builder",
    "retrieval",
    "rerank",
    "quality",
    "context_builder",
    "memory_context",
    "strategy",
    "generation",
    "debug_report",
)


EXPECTED_MODULE_NAMES = (
    "entry_node",
    "rag_query_builder",
    "retrieval_runner",
    "reranker",
    "retrieval_quality_evaluator",
    "rag_context_builder",
    "memory_context_provider",
    "answer_strategy_selector",
    "answer_generator",
    "dog_knowledge_debug_report",
)


def test_dog_knowledge_module_contracts_are_defined() -> None:
    """
    测试 DogKnowledgeAgent 模块契约已经定义。

    功能：
        确认 get_dog_knowledge_module_contracts 返回非空契约列表。

    参数：
        无。

    返回值：
        None。
    """

    contracts = get_dog_knowledge_module_contracts()

    assert contracts
    assert all(
        isinstance(
            contract,
            DogKnowledgeModuleContract,
        )
        for contract in contracts
    )


def test_expected_dog_knowledge_layers_are_complete() -> None:
    """
    测试 DogKnowledgeAgent 预期职责层完整。

    功能：
        确认 V1.7.2 规划中的职责层全部存在，且顺序符合标准执行链路。

    参数：
        无。

    返回值：
        None。
    """

    layers = get_expected_dog_knowledge_layers()

    assert layers == EXPECTED_LAYERS


@pytest.mark.parametrize(
    "layer",
    EXPECTED_LAYERS,
)
def test_get_contract_by_layer(
        layer: str,
) -> None:
    """
    测试按 layer 获取模块契约。

    功能：
        验证 get_contract_by_layer 可以根据层级名称找到对应契约。

    参数：
        layer:
            模块层级名称。

    返回值：
        None。
    """

    contract = get_contract_by_layer(
        layer,  # type: ignore[arg-type]
    )

    assert contract.layer == layer
    assert contract.module_name
    assert contract.chinese_name
    assert contract.responsibility
    assert contract.expected_input
    assert contract.expected_output


def test_get_contract_by_unknown_layer_raises_error() -> None:
    """
    测试未知 layer 会抛出异常。

    功能：
        如果传入不存在的模块层级，get_contract_by_layer 应该抛出 ValueError。

    参数：
        无。

    返回值：
        None。
    """

    with pytest.raises(
            ValueError,
    ):
        get_contract_by_layer(
            "unknown",  # type: ignore[arg-type]
        )


def test_each_contract_has_should_not_do_rules() -> None:
    """
    测试每个模块契约都有禁止职责说明。

    功能：
        确保每个职责层都明确说明自己不应该做什么。

    参数：
        无。

    返回值：
        None。
    """

    contracts = get_dog_knowledge_module_contracts()

    for contract in contracts:
        assert contract.should_not_do
        assert all(
            item
            for item in contract.should_not_do
        )


def test_each_contract_has_enterprise_reason() -> None:
    """
    测试每个模块契约都有企业级设计原因。

    功能：
        确保每个职责拆分都有工程解释。

    参数：
        无。

    返回值：
        None。
    """

    contracts = get_dog_knowledge_module_contracts()

    for contract in contracts:
        assert contract.enterprise_reason
        assert contract.enterprise_reason.strip()


def test_render_dog_knowledge_contract_markdown() -> None:
    """
    测试模块契约 Markdown 渲染。

    功能：
        验证职责契约可以被渲染成 Markdown 文档，并包含所有关键模块。

    参数：
        无。

    返回值：
        None。
    """

    markdown = render_dog_knowledge_contract_markdown()

    assert "DogKnowledgeAgent" in markdown
    assert (
        "entry -> query_builder -> retrieval -> rerank -> quality -> "
        "context_builder -> memory_context -> strategy -> generation -> debug_report"
    ) in markdown

    for module_name in EXPECTED_MODULE_NAMES:
        assert module_name in markdown
