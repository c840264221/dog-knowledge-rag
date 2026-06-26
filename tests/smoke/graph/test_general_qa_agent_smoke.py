"""
general_qa_agent 图冒烟测试。

Smoke Test（冒烟测试）：
不是详细测试每个 node 的内部逻辑，而是验证整张 Graph 能否被构建、
能否启动执行、能否走到结束节点。

本测试覆盖最小流程：

supervisor
    -> answer_gen
    -> supervisor
    -> finish
    -> END

注意：
    这里不测试工具调用链路。
    工具调用链路包含 interrupt，人机确认流程更复杂，
    后续可以单独做 tool flow integration test。
"""

import pytest

from src.agents.general_qa_agent.agent import (
    build_general_qa_agent,
)


class FakeLLMResponse:
    """
    测试用假 LLM 响应。

    LLM Response（大语言模型响应）：
    真实模型返回值通常带有 content 字段。
    """

    def __init__(
        self,
        content,
    ):
        """
        初始化假 LLM 响应。

        参数：
            content：
                模拟模型返回内容。

        返回值：
            None：
                构造函数无返回值。
        """

        self.content = content


class FakeLLMProvider:
    """
    测试用假 LLMProvider。

    LLMProvider（大语言模型提供者）：
    用于模拟 supervisor 和 answer_gen 对大语言模型的调用。

    本测试中 safe_ainvoke 会按顺序返回：
        1. answer_gen
        2. 这是最终回答
        3. finish
    """

    def __init__(
        self,
        outputs,
    ):
        """
        初始化假 LLMProvider。

        参数：
            outputs：
                按调用顺序返回的模型输出列表。

        返回值：
            None：
                构造函数无返回值。
        """

        self.main_llm = "fake_main_llm"
        self.backup_llm = "fake_backup_llm"
        self.outputs = list(
            outputs
        )
        self.calls = []

    async def safe_ainvoke(
        self,
        llm,
        prompt,
        fallback_response,
    ):
        """
        模拟异步调用 LLM。

        参数：
            llm：
                当前使用的大语言模型对象。

            prompt：
                传入模型的 Prompt（提示词）。

            fallback_response：
                模型不可用时的兜底响应。

        返回值：
            FakeLLMResponse：
                模拟模型响应对象。
        """

        self.calls.append(
            {
                "llm": llm,
                "prompt": prompt,
                "fallback_response": fallback_response,
            }
        )

        if not self.outputs:
            return FakeLLMResponse(
                fallback_response
            )

        return FakeLLMResponse(
            self.outputs.pop(
                0
            )
        )


class FakeCheckpointManager:
    """
    测试用假 CheckpointManager。

    CheckpointManager（检查点管理器）：
    用于模拟 Graph 执行过程中保存 checkpoint。
    """

    def __init__(
        self,
    ):
        """
        初始化假 CheckpointManager。

        参数：
            无。

        返回值：
            None：
                构造函数无返回值。
        """

        self.save_count = 0

    def save_checkpoint(
        self,
    ):
        """
        模拟保存 checkpoint。

        参数：
            无。

        返回值：
            None：
                无业务返回值。
        """

        self.save_count += 1


class FakeCheckpointProvider:
    """
    测试用假 CheckpointProvider。

    CheckpointProvider（检查点提供者）：
    对外暴露 manager，供 graph builder 注入 node。
    """

    def __init__(
        self,
    ):
        """
        初始化假 CheckpointProvider。

        参数：
            无。

        返回值：
            None：
                构造函数无返回值。
        """

        self.manager = FakeCheckpointManager()


@pytest.mark.asyncio
async def test_general_qa_agent_should_run_minimal_answer_flow_to_finish():
    """
    测试 general_qa_agent 是否能跑通最小回答流程。

    流程：
        1. supervisor 决策进入 answer_gen
        2. answer_gen 生成最终回答
        3. answer_gen 回到 supervisor
        4. supervisor 决策 finish
        5. Graph 结束

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    llm_provider = FakeLLMProvider(
        outputs=[
            "answer_gen",
            "这是最终回答",
            "finish",
        ]
    )

    checkpoint_provider = FakeCheckpointProvider()

    graph = build_general_qa_agent(
        llm_provider=llm_provider,
        memory_provider=None,
        checkpoint_provider=checkpoint_provider,
    )

    result = await graph.ainvoke(
        {
            "question": "你好，请直接回答我。",
            "messages": [],
            "tool_results": [],
            "tool_calls": [],
        },
        config={
            "recursion_limit": 10,
        },
    )

    assert result["answer"] == "这是最终回答"
    assert result["next_worker"] == "finish"

    assert len(
        llm_provider.calls
    ) == 3

    assert checkpoint_provider.manager.save_count == 3


@pytest.mark.asyncio
async def test_general_qa_agent_should_build_and_run_without_checkpoint_provider():
    """
    测试不传 checkpoint_provider 时，Graph 是否仍能跑通最小流程。

    场景：
        有些测试环境或降级运行环境可能没有 checkpoint。
        此时 node 应该跳过保存 checkpoint，而不是直接报错。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    llm_provider = FakeLLMProvider(
        outputs=[
            "answer_gen",
            "没有 checkpoint 也能回答",
            "finish",
        ]
    )

    graph = build_general_qa_agent(
        llm_provider=llm_provider,
        memory_provider=None,
        checkpoint_provider=None,
    )

    result = await graph.ainvoke(
        {
            "question": "你好",
            "messages": [],
            "tool_results": [],
            "tool_calls": [],
        },
        config={
            "recursion_limit": 10,
        },
    )

    assert result["answer"] == "没有 checkpoint 也能回答"
    assert result["next_worker"] == "finish"

    assert len(
        llm_provider.calls
    ) == 3