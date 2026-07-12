import pytest

from src.graph.nodes.memory_extract_node import (
    build_memory_extract_node,
)


class FakeMemoryManager:
    """
    测试用 MemoryManager（记忆管理器）假对象。

    功能：
        记录 save_memory 调用参数，并返回预设的保存结果或异常。

    参数：
        result：保存成功时返回的字典。
        error：需要在保存时抛出的异常。

    返回值：
        FakeMemoryManager：测试用记忆管理器。
    """

    def __init__(self, result=None, error=None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def save_memory(self, **kwargs):
        """
        模拟保存记忆。

        参数：
            **kwargs：节点传入 MemoryManager.save_memory 的业务参数。

        返回值：
            dict | None：预设的保存结果。
        """

        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeMemoryProvider:
    """
    测试用 MemoryProvider（记忆服务提供者）。

    参数：
        manager：注入节点的记忆管理器。

    返回值：
        FakeMemoryProvider：包含 manager 属性的假服务提供者。
    """

    def __init__(self, manager: FakeMemoryManager) -> None:
        self.manager = manager


class FakeCheckpointManager:
    """
    测试用 CheckpointManager（检查点管理器）。

    参数：
        error：保存检查点时需要抛出的异常。

    返回值：
        FakeCheckpointManager：可记录保存次数的假对象。
    """

    def __init__(self, error=None) -> None:
        self.error = error
        self.save_count = 0

    def save_checkpoint(self) -> None:
        """
        模拟保存检查点。

        参数：
            无。

        返回值：
            None。
        """

        self.save_count += 1
        if self.error is not None:
            raise self.error


def build_test_node(
        extract_result: dict,
        save_result: dict | None = None,
        save_error: Exception | None = None,
        checkpoint_error: Exception | None = None,
):
    """
    构建完全使用注入依赖的测试记忆抽取节点。

    参数：
        extract_result：假记忆抽取器的返回值。
        save_result：假记忆管理器的保存结果。
        save_error：保存记忆时的测试异常。
        checkpoint_error：保存检查点时的测试异常。

    返回值：
        tuple：节点、LLMProvider 假对象、MemoryManager 和 CheckpointManager。
    """

    llm_provider = object()
    manager = FakeMemoryManager(
        result=save_result,
        error=save_error,
    )
    checkpoint_manager = FakeCheckpointManager(
        error=checkpoint_error,
    )

    async def fake_memory_extractor(
            llm_provider,
            question,
    ):
        """
        返回预设的记忆抽取结果。

        参数：
            llm_provider：节点注入的 LLMProvider。
            question：当前用户问题。

        返回值：
            dict：预设的记忆抽取字典。
        """

        return extract_result

    node = build_memory_extract_node(
        llm_provider=llm_provider,
        memory_provider=FakeMemoryProvider(manager),
        checkpoint_manager=checkpoint_manager,
        runtime_context_getter=lambda: None,
        memory_extractor=fake_memory_extractor,
    )
    return node, llm_provider, manager, checkpoint_manager


@pytest.mark.asyncio
async def test_memory_extract_node_should_save_with_injected_dependencies() -> None:
    """
    测试 should_save=True 时使用注入的 MemoryProvider 保存记忆。

    参数：无。
    返回值：None。
    """

    node, _llm_provider, manager, checkpoint_manager = build_test_node(
        extract_result={
            "should_save": True,
            "memory_type": "favorite_dog",
            "content": "金毛",
            "confidence": 0.9,
            "importance": 0.8,
            "reason": "用户明确表达偏好。",
        },
        save_result={
            "action": "created",
            "memory_id": 12,
        },
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "我喜欢金毛。",
        }
    )

    assert result["memory_saved"] is True
    assert result["memory_save_result"]["memory_id"] == 12
    assert manager.calls == [
        {
            "user_id": "user_001",
            "memory_type": "favorite_dog",
            "content": "金毛",
            "confidence": 0.9,
            "importance": 0.8,
            "source": "conversation",
        }
    ]
    assert checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_memory_extract_node_should_skip_save_when_not_required() -> None:
    """
    测试 should_save=False 时不调用 MemoryManager。

    参数：无。
    返回值：None。
    """

    node, _llm_provider, manager, checkpoint_manager = build_test_node(
        extract_result={
            "should_save": False,
            "memory_type": "preference",
            "content": "",
            "confidence": 0.0,
            "importance": 0.0,
            "reason": "当前输入不是长期记忆。",
        }
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "今天天气怎么样？",
        }
    )

    assert result["memory_saved"] is False
    assert result["memory_save_result"] is None
    assert manager.calls == []
    assert checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_memory_extract_node_should_fallback_when_save_failed() -> None:
    """
    测试记忆保存异常时不阻断节点返回。

    参数：无。
    返回值：None。
    """

    node, _llm_provider, manager, _checkpoint_manager = build_test_node(
        extract_result={
            "should_save": True,
            "memory_type": "preference",
            "content": "用户喜欢安静的犬种",
            "confidence": 0.8,
            "importance": 0.6,
            "reason": "长期偏好。",
        },
        save_error=RuntimeError("save failed"),
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "我喜欢安静的犬种。",
        }
    )

    assert result["memory_saved"] is False
    assert result["memory_save_result"] is None
    assert len(manager.calls) == 1


@pytest.mark.asyncio
async def test_memory_extract_node_should_ignore_checkpoint_failure() -> None:
    """
    测试 checkpoint 保存失败时仍保留记忆业务结果。

    参数：无。
    返回值：None。
    """

    node, _llm_provider, _manager, checkpoint_manager = build_test_node(
        extract_result={
            "should_save": False,
            "reason": "无需保存。",
        },
        checkpoint_error=RuntimeError("checkpoint failed"),
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "你好。",
        }
    )

    assert result["memory_saved"] is False
    assert checkpoint_manager.save_count == 1
