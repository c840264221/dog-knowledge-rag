import pytest
from datetime import datetime, timezone

from src.graph.nodes.memory_extract_node import (
    build_memory_extract_node,
)
from src.memory.memory_schema import (
    PetProfileExtractionResult,
    PetProfileFact,
    PetProfileFactCandidate,
    PetProfileResolutionResult,
    PetProfileSaveResult,
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


class FakePetProfileService:
    """记录宠物档案保存参数并返回预设结果的测试服务。"""

    def __init__(self, result: PetProfileSaveResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    def save_extraction_result(self, **kwargs) -> PetProfileSaveResult:
        """
        记录节点交给宠物档案服务的数据。

        参数含义：
            **kwargs：用户标识、抽取结果、观测时间和当前宠物身份。

        返回值含义：
            PetProfileSaveResult：预设的宠物档案保存结果。
        """

        self.calls.append(kwargs)
        return self.result


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
    assert result["memory_retention_result"]["action"] == "accepted"
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
    assert result["memory_retention_result"]["action"] == "rejected"
    assert result["memory_save_result"] is None
    assert manager.calls == []
    assert checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_memory_extract_node_should_reject_candidate_below_policy_threshold() -> None:
    """
    验证 LLM 建议保存但未达到确定性门槛时不会调用 MemoryManager。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    node, _llm_provider, manager, checkpoint_manager = build_test_node(
        extract_result={
            "should_save": True,
            "memory_type": "preference",
            "content": "用户可能喜欢简短回答",
            "confidence": 0.60,
            "importance": 0.90,
            "reason": "模型认为可能是长期偏好。",
        }
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "也许以后回答短一点。",
        }
    )

    assert result["memory_saved"] is False
    assert result["memory_retention_result"]["action"] == "rejected"
    assert "可信度" in result["memory_retention_result"]["reason"]
    assert result["memory_save_result"] is None
    assert manager.calls == []
    assert checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_memory_extract_node_should_prefer_memory_source_text() -> None:
    """
    验证门卫准备的上下文文本优先于可变的业务 question。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    received_questions: list[str] = []

    async def capture_memory_extractor(
        llm_provider,
        question,
    ):
        """
        记录记忆抽取器收到的文本。

        参数含义：
            llm_provider:
                测试注入的模型服务占位对象。
            question:
                节点最终选择的记忆抽取文本。

        返回值含义：
            dict:
                不需要保存记忆的确定性结果。
        """

        _ = llm_provider
        received_questions.append(question)
        return {
            "should_save": False,
            "reason": "测试只检查输入来源。",
        }

    node = build_memory_extract_node(
        llm_provider=object(),
        memory_provider=FakeMemoryProvider(FakeMemoryManager()),
        checkpoint_manager=None,
        runtime_context_getter=lambda: None,
        memory_extractor=capture_memory_extractor,
    )

    await node(
        {
            "user_id": "user_001",
            "question": "6岁",
            "memory_source_text": (
                "旧任务正在询问：请补充狗狗年龄。\n"
                "用户本轮补充：6岁"
            ),
        }
    )

    assert received_questions == [
        "旧任务正在询问：请补充狗狗年龄。\n用户本轮补充：6岁"
    ]


@pytest.mark.asyncio
async def test_memory_extract_node_should_honor_explicit_empty_source() -> None:
    """
    验证门卫明确传入空记忆文本时不会回退到业务问题。

    功能：
        空字符串表示本轮输入是控制指令，普通记忆和宠物档案提取器都不能
        被调用；只有旧状态完全缺少该字段时才允许兼容 question。

    参数含义：无。
    返回值含义：None，pytest 根据两个提取器的调用次数判断是否通过。
    """

    memory_calls: list[str] = []
    profile_calls: list[str] = []

    async def capture_memory_extractor(llm_provider, question):
        """记录不应发生的普通记忆抽取调用。"""

        _ = llm_provider
        memory_calls.append(question)
        return {"should_save": False}

    async def capture_profile_extractor(llm_provider, user_text):
        """记录不应发生的宠物档案抽取调用。"""

        _ = llm_provider
        profile_calls.append(user_text)
        return default_pet_profile_extraction_result("测试")

    node = build_memory_extract_node(
        llm_provider=object(),
        memory_provider=FakeMemoryProvider(FakeMemoryManager()),
        checkpoint_manager=None,
        runtime_context_getter=lambda: None,
        memory_extractor=capture_memory_extractor,
        pet_profile_extractor=capture_profile_extractor,
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "简化执行",
            "memory_source_text": "",
        }
    )

    assert memory_calls == []
    assert profile_calls == []
    assert result["memory_extract_result"]["reason"] == "未执行 Memory 抽取"


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


@pytest.mark.asyncio
async def test_memory_extract_node_should_save_pet_profile_and_update_active_pet() -> None:
    """验证节点会保存宠物档案，并在唯一宠物明确时更新当前宠物身份。"""

    received_profile_texts: list[str] = []
    extraction_result = PetProfileExtractionResult(
        facts=[
            PetProfileFactCandidate(
                subject_reference="豆豆",
                attribute="weight_kg",
                value="30",
                confidence=0.98,
                evidence_text="豆豆现在30公斤",
            )
        ],
        reason="用户明确提供了宠物体重。",
    )
    resolved_fact = PetProfileFact(
        user_id="user_001",
        pet_key="pet_v1_test",
        pet_name="豆豆",
        attribute="weight_kg",
        value="30",
        confidence=0.98,
        evidence_text="豆豆现在30公斤",
        observed_at=datetime.now(timezone.utc),
    )
    profile_service = FakePetProfileService(
        PetProfileSaveResult(
            resolution=PetProfileResolutionResult(facts=[resolved_fact]),
            created_count=1,
            reason="保存成功。",
        )
    )

    async def fake_profile_extractor(*, llm_provider, user_text):
        """
        记录宠物档案抽取器收到的上下文文本。

        参数含义：
            llm_provider：节点注入的模型服务占位对象。
            user_text：节点选择的宠物档案抽取文本。

        返回值含义：
            PetProfileExtractionResult：预设候选事实。
        """

        _ = llm_provider
        received_profile_texts.append(user_text)
        return extraction_result

    node = build_memory_extract_node(
        llm_provider=object(),
        memory_provider=FakeMemoryProvider(FakeMemoryManager()),
        checkpoint_manager=None,
        runtime_context_getter=lambda: None,
        memory_extractor=lambda **kwargs: {
            "should_save": False,
            "reason": "不保存普通长期记忆。",
        },
        pet_profile_extractor=fake_profile_extractor,
        pet_profile_service=profile_service,
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "30公斤",
            "memory_source_text": "旧任务询问豆豆体重，用户补充：30公斤",
        }
    )

    assert received_profile_texts == [
        "旧任务询问豆豆体重，用户补充：30公斤"
    ]
    assert len(profile_service.calls) == 1
    assert profile_service.calls[0]["user_id"] == "user_001"
    assert result["pet_profile_save_result"]["created_count"] == 1
    assert result["active_pet_key"] == "pet_v1_test"
    assert result["active_pet_name"] == "豆豆"


@pytest.mark.asyncio
async def test_memory_extract_node_should_keep_memory_result_when_profile_fails() -> None:
    """验证宠物档案支线异常不会覆盖普通长期记忆处理结果。"""

    async def failing_profile_extractor(**kwargs):
        """
        模拟宠物档案抽取异常。

        参数含义：
            **kwargs：节点传入的模型服务和用户文本。

        返回值含义：
            无；该测试函数固定抛出异常。
        """

        _ = kwargs
        raise RuntimeError("profile failed")

    profile_service = FakePetProfileService(
        PetProfileSaveResult(
            resolution=PetProfileResolutionResult(),
        )
    )
    node = build_memory_extract_node(
        llm_provider=object(),
        memory_provider=FakeMemoryProvider(FakeMemoryManager()),
        checkpoint_manager=None,
        runtime_context_getter=lambda: None,
        memory_extractor=lambda **kwargs: {
            "should_save": False,
            "reason": "普通记忆处理完成。",
        },
        pet_profile_extractor=failing_profile_extractor,
        pet_profile_service=profile_service,
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "你好",
        }
    )

    assert result["memory_extract_result"]["reason"] == "普通记忆处理完成。"
    assert result["pet_profile_extraction_result"]["facts"] == []
    assert result["pet_profile_save_result"] == {}
    assert profile_service.calls == []
