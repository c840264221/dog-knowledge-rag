"""
memory_retrieve_node 单元测试。

Memory Retrieve Node（记忆召回节点）：
用于根据 user_id 和 question 召回用户长期记忆，
并把召回结果写入 state["memory_context"]。

Semantic Recall（语义召回）：
根据语义相似度查找相关长期记忆。

本测试覆盖：
1. 正常召回字符串记忆
2. 召回结果为 None 时兜底
3. 召回结果为 dict/list 时转字符串
4. user_id 优先级高于 session_id
5. 缺少 user_id 时使用 session_id
6. user_id 和 session_id 都缺失时使用 default_user
7. question 缺失时使用空字符串
8. semantic_recall.retrieve 异常时降级
9. checkpoint 保存
10. checkpoint 保存失败不影响返回
11. runtime_context_getter 返回 None 时不报错
12. 支持异步 retrieve 返回
"""

import pytest

from src.graph.nodes.memory_retrieve_node import (
    build_memory_retrieve_node,
    _format_memory_context,
)


class FakeStateScope:
    """
    测试用假 StateScope。

    StateScope（状态作用域）：
    用于记录当前 Graph 正在执行的 node。
    """

    def __init__(self):
        """
        初始化假 StateScope。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.current_node = None

    def set_node(self, node_name):
        """
        设置当前 node 名称。

        参数：
            node_name：
                当前执行的节点名称。

        返回值：
            None：无业务返回值。
        """

        self.current_node = node_name


class FakeTimelineScope:
    """
    测试用假 TimelineScope。

    TimelineScope（时间线作用域）：
    用于记录 node 执行事件。
    """

    def __init__(self):
        """
        初始化假 TimelineScope。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.events = []

    def add_event(self, event_type, name, metadata=None):
        """
        添加时间线事件。

        参数：
            event_type：
                事件类型，例如 node。

            name：
                事件名称，例如 memory_retrieve_node。

            metadata：
                事件附加信息，默认为 None。

        返回值：
            None：无业务返回值。
        """

        self.events.append(
            {
                "event_type": event_type,
                "name": name,
                "metadata": metadata,
            }
        )


class FakeRuntimeContext:
    """
    测试用假 RuntimeContext。

    RuntimeContext（运行时上下文）：
    表示一次请求执行过程中的运行时环境。
    """

    def __init__(self):
        """
        初始化假 RuntimeContext。

        参数：
            无。

        返回值：
            None：构造函数无返回值。
        """

        self.state_scope = FakeStateScope()
        self.timeline_scope = FakeTimelineScope()

    def state(self):
        """
        获取假 StateScope。

        参数：
            无。

        返回值：
            FakeStateScope：
                测试用状态作用域。
        """

        return self.state_scope

    def timeline(self):
        """
        获取假 TimelineScope。

        参数：
            无。

        返回值：
            FakeTimelineScope：
                测试用时间线作用域。
        """

        return self.timeline_scope


class FakeCheckpointManager:
    """
    测试用假 CheckpointManager。

    CheckpointManager（检查点管理器）：
    用于保存运行时 checkpoint。
    """

    def __init__(self, error=None):
        """
        初始化假 CheckpointManager。

        参数：
            error：
                save_checkpoint 需要抛出的异常。

        返回值：
            None：构造函数无返回值。
        """

        self.error = error
        self.save_count = 0

    def save_checkpoint(self):
        """
        模拟保存 checkpoint。

        参数：
            无。

        返回值：
            None：无业务返回值。
        """

        self.save_count += 1

        if self.error:
            raise self.error


class FakeSemanticRecall:
    """
    测试用假 SemanticRecall。

    SemanticRecall（语义召回服务）：
    用于根据 user_id 和 question 召回相关长期记忆。
    """

    def __init__(
        self,
        result=None,
        error=None,
    ):
        """
        初始化假 SemanticRecall。

        参数：
            result：
                retrieve 成功时返回的结果。

            error：
                retrieve 需要抛出的异常。

        返回值：
            None：构造函数无返回值。
        """

        self.result = result
        self.error = error
        self.calls = []

    def retrieve(
        self,
        user_id,
        question,
        limit,
    ):
        """
        模拟同步记忆召回。

        参数：
            user_id：
                用户 ID。

            question：
                用户问题。

            limit：
                召回数量上限。

        返回值：
            object：
                模拟召回结果。
        """

        self.calls.append(
            {
                "user_id": user_id,
                "question": question,
                "limit": limit,
            }
        )

        if self.error:
            raise self.error

        return self.result


class FakeAsyncSemanticRecall:
    """
    测试用假异步 SemanticRecall。

    Async SemanticRecall（异步语义召回服务）：
    retrieve 返回 awaitable，用于验证 node 是否兼容异步返回。
    """

    def __init__(
        self,
        result=None,
    ):
        """
        初始化假异步 SemanticRecall。

        参数：
            result：
                retrieve 成功时返回的结果。

        返回值：
            None：构造函数无返回值。
        """

        self.result = result
        self.calls = []

    async def retrieve(
        self,
        user_id,
        question,
        limit,
    ):
        """
        模拟异步记忆召回。

        参数：
            user_id：
                用户 ID。

            question：
                用户问题。

            limit：
                召回数量上限。

        返回值：
            object：
                模拟召回结果。
        """

        self.calls.append(
            {
                "user_id": user_id,
                "question": question,
                "limit": limit,
            }
        )

        return self.result


def build_test_node(
    recall_result=None,
    recall_error=None,
    with_checkpoint=True,
    checkpoint_error=None,
    with_runtime_context=True,
    semantic_recall=None,
):
    """
    构建测试用 memory_retrieve_node。

    功能：
        创建 fake semantic_recall、fake checkpoint_manager、fake runtime_context。
        然后通过 build_memory_retrieve_node 注入依赖，得到真正可执行的 node。

    参数：
        recall_result：
            记忆召回成功时返回的结果。

        recall_error：
            记忆召回需要抛出的异常。

        with_checkpoint：
            是否传入 checkpoint_manager。

        checkpoint_error：
            checkpoint_manager.save_checkpoint 需要抛出的异常。

        with_runtime_context：
            是否提供 fake runtime context。

        semantic_recall：
            自定义 semantic_recall。
            如果传入，则优先使用它。

    返回值：
        tuple：
            node, fake_ctx, fake_semantic_recall, fake_checkpoint_manager。
    """

    fake_ctx = FakeRuntimeContext()

    fake_semantic_recall = (
        semantic_recall
        if semantic_recall is not None
        else FakeSemanticRecall(
            result=recall_result,
            error=recall_error,
        )
    )

    fake_checkpoint_manager = (
        FakeCheckpointManager(
            error=checkpoint_error,
        )
        if with_checkpoint
        else None
    )

    def runtime_context_getter():
        """
        获取测试用 RuntimeContext。

        参数：
            无。

        返回值：
            FakeRuntimeContext | None：
                根据 with_runtime_context 决定是否返回 fake_ctx。
        """

        if not with_runtime_context:
            return None

        return fake_ctx

    node = build_memory_retrieve_node(
        semantic_recall=fake_semantic_recall,
        checkpoint_manager=fake_checkpoint_manager,
        runtime_context_getter=runtime_context_getter,
    )

    return (
        node,
        fake_ctx,
        fake_semantic_recall,
        fake_checkpoint_manager,
    )


def test_format_memory_context_should_return_default_when_none():
    """
    测试召回结果为 None 时，是否返回默认空记忆文本。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    assert _format_memory_context(
        None,
    ) == "暂无用户记忆"


def test_format_memory_context_should_return_default_when_empty_string():
    """
    测试召回结果为空字符串时，是否返回默认空记忆文本。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    assert _format_memory_context(
        "",
    ) == "暂无用户记忆"


def test_format_memory_context_should_keep_string():
    """
    测试召回结果为字符串时，是否原样返回。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    assert _format_memory_context(
        "用户喜欢边牧"
    ) == "用户喜欢边牧"


def test_format_memory_context_should_convert_object_to_string():
    """
    测试召回结果为 dict 时，是否转换成字符串。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    result = _format_memory_context(
        {
            "preference": "用户喜欢边牧"
        }
    )

    assert result == "{'preference': '用户喜欢边牧'}"


@pytest.mark.asyncio
async def test_memory_retrieve_node_should_retrieve_memory_with_user_id():
    """
    测试有 user_id 时，是否使用 user_id 召回记忆。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        fake_ctx,
        fake_semantic_recall,
        fake_checkpoint_manager,
    ) = build_test_node(
        recall_result="用户喜欢边牧"
    )

    state = {
        "user_id": "user_001",
        "session_id": "session_001",
        "question": "我喜欢什么狗？",
    }

    result = await node(
        state,
    )

    assert result == {
        "memory_context": "用户喜欢边牧"
    }

    assert fake_semantic_recall.calls == [
        {
            "user_id": "user_001",
            "question": "我喜欢什么狗？",
            "limit": 5,
        }
    ]

    assert fake_ctx.state_scope.current_node == "memory_retrieve_node"

    assert fake_ctx.timeline_scope.events == [
        {
            "event_type": "node",
            "name": "memory_retrieve_node",
            "metadata": None,
        }
    ]

    assert fake_checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_memory_retrieve_node_should_use_session_id_when_user_id_missing():
    """
    测试缺少 user_id 时，是否使用 session_id。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_semantic_recall,
        _fake_checkpoint_manager,
    ) = build_test_node(
        recall_result="用户喜欢金毛"
    )

    state = {
        "session_id": "session_001",
        "question": "我喜欢什么狗？",
    }

    result = await node(
        state,
    )

    assert result == {
        "memory_context": "用户喜欢金毛"
    }

    assert fake_semantic_recall.calls[0]["user_id"] == "session_001"


@pytest.mark.asyncio
async def test_memory_retrieve_node_should_use_default_user_when_user_and_session_missing():
    """
    测试 user_id 和 session_id 都缺失时，是否使用 default_user。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_semantic_recall,
        _fake_checkpoint_manager,
    ) = build_test_node(
        recall_result="暂无用户记忆"
    )

    state = {
        "question": "我喜欢什么狗？",
    }

    await node(
        state,
    )

    assert fake_semantic_recall.calls[0]["user_id"] == "default_user"


@pytest.mark.asyncio
async def test_memory_retrieve_node_should_use_empty_question_when_question_missing():
    """
    测试 question 缺失时，是否使用空字符串。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_semantic_recall,
        _fake_checkpoint_manager,
    ) = build_test_node(
        recall_result="暂无用户记忆"
    )

    state = {
        "user_id": "user_001",
    }

    await node(
        state,
    )

    assert fake_semantic_recall.calls[0]["question"] == ""


@pytest.mark.asyncio
async def test_memory_retrieve_node_should_return_default_when_recall_none():
    """
    测试召回结果为 None 时，是否返回暂无用户记忆。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        _fake_semantic_recall,
        _fake_checkpoint_manager,
    ) = build_test_node(
        recall_result=None,
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "你好",
        }
    )

    assert result == {
        "memory_context": "暂无用户记忆"
    }


@pytest.mark.asyncio
async def test_memory_retrieve_node_should_convert_non_string_recall_result():
    """
    测试召回结果不是字符串时，是否转换成字符串。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        _fake_semantic_recall,
        _fake_checkpoint_manager,
    ) = build_test_node(
        recall_result=[
            "用户喜欢边牧",
            "用户不喜欢哈士奇",
        ],
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "我喜欢什么狗？",
        }
    )

    assert result == {
        "memory_context": "['用户喜欢边牧', '用户不喜欢哈士奇']"
    }


@pytest.mark.asyncio
async def test_memory_retrieve_node_should_support_async_retrieve():
    """
    测试 semantic_recall.retrieve 是异步方法时，节点是否支持 awaitable 返回。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    async_semantic_recall = FakeAsyncSemanticRecall(
        result="异步召回结果"
    )

    (
        node,
        _fake_ctx,
        fake_semantic_recall,
        _fake_checkpoint_manager,
    ) = build_test_node(
        semantic_recall=async_semantic_recall,
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "测试异步召回",
        }
    )

    assert result == {
        "memory_context": "异步召回结果"
    }

    assert fake_semantic_recall.calls == [
        {
            "user_id": "user_001",
            "question": "测试异步召回",
            "limit": 5,
        }
    ]


@pytest.mark.asyncio
async def test_memory_retrieve_node_should_fallback_when_recall_failed():
    """
    测试 semantic_recall.retrieve 抛异常时，是否降级为空记忆。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        fake_semantic_recall,
        fake_checkpoint_manager,
    ) = build_test_node(
        recall_error=RuntimeError(
            "recall failed"
        )
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "你好",
        }
    )

    assert result == {
        "memory_context": "暂无用户记忆"
    }

    assert len(
        fake_semantic_recall.calls
    ) == 1

    assert fake_checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_memory_retrieve_node_should_not_break_when_checkpoint_failed():
    """
    测试 checkpoint 保存失败时，节点是否仍返回 memory_context。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        _fake_semantic_recall,
        fake_checkpoint_manager,
    ) = build_test_node(
        recall_result="用户喜欢边牧",
        checkpoint_error=RuntimeError(
            "checkpoint failed"
        ),
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "我喜欢什么狗？",
        }
    )

    assert result == {
        "memory_context": "用户喜欢边牧"
    }

    assert fake_checkpoint_manager.save_count == 1


@pytest.mark.asyncio
async def test_memory_retrieve_node_should_work_without_checkpoint_manager():
    """
    测试 checkpoint_manager 为 None 时，节点是否仍可正常工作。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        _fake_ctx,
        _fake_semantic_recall,
        fake_checkpoint_manager,
    ) = build_test_node(
        recall_result="没有 checkpoint 也能召回",
        with_checkpoint=False,
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "你好",
        }
    )

    assert result == {
        "memory_context": "没有 checkpoint 也能召回"
    }

    assert fake_checkpoint_manager is None


@pytest.mark.asyncio
async def test_memory_retrieve_node_should_work_without_runtime_context():
    """
    测试 runtime_context_getter 返回 None 时，节点是否仍可正常工作。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    (
        node,
        fake_ctx,
        _fake_semantic_recall,
        fake_checkpoint_manager,
    ) = build_test_node(
        recall_result="没有 runtime context 也能召回",
        with_runtime_context=False,
    )

    result = await node(
        {
            "user_id": "user_001",
            "question": "你好",
        }
    )

    assert result == {
        "memory_context": "没有 runtime context 也能召回"
    }

    assert fake_ctx.state_scope.current_node is None
    assert fake_ctx.timeline_scope.events == []
    assert fake_checkpoint_manager.save_count == 1