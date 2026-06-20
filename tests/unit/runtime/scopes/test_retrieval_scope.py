"""
retrieval scope 单元测试。

RetrievalScope（检索作用域）：
用于管理一次请求执行过程中的 retrieval docs（检索文档）。

retrieval（检索）：
在 RAG（Retrieval-Augmented Generation，检索增强生成）中，
根据用户问题从知识库、向量数据库或搜索系统中找出相关文档的过程。

docs（文档列表）：
检索阶段返回的文档集合，后续会被用于 answer generation（答案生成）。

RequestScope（请求作用域）：
底层 key-value store（键值存储），用于保存一次请求内的临时数据。

KEY（键）：
RetrievalScope 使用固定 key "retrieval" 把检索文档存入 RequestScope。
"""

import pytest

from src.runtime.context.request_scope import RequestScope

from src.runtime.scopes.retrieval_scope import RetrievalScope


def test_retrieval_scope_can_be_created():
    """
    测试 RetrievalScope 是否可以正常创建。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    retrieval_scope = RetrievalScope(
        request_scope,
    )

    assert retrieval_scope is not None
    assert retrieval_scope.scope is request_scope


def test_retrieval_scope_key_should_be_retrieval():
    """
    测试 RetrievalScope.KEY 是否为 retrieval。

    KEY（键）：
    用于在 RequestScope 中保存检索结果的固定名称。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    assert RetrievalScope.KEY == "retrieval"


def test_retrieval_scope_get_docs_should_return_empty_list_by_default():
    """
    测试未设置 docs 时，get_docs 是否默认返回空列表。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    retrieval_scope = RetrievalScope(
        request_scope,
    )

    assert retrieval_scope.get_docs() == []


def test_retrieval_scope_set_and_get_docs():
    """
    测试 set_docs 和 get_docs 是否可以正确写入和读取检索文档。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    retrieval_scope = RetrievalScope(
        request_scope,
    )

    docs = [
        {
            "page_content": "金毛寻回犬性格温顺，适合家庭饲养。",
            "metadata": {
                "breed": "Golden Retriever",
            },
        },
        {
            "page_content": "边境牧羊犬智商高，但运动需求较大。",
            "metadata": {
                "breed": "Border Collie",
            },
        },
    ]

    retrieval_scope.set_docs(
        docs,
    )

    assert retrieval_scope.get_docs() == docs


def test_retrieval_scope_set_docs_should_store_data_in_request_scope():
    """
    测试 set_docs 是否真的把数据保存到底层 RequestScope。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    retrieval_scope = RetrievalScope(
        request_scope,
    )

    docs = [
        {
            "page_content": "拉布拉多适合新手。",
            "metadata": {
                "breed": "Labrador Retriever",
            },
        }
    ]

    retrieval_scope.set_docs(
        docs,
    )

    assert request_scope.get(
        RetrievalScope.KEY,
    ) == docs


def test_retrieval_scope_get_docs_should_read_from_request_scope():
    """
    测试 get_docs 是否从底层 RequestScope 读取数据。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    retrieval_scope = RetrievalScope(
        request_scope,
    )

    docs = [
        {
            "page_content": "柯基腿短，性格活泼。",
            "metadata": {
                "breed": "Corgi",
            },
        }
    ]

    request_scope.set(
        RetrievalScope.KEY,
        docs,
    )

    assert retrieval_scope.get_docs() == docs


def test_retrieval_scope_set_docs_should_override_old_docs():
    """
    测试重复 set_docs 时是否会覆盖旧检索结果。

    override（覆盖）：
    新检索结果替换旧检索结果。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    retrieval_scope = RetrievalScope(
        request_scope,
    )

    old_docs = [
        {
            "page_content": "旧检索结果",
        }
    ]

    new_docs = [
        {
            "page_content": "新检索结果",
        }
    ]

    retrieval_scope.set_docs(
        old_docs,
    )

    retrieval_scope.set_docs(
        new_docs,
    )

    assert retrieval_scope.get_docs() == new_docs


def test_retrieval_scope_clear_should_remove_docs():
    """
    测试 clear 是否可以清空检索文档。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    retrieval_scope = RetrievalScope(
        request_scope,
    )

    docs = [
        {
            "page_content": "用户问题相关文档",
        }
    ]

    retrieval_scope.set_docs(
        docs,
    )

    assert retrieval_scope.get_docs() == docs

    retrieval_scope.clear()

    assert retrieval_scope.get_docs() == []
    assert request_scope.get(RetrievalScope.KEY) is None


def test_retrieval_scope_clear_without_docs_should_not_raise_error():
    """
    测试没有 docs 时调用 clear 是否不会报错。

    当前 clear 内部调用 RequestScope.remove，
    RequestScope.remove 使用 pop(key, None)，所以 key 不存在时不会报错。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    retrieval_scope = RetrievalScope(
        request_scope,
    )

    retrieval_scope.clear()

    assert retrieval_scope.get_docs() == []


def test_retrieval_scope_should_share_data_when_using_same_request_scope():
    """
    测试多个 RetrievalScope 使用同一个 RequestScope 时是否共享 docs。

    shared scope（共享作用域）：
    多个 scope wrapper（作用域包装器）底层使用同一个 RequestScope，
    因此可以读写同一份请求级数据。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()

    first_retrieval_scope = RetrievalScope(
        request_scope,
    )

    second_retrieval_scope = RetrievalScope(
        request_scope,
    )

    docs = [
        {
            "page_content": "共享检索结果",
        }
    ]

    first_retrieval_scope.set_docs(
        docs,
    )

    assert second_retrieval_scope.get_docs() == docs


def test_retrieval_scope_should_be_isolated_when_using_different_request_scopes():
    """
    测试不同 RequestScope 下的 RetrievalScope 是否互不污染。

    isolated（隔离）：
    一个请求作用域中的检索结果，不应该影响另一个请求作用域。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    first_request_scope = RequestScope()
    second_request_scope = RequestScope()

    first_retrieval_scope = RetrievalScope(
        first_request_scope,
    )

    second_retrieval_scope = RetrievalScope(
        second_request_scope,
    )

    first_retrieval_scope.set_docs(
        [
            {
                "page_content": "第一个请求的检索结果",
            }
        ]
    )

    assert first_retrieval_scope.get_docs() == [
        {
            "page_content": "第一个请求的检索结果",
        }
    ]

    assert second_retrieval_scope.get_docs() == []


@pytest.mark.asyncio
async def test_retrieval_scope_startup_should_not_change_docs():
    """
    测试 startup 是否不会改变已有 docs。

    当前 RetrievalScope.startup 是 pass，
    所以 startup 前后 docs 应保持一致。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    retrieval_scope = RetrievalScope(
        request_scope,
    )

    docs = [
        {
            "page_content": "startup 前已有检索结果",
        }
    ]

    retrieval_scope.set_docs(
        docs,
    )

    await retrieval_scope.startup()

    assert retrieval_scope.get_docs() == docs


@pytest.mark.asyncio
async def test_retrieval_scope_shutdown_should_clear_docs():
    """
    测试 shutdown 是否会清空 docs。

    当前 RetrievalScope.shutdown 内部调用 self.clear()，
    所以 shutdown 后 get_docs 应返回空列表。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    request_scope = RequestScope()
    retrieval_scope = RetrievalScope(
        request_scope,
    )

    docs = [
        {
            "page_content": "shutdown 前已有检索结果",
        }
    ]

    retrieval_scope.set_docs(
        docs,
    )

    await retrieval_scope.shutdown()

    assert retrieval_scope.get_docs() == []
    assert request_scope.get(RetrievalScope.KEY) is None