"""
inspect_dog_rag_index.py 单元测试。

本测试只验证检查脚本自身逻辑：
1. 不启动真实 container
2. 不连接真实 Chroma
3. 不调用真实 Embedding
4. 使用 FakeVectorStore / FakeContainer 替代真实依赖
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from scripts.rag import inspect_dog_rag_index as inspect_script


class FakeDocument:
    """
    Fake Document（假的 LangChain Document）。

    功能：
        模拟 similarity_search 返回的 Document 对象。
    """

    def __init__(
            self,
            page_content: str,
            metadata: dict[str, Any],
    ):
        """
        初始化 FakeDocument。

        参数：
            page_content: str
                文档内容。

            metadata: dict[str, Any]
                文档 metadata。

        返回值：
            None：
                构造函数无返回值。
        """

        self.page_content = page_content
        self.metadata = metadata


class FakeCollection:
    """
    Fake Collection（假的 Chroma Collection）。

    功能：
        模拟 Chroma collection 的 count 和 get 方法。
    """

    def __init__(
            self,
            records: list[dict[str, Any]],
    ):
        """
        初始化 FakeCollection。

        参数：
            records: list[dict[str, Any]]
                假数据列表。

        返回值：
            None：
                构造函数无返回值。
        """

        self.records = records

    def count(self) -> int:
        """
        返回假 collection 中的数据量。

        参数：
            无。

        返回值：
            int：
                数据条数。
        """

        return len(
            self.records,
        )

    def get(
            self,
            limit: int = 5,
            include: list[str] | None = None,
            where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        模拟 Chroma collection.get。

        参数：
            limit: int
                最多返回多少条。

            include: list[str] | None
                兼容 Chroma 参数，本测试中不实际使用。

            where: dict[str, Any] | None
                metadata filter。

        返回值：
            dict[str, Any]：
                模拟 Chroma get 返回结构。
        """

        matched_records = [
            record
            for record in self.records
            if match_where(
                metadata=record[
                    "metadata"
                ],
                where=where,
            )
        ]

        limited_records = matched_records[
            :limit
        ]

        return {
            "ids": [
                record[
                    "id"
                ]
                for record in limited_records
            ],
            "documents": [
                record[
                    "document"
                ]
                for record in limited_records
            ],
            "metadatas": [
                record[
                    "metadata"
                ]
                for record in limited_records
            ],
        }


class FakeVectorStore:
    """
    Fake Vector Store（假的向量库）。

    功能：
        模拟 LangChain Chroma。
        提供 _collection、get、similarity_search。
    """

    def __init__(
            self,
            records: list[dict[str, Any]],
    ):
        """
        初始化 FakeVectorStore。

        参数：
            records: list[dict[str, Any]]
                假数据列表。

        返回值：
            None：
                构造函数无返回值。
        """

        self.records = records
        self._collection = FakeCollection(
            records=records,
        )

    def get(
            self,
            limit: int = 5,
            include: list[str] | None = None,
            where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        模拟 LangChain Chroma.get。

        参数：
            limit: int
                最多返回多少条。

            include: list[str] | None
                兼容 Chroma 参数。

            where: dict[str, Any] | None
                metadata filter。

        返回值：
            dict[str, Any]：
                模拟 Chroma get 返回结构。
        """

        return self._collection.get(
            limit=limit,
            include=include,
            where=where,
        )

    def similarity_search(
            self,
            query: str,
            k: int = 5,
            filter: dict[str, Any] | None = None,
    ) -> list[FakeDocument]:
        """
        模拟 LangChain Chroma.similarity_search。

        参数：
            query: str
                查询文本。

            k: int
                返回数量。

            filter: dict[str, Any] | None
                metadata filter。

        返回值：
            list[FakeDocument]：
                模拟检索结果。
        """

        matched_records = [
            record
            for record in self.records
            if match_where(
                metadata=record[
                    "metadata"
                ],
                where=filter,
            )
        ]

        return [
            FakeDocument(
                page_content=record[
                    "document"
                ],
                metadata=record[
                    "metadata"
                ],
            )
            for record in matched_records[
                :k
            ]
        ]


class FakeVectorStoreProvider:
    """
    Fake Vector Store Provider（假的向量库 Provider）。

    功能：
        模拟 VectorStoreProvider，默认提供 db 属性。
    """

    def __init__(
            self,
            db: Any,
    ):
        """
        初始化 FakeVectorStoreProvider。

        参数：
            db: Any
                假 vector store。

        返回值：
            None：
                构造函数无返回值。
        """

        self.db = db


class FakeContainer:
    """
    Fake Container（假的运行时容器）。

    功能：
        模拟 RuntimeContainer 的 startup、get、shutdown 方法。
    """

    def __init__(
            self,
            provider: Any,
    ):
        """
        初始化 FakeContainer。

        参数：
            provider: Any
                get 方法返回的 Provider。

        返回值：
            None：
                构造函数无返回值。
        """

        self.provider = provider
        self.started = False
        self.shutdown_called = False
        self.received_service_name = None

    async def startup(self) -> None:
        """
        模拟异步启动容器。

        参数：
            无。

        返回值：
            None：
                启动方法无返回值。
        """

        self.started = True

    def get(
            self,
            service_name: str,
    ) -> Any:
        """
        模拟从容器中获取服务。

        参数：
            service_name: str
                服务注册名。

        返回值：
            Any：
                预设 Provider。
        """

        self.received_service_name = service_name

        return self.provider

    async def shutdown(self) -> None:
        """
        模拟异步关闭容器。

        参数：
            无。

        返回值：
            None：
                关闭方法无返回值。
        """

        self.shutdown_called = True


def match_where(
        metadata: dict[str, Any],
        where: dict[str, Any] | None,
) -> bool:
    """
    匹配简化版 Chroma where filter。

    功能：
        支持测试中用到的：
        1. $and
        2. $eq
        3. $lte

    参数：
        metadata: dict[str, Any]
            文档 metadata。

        where: dict[str, Any] | None
            metadata filter。

    返回值：
        bool：
            True 表示匹配；
            False 表示不匹配。
    """

    if not where:
        return True

    if "$and" in where:
        return all(
            match_where(
                metadata=metadata,
                where=condition,
            )
            for condition in where[
                "$and"
            ]
        )

    for key, condition in where.items():

        actual_value = metadata.get(
            key,
        )

        if isinstance(
                condition,
                dict,
        ):

            if "$eq" in condition and actual_value != condition[
                "$eq"
            ]:
                return False

            if "$lte" in condition and actual_value > condition[
                "$lte"
            ]:
                return False

            continue

        if actual_value != condition:
            return False

    return True


@pytest.fixture
def fake_records() -> list[dict[str, Any]]:
    """
    构建测试用假数据。

    参数：
        无。

    返回值：
        list[dict[str, Any]]：
            模拟 Chroma 中的记录。
    """

    return [
        {
            "id": "chunk-affenpinscher-001",
            "document": "Affenpinscher is a small confident dog.",
            "metadata": {
                "dog_name": "Affenpinscher",
                "size": "small",
                "energy_level": 3,
                "barking_level": 3,
                "good_for_apartment": True,
            },
        },
        {
            "id": "chunk-beagle-001",
            "document": "Beagle is friendly but can be vocal.",
            "metadata": {
                "dog_name": "Beagle",
                "size": "small",
                "energy_level": 4,
                "barking_level": 5,
                "good_for_apartment": False,
            },
        },
    ]


@pytest.fixture
def fake_vector_store(
        fake_records,
) -> FakeVectorStore:
    """
    构建 FakeVectorStore。

    参数：
        fake_records:
            pytest fixture 注入的假记录。

    返回值：
        FakeVectorStore：
            假向量库。
    """

    return FakeVectorStore(
        records=fake_records,
    )


@pytest.fixture
def fake_provider(
        fake_vector_store,
) -> FakeVectorStoreProvider:
    """
    构建 FakeVectorStoreProvider。

    参数：
        fake_vector_store:
            pytest fixture 注入的假向量库。

    返回值：
        FakeVectorStoreProvider：
            假 Provider。
    """

    return FakeVectorStoreProvider(
        db=fake_vector_store,
    )


@pytest.fixture
def fake_container(
        fake_provider,
) -> FakeContainer:
    """
    构建 FakeContainer。

    参数：
        fake_provider:
            pytest fixture 注入的假 Provider。

    返回值：
        FakeContainer：
            假容器。
    """

    return FakeContainer(
        provider=fake_provider,
    )


@pytest.fixture
def args() -> argparse.Namespace:
    """
    构建测试用命令行参数。

    参数：
        无。

    返回值：
        argparse.Namespace：
            模拟命令行参数对象。
    """

    return argparse.Namespace(
        vector_store_provider="vectorstore",
        vector_store_attr="db",
        sample_limit=5,
        dog_name="Affenpinscher",
        size="small",
        max_energy=3,
        max_barking=3,
        query="small apartment dog",
        top_k=5,
    )


def test_parse_args_should_use_default_values():
    """
    测试命令行参数默认值。

    参数：
        无。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    parsed_args = inspect_script.parse_args(
        argv=[],
    )

    assert parsed_args.vector_store_provider == "vectorstore"
    assert parsed_args.vector_store_attr == "db"
    assert parsed_args.sample_limit == 5
    assert parsed_args.dog_name == "Golden Retriever"
    assert parsed_args.size == "small"
    assert parsed_args.max_energy == 3
    assert parsed_args.max_barking == 3
    assert parsed_args.top_k == 5


def test_get_vector_store_from_provider_should_read_db_attr(
        fake_provider,
        fake_vector_store,
):
    """
    测试是否能从 provider.db 获取 vector store。

    参数：
        fake_provider:
            pytest fixture 注入的假 Provider。

        fake_vector_store:
            pytest fixture 注入的假向量库。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = inspect_script.get_vector_store_from_provider(
        vector_store_provider=fake_provider,
        vector_store_attr="db",
    )

    assert result is fake_vector_store


def test_build_dog_name_filter_should_create_eq_filter():
    """
    测试 dog_name filter 构造。

    参数：
        无。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = inspect_script.build_dog_name_filter(
        dog_name="Affenpinscher",
    )

    assert result == {
        "dog_name": {
            "$eq": "Affenpinscher",
        }
    }


def test_build_metadata_filter_should_create_and_filter():
    """
    测试组合 metadata filter 构造。

    参数：
        无。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = inspect_script.build_metadata_filter(
        size="small",
        max_energy=3,
        max_barking=3,
    )

    assert result == {
        "$and": [
            {
                "size": {
                    "$eq": "small",
                }
            },
            {
                "energy_level": {
                    "$lte": 3,
                }
            },
            {
                "barking_level": {
                    "$lte": 3,
                }
            },
        ]
    }


def test_inspect_vector_store_should_return_check_result(
        fake_vector_store,
):
    """
    测试 inspect_vector_store 是否返回完整检查结果。

    参数：
        fake_vector_store:
            pytest fixture 注入的假向量库。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = inspect_script.inspect_vector_store(
        vector_store=fake_vector_store,
        sample_limit=5,
        dog_name="Affenpinscher",
        size="small",
        max_energy=3,
        max_barking=3,
        query="small apartment dog",
        top_k=5,
    )

    assert result[
        "collection_count"
    ] == 2

    assert len(
        result[
            "sample_check"
        ][
            "records"
        ]
    ) == 2

    assert result[
        "dog_name_filter_check"
    ][
        "match_count"
    ] == 1

    assert result[
        "metadata_filter_check"
    ][
        "match_count"
    ] == 1

    assert result[
        "vector_search_check"
    ][
        "match_count"
    ] == 1

    assert result[
        "vector_search_check"
    ][
        "records"
    ][
        0
    ][
        "metadata"
    ][
        "dog_name"
    ] == "Affenpinscher"


@pytest.mark.asyncio
async def test_run_inspection_from_args_should_start_and_shutdown_container(
        args,
        fake_container,
):
    """
    测试 run_inspection_from_args 是否启动并关闭 container。

    参数：
        args:
            pytest fixture 注入的命令行参数。

        fake_container:
            pytest fixture 注入的假容器。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = await inspect_script.run_inspection_from_args(
        args=args,
        runtime_container=fake_container,
    )

    assert fake_container.started is True
    assert fake_container.shutdown_called is True
    assert fake_container.received_service_name == "vectorstore"

    assert result[
        "collection_count"
    ] == 2

    assert result[
        "dog_name_filter_check"
    ][
        "match_count"
    ] == 1