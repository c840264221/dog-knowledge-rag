"""
rebuild_dog_rag_index.py 单元测试。

本测试只验证脚本编排逻辑：
1. 不启动真实 container
2. 不连接真实 Chroma
3. 不调用真实 Embedding
4. 不读取真实 data/dog_markdown
5. 使用 FakeContainer / FakePipeline 替代真实依赖
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from scripts.rag import rebuild_dog_rag_index as rebuild_script


class FakeVectorStore:
    """
    Fake Vector Store（假的向量库）。

    功能：
        用于模拟 Chroma vector store。
    """

    pass


class FakeVectorStoreProvider:
    """
    Fake Vector Store Provider（假的向量库 Provider）。

    功能：
        模拟项目中的 VectorStoreProvider。
        默认提供 db 属性。
    """

    def __init__(
            self,
            db: Any,
    ):
        """
        初始化 FakeVectorStoreProvider。

        参数：
            db: Any
                假的 vector store 对象。

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
                get 方法要返回的 Provider。

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
                预设的 Provider。
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


class FakePipeline:
    """
    Fake Pipeline（假的入库流水线）。

    功能：
        模拟当前版本 RagIndexPipeline。
        当前 Pipeline 对外执行入口是 index()，不是 index_dir()。
    """

    def __init__(self):
        """
        初始化 FakePipeline。

        参数：
            无。

        返回值：
            None：
                构造函数无返回值。
        """

        self.called = False

    def index(self) -> dict[str, Any]:
        """
        模拟 RagIndexPipeline.index。

        功能：
            用于确认 rebuild_script.run_pipeline 是否调用 pipeline.index()。

        参数：
            无。

        返回值：
            dict[str, Any]：
                模拟 Pipeline 入库结果。
        """

        self.called = True

        return {
            "pipeline": "fake_pipeline",
            "source": "fake_source",
            "loaded_documents": 1,
            "enhanced_documents": 1,
            "created_chunks": 1,
            "index_result": {
                "indexed_chunks": 1,
                "skipped_chunks": 0,
                "batch_count": 1,
                "index_ids": [
                    "chunk-001",
                ],
            },
        }


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    """
    构建 FakeVectorStore。

    参数：
        无。

    返回值：
        FakeVectorStore：
            假向量库对象。
    """

    return FakeVectorStore()


@pytest.fixture
def fake_vector_store_provider(
        fake_vector_store,
) -> FakeVectorStoreProvider:
    """
    构建 FakeVectorStoreProvider。

    参数：
        fake_vector_store:
            pytest fixture 注入的假向量库。

    返回值：
        FakeVectorStoreProvider：
            假向量库 Provider。
    """

    return FakeVectorStoreProvider(
        db=fake_vector_store,
    )


@pytest.fixture
def fake_container(
        fake_vector_store_provider,
) -> FakeContainer:
    """
    构建 FakeContainer。

    参数：
        fake_vector_store_provider:
            pytest fixture 注入的假 Provider。

    返回值：
        FakeContainer：
            假运行时容器。
    """

    return FakeContainer(
        provider=fake_vector_store_provider,
    )


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    """
    构建 FakePipeline。

    参数：
        无。

    返回值：
        FakePipeline：
            假入库流水线。
    """

    return FakePipeline()


@pytest.fixture
def args(
        tmp_path,
) -> argparse.Namespace:
    """
    构建测试用命令行参数对象。

    参数：
        tmp_path:
            pytest 提供的临时目录 fixture。
            run_rebuild_from_args 内部会检查 data_dir 是否存在，
            所以这里使用真实存在的 tmp_path。

    返回值：
        argparse.Namespace：
            模拟命令行参数。
    """

    return argparse.Namespace(
        data_dir=str(
            tmp_path,
        ),
        vector_store_provider="vector_store_provider",
        vector_store_attr="db",
        batch_size=64,
        overwrite_existing=True,
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

    parsed_args = rebuild_script.parse_args(
        argv=[],
    )

    assert parsed_args.data_dir == "data/dog_markdown"
    assert parsed_args.vector_store_provider == "vector_store_provider"
    assert parsed_args.vector_store_attr == "db"
    assert parsed_args.batch_size == 64
    assert parsed_args.overwrite_existing is True


def test_parse_args_should_accept_custom_values():
    """
    测试命令行参数自定义值。

    参数：
        无。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    parsed_args = rebuild_script.parse_args(
        argv=[
            "--data-dir",
            "custom_dir",
            "--vector-store-provider",
            "custom_provider",
            "--vector-store-attr",
            "vector_store",
            "--batch-size",
            "10",
            "--no-overwrite-existing",
        ],
    )

    assert parsed_args.data_dir == "custom_dir"
    assert parsed_args.vector_store_provider == "custom_provider"
    assert parsed_args.vector_store_attr == "vector_store"
    assert parsed_args.batch_size == 10
    assert parsed_args.overwrite_existing is False


def test_get_vector_store_from_provider_should_read_db_attr(
        fake_vector_store,
        fake_vector_store_provider,
):
    """
    测试是否能从 provider.db 中获取 vector store。

    参数：
        fake_vector_store:
            pytest fixture 注入的假向量库。

        fake_vector_store_provider:
            pytest fixture 注入的假 Provider。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = rebuild_script._get_vector_store_from_provider(
        vector_store_provider=fake_vector_store_provider,
        vector_store_attr="db",
    )

    assert result is fake_vector_store


def test_get_vector_store_from_provider_should_return_provider_when_attr_is_self(
        fake_vector_store_provider,
):
    """
    测试 vector_store_attr=self 时是否直接返回 provider。

    参数：
        fake_vector_store_provider:
            pytest fixture 注入的假 Provider。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = rebuild_script._get_vector_store_from_provider(
        vector_store_provider=fake_vector_store_provider,
        vector_store_attr="self",
    )

    assert result is fake_vector_store_provider


def test_get_vector_store_from_provider_should_raise_error_when_attr_missing(
        fake_vector_store_provider,
):
    """
    测试 provider 缺少指定属性时是否抛出异常。

    参数：
        fake_vector_store_provider:
            pytest fixture 注入的假 Provider。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    with pytest.raises(
            AttributeError,
    ):
        rebuild_script._get_vector_store_from_provider(
            vector_store_provider=fake_vector_store_provider,
            vector_store_attr="missing_attr",
        )


def test_run_pipeline_should_call_index(
        fake_pipeline,
):
    """
    测试 run_pipeline 是否调用 pipeline.index。

    参数：
        fake_pipeline:
            pytest fixture 注入的假 Pipeline。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = rebuild_script.run_pipeline(
        pipeline=fake_pipeline,
    )

    assert fake_pipeline.called is True

    assert result[
        "pipeline"
    ] == "fake_pipeline"

    assert result[
        "index_result"
    ][
        "indexed_chunks"
    ] == 1


@pytest.mark.asyncio
async def test_maybe_await_should_support_normal_value():
    """
    测试 _maybe_await 是否支持普通值。

    参数：
        无。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    result = await rebuild_script._maybe_await(
        value="normal",
    )

    assert result == "normal"


@pytest.mark.asyncio
async def test_maybe_await_should_support_coroutine():
    """
    测试 _maybe_await 是否支持 coroutine。

    参数：
        无。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    async def async_value():
        """
        构建测试 coroutine。

        参数：
            无。

        返回值：
            str：
                测试字符串。
        """

        return "async-result"

    result = await rebuild_script._maybe_await(
        value=async_value(),
    )

    assert result == "async-result"


@pytest.mark.asyncio
async def test_run_rebuild_from_args_should_start_container_and_shutdown(
        monkeypatch,
        args,
        fake_container,
        fake_pipeline,
        fake_vector_store,
):
    """
    测试 run_rebuild_from_args 是否启动容器、执行 Pipeline、关闭容器。

    参数：
        monkeypatch:
            pytest 提供的 monkeypatch fixture。

        args:
            pytest fixture 注入的命令行参数。

        fake_container:
            pytest fixture 注入的假容器。

        fake_pipeline:
            pytest fixture 注入的假 Pipeline。

        fake_vector_store:
            pytest fixture 注入的假向量库。

    返回值：
        None：
            pytest 根据 assert 判断测试是否通过。
    """

    captured_build_args: dict[str, Any] = {}

    def fake_build_pipeline(
            input_path: str | Path,
            vector_store: Any,
            batch_size: int,
            overwrite_existing: bool,
    ) -> FakePipeline:
        """
        替换真实 build_pipeline。

        参数：
            input_path: str | Path
                Markdown 输入路径。

            vector_store: Any
                假向量库。

            batch_size: int
                批大小。

            overwrite_existing: bool
                是否覆盖已有数据。

        返回值：
            FakePipeline：
                假 Pipeline。
        """

        captured_build_args[
            "input_path"
        ] = input_path

        captured_build_args[
            "vector_store"
        ] = vector_store

        captured_build_args[
            "batch_size"
        ] = batch_size

        captured_build_args[
            "overwrite_existing"
        ] = overwrite_existing

        return fake_pipeline

    monkeypatch.setattr(
        rebuild_script,
        "build_pipeline",
        fake_build_pipeline,
    )

    result = await rebuild_script.run_rebuild_from_args(
        args=args,
        runtime_container=fake_container,
    )

    assert fake_container.started is True
    assert fake_container.shutdown_called is True
    assert fake_container.received_service_name == "vector_store_provider"

    assert captured_build_args[
        "input_path"
    ] == Path(
        args.data_dir,
    )

    assert captured_build_args[
        "vector_store"
    ] is fake_vector_store

    assert captured_build_args[
        "batch_size"
    ] == 64

    assert captured_build_args[
        "overwrite_existing"
    ] is True

    assert fake_pipeline.called is True

    assert result[
        "index_result"
    ][
        "indexed_chunks"
    ] == 1