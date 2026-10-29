"""PDF项目接入测试：只用生成的空白PDF和替身对象，不下载模型、不连接向量库。"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pypdf import PdfWriter
from pypdf._page import PageObject

from src.rag.chunking.pdf_text_chunker import PdfTextChunker
from src.rag.indexers.chroma_indexer import RagChromaIndexer
from src.rag.loaders.pdf_loader import PdfDocumentLoader
from src.rag.pipelines.index_pipeline import RagIndexPipeline
from src.rag.retrievers.metadata_filter_retriever import MetadataFilterRetriever
from src.rag.schemas import RagRetrievedChunk
from src.runtime.container.providers import vectorstore_provider as vector_module
from src.runtime.container.providers.embedding_provider import EmbeddingProvider


def make_loader(tmp_path: Path, monkeypatch) -> PdfDocumentLoader:
    """生成三页测试PDF并替换文字提取；参数为临时目录和补丁夹具；返回选第2、3页的加载器。"""
    path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    for width in (100, 101, 102):
        writer.add_blank_page(width=width, height=200)
    with path.open("wb") as stream:
        writer.write(stream)
    monkeypatch.setattr(PageObject, "extract_text", lambda page: f"text {int(page.mediabox.width)}")
    return PdfDocumentLoader(
        path, doc_id="cat_sample", title="Test cat guide", included_pages=[2, 3],
        metadata={"species": "cat", "review_status": "experiment_selected_not_pagewise_reviewed"},
    )


def test_loader_retains_raw_pages_and_selects_body(tmp_path, monkeypatch):
    """验证选页正文与全部原始页分开保存；参数为测试夹具；断言失败时抛错，无返回值。"""
    document = make_loader(tmp_path, monkeypatch).load()[0]
    assert document.content == "text 101\n\ntext 102"
    assert len(document.metadata["pdf_pages"]) == 3
    assert document.metadata["selected_page_numbers"] == [2, 3]
    assert document.metadata["species"] == "cat"
    assert "dog_name" not in document.metadata


@pytest.mark.parametrize("pages", [[], [0], [True], [1, 1]])
def test_invalid_page_selection(pages):
    """验证非法选页不会静默扩为全选；pages为非法配置；无返回值。"""
    with pytest.raises(ValueError):
        PdfDocumentLoader("unused.pdf", doc_id="sample", title="sample", included_pages=pages)


def test_page_out_of_range(tmp_path, monkeypatch):
    """验证页码越界报错；参数为测试夹具；无返回值。"""
    loader = make_loader(tmp_path, monkeypatch)
    loader.included_pages = [4]
    with pytest.raises(ValueError):
        loader.load()


def test_chunk_page_metadata_and_stable_ids(tmp_path, monkeypatch):
    """验证块编号稳定、继承物种及真实页码，不携带整份原始页；参数为测试夹具；无返回值。"""
    document = make_loader(tmp_path, monkeypatch).load()[0]
    chunker = PdfTextChunker()
    chunks = chunker.chunk(document)
    assert [chunk.metadata["page_number"] for chunk in chunks] == [2, 3]
    assert len({chunk.chunk_id for chunk in chunks}) == 2
    assert [chunk.chunk_id for chunk in chunks] == [chunk.chunk_id for chunk in chunker.chunk(document)]
    assert all(chunk.doc_id == "cat_sample" and chunk.metadata["species"] == "cat" for chunk in chunks)
    assert all("pdf_pages" not in chunk.metadata for chunk in chunks)


def test_pipeline_without_breed_extractor(tmp_path, monkeypatch):
    """验证正式管线允许跳过犬种提取且索引器保留页码；参数为测试夹具；无返回值。"""
    store = Mock()
    pipeline = RagIndexPipeline(
        loader=make_loader(tmp_path, monkeypatch), metadata_extractor=None,
        chunker=PdfTextChunker(), indexer=RagChromaIndexer(store, overwrite_existing=False),
    )
    result = pipeline.index()
    assert result["loaded_documents"] == 1
    assert result["created_chunks"] == 2
    written = store.add_documents.call_args.kwargs["documents"]
    assert [doc.metadata["page_number"] for doc in written] == [2, 3]
    assert all("dog_name" not in doc.metadata for doc in written)
    store.delete.assert_not_called()


def test_context_includes_physical_page(tmp_path, monkeypatch):
    """验证现有检索器上下文展示PDF页码；参数为测试夹具；无返回值。"""
    chunk = PdfTextChunker().chunk(make_loader(tmp_path, monkeypatch).load()[0])[0]
    context = MetadataFilterRetriever(Mock())._build_context_text([RagRetrievedChunk(chunk=chunk)])
    assert "page_number: 2" in context
    assert "title: Test cat guide" in context


def test_provider_default_collection_unchanged(monkeypatch):
    """验证不指定集合时不新增集合参数；monkeypatch为补丁夹具；无返回值。"""
    constructor = Mock()
    monkeypatch.setattr(vector_module, "Chroma", constructor)
    provider = vector_module.VectorStoreProvider(SimpleNamespace(embedding=object()))
    _ = provider.db
    assert "collection_name" not in constructor.call_args.kwargs
    assert provider.enable_memory is True


def test_provider_explicit_collection_without_memory(monkeypatch, tmp_path):
    """验证集合和目录被传给Chroma且关闭记忆预热；参数为测试夹具；无返回值。"""
    constructor = Mock()
    monkeypatch.setattr(vector_module, "Chroma", constructor)
    provider = vector_module.VectorStoreProvider(
        SimpleNamespace(embedding=object()), persist_directory=tmp_path,
        collection_name="pet_care_v1", enable_memory=False,
    )
    asyncio.run(provider.startup())
    assert constructor.call_count == 1
    assert constructor.call_args.kwargs["collection_name"] == "pet_care_v1"
    assert constructor.call_args.kwargs["persist_directory"] == str(tmp_path)
    assert provider._memory_db is None


def test_embedding_without_memory(monkeypatch):
    """验证只预热知识向量时不创建记忆模型；monkeypatch为补丁夹具；无返回值。"""
    calls = Mock(return_value=object())
    monkeypatch.setattr(EmbeddingProvider, "embedding", property(lambda self: calls("knowledge")))
    monkeypatch.setattr(EmbeddingProvider, "memory_embedding_service", property(lambda self: calls("memory")))
    asyncio.run(EmbeddingProvider(enable_memory=False).startup())
    calls.assert_called_once_with("knowledge")


def test_long_page_chunking_terminates_and_respects_size():
    """验证重叠较大仍可推进且块长有上限；无参数；断言不通过时抛错，无返回值。"""
    chunker = PdfTextChunker(chunk_size=24, chunk_overlap=20)
    parts = chunker._split_text("one two three four five six seven eight nine ten")
    assert parts and all(len(part) <= 24 for part in parts)
    assert parts[-1].endswith("ten")


def test_query_provider_disables_collection_creation(monkeypatch):
    """验证查询配置显式禁止创建集合；monkeypatch为补丁夹具；无返回值。"""
    constructor = Mock()
    monkeypatch.setattr(vector_module, "Chroma", constructor)
    provider = vector_module.VectorStoreProvider(
        SimpleNamespace(embedding=object()), collection_name="pet_care_v1",
        enable_memory=False, create_collection_if_not_exists=False,
    )
    _ = provider.db
    assert constructor.call_args.kwargs["create_collection_if_not_exists"] is False


def test_missing_collection_error_is_not_hidden(monkeypatch):
    """验证集合不存在的异常继续向上传播，不回退旧集合；monkeypatch为补丁夹具；无返回值。"""
    constructor = Mock(side_effect=ValueError("collection missing"))
    monkeypatch.setattr(vector_module, "Chroma", constructor)
    provider = vector_module.VectorStoreProvider(
        SimpleNamespace(embedding=object()), collection_name="missing_collection",
        enable_memory=False, create_collection_if_not_exists=False,
    )
    with pytest.raises(ValueError, match="collection missing"):
        _ = provider.db
    constructor.assert_called_once()
