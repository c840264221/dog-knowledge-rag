import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.graph.nodes.memory_retrieve_node import build_memory_retrieve_node
from src.memory.memory_manager import MemoryManager
from src.memory.memory_semantic_recall import MemorySemanticRecallService
from src.memory.sqlite_memory_store import SQLiteMemoryStore
from src.memory.v111_smoke_checks import (
    render_memory_pipeline_smoke_markdown,
    validate_memory_pipeline_smoke,
)


class SmokeMemoryVectorDb:
    """
    V1.11 冒烟测试使用的确定性向量库替身。

    功能：
        保存 MemoryManager 同步的 Document（文档），并根据测试问题
        返回固定距离，避免真实 Embedding（向量化模型）波动导致冒烟测试不稳定。

    参数：
        无。

    返回值：
        SmokeMemoryVectorDb：可提供 add_documents、delete 和
        similarity_search_with_score 的测试向量库。
    """

    def __init__(self) -> None:
        self.documents: list[Any] = []

    def delete(self, ids: list[str]) -> None:
        """
        模拟删除旧向量。

        参数：
            ids：需要删除的向量 ID 列表。

        返回值：
            None。
        """

        id_set = set(ids)
        self.documents = [
            document
            for document in self.documents
            if f"memory_{document.metadata.get('memory_id')}" not in id_set
        ]

    def add_documents(
            self,
            documents: list[Any],
            ids: list[str],
    ) -> None:
        """
        记录 MemoryManager 同步的向量文档。

        参数：
            documents：待写入的 LangChain Document 列表。
            ids：与文档对应的向量 ID 列表。

        返回值：
            None。
        """

        del ids
        self.documents.extend(documents)

    def similarity_search_with_score(
            self,
            query: str,
            k: int,
            filter: dict,
    ) -> list[tuple[Any, float]]:
        """
        根据冒烟测试问题返回确定性距离。

        参数：
            query：当前语义检索问题。
            k：最大候选数量。
            filter：Chroma metadata（向量元数据）过滤条件。

        返回值：
            list[tuple[Any, float]]：Document 与距离组成的列表。
        """

        del filter
        distance = 0.2 if "喜欢什么狗" in query else 2.0
        return [
            (document, distance)
            for document in self.documents[:k]
        ]


class SmokeVectorStoreProvider:
    """
    向 MemoryManager 和语义召回服务提供测试 memory_db。

    参数：
        memory_db：V1.11 冒烟测试使用的向量库替身。

    返回值：
        SmokeVectorStoreProvider：包含 memory_db 属性的服务提供者。
    """

    def __init__(self, memory_db: SmokeMemoryVectorDb) -> None:
        self.memory_db = memory_db


async def run_memory_pipeline_smoke() -> int:
    """
    执行 V1.11 记忆管线冒烟测试。

    功能：
        在临时 SQLite 数据库中保存记忆，同步向量 metadata，
        然后分别执行相关问题和不相关问题的记忆召回节点。

    参数：
        无。

    返回值：
        int：0 表示全部检查通过，1 表示存在错误。
    """

    with TemporaryDirectory(prefix="v111_memory_smoke_") as temp_dir:
        store = SQLiteMemoryStore(
            db_path=Path(temp_dir) / "memory_smoke.sqlite3"
        )

        try:
            memory_db = SmokeMemoryVectorDb()
            vectorstore_provider = SmokeVectorStoreProvider(memory_db)
            manager = MemoryManager(
                store=store,
                vectorstore_provider=vectorstore_provider,
            )
            save_result = manager.save_memory(
                user_id="smoke_user_v111",
                memory_type="favorite_dog",
                content="用户喜欢金毛寻回犬",
                confidence=0.95,
                importance=0.8,
                source="conversation",
            )

            semantic_recall = MemorySemanticRecallService(
                store=store,
                vectorstore_provider=vectorstore_provider,
                minimum_semantic_score=0.45,
            )
            node = build_memory_retrieve_node(
                semantic_recall=semantic_recall,
                checkpoint_manager=None,
                runtime_context_getter=lambda: None,
            )
            related_state = await node(
                {
                    "user_id": "smoke_user_v111",
                    "question": "我喜欢什么狗？",
                }
            )
            unrelated_state = await node(
                {
                    "user_id": "smoke_user_v111",
                    "question": "今天成都天气怎么样？",
                }
            )

            result = validate_memory_pipeline_smoke(
                save_result=save_result,
                vector_documents=memory_db.documents,
                related_state=related_state,
                unrelated_state=unrelated_state,
            )
            print(render_memory_pipeline_smoke_markdown(result))
            return 0 if result.passed else 1
        finally:
            store.close()


def main() -> int:
    """
    运行 V1.11 记忆管线冒烟测试命令行入口。

    参数：
        无。

    返回值：
        int：冒烟测试进程退出码。
    """

    return asyncio.run(run_memory_pipeline_smoke())


if __name__ == "__main__":
    raise SystemExit(main())
