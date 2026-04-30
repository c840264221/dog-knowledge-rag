from langchain_chroma import Chroma
import shutil
import os


def build_vector_store(docs, embedding, db_path):
    print(f"创建数据库   路径为{db_path}")
    print(f"创建数据库: {db_path}")
    print(f"docs: {len(docs)}")

    if len(docs) == 0:
        raise ValueError("❌ 没有文档")

    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=db_path
    )

    # ✅ 强制写入
    count = db._collection.count()
    print("写入数量:", count)

    # ✅ 强制触发 index
    db.similarity_search("test", k=1)

    print("✅ 索引构建完成")

    return db

def load_vector_store(embedding, db_path):
    print("载入向量数据库...")
    print(f"加载路径为：{db_path}")
    return Chroma(
        persist_directory=db_path,
        embedding_function=embedding
    )

# 清理旧数据库
def reset_chroma(db_dir):
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
        print("🗑️ 已删除旧数据库")


if __name__ == "__main__":
    from src.config import CHROMA_DB_DIR
    from src.embedding.embedder import get_embedding
    embedding = get_embedding()
    db = load_vector_store(embedding, CHROMA_DB_DIR)
    print(len(embedding.embed_query("test")))
    print("文档数量:", db._collection.count())

