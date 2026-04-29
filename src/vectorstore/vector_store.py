from langchain_chroma import Chroma
import shutil
import os


def build_vector_store(docs, embedding):
    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory="chroma_db"   # 👈 关键
    )

    # 👈 保存到磁盘
    # db.persist()  # 新版chroma已经支持自动持久化，所以这个弃用了
    return db

def load_vector_store(embedding):
    print("载入向量数据库...")
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding
    )

# 清理旧数据库
def reset_chroma(db_dir):
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
        print("🗑️ 已删除旧数据库")