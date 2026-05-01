from langchain_chroma import Chroma
from langchain_core.documents import Document
import shutil
import os


# 创建向量数据库
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

# 载入向量数据库
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

def build_documents(chunks, dog_map):
    new_docs = []
    current_title = None

    for doc in chunks:

        # 从metadata的title中提取名字
        if "title" in doc.metadata:
            current_title = doc.metadata["title"].strip()

        # 如果没有title就用上一次保存的title作为name
        name = current_title
        print("doc_name:",name)

        if len(name) == 0:
            print("❌️ 未正确提取到狗狗名字！")
            continue

        text = doc.page_content

        meta = dog_map.get(name, {})

        # 类型转换
        metadata = {
            "name": name,
            "trainability": int(float(meta.get("trainability", 0))),
            "barking": int(float(meta.get("barking", 0))),
            "shedding": int(float(meta.get("shedding", 0))),
            "height":float(meta.get("height", 0)),
            "section": doc.metadata.get("section", "")
        }

        new_doc = Document(
            page_content=text,
            metadata=metadata
        )

        new_docs.append(new_doc)

    return new_docs


if __name__ == "__main__":
    from src.config import CHROMA_DB_DIR
    from src.embedding.embedder import get_embedding
    embedding = get_embedding()
    db = load_vector_store(embedding, CHROMA_DB_DIR)
    print(len(embedding.embed_query("test")))
    print("文档数量:", db._collection.count())

