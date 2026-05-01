from src.ingestion.markdown_file_loader import load_markdown_files
from src.ingestion.splitter import split_markdown
from src.embedding.embedder import get_embedding
from src.vectorstore.vector_store import build_vector_store, reset_chroma, build_documents, load_vector_store
from src.config import CHROMA_DB_DIR, DOG_MD_DATA_DIR
from src.service.dog_data_loader import load_json_data

# reset_chroma(CHROMA_DB_DIR)
# docs = load_markdown_files(DOG_MD_DATA_DIR)
# print(len(docs))
# chunks = split_markdown(docs)
# print(len(chunks))
#
# embeddings = get_embedding()
# vec = embeddings.embed_query("test")
# print("向量维度:", len(vec))
#
# db = build_vector_store(chunks, embeddings, CHROMA_DB_DIR)
# count = db._collection.count()
# print("写入数量:", count)
#
# if count == 0:
#     raise ValueError("❌ 向量库为空，构建失败")

def build_db():
    docs = load_markdown_files(DOG_MD_DATA_DIR)
    chunks = split_markdown(docs)

    # ✅ 新增
    dog_map = load_json_data()

    chunks = build_documents(chunks, dog_map)

    reset_chroma(CHROMA_DB_DIR)

    embeddings = get_embedding()

    db = build_vector_store(chunks, embeddings, CHROMA_DB_DIR)

    return db

if __name__ == "__main__":
    db = build_db()
    # embedding = get_embedding()
    # db = load_vector_store(embedding, CHROMA_DB_DIR)
    # docs = db.similarity_search("适合新手的狗", k=2)
    #
    # for d in docs:
    #     print(d.metadata)