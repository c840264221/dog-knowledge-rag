from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL, CACHE_DIR


_embedding_instance = None

def get_embedding():
    global _embedding_instance

    if _embedding_instance is None:
        print("🚀 加载 Embedding 模型...", flush=True)

        _embedding_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            cache_folder=CACHE_DIR
        )

        print("✅ Embedding 加载完成", flush=True)

    return _embedding_instance

# def get_embedding():
#     print("实例化向量工具......")
#     return HuggingFaceEmbeddings(
#         # model_name="sentence-transformers/all-MiniLM-L6-v2"  # 体量小，速度快，但效果一般
#         model_name="BAAI/bge-small-zh"  # 中文语义更好，但是会慢
#     )


if __name__ == "__main__":
    embedding = get_embedding()
