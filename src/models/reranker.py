from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL, CACHE_DIR, HF_TOKEN

_reranker_instance = None

def get_reranker():
    global _reranker_instance

    if _reranker_instance is None:
        print("🚀 加载 Reranker 模型...", flush=True)

        _reranker_instance = CrossEncoder(
            RERANKER_MODEL,
            cache_folder=CACHE_DIR,
            token=HF_TOKEN
        )

        print("✅ Reranker 加载完成", flush=True)

    return _reranker_instance

def close_reranker():
    global _reranker_instance
    if _reranker_instance:
        del _reranker_instance