from sentence_transformers import CrossEncoder
# from src.config import RERANKER_MODEL, CACHE_DIR, HF_TOKEN

from src.settings import settings

from src.settings import settings

_reranker_instance = None

def get_reranker():
    global _reranker_instance

    if _reranker_instance is None:
        print("🚀 加载 Reranker 模型...", flush=True)

        # 准备传递给底层模型的参数
        model_kwargs = {}
        if settings.path.CACHE_DIR:
            model_kwargs['cache_dir'] = str(settings.path.CACHE_DIR)  # 指定缓存目录

        if settings.reranker.huggingface_token:
            # 根据你的 sentence-transformers / transformers 版本选择其中一个
            # 新版本推荐 'token'，旧版本则用 'use_auth_token'
            model_kwargs["token"] = (
                settings.reranker.huggingface_token
                .get_secret_value()
            )
            # model_kwargs['use_auth_token'] = HF_TOKEN

        _reranker_instance = CrossEncoder(
            settings.reranker.model_name,
            model_kwargs=model_kwargs,  # 统一传入
            device='cpu'  # 可选：'cuda' 或 'cpu'
        )

        print("✅ Reranker 加载完成", flush=True)

    return _reranker_instance

def close_reranker():
    global _reranker_instance
    if _reranker_instance:
        del _reranker_instance


if __name__ == "__main__":
    reranker = get_reranker()