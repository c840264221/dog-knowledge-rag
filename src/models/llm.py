from langchain_ollama import ChatOllama

_llm_instance = None

def get_llm():
    global _llm_instance

    if _llm_instance is None:
        print("🚀 初始化 LLM...", flush=True)

        _llm_instance = ChatOllama(
            # model="deepseek-r1:latest",
            model="deepseek-r1:14b",
            temperature=0
        )

    return _llm_instance