from src.embedding.embedder import get_embedding
from src.vectorstore.vector_store import load_vector_store
from src.qa_chain import build_rag_chain


def chat():
    print("进入程序...")
    embedding = get_embedding()
    db = load_vector_store(embedding)

    rag_chain = build_rag_chain(db)

    while True:
        q = input("请输入问题：").strip()
        # print(f"DEBUG: {repr(q)}")
        if q.lower() == "exit":
            print("⚠️ 即将 break")
            # 可选：清理资源
            try:
                db._client._system.stop()
            except:
                pass

            import os
            os._exit(0)

            print("👋 已释放资源")
        try:
            answer = rag_chain.invoke(q)
            print("🤖:", answer)
        except Exception as e:
            print("❌ 出错:", e)

if __name__ == '__main__':
    chat()