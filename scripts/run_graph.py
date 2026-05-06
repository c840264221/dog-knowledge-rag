from src.graph.graph_run import run

def chat():
    print("进入程序...")

    while True:
        q = input("请输入问题：").strip()
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
            answer = run(q)
            print("🤖:", answer)
        except Exception as e:
            print("❌ 出错:", e)

if __name__ == '__main__':
    chat()