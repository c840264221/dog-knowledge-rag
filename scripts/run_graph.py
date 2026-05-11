from src.graph.graph_run import run_main_graph, run_main_graph_with_stream
from dotenv import load_dotenv
import os

load_success = load_dotenv(override=True)
print(f"Load .env success: {load_success}")

# 或者尝试清除 langsmith 的特定缓存
try:
    from langsmith import utils
    utils.get_env_var.cache_clear()
except ImportError:
    pass

# 2. 打印键值，验证变量是否加载到 os.environ
api_key = os.getenv("LANGSMITH_API_KEY")
print(f"API Key: {api_key}")

assert os.environ.get("LANGSMITH_API_KEY"), "请设置 LANGSMITH_API_KEY"


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
            # answer = run(q)
            # answer = run_main_graph(q)
            answer = run_main_graph_with_stream(q)
            print("🤖:", answer)
        except Exception as e:
            print("❌ 出错:", e)

if __name__ == '__main__':
    chat()