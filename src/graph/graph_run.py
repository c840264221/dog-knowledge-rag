from src.graph.build_graph import build_graph
from src.vectorstore.vector_store import load_vector_store
from src.embedding.embedder import get_embedding
from src.config import CHROMA_DB_DIR
from src.graph.sub_graphs.main_graph import build_main_graph
import uuid


# embedding = get_embedding()
# db = load_vector_store(embedding, CHROMA_DB_DIR)
# app = build_graph(db)


# def run(question: str):
#     result = app.invoke({
#         "question": question
#     })
#
#     return result["answer"]

app_2 = build_main_graph()
def run_main_graph(question: str):
    result = app_2.invoke({
        "question": question
    })
    return result["answer"]

def run_main_graph_with_stream(question: str) -> str:
    from langgraph.types import Command
    state = {
        "question": question,
        "retry_count": 0,
        "has_asked_user": False,
        "docs": [],
        "filters": {},
        "answer": ""
    }
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}

    # 第一次调用，可能因中断而提前结束迭代
    events = list(app_2.stream(state, config, stream_mode="values"))
    # 检查最后的事件中是否有 answer
    for ev in reversed(events):
        if "answer" in ev and ev["answer"]:
            return ev["answer"]

    # 如果没有 answer，说明图在中断点暂停了，需要循环处理
    while True:
        # 获取当前状态，检查中断信息
        current_state = app_2.get_state(config)
        if not current_state.next:  # 图已结束
            return current_state.values.get("answer", "无答案")

        # 提取中断附带的消息（由 ask_user_node 中的 interrupt() 传入）
        # 在 LangGraph 中，中断信息存储在 current_state.tasks[0].interrupts 或 __interrupt__ 字段中
        # 不同版本略有差异，常用方式：
        if hasattr(current_state, 'tasks') and current_state.tasks:
            interrupts = current_state.tasks[0].interrupts
            if interrupts:
                # 取第一个中断的消息
                prompt_message = interrupts[0].value  # 这就是 interrupt(question) 中的 question
            else:
                prompt_message = "请做出选择："
        else:
            # 备用：从状态中的某个字段获取（但这需要你在 ask_user_node 中额外保存）
            prompt_message = "请做出选择："

        # 显示从节点中提取的提示，而不是硬编码
        print("\n" + prompt_message)
        user_input = input("您的输入：").strip()

        # 恢复执行
        for event in app_2.stream(Command(resume=user_input), config, stream_mode="values"):
            if "answer" in event and event["answer"]:
                return event["answer"]