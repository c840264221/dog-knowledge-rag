from click import Command

from src.graph.build_graph import build_graph
from src.vectorstore.vector_store import load_vector_store
from src.embedding.embedder import get_embedding
from src.config import CHROMA_DB_DIR
from src.graph.sub_graphs.main_graph import build_main_graph
import uuid
from src.logger import logger


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

def run_main_graph_with_stream(question: str, user_id:str="default_user") -> str:
    if isinstance(question, list) and len(question) > 0 and "text" in question[0]:
        question = question[0]["text"]

    logger.info(f"收到用户 [{user_id}] 问题: {question}")

    # 用user_id作为线程的id 每次对话都可以根据线程id来获取之前对话历史  达到记忆目的
    config = {"configurable": {"thread_id": user_id},
              "run_name": f"query_{question[:20]}",  # LangSmith 显示的名称
              "tags": ["dog_agent", "memory_test"],
              }
    from langgraph.types import Command
    # 根据question判断是恢复断点还是新的问题
    if question.startswith("RESUME:"):
        user_input = question[7:].strip()
        events = list(app_2.stream(Command(resume=user_input), config, stream_mode="values"))
        # 遍历事件  获取最后的answer
        for ev in events:
            if "answer" in ev and ev["answer"]:
                return ev["answer"]
        # 恢复后又中断了  虽然不太可能  但为了保险起见
        current = app_2.get_state(config)
        if current.next:
            prompt = extract_interrupt_prompt(current)
            return f"__INTERRUPT__:{prompt}"
        else:
            return current.values.get("answer","无答案")
    else:
        state = {
            "question": question,
            "retry_count": 0,
            "has_asked_user": False,
            "docs": [],
            "filters": {},
            "answer": "",
            "dog_name":None,
            "strategy":None
        }
        # 使用uuid作为线程id 每次对话创建一个新的id
        # config = {"configurable": {"thread_id": uuid.uuid4().hex}}
        try:
        # 第一次调用，可能因中断而提前结束迭代
            events = list(app_2.stream(state, config, stream_mode="values"))
            logger.debug(f"图执行产生 {len(events)} 个状态快照")
        except Exception as e:
            logger.exception(f"图执行异常: {e}")
            raise
        # 检查最后的事件中是否有 answer
        for ev in reversed(events):
            if "answer" in ev and ev["answer"]:
                logger.info(f"返回答案长度: {len(ev["answer"])} 字符")
                return ev["answer"]
        # 如果没有 answer，说明图在中断点暂停了 此处配合gradio，所以不采用循环处理
        current = app_2.get_state(config)
        if current.next:
            prompt = extract_interrupt_prompt(current)
            # 返回一个带有特定表示的数据 前端可以根据标识知道这是断点  需要用户输入
            return f"__INTERRUPT__:{prompt}"
        else:
            return current.values.get("answer", "无答案")


        # 如果没有 answer，说明图在中断点暂停了，需要循环处理
        # while True:
        #     # 获取当前状态，检查中断信息
        #     current_state = app_2.get_state(config)
        #     if not current_state.next:  # 图已结束
        #         return current_state.values.get("answer", "无答案")
        #
        #     # 提取中断附带的消息（由 ask_user_node 中的 interrupt() 传入）
        #     # 在 LangGraph 中，中断信息存储在 current_state.tasks[0].interrupts 或 __interrupt__ 字段中
        #     # 不同版本略有差异，常用方式：
        #     if hasattr(current_state, 'tasks') and current_state.tasks:
        #         interrupts = current_state.tasks[0].interrupts
        #         if interrupts:
        #             # 取第一个中断的消息
        #             prompt_message = interrupts[0].value  # 这就是 interrupt(question) 中的 question
        #             logger.debug(f"第一个中断的信息: {prompt_message}")
        #         else:
        #             prompt_message = "请做出选择："
        #     else:
        #         # 备用：从状态中的某个字段获取（但这需要你在 ask_user_node 中额外保存）
        #         prompt_message = "请做出选择："
        #
        #     # 显示从节点中提取的提示，而不是硬编码
        #     logger.debug(f"节点中提取的提示信息: {prompt_message}")
        #     user_input = input("您的输入：").strip()
        #
        #     # 恢复执行
        #     for event in app_2.stream(Command(resume=user_input), config, stream_mode="values"):
        #         if "answer" in event and event["answer"]:
        #             return event["answer"]


def extract_interrupt_prompt(current_state):
    if hasattr(current_state, 'tasks') and current_state.tasks:
        interrupts = current_state.tasks[0].interrupts
        if interrupts:
            return interrupts[0].value
    return "请做出选择（1/2/3）："