# from langchain_classic.chains.question_answering.map_reduce_prompt import messages
from gradio.themes.builder_app import history
from langchain_core.messages import AIMessage

from src.models.llm import get_llm
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


llm = get_llm()

def build_context(docs):
    context = []

    for d in docs:
        item = {
            "name": d.metadata.get("name"),
            "structured": {
                "barking": d.metadata.get("barking"),
                "trainability": d.metadata.get("trainability"),
                "shedding": d.metadata.get("shedding"),
            },
            "text": d.page_content[:300]  # 截断避免太长
        }
        context.append(item)

    return json.dumps(context, ensure_ascii=False, indent=2)

def generate_node(state):
    print("开始generate_node......")
    print("当前state为：", state)
    context = build_context(state["docs"])
    print("context:", context)

    prompt = ChatPromptTemplate.from_template("""
你是一个严谨的狗狗百科助手。

【任务】
根据 intent 决定行为：
- recommendation → 推荐狗狗
- 其他 → 回答问题

【严格要求】
1. 必须使用数据中的 "name" 字段作为狗狗名称
2. 严禁使用“品种一/二”等编号
3. 只能基于提供数据，不得编造
4. 每条推荐必须包含名称 + 原因
5. 至少3条数据，但最多不超过5条
6. 如果intent不是general，只回答该intent相关内容
7. 你可以将历史信息作为依赖来回答用户的问题


intent: {intent}

数据（JSON格式）：
{context}

历史信息：
{history_text}

问题：
{question}

输出规则：
- 如果是推荐：最多5个，名称+原因
- 如果是问答：直接回答，不要推荐
""")

    # answer = llm.invoke(prompt)
    history_text = "\n".join([f"用户: {m.content}" if isinstance(m, HumanMessage) else f"助手: {m.content}" for m in state["messages"]])
    answer = (prompt | llm | StrOutputParser()).invoke({
        "intent": state["intent"],
        "context": context,
        "question": state["question"],
        "history_text": history_text
    })
    messages = state.get("messages",[])
    messages.append(AIMessage(content=answer))
    print("generate_node结束，结果为：", answer)
    print("当前state为：", state)

    return {"answer": answer}