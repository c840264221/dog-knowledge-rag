# from langchain_classic.chains.question_answering.map_reduce_prompt import messages
from gradio.themes.builder_app import history
from langchain_core.messages import AIMessage

from src.models.llm import get_instance_llm, safe_llm_invoke
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from src.logger import logger

# 导入记忆模块  用历史记忆注入prompt
from src.memory.memory_retrieve import retrieve_user_memory


llm = get_instance_llm()

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

    logger.info(f"进入generate_node节点，state为："
        f"state: "
        f"question:{state['question']}, "
        f"intent:{state['intent']}, "
        f"strategy:{state['strategy']}, "
        f"filters:{state['filters']}, "
        f"tags:{state['tags']}, "
        f"dog_name:{state['dog_name']}, "
        f"docs len:{len(state['docs'])},"
        f"user_id:{state['user_id']} "
                )

    # 记忆检索
    memory_text = retrieve_user_memory(user_id=state['user_id'])

    context = build_context(state["docs"])
    logger.debug(f"context<UNK>{context}")

    prompt = ChatPromptTemplate.from_template("""
你是一个严谨的狗狗百科助手。


# 用户长期记忆（Memory）
{memory_text}


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
    logger.debug(f"history_text:{history_text}")

    safe_llm = RunnableLambda(
        lambda x: safe_llm_invoke(
            llm=llm,
            prompt=x,
            fallback_response="模型暂时不可用"
        )
    )

    # answer = (prompt | llm | StrOutputParser()).invoke({
    #     "intent": state["intent"],
    #     "context": context,
    #     "question": state["question"],
    #     "history_text": history_text
    # })

    # 采用更安全的llm调用 支持降级和重试
    answer = (prompt | safe_llm | StrOutputParser()).invoke({
        "memory_text": memory_text,
        "intent": state["intent"],
        "context": context,
        "question": state["question"],
        "history_text": history_text
    })

    messages = state.get("messages",[])
    logger.debug(f"messages<UNK>{messages}")
    messages.append(AIMessage(content=answer))
    logger.info(f"generate_node节点完成，结果answer为：{answer}")

    return {"answer": answer}