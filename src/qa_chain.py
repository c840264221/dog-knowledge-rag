from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.retrieval.retriever import get_smart_retriever
from src.models.llm import get_llm

def build_rag_chain(db):
    print("开始实例化rag链......")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("""
你是一个严谨的狗狗百科助手。
【任务】
根据提供的资料回答用户问题。

【严格要求】：
1. 只能根据提供的资料回答
2. 不要编造任何信息
3. 不要重复内容
4. 控制在5点以内
5. 用清晰条目输出

intent:{intent}

资料：
{context}

问题：
{question}

请输出：
- 用编号列出要点
- 每点一句话
""")
    # def format_docs(docs):
    #     return "\n\n".join([doc.page_content for doc in docs])

    def format_docs(docs, intent=None):
        results = []
        for doc in docs:
            # 轻过滤  因为数据源不是标准格式且数据完整度不够 暂不采用轻过滤
            # text = doc.page_content.lower()
            #
            # if intent == "temperament" and "temperament" not in text:
            #     continue
            # if intent == "trainability" and "train" not in text:
            #     continue
            # if intent == "shedding" and "shedding" not in text:
            #     continue

            results.append(doc.page_content)

        return "\n\n".join(results)

    smart_retriever = RunnableLambda(
        lambda question: get_smart_retriever(question, db)
    )
    print("开始创建rag_chain......")

    rag_chain = (
        smart_retriever
        |{
            "context": lambda x: format_docs(x["docs"], x["intent"]),
            "question": lambda x: x["question"],
            "intent": lambda x: x["intent"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain