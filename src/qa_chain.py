from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.retrieval.retriever import get_smart_retriever
from src.models.llm import get_llm

def build_rag_chain(db):
    print("开始实例化rag链......")
    # llm = ChatOllama(
    #     model='qwen:1.8b',
    #     temperature=0
    # )
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("""
你是一个严谨的狗狗百科助手。

【严格要求】：
1. 只能根据提供的资料回答
2. 不要编造任何信息
3. 不要重复内容
4. 控制在5点以内
5. 用清晰条目输出

资料：
{context}

问题：
{question}

请输出：
- 用编号列出要点
- 每点一句话
""")
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    smart_retriever = RunnableLambda(
        lambda question: get_smart_retriever(question, db)
    )
    print("开始创建rag_chain......")

    rag_chain = (
        {
            "context": smart_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain