import json


def get_retriever(db):
    # db = Chroma(
    #     persist_directory="chroma_db",
    #     embedding_function=get_embeddings()
    # )

    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

from src.retrieval.alias_loader import get_alias_dict

def detect_dog(question):
    alias_dict = get_alias_dict()

    q = question.lower()

    for dog, aliases in alias_dict.items():
        for a in aliases:
            if a in q:
                return dog
    return None

def detect_section(question):
    sections = ["性格", "标签", "基本信息", "关于", "历史", "指南"]

    for section in sections:
        if section.lower() in question.lower():
            return section

    return None

from src.config import FILTER_RULES_PATH

with open(FILTER_RULES_PATH, "r", encoding="utf-8") as f:
    FILTER_RULES = json.load(f)

def detect_filters(question):
    filters = {}

    for rule in FILTER_RULES:
        for kw in rule["keywords"]:
            if kw in question:
                filters[rule["field"]] = {
                    rule["op"]: rule["value"]
                }
                break

    return filters

from src.models.reranker import get_reranker

# rerank重新将向量数据库返回的数据进行一次精准排序
def rerank_docs(question, docs, top_k=3):
    pairs = [(question, doc.page_content) for doc in docs]
    reranker_model = get_reranker()
    scores = reranker_model.predict(pairs)

    # 组合 doc + score
    scored_docs = list(zip(docs, scores))

    # 按分数排序（从高到低）
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # 取前k个
    reranked_docs = [doc for doc, _ in scored_docs[:top_k]]

    return reranked_docs

# 添加过滤功能  更精准匹配数据
def get_smart_retriever(question: str, db):
    section = detect_section(question)
    dog_name = detect_dog(question)

    filter_dict = detect_filters(question)

    if section:
        filter_dict["section"] = section
    if dog_name:
        filter_dict["name"] = dog_name

    retriever = db.as_retriever(
        search_kwargs={
            # "k": 3,
            "k": 8,
            "filter": filter_dict if filter_dict else None
        }
    )
    docs = retriever.invoke(question)
    print(len(docs))
    for doc in docs:
        print(doc.metadata)

    # 添加fallback  防止过滤过于严格导致未查到数据
    if not docs:
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)

    # return retriever.invoke(question)
    # return retriever
    # rerank精细排序 取出相似度最高的前三个
    docs = rerank_docs(question, docs, top_k=3)
    return docs


if __name__ == "__main__":
    result = detect_filters("新手可以养的不爱叫的狗狗")
    print(result)