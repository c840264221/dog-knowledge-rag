import json
from src.retrieval.filter_by_tags import filter_by_tags


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
    sections = []

    for section in sections:
        if section.lower() in question.lower():
            return section

    return None

from src.config import FILTER_RULES_PATH

with open(FILTER_RULES_PATH, "r", encoding="utf-8") as f:
    FILTER_RULES = json.load(f)

from src.config import INTENT_RULES_PATH
with open(INTENT_RULES_PATH, "r", encoding="utf-8") as f:
    INTENT_RULES = json.load(f)

# 根据问题提取关键字段 然后创建filed用于精确过滤
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

# 根据问题提取intent  然后返回intent结合prompt  用于控制LLM的输出
# def detect_intent(question: str) -> str:
#     q = question.lower()
#
#     if "性格" in q or "temperament" in q:
#         return "temperament"
#
#     if "训练" in q or "train" in q:
#         return "trainability"
#
#     if "掉毛" in q or "shedding" in q:
#         return "shedding"
#
#     if "叫" in q or "bark" in q:
#         return "barking"
#
#     return "general"

def detect_intent_and_tags(question: str):
    for rule in INTENT_RULES:
        for kw in rule["keywords"]:
            if kw in question:
                return {
                    "intent": rule["intent"],
                    "tags": rule.get("tags", []),
                    "field": rule.get("field")
                }
    return {"intent": "general", "tags": [], "field": None}

from src.models.reranker import get_reranker

# rerank重新将向量数据库返回的数据进行一次精准排序
def rerank_docs(question, docs, intent, top_k=3):
    pairs = [(question, doc.page_content) for doc in docs]
    reranker_model = get_reranker()
    scores = reranker_model.predict(pairs)

    # 组合 doc + score
    scored_docs = list(zip(docs, scores))

    scored = []
    # 根据intent和tags做一次加权
    for doc, score in scored_docs:
        text = doc.page_content.lower()
        tags = doc.metadata.get("tags", [])

        # 🔹 1. 语义关键词加分（简单版）
        if intent != "general" and intent in text:
            score += 2

        # 🔹 2. tags 命中加权（关键）
        if intent in tags:
            score += 3

        scored.append((doc, score))

    # 按分数排序（从高到低）
    scored.sort(key=lambda x: x[1], reverse=True)

    # 取前k个
    reranked_docs = [doc for doc, _ in scored_docs[:top_k]]

    return reranked_docs

# 添加过滤功能  更精准匹配数据
def get_smart_retriever(question: str, db):
    section = detect_section(question)
    dog_name = detect_dog(question)

    parsed = detect_intent_and_tags(question)
    intent = parsed["intent"]
    tags = parsed["tags"]

    filter_dict = detect_filters(question)

    print("intent:", intent)
    print("tags:", tags)
    print("filter_dict:", filter_dict)

    if section:
        filter_dict["section"] = section
    if dog_name:
        filter_dict["name"] = dog_name

    print("filter_dict:",filter_dict)

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

    # 手动过滤 有的向量数据库不支持$all $in这种 所以手动过滤一下 以后会加上数据库和手动混合过滤
    if tags:
        docs = filter_by_tags(docs, tags)
    print("手动按tag过滤后的结果数量:",len(docs))

    # 添加fallback  防止过滤过于严格导致未查到数据
    if not docs:
        print("过滤过为严格，数量为0，fallback......")
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)

    # rerank精细排序 取出相似度最高的前三个
    docs = rerank_docs(question, docs, intent, top_k=3)
    print("结果：",{
        "docs": docs,
        "question": question,
        "intent": intent
    })
    return {
        "docs": docs,
        "question": question,
        "intent": intent
    }


if __name__ == "__main__":
    result = detect_filters("新手可以养的不爱叫的狗狗")
    print(result)