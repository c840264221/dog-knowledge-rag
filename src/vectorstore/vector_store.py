from langchain_chroma import Chroma
from langchain_core.documents import Document
import shutil
import os
import json


# 创建向量数据库
def build_vector_store(docs, embedding, db_path):
    print(f"创建数据库   路径为{db_path}")
    print(f"创建数据库: {db_path}")
    print(f"docs: {len(docs)}")

    if len(docs) == 0:
        raise ValueError("❌ 没有文档")

    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=db_path
    )

    # ✅ 强制写入
    count = db._collection.count()
    print("写入数量:", count)

    # ✅ 强制触发 index
    db.similarity_search("test", k=1)

    print("✅ 索引构建完成")

    return db

# 载入向量数据库
def load_vector_store(embedding, db_path):
    print("载入向量数据库...")
    print(f"加载路径为：{db_path}")
    return Chroma(
        persist_directory=db_path,
        embedding_function=embedding
    )

# 清理旧数据库
def reset_chroma(db_dir):
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
        print("🗑️ 已删除旧数据库")

def build_metadata_feature(meta):
    feature = set()
    if int(float(meta.get("trainability", 0))) > 3:
        feature.add("trainability")
    else:
        feature.add("Untrainable")
    if int(float(meta.get("shedding", 0))) < 3:
        feature.add("no_shedding")
    else:
        feature.add("shedding")
    if int(float(meta.get("barking", 0))) > 3:
        feature.add("barking")
    else:
        feature.add("no_barking")
    if int(float(meta.get("affectionate_with_family", 0))) > 3:
        feature.add("affectionate")
    else:
        feature.add("aloof")
    if int(float(meta.get("good_with_young_children", 0))) > 3:
        feature.add("good_with_young_children")
    else:
        feature.add("no_good_with_young_children")
    if int(float(meta.get("good_with_other_dogs", 0))) > 3:
        feature.add("good_with_other_dogs")
    else:
        feature.add("no_good_with_other_dogs")
    if int(float(meta.get("coat_grooming_frequency", 0))) > 3:
        feature.add("often_coat_groom")
    else:
        feature.add("seldom_coat_groom")
    if int(float(meta.get("drooling", 0))) > 3:
        feature.add("drooling")
    else:
        feature.add("no_drooling")
    if int(float(meta.get("openness_to_strangers", 0))) > 3:
        feature.add("open_to_strangers")
    else:
        feature.add("no_open_to_strangers")
    if int(float(meta.get("playfulness", 0))) > 3:
        feature.add("playfulness")
    else:
        feature.add("no_playfulness")
    if int(float(meta.get("watchdog", 0))) > 3:
        feature.add("watchdog")
    else:
        feature.add("no_watchdog")
    if int(float(meta.get("adaptability", 0))) > 3:
        feature.add("adaptable")
    else:
        feature.add("no_adaptable")
    if int(float(meta.get("energy", 0))) > 3:
        feature.add("energy")
    else:
        feature.add("no_energy")
    if int(float(meta.get("mental_stimulation_needs", 0))) > 3:
        feature.add("need_mental_stimulation")
    else:
        feature.add("no_need_mental_stimulation")
    if int(float(meta.get("height", 0))) < 13:
        feature.add("small")
    elif 13 <= int(float(meta.get("height", 0))) <= 24:
        feature.add("medium")
    elif int(float(meta.get("height", 0))) > 24:
        feature.add("big")

    return list(feature)

from src.config import TAG_RULES_PATH
with open(TAG_RULES_PATH, "r", encoding="utf-8") as f:
    TAG_RULES = json.load(f)

def extract_tags(text: str, section: str):
    tags = set()

    text_lower = text.lower()

    # 文本关键词
    for tag, keywords in TAG_RULES.items():
        if any(kw in text_lower for kw in keywords):
            tags.add(tag)

    # section 辅助（结构信号）
    if "标签" in section:
        tags.add("temperament")

    if not tags:
        tags.add("general")
    return list(tags)

def build_documents(chunks, dog_map):
    new_docs = []
    current_title = None

    for doc in chunks:

        # 从metadata的title中提取名字
        if "title" in doc.metadata:
            current_title = doc.metadata["title"].strip()

        # 如果没有title就用上一次保存的title作为name
        name = current_title
        print("doc_name:",name)

        if len(name) == 0:
            print("❌️ 未正确提取到狗狗名字！")
            continue

        text = doc.page_content

        meta = dog_map.get(name, {})

        def map_section_to_field(section: str) -> str:
            if "标签" in section:
                return "tag"
            elif "基本信息" in section:
                return "base_information"
            elif "指南" in section:
                return "guide"
            elif "关于" in section:
                return "about"
            elif "特征" in section:
                return "feature"
            elif "历史" in section:
                return "history"
            else:
                return "other"

        # 类型转换
        metadata = {
            "name": name,
            "trainability": int(float(meta.get("trainability", 0))),
            "barking": int(float(meta.get("barking", 0))),
            "shedding": int(float(meta.get("shedding", 0))),
            "height":float(meta.get("height", 0)),
            "section": doc.metadata.get("section", ""),
            "field": map_section_to_field(doc.metadata.get("section", "")),
            "feature": build_metadata_feature(meta),
            "tags": extract_tags(text,doc.metadata.get("section"))
        }

        new_doc = Document(
            page_content=text,
            metadata=metadata
        )

        new_docs.append(new_doc)

    return new_docs


if __name__ == "__main__":
    from src.config import CHROMA_DB_DIR
    from src.embedding.embedder import get_embedding
    embedding = get_embedding()
    db = load_vector_store(embedding, CHROMA_DB_DIR)
    print(len(embedding.embed_query("test")))
    print("文档数量:", db._collection.count())

