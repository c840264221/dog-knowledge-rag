"""逐条比较向量的诊断工具，只用于小规模离线排查。"""

import math


def exact_l2_ranking(query_vector, records: dict, *, dog_name: str | None = None) -> list[dict]:
    """逐条计算平方欧氏距离并排序，不经过快速查找索引。

    参数：query_vector为问题向量；records含等长ids、metadatas、embeddings三列；
        dog_name不为空时只比较元数据中犬种完全相等的记录，与数据库等值过滤对应。
    返回值：按距离升序排列的编号、距离字典列表；同距离按编号排序以稳定显示。
        距离采用各维差值平方之和，与当前Chroma的l2返回口径一致，越小越近。
        空向量、非有限数值、维度不同及记录列不齐均报错，不静默跳过。
    """
    query = [float(value) for value in query_vector]
    if not query or not all(math.isfinite(value) for value in query):
        raise ValueError("问题向量必须非空且数值有限。")
    ids = records["ids"]
    if len(set(ids)) != len(ids) or any(records[key] is None or len(records[key]) != len(ids)
                                      for key in ("metadatas", "embeddings")):
        raise ValueError("记录编号重复或各列长度不一致。")
    ranked = []
    for index, record_id in enumerate(ids):
        metadata = records["metadatas"][index] or {}
        if dog_name is not None and metadata.get("dog_name") != dog_name:
            continue
        vector = [float(value) for value in records["embeddings"][index]]
        if len(vector) != len(query) or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"记录向量维度不同或数值无效：{record_id}")
        distance = sum((left - right) ** 2 for left, right in zip(query, vector))
        ranked.append({"chunk_id": record_id, "distance": distance})
    return sorted(ranked, key=lambda item: (item["distance"], item["chunk_id"]))
