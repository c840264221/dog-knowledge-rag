"""维护用集合复制：排除纯标题块，逐字保留正文、元数据和原向量。"""

import hashlib
import json

from src.rag.evaluation.index_quality_audit import is_heading_only


def _vector_list(vector) -> list[float]:
    """把原向量转换为可传给Chroma的浮点列表；vector为数组或列表；返回数值不变的列表。"""
    return [float(value) for value in vector]


def records_fingerprint(records: dict) -> str:
    """生成含向量的记录快照指纹，用于核对源集合是否变化。

    参数：records为含ids、documents、metadatas、embeddings的等长记录。
    返回值：按编号排序后逐条计算的SHA-256；不把返回顺序变化误判为数据变化。
    """
    ids = records["ids"]
    if len(set(ids)) != len(ids) or any(records[key] is None or len(records[key]) != len(ids)
                                      for key in ("documents", "metadatas", "embeddings")):
        raise ValueError("快照编号重复、向量缺失或记录列长度不一致。")
    digest = hashlib.sha256()
    for index in sorted(range(len(ids)), key=lambda item: ids[item]):
        value = [ids[index], records["documents"][index], records["metadatas"][index],
                 _vector_list(records["embeddings"][index])]
        digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_hnsw_configuration(collection) -> dict:
    """读取源集合的HNSW距离和参数，不猜测默认距离。

    参数：collection为原生Chroma集合；本入口只支持已记录HNSW参数的集合。
    返回值：可传入collection_configuration的hnsw字典；未知配置直接报错，
        不将未知模型配置补写为当前向量模型。
    """
    schema = collection.schema
    serialized = schema.serialize_to_json() if schema is not None else {}
    vector_config = (serialized.get("keys", {}).get("#embedding", {})
                     .get("float_list", {}).get("vector_index", {}).get("config", {}))
    hnsw = vector_config.get("hnsw")
    if not hnsw or vector_config.get("space") not in {"l2", "cosine", "ip"}:
        raise ValueError("无法确认源集合HNSW参数与距离类型，停止复制。")
    fields = {"ef_construction", "max_neighbors", "ef_search", "num_threads", "batch_size", "sync_threshold", "resize_factor"}
    return {"hnsw": {"space": vector_config["space"], **{key: value for key, value in hnsw.items() if key in fields}}}


def copy_without_headings(source, target, records: dict, *, batch_size: int = 128) -> dict:
    """将非纯标题记录连同原向量写入空的对照集合并完整回查。

    参数：source、target为不同的原生集合；records为本轮源集合含向量快照；
        batch_size为每批复制条数，默认128。
    返回值：复制数量、排除数量、内容及向量核对结果、源集合未变核对结果。
        目标非空时拒绝，不删除或覆盖；不调用向量模型。复制失败可能留下部分
        目标记录，不能用于正式流量；本函数不自动清理，也不支持并发维护。
    """
    if batch_size < 1 or source.name == target.name:
        raise ValueError("批量大小必须为正，且源与目标不能同名。")
    if target.count() != 0:
        raise ValueError("目标集合已有数据，拒绝覆盖或继续部分复制。")
    if source_hnsw_configuration(source) != source_hnsw_configuration(target):
        raise ValueError("源与目标HNSW配置不同，停止写入。")
    before = records_fingerprint(records)
    kept = [index for index, text in enumerate(records["documents"]) if not is_heading_only(text or "")]
    if not kept or any(not isinstance(records["metadatas"][index], dict) or not records["metadatas"][index]
                       or records["documents"][index] is None for index in kept):
        raise ValueError("没有可复制记录，或记录缺少可原样复制的正文/元数据。")
    if before != records_fingerprint(source.get(include=["documents", "metadatas", "embeddings"])):
        raise ValueError("预览快照后源集合发生变化，停止复制。")
    for start in range(0, len(kept), batch_size):
        indices = kept[start:start + batch_size]
        target.add(
            ids=[records["ids"][index] for index in indices],
            documents=[records["documents"][index] for index in indices],
            metadatas=[records["metadatas"][index] for index in indices],
            embeddings=[_vector_list(records["embeddings"][index]) for index in indices],
        )
    expected = {key: [records[key][index] for index in kept]
                for key in ("ids", "documents", "metadatas", "embeddings")}
    actual = target.get(include=["documents", "metadatas", "embeddings"])
    if records_fingerprint(actual) != records_fingerprint(expected):
        raise RuntimeError("目标的编号、正文、元数据或向量核对不一致。")
    if before != records_fingerprint(source.get(include=["documents", "metadatas", "embeddings"])):
        raise RuntimeError("复制期间源集合发生变化，本轮对照不可直接采用。")
    return {
        "copied_records": len(kept), "excluded_heading_records": len(records["ids"]) - len(kept),
        "verified_documents_metadata_embeddings": True, "source_unchanged_verified": True,
        "embedding_reused": True,
    }
