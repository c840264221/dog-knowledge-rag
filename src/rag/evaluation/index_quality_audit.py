"""索引记录质量审计：识别纯Markdown标题块，并检查指定犬种的运动证据是否存在。"""

from collections import Counter
from pathlib import PurePosixPath
import re


def is_heading_only(text: str) -> bool:
    """判断非空内容是否全部由Markdown标题行组成。

    参数：text为索引中的实际正文，不读取外部源文件。
    返回值：所有非空行都符合井号加空白的标题格式时为True；空正文返回False，
        单独统计空记录。不会因内容短、包含某个关键词就认定为纯标题。
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(re.fullmatch(r"#{1,6}[ \t]+\S.*", line) is not None for line in lines)


def audit_index_records(records: dict, *, target_breed: str = "Golden Retriever") -> dict:
    """审计一次向量库get返回的记录快照，不调用模型或修改数据。

    参数：records包含等长ids、documents、metadatas；target_breed为检查犬种名。
    返回值：总记录数、空正文数、纯标题数及名单、受影响文档数、重复标题分布，
        以及目标犬种全部记录和运动章节命中。文档范围按doc_id优先、source其次统计，
        缺失身份的记录单独计数。运动证据检测仅针对明确Exercise标题，不证明
        其他表达的相关段落不存在，也不证明检索一定能召回。
    """
    ids = records["ids"]
    documents = records["documents"]
    metadatas = records["metadatas"]
    if not (len(ids) == len(documents) == len(metadatas)):
        raise ValueError("索引快照的编号、正文和元数据数量不一致。")
    heading_records = []
    empty_ids = []
    affected = set()
    unknown_document_heading_count = 0
    patterns = Counter()
    target_records = []
    exercise_records = []
    target_key = target_breed.strip().casefold()
    for record_id, text, metadata in zip(ids, documents, metadatas):
        text = text or ""
        metadata = metadata or {}
        record = {"record_id": record_id, "content": text, "metadata": dict(metadata)}
        if not text.strip():
            empty_ids.append(record_id)
        if is_heading_only(text):
            heading_records.append(record)
            patterns["\n".join(line.strip() for line in text.splitlines() if line.strip())] += 1
            identity = metadata.get("doc_id") or metadata.get("source")
            if identity:
                affected.add(str(identity))
            else:
                unknown_document_heading_count += 1
        source = str(metadata.get("source", "")).replace("\\", "/")
        source_name = PurePosixPath(source).stem.replace("_", " ").strip().casefold()
        names = {str(metadata.get(key, "")).strip().casefold() for key in ("dog_name", "title")}
        if target_key not in names and source_name != target_key:
            continue
        target_records.append(record)
        explicit_exercise = (
            str(metadata.get("section_title", "")).strip().casefold() == "exercise"
            or re.search(r"(?im)^\s*#{1,6}[ \t]+Exercise[ \t]*$", text) is not None
        )
        if explicit_exercise and text.strip() and not is_heading_only(text):
            exercise_records.append(record)
    return {
        "record_count": len(ids),
        "empty_content_count": len(empty_ids),
        "empty_content_record_ids": empty_ids,
        "heading_only_count": len(heading_records),
        "heading_only_fraction": len(heading_records) / len(ids) if ids else 0.0,
        "affected_document_count": len(affected),
        "heading_records_without_document_identity": unknown_document_heading_count,
        "heading_patterns": [{"content": text, "count": count} for text, count in patterns.most_common()],
        "heading_only_records": heading_records,
        "target_breed": target_breed,
        "target_record_count": len(target_records),
        "target_records": target_records,
        "explicit_exercise_record_count": len(exercise_records),
        "explicit_exercise_records": exercise_records,
        "notice": "仅审计本次库中记录。标题检测不是通用语义审核；未删除记录，也未证明它是召回失败的唯一原因。",
    }
