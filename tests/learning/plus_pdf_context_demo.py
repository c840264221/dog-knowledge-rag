"""PDF 入库学习第七步：按重排结果回查图文原料，构建带来源的回答上下文。"""

from copy import deepcopy
import json
from pathlib import Path


def build_pdf_context_from_candidates(
    *, question: str, candidates: list[dict], bundle: dict,
    mode: str = "without_notes", top_k: int = 1,
) -> dict:
    """选择重排候选，回查完整原料，去重注意事项并建立程序生成的引用映射。

    参数：
        question：本轮实际问题。
        candidates：本轮重排后的候选列表，包含块编号、说明和重排排名。
        bundle：rag_chunk_bundle.json，包含标准块和按块编号保存的结构化记录。
        mode：选择哪组实验结果，默认without_notes，不代表更改项目检索策略。
        top_k：最多选择几块，默认1，仅用于当前单图示问题的教学演示。
    返回值：
        包含question、selected_chunk_ids、evidence、common_notes、citation_map、
        image_inputs和context_text的字典。原文与视觉描述分开，限制条件恢复到
        回答上下文，引用和页码由程序添加。此函数不调用模型、不加载图片、
        不保证资料足够，也不实现通用预算控制或自动相关性阈值。
    """
    if top_k < 1:
        raise ValueError("选择数量必须大于等于1。")
    selected = sorted(candidates, key=lambda item: item["rerank_rank"])[:top_k]
    if not selected:
        raise ValueError("没有候选资料，不能构建本次回答上下文。")

    evidence = []
    citation_map = {}
    note_sources = {}
    image_sources = {}
    sections = []

    for index, candidate in enumerate(selected, start=1):
        chunk_id = candidate["chunk_id"]
        record = bundle["structured_content_by_chunk_id"][chunk_id]
        content = deepcopy(record["structured_content"])
        metadata = record["original_metadata"]
        if content["caption"] != candidate["caption"]:
            raise ValueError("候选说明与当前原料不一致，请重新生成适配和重排结果。")
        citation_id = f"S{index}"

        citation_map[citation_id] = {
            "chunk_id": chunk_id,
            "doc_id": record["doc_id"],
            "source": metadata["source"],
            "page_number": metadata["page_number"],
            "image_path": metadata["image_path"],
            "section_index": metadata["section_index"],
            "item_index": metadata["item_index"],
        }
        evidence.append({
            "citation_id": citation_id,
            "chunk_id": chunk_id,
            "rerank_rank": candidate["rerank_rank"],
            "content": content,
            "extraction_uncertainties": deepcopy(metadata["extraction_uncertainties"]),
        })

        lines = [
            f"[{citation_id}]",
            f"文档标题：{content['page_title'] or '未提取到标题'}",
            f"来源页码：PDF第{metadata['page_number']}页",
            f"分组标题原文：{content['section_heading'] or '分组不明确'}",
        ]
        if content["section_paragraphs"]:
            lines.append("分组说明原文：\n" + "\n".join(content["section_paragraphs"]))
        if content["caption"]:
            lines.append("图示说明原文：" + content["caption"])
        if content["visual_description"]:
            lines.append("视觉描述（来自模型提取及人工修正，不是原文）：" + content["visual_description"])
        sections.append("\n".join(lines))

        for note in content["notes"]:
            ids = note_sources.setdefault(note, [])
            if citation_id not in ids:
                ids.append(citation_id)
        image_sources.setdefault(metadata["image_path"], []).append(citation_id)

    common_notes = [
        {"text": note, "citation_ids": ids}
        for note, ids in note_sources.items()
    ]
    if common_notes:
        note_lines = [
            "".join(f"[{citation_id}]" for citation_id in note["citation_ids"]) + " " + note["text"]
            for note in common_notes
        ]
        sections.append("资料适用限制与注意事项原文（去重后）：\n" + "\n".join(note_lines))

    return {
        "question": question,
        "mode": mode,
        "selection_policy": "top_k_for_single_figure_demo",
        "review_status": "needs_context_review",
        "selected_chunk_ids": [item["chunk_id"] for item in selected],
        "evidence": evidence,
        "common_notes": common_notes,
        "citation_map": citation_map,
        "image_inputs": [
            {"image_path": path, "citation_ids": ids}
            for path, ids in image_sources.items()
        ],
        "context_text": "\n\n".join(sections),
    }


def build_pdf_answer_context(
    *, rerank_report: dict, bundle: dict, mode: str = "without_notes", top_k: int = 1
) -> dict:
    """兼容旧对照报告入口，委托通用函数构建上下文。

    参数：rerank_report为旧重排报告，bundle为结构化原料，mode为实验组，
        top_k为最多入选块数。
    返回值：与原入口相同的上下文字典；不调用模型或修改向量库。
    """
    return build_pdf_context_from_candidates(
        question=rerank_report["question"],
        candidates=rerank_report["variants"][mode]["ranking"],
        bundle=bundle, mode=mode, top_k=top_k,
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output/learning/pdf_pages/dogs_trust_body_language"
    rerank_report = json.loads((output_dir / "rerank_comparison.json").read_text(encoding="utf-8-sig"))
    bundle = json.loads((output_dir / "rag_chunk_bundle.json").read_text(encoding="utf-8-sig"))
    context = build_pdf_answer_context(
        rerank_report=rerank_report, bundle=bundle, mode="without_notes", top_k=1,
    )
    output_path = output_dir / "answer_context.json"
    output_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"问题：{context['question']}")
    print(f"入选块数：{len(context['selected_chunk_ids'])}")
    print(f"去重后的注意事项数：{len(context['common_notes'])}")
    print("\n待交给回答模型的文字资料：\n")
    print(context["context_text"])
    print("\n引用映射：")
    print(json.dumps(context["citation_map"], ensure_ascii=False, indent=2))
    print(f"\n上下文报告已保存：{output_path}")
    print("图片目前仅保留引用，尚未发送模型；本步没有调用模型或修改向量库。")
