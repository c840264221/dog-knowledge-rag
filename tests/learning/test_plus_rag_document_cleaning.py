"""页面文本清洗教学：仅用模拟文本，不解析真实 PDF，不执行跨页合并。"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import re
from typing import Any


def clean_confirmed_page_noise(
    *, source: str, pages: list[dict[str, Any]], confirmed_headers: set[str],
    edge_line_count: int = 2, minimum_header_pages: int = 2,
) -> dict[str, Any]:
    """
    功能：删除已确认且在多页顶部重复的页眉，以及底部与当前页一致的明确页码。
    参数含义：source 是原始来源；pages 包含 page_number 和 text；confirmed_headers
        是外部审核确认的页眉白名单；edge_line_count 是首尾非空行检查范围；
        minimum_header_pages 是候选页眉至少出现在多少页顶部才允许移除。
    返回值含义：返回原始页面副本、按页清洗块、删除审计记录及页眉出现页数。
        只对匹配文本归一化，不改写保留原文；locations 仅保存页码，不伪造 PDF 行号。
        本规则无法代替版面分析，既不自动识别任意页眉，也不证明跨页正文连续。
    """
    if edge_line_count < 1 or minimum_header_pages < 2:
        raise ValueError("边缘行范围必须大于零，重复页数阈值至少为二")
    approved = {" ".join(header.split()) for header in confirmed_headers}
    header_page_counts = Counter()
    # 每页同一页眉只计一次，避免同页重复被误认为跨页重复。
    for page in pages:
        nonempty_lines = [line for line in page["text"].splitlines() if line.strip()]
        top_texts = {" ".join(line.split()) for line in nonempty_lines[:edge_line_count]}
        header_page_counts.update(top_texts & approved)

    cleaned_blocks, removed_records = [], []
    for page in pages:
        lines = page["text"].splitlines()
        nonempty_indexes = [index for index, line in enumerate(lines) if line.strip()]
        top_indexes = set(nonempty_indexes[:edge_line_count])
        bottom_indexes = set(nonempty_indexes[-edge_line_count:])
        retained_lines = []
        for index, line in enumerate(lines):
            normalized = " ".join(line.split())
            reason = None
            if (index in top_indexes and normalized in approved
                    and header_page_counts[normalized] >= minimum_header_pages):
                reason = "confirmed_repeated_header"
            # 不删除裸数字，也不把“运动安排见第2页”当作页码。
            page_marker = re.fullmatch(r"第\s*(\d+)\s*页", normalized)
            if (reason is None and index in bottom_indexes and page_marker
                    and int(page_marker.group(1)) == page["page_number"]):
                reason = "explicit_page_marker"
            if reason is None:
                retained_lines.append(line)
            else:
                removed_records.append({
                    "source": source, "page_number": page["page_number"],
                    "extracted_line_index": index, "original_text": line, "reason": reason,
                })
        cleaned_blocks.append({
            "text": "\n".join(retained_lines),
            "locations": [{"source": source, "page_number": page["page_number"]}],
        })
    return {
        "original_pages": deepcopy(pages), "cleaned_blocks": cleaned_blocks,
        "removed_records": removed_records,
        "header_page_counts": dict(header_page_counts),
        "notice": "页面未合并；extracted_line_index 仅用于定位抽取文本，不是 PDF 原始行号。",
    }


def run_document_cleaning_demo() -> dict[str, Any]:
    """
    功能：展示重复页眉、独立页码、正文同名文字和普通数字的不同处理。
    参数含义：无参数；模拟页面与已确认页眉由人工提供，不读取真实文件。
    返回值含义：返回含原文、清洗结果及删除原因的报告；不含检索、模型调用或跨页合并。
    """
    pages = [
        {"page_number": 1, "text": "犬类护理资料——内部整理版\n运动安排\n"
         "教学占位：建议应结合年龄与健康状况，\n第 1 页"},
        {"page_number": 2, "text": "犬类护理资料——内部整理版\n"
         "教学占位：并保留与建议相关的限制条件。\n"
         "正文引用标题：\n犬类护理资料——内部整理版\n第 2 页"},
        {"page_number": 3, "text": "犬类护理资料——内部整理版\n训练记录\n"
         "教学占位：运动安排见第2页。\n2026\n第 3 页"},
    ]
    return clean_confirmed_page_noise(
        source="synthetic_teaching_pages", pages=pages,
        confirmed_headers={"犬类护理资料——内部整理版"},
    )


if __name__ == "__main__":
    print(json.dumps(run_document_cleaning_demo(), ensure_ascii=False, indent=2))
