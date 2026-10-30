"""离线检查两条PDF查询报告；不调用模型、不读取或修改向量库。"""

import json
from pathlib import Path
from uuid import uuid4


def read_json(path: Path) -> dict:
    """读取一个JSON报告；参数path为文件路径；返回解析字典，文件错误向上抛出。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_pdf_case(*, case: dict, run_dir: Path) -> dict:
    """核对指定用例的三份运行报告，独立检查来源映射。

    参数：case为用例预期；run_dir为一次完整查询的目录，不自动选择最新报告。
    返回值：包含逐项布尔检查、规则通过状态和人工复核清单的字典。
        预期答案只用于运行结束后的检查，不进入模型提示词；规则通过不代表
        语义审核通过，也不证明模型整体的正确率或拒答率。
    """
    retrieval = read_json(run_dir / "retrieval_report.json")
    context = read_json(run_dir / "answer_context.json")
    answer = read_json(run_dir / "pdf_answer_report.json")
    draft = answer["model_draft"]
    citation_check = answer["citation_check"]
    selected_ids = context["selected_chunk_ids"]
    evidence = context["evidence"]
    citation_map = context["citation_map"]
    resolved = citation_check["resolved_paragraphs"]

    # 不只相信报告中的passed，重新核对段落编号与程序来源映射。
    references_match = True
    for paragraph in resolved:
        ids = paragraph["citation_ids"]
        if not ids or len(ids) != len(set(ids)) or any(key not in citation_map for key in ids):
            references_match = False
            continue
        expected_sources = [
            {"citation_id": key, **citation_map[key]} for key in ids
        ]
        if paragraph["sources"] != expected_sources:
            references_match = False
    draft_paragraphs = draft["paragraphs"]
    paragraphs_match = len(resolved) == len(draft_paragraphs) and all(
        saved["text"] == original["text"]
        and saved["citation_ids"] == list(dict.fromkeys(original["citation_ids"]))
        for saved, original in zip(resolved, draft_paragraphs)
    )
    checks = {
        "同一运行编号": all(
            report["run_id"] == run_dir.name for report in (retrieval, context, answer)
        ),
        "三阶段问题与用例一致": all(
            report["question"] == case["question"] for report in (retrieval, context, answer)
        ),
        "回答状态符合预期": draft["status"] == case["expected_status"],
        "有入选资料且回答沿用这些块": bool(selected_ids)
        and selected_ids == answer["selected_chunk_ids"],
        "入选块来自本轮重排候选": set(selected_ids).issubset(
            {item["chunk_id"] for item in retrieval["reranked_candidates"]}
        ),
        "入选证据编号一致": selected_ids == [item["chunk_id"] for item in evidence],
        "预期图示与分组进入上下文": any(
            item["content"]["caption"] == case["expected_caption"]
            and item["content"]["section_heading"] == case["expected_section"]
            for item in evidence
        ),
        "引用映射指向入选块与预期页码": bool(citation_map) and all(
            source["chunk_id"] in selected_ids
            and source["page_number"] == case["expected_page"]
            for source in citation_map.values()
        ),
        "原报告引用检查通过": citation_check["passed"] is True,
        "段落与草稿一致且来源映射一致": references_match and paragraphs_match,
        "最终回答包含预期文字": all(
            text in citation_check["final_answer"] for text in case["answer_must_include"]
        ),
        "资料不足时明确填写限制": draft["status"] != "insufficient_evidence"
        or bool(draft["limitation"].strip()),
    }
    return {
        "case_id": case["case_id"],
        "run_id": run_dir.name,
        "checks": checks,
        "rule_checks_passed": all(checks.values()),
        "final_answer": citation_check["final_answer"],
        "manual_review_status": "pending",
        "manual_review": case["manual_review"],
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output/learning/pdf_pages/dogs_trust_body_language"
    cases = read_json(Path(__file__).with_name("pdf_regression_cases.json"))["cases"]
    results = []
    for case in cases:
        print(f"\n用例：{case['case_id']}\n问题：{case['question']}")
        run_id = input(
            f"输入待检查运行编号，回车检查已验证的基线 {case['baseline_run_id']}："
        ).strip() or case["baseline_run_id"]
        # 只接收运行编号，避免意外读取query_runs之外的目录。
        if len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id):
            raise SystemExit("运行编号应为32位小写十六进制字符串。")
        try:
            result = check_pdf_case(case=case, run_dir=output_dir / "query_runs" / run_id)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result = {
                "case_id": case["case_id"], "run_id": run_id,
                "rule_checks_passed": False, "report_error": str(error),
                "manual_review_status": "pending", "manual_review": case["manual_review"],
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    report_path = output_dir / f"regression_check_{uuid4().hex}.json"
    report_path.write_text(json.dumps({
        "dataset_version": 1,
        "evaluation_mode": "offline_saved_reports",
        "results": results,
        "rule_checks_passed_count": sum(item["rule_checks_passed"] for item in results),
        "case_count": len(results),
        "notice": "仅检查指定历史运行，不代表重新运行模型，也不代表语义评估通过。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n检查报告：{report_path}")
    print("本脚本未调用模型。请完成报告中的人工复核，不将规则通过比例当成回答准确率。")
    raise SystemExit(0 if all(item["rule_checks_passed"] for item in results) else 1)
