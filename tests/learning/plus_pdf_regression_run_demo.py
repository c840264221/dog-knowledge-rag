"""真实运行PDF用例，再检查新报告；需要模型密钥，会产生真实调用费用。"""

import asyncio
import json
import os
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse
from uuid import uuid4

from src.settings import settings
from tests.learning.plus_pdf_chat_demo import run_pdf_chat_demo
from tests.learning.plus_pdf_regression_demo import check_pdf_case, read_json


async def run_pdf_regression(*, model_name: str, base_url: str, api_key: str) -> dict:
    """串行真实运行数据集，并对每个新运行执行离线规则检查。

    参数：model_name为回答模型名，base_url为服务地址，api_key为密钥。
    返回值：包含批次编号、配置快照、逐用例结果和通过数量的汇总字典。
        只把问题传入查询链路，不把预期答案传给模型；固定不发送图片。
        单个用例异常会记录并继续，不自动重试；取消整个程序则向上传播。
        耗时包含模型加载及报告检查，不代表线上单请求推理延迟。
    """
    dataset_path = Path(__file__).with_name("pdf_regression_cases.json")
    dataset = read_json(dataset_path)
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output/learning/pdf_pages/dogs_trust_body_language"
    batch_id = uuid4().hex
    batch_dir = output_dir / "regression_runs" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    # 保存本批实际使用的预期，避免日后修改用例文件后无法复盘。
    (batch_dir / "dataset_snapshot.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    summary = {
        "batch_id": batch_id,
        "evaluation_mode": "fresh_queries_then_rule_checks",
        "dataset_version": dataset["dataset_version"],
        "case_count": len(dataset["cases"]),
        "model": model_name,
        "base_url": base_url,
        "include_images": False,
        "results": [],
        "rule_checks_passed_count": 0,
        "execution_error_count": 0,
        "completed_case_count": 0,
        "batch_status": "running",
        "report_path": str(batch_dir / "summary.json"),
        "notice": "规则通过不代表内容审核通过；两个样本不能代表整体准确率。",
    }
    report_path = batch_dir / "summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"批次报告目录：{batch_dir}", flush=True)
    for index, case in enumerate(dataset["cases"], start=1):
        print(f"\n[{index}/{summary['case_count']}] 正在真实运行：{case['case_id']}", flush=True)
        started_at = perf_counter()
        run_id = None
        try:
            answer = await run_pdf_chat_demo(
                case["question"], model_name=model_name, base_url=base_url,
                api_key=api_key, include_images=False,
            )
            run_id = answer["run_id"]
            result = check_pdf_case(
                case=case, run_dir=output_dir / "query_runs" / run_id,
            )
            result["execution_status"] = "completed"
        except Exception as error:
            # 批量评估边界隔离单个用例故障；不把故障伪装成正常的资料不足。
            result = {
                "case_id": case["case_id"],
                "run_id": run_id,
                "execution_status": "error",
                "error_type": type(error).__name__,
                "rule_checks_passed": False,
                "manual_review_status": "pending",
                "manual_review": case["manual_review"],
            }
            # 不直接保存异常文本，避免第三方异常携带请求头或密钥。
        result["elapsed_seconds_including_startup"] = round(perf_counter() - started_at, 4)
        summary["results"].append(result)
        summary["completed_case_count"] = len(summary["results"])
        summary["rule_checks_passed_count"] += int(result["rule_checks_passed"])
        summary["execution_error_count"] += int(result["execution_status"] == "error")
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    summary["batch_status"] = "finished"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    model_name = os.environ.get("VISION_MODEL", "deepseek-flash").strip()
    base_url = os.environ.get("VISION_BASE_URL", "https://api.deepseek.com").strip()
    api_key = os.environ.get("VISION_API_KEY", "").strip()
    endpoint = urlparse(base_url)
    if not api_key and endpoint.scheme == "https" and endpoint.hostname == "api.deepseek.com" and endpoint.port in (None, 443):
        project_key = settings.llm.deepseek_api_key
        api_key = project_key.get_secret_value().strip() if project_key else ""
    if not api_key:
        raise SystemExit("未找到密钥，请配置VISION_API_KEY；DeepSeek官方接口可复用项目DEEPSEEK_API_KEY。")
    print("将真实运行两个用例，通常共调用两次回答模型；不自动重试、不发送图片。", flush=True)
    confirmation = input("输入y开始，其他输入退出：").strip().casefold()
    if confirmation != "y":
        raise SystemExit("已退出，未运行用例。")
    summary = asyncio.run(run_pdf_regression(
        model_name=model_name, base_url=base_url, api_key=api_key,
    ))
    print(f"\n规则检查通过：{summary['rule_checks_passed_count']}/{summary['case_count']}")
    print(f"执行异常数：{summary['execution_error_count']}")
    print(f"汇总报告：{summary['report_path']}")
    print("仍需人工复核答案；本次未与历史方案计算效果提升比例。")
    raise SystemExit(0 if summary["rule_checks_passed_count"] == summary["case_count"] else 1)
