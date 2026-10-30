"""PDF统一查询第二步：新问题检索、重排、构建上下文并生成带引用的答案。"""

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from src.runtime.container.core import RuntimeContainer
from src.settings import settings
from tests.learning.plus_pdf_answer_demo import generate_pdf_answer
from tests.learning.plus_pdf_context_demo import build_pdf_context_from_candidates
from tests.learning.plus_pdf_query_demo import run_pdf_query_demo
from tests.learning.plus_pdf_vision_demo import DemoVisionProvider


async def run_pdf_chat_demo(
    question: str, *, model_name: str, base_url: str, api_key: str,
    include_images: bool = False,
) -> dict:
    """将本轮真实检索结果直接交给上下文构建和回答函数。

    参数：question为本轮输入；model_name、base_url、api_key为模型连接配置；
        include_images为手动图片开关，默认只发送提取后的文字资料。
    返回值：本轮答案报告，包含运行编号、问题、入选块和引用检查。
        所有报告保存到同一运行目录；只查询已有库，不重新入库。
        当前只选重排第一块，适用于单图示演示，不具备通用多问题覆盖策略。
    """
    retrieval = await run_pdf_query_demo(question)
    run_dir = Path(retrieval["report_path"]).parent
    if not retrieval["reranked_candidates"]:
        report = {
            "run_id": retrieval["run_id"],
            "question": retrieval["question"],
            "status": "insufficient_evidence",
            "final_answer": "现有资料未召回候选，无法根据资料回答本次问题。",
        }
        (run_dir / "pdf_answer_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return report

    bundle = json.loads(Path(retrieval["bundle_path"]).read_text(encoding="utf-8-sig"))
    context = build_pdf_context_from_candidates(
        question=retrieval["question"], candidates=retrieval["reranked_candidates"],
        bundle=bundle, mode=retrieval["retrieval_mode"], top_k=1,
    )
    context["run_id"] = retrieval["run_id"]
    (run_dir / "answer_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"本轮入选块：{context['selected_chunk_ids']}", flush=True)
    print(f"发送页面图片：{include_images}；开始调用一次回答模型。", flush=True)

    container = RuntimeContainer()
    container.register("vision", DemoVisionProvider(
        model_name=model_name, base_url=base_url, api_key=api_key,
    ))
    try:
        await container.startup()
        return await generate_pdf_answer(
            context=context, provider=container.get("vision"), output_dir=run_dir,
            model_name=model_name, include_images=include_images,
        )
    finally:
        await container.shutdown()


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

    question = input("请输入本轮要回答的问题：")
    include_images = input("本次是否发送页面图片？输入y发送，直接回车只发文字：").strip().casefold() == "y"
    report = asyncio.run(run_pdf_chat_demo(
        question, model_name=model_name, base_url=base_url,
        api_key=api_key, include_images=include_images,
    ))
    print(f"\n运行编号：{report['run_id']}")
    print(f"本轮问题：{report['question']}")
    if "citation_check" in report:
        check = report["citation_check"]
        print(f"模型状态：{report['model_draft']['status']}；引用检查：{check['passed']}")
        print(check["final_answer"] if check["passed"] else "引用检查失败，不展示草稿：" + str(check["errors"]))
        print("\n程序解析的来源：")
        print(json.dumps(check["resolved_paragraphs"], ensure_ascii=False, indent=2))
    else:
        print(report["final_answer"])
    print(f"\n报告目录：output/learning/pdf_pages/dogs_trust_body_language/query_runs/{report['run_id']}")
    print("当前为单图示教学链路，引用检查通过不代表内容一定正确，请人工核对原文。")
