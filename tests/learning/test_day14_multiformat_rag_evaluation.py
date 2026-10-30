import json
from typing import Any

from src.rag.schemas import RagChunk

from tests.learning.test_day13_rag_pipeline_adapter import (
    build_index_safe_rag_chunks,
    build_multiformat_rag_index_preview,
)

def extract_rag_chunk_locations(
    chunk: RagChunk,
) -> list[dict[str, Any]]:
    """
    从 RagChunk metadata 中读取并统一返回结构化来源位置。

    功能：
        读取 chunk.metadata 中的 locations 字段。

        原始 RagChunk 的 locations 通常是 list[dict[str, Any]]，
        可以直接逐项复制后返回。

        索引安全 RagChunk 的 locations 是 JSON 字符串，
        需要先使用 json.loads 还原，再检查还原结果必须是字典列表。

        如果 metadata 中不存在 locations，返回空列表。

        本函数只读取和转换来源位置，不修改传入的 RagChunk，
        也不修改其原始 metadata。

    参数：
        chunk:
            需要读取来源位置的正式 RagChunk。
            可以是保留结构化 metadata 的原始 Chunk，
            也可以是经过 Day 13 索引安全转换的 Chunk。

    返回值：
        list[dict[str, Any]]:
            统一后的结构化来源位置列表。

            PDF 示例：
            [
                {
                    "source": "docs/dogs.pdf",
                    "page_number": 2,
                }
            ]

            Excel 示例：
            [
                {
                    "source": "docs/dogs.xlsx",
                    "sheet_name": "犬种资料",
                    "row_start": 2,
                    "row_end": 2,
                }
            ]

            当 locations 不存在时返回空列表。

    异常：
        ValueError:
            当 locations 是 JSON 字符串，但字符串内容不是合法 JSON，
            或 JSON 还原结果不是列表时抛出。

        TypeError:
            当 locations 既不是列表也不是字符串，
            或列表中的某一项不是字典时抛出。
    """
    raw_locations = chunk.metadata.get("locations")
    normalized_locations: list[dict[str, Any]] = []
    locations_value: Any
    if raw_locations is None:
        return []

    if isinstance(raw_locations, str):
        stripped_locations = raw_locations.strip()
        if not stripped_locations:
            return []
        try:
            locations_value  = json.loads(raw_locations.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(
                "locations 不是合法的 JSON 字符串"
            ) from exc

    elif isinstance(raw_locations, list):
        locations_value = raw_locations
    else:
        raise TypeError(
            f"locations 必须是列表或 JSON 字符串，"
            f"实际类型是 {type(raw_locations).__name__}"
        )

    if not isinstance(locations_value, list):
        raise ValueError("locations 的 JSON 还原结果必须是列表")

    for location_index, location in enumerate(locations_value):
        if not isinstance(location, dict):
            raise TypeError(
                f"locations 第 {location_index} 项必须是字典，"
                f"实际类型是 {type(location).__name__}"
            )
        normalized_locations.append(dict(location))

    return normalized_locations

def evaluate_rag_chunk_locations(
    *,
    chunk: RagChunk,
    expected_locations: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    评估一个 RagChunk 是否完整保留预期来源位置。

    功能：
        将 expected_locations 作为当前 Chunk 的黄金标准来源位置。

        调用 extract_rag_chunk_locations，从实际 RagChunk 中读取统一格式的
        actual_locations。

        当前 MVP 使用严格比较：
        实际来源位置与预期来源位置的数量、顺序和每个字典的内容都必须一致，
        才认为评估通过。

        返回结果中同时保存预期位置和实际位置，
        方便评估失败时查看具体差异。

        本函数只执行内存比较，不修改 RagChunk，
        不调用 Retriever、Embedding 或向量数据库。

    参数：
        chunk:
            需要评估来源位置完整性的正式 RagChunk。
            可以是原始 RagChunk，也可以是索引安全 RagChunk。

        expected_locations:
            当前 Chunk 应当保留的黄金标准来源位置列表。
            列表中的每一项都必须是字典，排列顺序表示原文顺序。

    返回值：
        dict[str, Any]:
            单个 Chunk 的来源位置评估结果，格式如下：

            {
                "chunk_id": "PDF-DOC-chunk-0",
                "passed": True,
                "expected_locations": [
                    {
                        "source": "docs/dogs.pdf",
                        "page_number": 1,
                    }
                ],
                "actual_locations": [
                    {
                        "source": "docs/dogs.pdf",
                        "page_number": 1,
                    }
                ],
            }

            passed 为 True：
                实际来源位置与黄金标准完全一致。

            passed 为 False：
                数量、顺序或某个字段存在差异。

    异常：
        TypeError:
            当 expected_locations 中的某一项不是字典时抛出。

        ValueError:
            当 Chunk 中的 locations JSON 不合法，
            或 JSON 还原结果不是列表时，
            由 extract_rag_chunk_locations 抛出。
    """
    normalized_expected_locations: list[dict[str, Any]] = []

    for location_index, expected_location in enumerate(expected_locations):
        if not isinstance(expected_location, dict):
            raise TypeError(
                f"locations 第 {location_index} 项必须是字典，"
                f"实际类型是 {type(expected_location).__name__}"
            )

        normalized_expected_locations.append(dict(expected_location))

    actual_locations = extract_rag_chunk_locations(chunk)

    passed = actual_locations == normalized_expected_locations

    evaluation_result = {
        "chunk_id": chunk.chunk_id,
        "passed": passed,
        "expected_locations": normalized_expected_locations,
        "actual_locations": actual_locations,
    }
    return evaluation_result

def build_minimal_multiformat_location_golden_set(
) -> dict[str, list[dict[str, Any]]]:
    """
    创建 Day 14 多格式来源位置评估的最小黄金数据集。

    功能：
        为 PDF、CSV 和 Excel 各选择一个具有代表性的 RagChunk，
        记录其预期保留的结构化来源位置。

        字典的键是 Chunk 唯一编号，
        字典的值是该 Chunk 应当保留的来源位置列表。

        PDF 用例验证跨页 Chunk 能同时保留第 1、2 页。

        CSV 用例验证原文件中间存在空行时，
        有效数据仍然保留真实原始行号 6，而不是重新编号后的行号。

        Excel 用例验证跨工作表解析后，
        Chunk 能保留“护理计划”工作表名称及原始第 2 行。

        本函数只定义黄金标准数据，
        不读取真实文件、不执行转换，也不运行评估。

    参数：
        无。

    返回值：
        dict[str, list[dict[str, Any]]]:
            以 chunk_id 为键、预期来源位置列表为值的黄金数据集。

            数据集包含三个 Chunk：
            - PDF-day10_text_sample-chunk-0
            - CSV-day12_dog_breeds-CHUNK-2
            - EXCEL-day12_dog_tables-CHUNK-3
    """
    golden_set = {
        "PDF-day10_text_sample-chunk-0": [
            {
                "source": "output/pdf/day10_text_sample.pdf",
                "page_number": 1,
            },
            {
                "source": "output/pdf/day10_text_sample.pdf",
                "page_number": 2,
            },
        ],
        "CSV-day12_dog_breeds-CHUNK-2": [
            {
                "source": "output/csv/day12_dog_breeds.csv",
                "row_start": 6,
                "row_end": 6,
            },
        ],
        "EXCEL-day12_dog_tables-CHUNK-3": [
            {
                "source": "output/excel/day12_dog_tables.xlsx",
                "sheet_name": "护理计划",
                "row_start": 2,
                "row_end": 2,
            },
        ],
    }

    return golden_set

def build_rag_chunk_lookup(
    chunks: list[RagChunk],
) -> dict[str, RagChunk]:
    """
    根据 RagChunk.chunk_id 创建唯一 Chunk 查找表。

    功能：
        按照传入顺序遍历 RagChunk 列表，
        使用每个 Chunk 的 chunk_id 作为字典键，
        使用 RagChunk 对象本身作为字典值。

        创建查找表后，评估代码可以根据黄金数据中的 chunk_id，
        直接找到对应的实际 RagChunk。

        chunk_id 会先清理首尾空格。
        如果清理后为空，说明该 Chunk 无法被稳定定位，应当抛出异常。

        如果多个 Chunk 使用相同的 chunk_id，也应当抛出异常，
        避免后加入的 Chunk 静默覆盖前一个 Chunk。

        本函数只创建新的字典，不修改传入列表或 RagChunk。

    参数：
        chunks:
            准备参与来源位置评估的 RagChunk 列表。
            列表可以为空。

    返回值：
        dict[str, RagChunk]:
            以清理后的 chunk_id 为键、RagChunk 对象为值的查找表。

            如果 chunks 为空，则返回空字典。

    异常：
        ValueError:
            当某个 Chunk 的 chunk_id 为空，
            或者列表中存在重复 chunk_id 时抛出。
    """
    chunk_lookup: dict[str, RagChunk] = {}
    for chunk in chunks:
        normalized_chunk_id = chunk.chunk_id.strip()
        if not normalized_chunk_id:
            raise ValueError("chunk_id 为空")
        if normalized_chunk_id in chunk_lookup:
            raise ValueError(f"编号为{normalized_chunk_id}的chunk重复了")
        chunk_lookup[normalized_chunk_id] = chunk
    return chunk_lookup


def evaluate_multiformat_location_golden_set(
    *,
    chunks: list[RagChunk],
    golden_set: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    使用最小黄金数据集批量评估多格式 RagChunk 的来源位置完整性。

    功能：
        首先调用 build_rag_chunk_lookup，
        根据实际 RagChunk 列表创建 chunk_id 查找表。

        随后按照黄金数据集的顺序逐条处理：
        根据黄金数据中的 chunk_id 查找实际 RagChunk。

        找到实际 Chunk 时，调用 evaluate_rag_chunk_locations，
        比较预期来源位置和实际来源位置。

        找不到实际 Chunk 时，将该编号记录到 missing_chunk_ids，
        并把该条黄金用例计为失败。

        最后统计总用例数、通过数、失败数和是否全部通过。

        本函数只执行内存评估，
        不读取文件、不调用 Retriever、Embedding 或向量数据库。

    参数：
        chunks:
            准备参与评估的实际 RagChunk 列表。
            可以包含黄金数据集未选中的其他 Chunk；
            这些额外 Chunk 不计为失败。

        golden_set:
            来源位置黄金数据集。
            键是需要评估的 chunk_id，
            值是该 Chunk 的预期来源位置列表。
            数据集不能为空。

    返回值：
        dict[str, Any]:
            批量来源位置评估结果，格式如下：

            {
                "total_cases": 3,
                "passed_cases": 2,
                "failed_cases": 1,
                "all_passed": False,
                "missing_chunk_ids": [],
                "results": [
                    {
                        "chunk_id": "PDF-...",
                        "passed": True,
                        "expected_locations": [...],
                        "actual_locations": [...],
                    },
                    {
                        "chunk_id": "EXCEL-...",
                        "passed": False,
                        "expected_locations": [...],
                        "actual_locations": [...],
                    },
                ],
            }

            results 只包含成功找到实际 Chunk 并完成比较的用例。
            未找到的 Chunk 单独记录在 missing_chunk_ids 中，
            但仍会计入 failed_cases。

    异常：
        ValueError:
            当 golden_set 为空，
            或实际 Chunk 存在空编号、重复编号时抛出。

        TypeError:
            当黄金数据中的预期来源位置格式不正确时，
            由 evaluate_rag_chunk_locations 抛出。
    """
    if not golden_set:
        raise ValueError("黄金集为空")

    chunk_lookup = build_rag_chunk_lookup(chunks)

    evaluation_results: list[dict[str, Any]] = []

    missing_chunk_ids: list[str] = []

    for golden_chunk_id, expected_locations in golden_set.items():
        normalized_chunk_id = golden_chunk_id.strip()
        actual_chunk = chunk_lookup.get(normalized_chunk_id)
        if actual_chunk is None:
            missing_chunk_ids.append(normalized_chunk_id)
        else:
            evaluation_result = evaluate_rag_chunk_locations(
                chunk=actual_chunk,
                expected_locations=expected_locations,
            )
            evaluation_results.append(evaluation_result)
    total_cases = len(golden_set)
    passed_cases = len(
        [
            passed_result
            for passed_result in evaluation_results
            if passed_result["passed"]
        ]
    )
    failed_cases = total_cases - passed_cases
    all_passed = failed_cases == 0

    batch_result = {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "missing_chunk_ids": missing_chunk_ids,
        "all_passed": all_passed,
        "results": evaluation_results,
    }
    return batch_result

def build_rag_chunks_from_index_preview(
    preview_result: dict[str, Any],
) -> list[RagChunk]:
    """
    从 Day 13 的入库前预览结果中恢复正式 RagChunk 列表。

    功能：
        读取 preview_result 中的 chunks 字段。

        检查 chunks 必须是列表，
        并检查列表中的每一项都必须是普通字典。

        将每个 Chunk 预览字典作为参数传给正式 RagChunk，
        由 Pydantic 模型检查必填字段和字段类型，
        最终生成可以交给 Day 14 评估函数使用的 RagChunk 列表。

        本函数只进行内存模型转换，
        不读取原始文件、不调用 Embedding，也不连接向量数据库。

    参数：
        preview_result:
            Day 13 的 build_multiformat_rag_index_preview
            返回的入库前预览字典。

            其中必须包含 chunks 字段；
            chunks 应当是由 Chunk 预览字典组成的列表。

    返回值：
        list[RagChunk]:
            按照 preview_result["chunks"] 原有顺序恢复的正式 RagChunk 列表。

            当 chunks 是空列表时，返回空列表。

    异常：
        TypeError:
            当 preview_result["chunks"] 不是列表，
            或列表中的某一项不是字典时抛出。

        ValidationError:
            当某个 Chunk 预览缺少 RagChunk 必填字段，
            或字段类型不符合 RagChunk 模型要求时，
            由 Pydantic 自动抛出。
    """
    raw_chunk_previews = preview_result.get("chunks")
    if not isinstance(raw_chunk_previews, list):
        raise TypeError(f"chunks不是列表，类型为：{type(raw_chunk_previews).__name__}")

    rag_chunks: list[RagChunk] = []

    for chunk_index, chunk_preview in enumerate(raw_chunk_previews):
        if not isinstance(chunk_preview, dict):
            raise TypeError(f"chunk_preview不是字典，"
                            f"实际类型为：{type(chunk_preview).__name__}"
                            f"下标为：{chunk_index}")

        rag_chunk = RagChunk(**chunk_preview)
        rag_chunks.append(rag_chunk)

    return rag_chunks

def evaluate_multiformat_files(
    *,
    file_paths: list[str],
    golden_set: dict[str, list[dict[str, Any]]],
    pdf_max_characters: int = 1000,
) -> dict[str, Any]:
    """
    读取多格式文件，并使用黄金数据集评估来源位置是否完整保留。

    功能：
        依次调用 Day 13 的统一预览入口处理每个文件，
        将预览字典恢复成 RagChunk，汇总所有文件的 Chunk，
        最后调用 evaluate_multiformat_location_golden_set 完成批量评估。

        每份文件只调用一次统一预览入口。
        本函数不修改黄金数据，不连接向量数据库。

    参数：
        file_paths:
            待评估的 PDF、CSV 或 .xlsx 文件路径列表，不允许为空。

        golden_set:
            人工确认的来源位置黄金数据集，不允许为空。
            键为 Chunk 编号，值为预期来源位置列表。

        pdf_max_characters:
            PDF 切块的最大字符数量，默认 1000。
            应与建立 PDF 黄金数据时的切块配置保持一致；
            CSV 和 Excel 不使用该参数。

    返回值：
        dict[str, Any]:
            批量来源评估结果，包含：
            total_cases、passed_cases、failed_cases、all_passed、
            missing_chunk_ids 和 results。

            总用例数由 golden_set 决定，不等于实际 Chunk 总数。

    异常：
        ValueError:
            当文件路径列表或黄金数据集为空时抛出。
            下游格式识别、切块、重复编号检查产生的异常继续向上传递。

        TypeError:
            当预览结构或来源位置的数据类型不正确时，
            由对应转换或评估函数抛出。

        FileNotFoundError:
            当文件不存在时，由具体解析器抛出。
    """
    if not golden_set:
        raise ValueError("黄金评估集不能为空")

    if not file_paths:
        raise ValueError("文档路径不能为空")

    all_chunks: list[RagChunk] = []

    for file_path in file_paths:
        preview_result = build_multiformat_rag_index_preview(
            file_path=file_path,
            pdf_max_characters=pdf_max_characters,
        )
        file_chunks = build_rag_chunks_from_index_preview(
            preview_result=preview_result,
        )
        all_chunks.extend(file_chunks)

    evaluation_result = evaluate_multiformat_location_golden_set(
        chunks=all_chunks,
        golden_set=golden_set,
    )
    return evaluation_result


def evaluate_location_serialization_round_trip(
    *,
    chunks: list[RagChunk],
    golden_set: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    使用同一份黄金数据评估 JSON 序列化前后的来源位置完整性。

    功能：
        先对原始 RagChunk 列表执行来源位置评估。

        再调用 Day 13 的 build_index_safe_rag_chunks，
        创建 metadata 中复杂结构已转换成 JSON 字符串的新 Chunk 列表，
        然后使用同一份黄金数据再次评估。

        第二次评估通过 extract_rag_chunk_locations 自动还原 JSON 字符串，
        从而检查序列化和反序列化后，来源位置是否仍符合黄金标准。

        只有转换前和转换后的评估都全部通过，
        本函数的 all_passed 才为 True。

        本函数只处理内存对象，不修改原始 Chunk 或黄金数据，
        不读取文件、不调用向量化服务或向量数据库。

    参数：
        chunks:
            包含原始结构化 metadata 的 RagChunk 列表。

        golden_set:
            人工确认的来源位置黄金数据集，不允许为空。
            同一份黄金数据用于转换前和转换后的两次评估。

    返回值：
        dict[str, Any]:
            包含以下字段的结果字典：

            original_result:
                原始 Chunk 的批量评估结果。

            safe_result:
                索引安全 Chunk 经 JSON 还原后的批量评估结果。

            all_passed:
                两次评估的 all_passed 均为 True 时为 True，
                否则为 False。

    异常：
        ValueError:
            当黄金数据集为空、Chunk 编号不合法，
            或来源位置 JSON 不合法时，由下游函数抛出。

        TypeError:
            当 metadata 包含不支持的类型，
            或来源位置结构不符合要求时，由下游函数抛出。
    """
    original_result = evaluate_multiformat_location_golden_set(
        chunks=chunks,
        golden_set=golden_set,
    )
    safe_chunks = build_index_safe_rag_chunks(
        chunks=chunks,
    )

    safe_result = evaluate_multiformat_location_golden_set(
        chunks=safe_chunks,
        golden_set=golden_set,
    )
    all_passed = (
            original_result["all_passed"]
            and safe_result["all_passed"]
    )
    return {
        "original_result": original_result,
        "safe_result": safe_result,
        "all_passed": all_passed,
    }

def evaluate_pdf_missing_page_case(
    *,
    chunks: list[RagChunk],
    golden_set: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    构造 PDF 丢失第 2 页来源位置的反向用例，并返回评估结果。

    功能：
        从实际 Chunk 列表中找到 PDF-day10_text_sample-chunk-0。

        深复制该 Chunk，读取其结构化来源位置，
        从副本的 locations 中移除 page_number 为 2 的位置记录。

        使用未修改的黄金标准评估这个副本，
        检查评估器能否发现来源位置缺失。

        本函数只移除副本中的来源位置记录，不修改正文，
        不修改原始 Chunk、原列表或黄金数据。

    参数：
        chunks:
            从真实样例文件生成的 RagChunk 列表，
            应包含目标 PDF Chunk。

        golden_set:
            人工确认的来源位置黄金数据集，
            应包含目标 PDF Chunk 的第 1、2 页预期位置。

    返回值：
        dict[str, Any]:
            evaluate_rag_chunk_locations 返回的单条评估结果，
            包含 chunk_id、passed、expected_locations 和 actual_locations。

            对当前样例，预期 passed 为 False：
            expected_locations 包含第 1、2 页，
            actual_locations 只包含第 1 页。

    异常：
        ValueError:
            当目标 Chunk 或对应黄金标准不存在，
            或实际来源位置中不存在第 2 页时抛出，
            避免没有真正构造缺页情况却继续评估。
    """
    target_chunk_id = "PDF-day10_text_sample-chunk-0"
    chunk_lookup = build_rag_chunk_lookup(
        chunks=chunks,
    )
    if not chunk_lookup:
        raise ValueError("找不到对应chunk")

    expected_locations = golden_set.get(target_chunk_id, None)
    if expected_locations is None:
        raise ValueError(f"黄金集中没有{target_chunk_id}对应的数据")

    original_chunk = chunk_lookup.get(target_chunk_id)

    if original_chunk is None:
        raise ValueError(
            f"原 chunks 中没有编号为 {target_chunk_id} 的 Chunk"
        )

    broken_chunk = original_chunk.model_copy(deep=True)

    remaining_locations: list[dict[str, Any]] = []

    original_locations = extract_rag_chunk_locations(broken_chunk)

    for location in original_locations:
        if location.get("page_number") != 2:
            remaining_locations.append(location)

    if remaining_locations == original_locations:
        raise ValueError("原数据中没有页码为2的数据")

    broken_chunk.metadata["locations"] = remaining_locations

    evaluation_result = evaluate_rag_chunk_locations(
        chunk=broken_chunk,
        expected_locations=expected_locations,
    )

    return evaluation_result

def evaluate_missing_chunk_case(
    *,
    chunks: list[RagChunk],
    golden_set: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    构造整个 Excel Chunk 缺失的反向用例，并返回批量评估结果。

    功能：
        确认实际 Chunk 列表和黄金数据集中都存在
        EXCEL-day12_dog_tables-CHUNK-3。

        创建新的 Chunk 列表，从中排除该目标 Chunk，
        然后使用未修改的黄金数据集执行批量来源位置评估。

        用于检查评估器是否把缺失 Chunk 计为失败，
        并在 missing_chunk_ids 中记录缺失编号。

        本函数不修改原始列表、Chunk 对象或黄金数据，
        不读取文件，也不调用向量数据库。

    参数：
        chunks:
            包含目标 Excel Chunk 的实际 RagChunk 列表。

        golden_set:
            包含目标 Excel Chunk 的来源位置黄金数据集。

    返回值：
        dict[str, Any]:
            evaluate_multiformat_location_golden_set 返回的批量结果。

            对当前三个黄金用例，预期：
            total_cases 为 3；
            passed_cases 为 2；
            failed_cases 为 1；
            all_passed 为 False；
            missing_chunk_ids 包含目标 Excel Chunk 编号。

    异常：
        ValueError:
            当目标 Chunk 在实际列表或黄金数据集中不存在时抛出，
            避免没有真正移除目标就继续评估。
            实际 Chunk 的空编号和重复编号由查找表构建函数检查。
    """
    target_chunk_id = "EXCEL-day12_dog_tables-CHUNK-3"

    chunk_lookup = build_rag_chunk_lookup(
        chunks=chunks,
    )
    if target_chunk_id not in chunk_lookup or target_chunk_id not in golden_set:
        raise ValueError("目标chunk不在chunks中或者不在黄金集中")

    remaining_chunks: list[RagChunk] = []

    for chunk in chunks:
        if chunk.chunk_id.strip() != target_chunk_id:
            remaining_chunks.append(chunk)

    evaluation_result = evaluate_multiformat_location_golden_set(
        chunks=remaining_chunks,
        golden_set=golden_set,
    )

    return evaluation_result

def verify_day14_evaluation_behaviors(
    *,
    round_trip_result: dict[str, Any],
    missing_page_result: dict[str, Any],
    missing_chunk_result: dict[str, Any],
) -> dict[str, bool]:
    """
    汇总并验证 Day14 多格式来源位置评估器的关键行为。

    功能：
        检查正常数据在 JSON 索引安全转换前后是否都通过评估。

        检查 PDF 缺少第 2 页来源位置时，
        评估器是否成功将该 Chunk 判定为不通过。

        检查 Excel Chunk 整体缺失时，
        批量评估是否判定为不通过，
        并在 missing_chunk_ids 中记录目标 Chunk 编号。

        最后汇总三个检查结果。
        只有正常数据通过、PDF 缺页被发现、
        Excel Chunk 缺失也被发现时，
        all_expected_behaviors 才为 True。

        本函数只读取已有评估结果，
        不重新处理文件、不修改传入字典，
        也不调用 Embedding、Retriever 或向量数据库。

    参数：
        round_trip_result:
            evaluate_location_serialization_round_trip 返回的结果。
            其中 all_passed 应当为 True。

        missing_page_result:
            evaluate_pdf_missing_page_case 返回的单条评估结果。
            因为它是故意构造的错误数据，
            所以其中 passed 应当为 False。

        missing_chunk_result:
            evaluate_missing_chunk_case 返回的批量评估结果。
            其中 all_passed 应当为 False，
            missing_chunk_ids 应当包含目标 Excel Chunk 编号。

    返回值：
        dict[str, bool]:
            Day14 关键行为验收结果，格式如下：

            {
                "round_trip_preserved": True,
                "missing_page_detected": True,
                "missing_chunk_detected": True,
                "all_expected_behaviors": True,
            }

            round_trip_preserved：
                JSON 转换前后的来源位置都通过评估。

            missing_page_detected：
                PDF 第 2 页来源位置缺失被成功发现。

            missing_chunk_detected：
                Excel Chunk 整体缺失被成功发现并记录。

            all_expected_behaviors：
                上述三个行为全部符合预期。
    """
    target_missing_chunk_id = "EXCEL-day12_dog_tables-CHUNK-3"

    round_trip_preserved = round_trip_result["all_passed"]

    missing_page_detected = missing_page_result["passed"] is False

    missing_chunk_ids = missing_chunk_result["missing_chunk_ids"]

    missing_chunk_detected = (target_missing_chunk_id in missing_chunk_ids and
                              not missing_chunk_result["all_passed"])

    all_expected_behaviors = all([round_trip_preserved, missing_page_detected, missing_chunk_detected])

    return {
        "all_expected_behaviors": all_expected_behaviors,
        "round_trip_preserved": round_trip_preserved,
        "missing_page_detected": missing_page_detected,
        "missing_chunk_detected": missing_chunk_detected,
    }



if __name__ == "__main__":
    file_paths = [
        "output/pdf/day10_text_sample.pdf",
        "output/csv/day12_dog_breeds.csv",
        "output/excel/day12_dog_tables.xlsx",
    ]
    golden_set = build_minimal_multiformat_location_golden_set()
    all_chunks: list[RagChunk] = []

    for file_path in file_paths:
        preview_result = build_multiformat_rag_index_preview(
            file_path=file_path,
            pdf_max_characters=1000,
        )
        file_chunks = build_rag_chunks_from_index_preview(
            preview_result=preview_result,
        )
        all_chunks.extend(file_chunks)

    round_trip_result = evaluate_location_serialization_round_trip(
        chunks=all_chunks,
        golden_set=golden_set,
    )

    print(
        json.dumps(
            round_trip_result,
            ensure_ascii=False,
            indent=2,
        )
    )

    missing_page_result = evaluate_pdf_missing_page_case(
        chunks=all_chunks,
        golden_set=golden_set,
    )

    print("PDF 缺页反向用例：")
    print(
        json.dumps(
            missing_page_result,
            ensure_ascii=False,
            indent=2,
        )
    )

    original_pdf_chunk = build_rag_chunk_lookup(all_chunks)[
        "PDF-day10_text_sample-chunk-0"
    ]

    print(
        "原始 PDF 来源位置：",
        extract_rag_chunk_locations(original_pdf_chunk),
    )

    missing_chunk_result = evaluate_missing_chunk_case(
        chunks=all_chunks,
        golden_set=golden_set,
    )

    print("Excel Chunk 缺失反向用例：")
    print(
        json.dumps(
            missing_chunk_result,
            ensure_ascii=False,
            indent=2,
        )
    )

    original_chunk_lookup = build_rag_chunk_lookup(
        chunks=all_chunks,
    )

    print(
        "原始列表仍包含目标 Chunk：",
        "EXCEL-day12_dog_tables-CHUNK-3" in original_chunk_lookup,
    )

    behavior_summary = verify_day14_evaluation_behaviors(
        round_trip_result=round_trip_result,
        missing_page_result=missing_page_result,
        missing_chunk_result=missing_chunk_result,
    )

    print("Day14 关键行为验收结果：")
    print(
        json.dumps(
            behavior_summary,
            ensure_ascii=False,
            indent=2,
        )
    )


