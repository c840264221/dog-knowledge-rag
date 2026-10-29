"""逐条距离排序的确定性测试，不初始化模型或数据库。"""

import pytest

from src.rag.evaluation.exact_vector_ranking import exact_l2_ranking


def sample_records():
    """准备乱序编号和两种犬种向量；无参数；返回独立记录字典。"""
    return {"ids": ["b", "a", "c"], "embeddings": [[1, 0], [0, 0], [0, 2]],
            "metadatas": [{"dog_name": "Other"}, {"dog_name": "Golden Retriever"},
                          {"dog_name": "Golden Retriever"}]}


def test_all_records_and_squared_distance():
    """验证逐条计算并使用平方距离排序；无参数，无返回值。"""
    ranked = exact_l2_ranking([0, 0], sample_records())
    assert [item["chunk_id"] for item in ranked] == ["a", "b", "c"]
    assert [item["distance"] for item in ranked] == [0, 1, 4]


def test_filter_and_stable_ties():
    """验证主体等值过滤及同距离稳定排序；无参数，无返回值。"""
    ranked = exact_l2_ranking([0, 1], sample_records(), dog_name="Golden Retriever")
    assert [item["chunk_id"] for item in ranked] == ["a", "c"]
    assert ranked[0]["distance"] == ranked[1]["distance"] == 1
    assert exact_l2_ranking([0, 0], sample_records(), dog_name="Missing") == []


def test_invalid_vector_and_columns():
    """验证维度错误、非有限值及列不齐都报错；无参数，无返回值。"""
    for vector in ([], [1], [float("nan"), 0]):
        with pytest.raises(ValueError):
            exact_l2_ranking(vector, sample_records())
    records = sample_records()
    records["metadatas"] = []
    with pytest.raises(ValueError):
        exact_l2_ranking([0, 0], records)
