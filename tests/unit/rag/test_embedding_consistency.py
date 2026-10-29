"""向量数值比较测试，不加载模型、不连接数据库。"""

import pytest

from src.rag.evaluation.embedding_consistency import compare_vectors


def test_identical_vectors():
    """验证相同向量误差为零；无参数，无返回值。"""
    result = compare_vectors([1, 0], [1, 0])
    assert result["numerically_close"] is True
    assert result["l2_distance"] == 0
    assert result["cosine_similarity"] == 1


def test_same_direction_different_scale():
    """验证方向相同不等于数值一致；无参数，无返回值。"""
    result = compare_vectors([1, 0], [2, 0])
    assert result["cosine_similarity"] == 1
    assert result["numerically_close"] is False
    assert result["stored_norm"] != result["recomputed_norm"]


def test_dimension_mismatch():
    """验证不同维度不能计算成可比较向量；无参数，无返回值。"""
    assert compare_vectors([1], [1, 0])["dimension_matches"] is False


@pytest.mark.parametrize("vector", [[], [float("nan")], [float("inf")]])
def test_invalid_vector(vector):
    """验证空或非有限向量被拒绝；vector为非法样本；无返回值。"""
    with pytest.raises(ValueError):
        compare_vectors(vector, [1])


def test_zero_norm_has_no_cosine():
    """验证零向量没有可定义的余弦值；无参数，无返回值。"""
    assert compare_vectors([0, 0], [1, 0])["cosine_similarity"] is None
