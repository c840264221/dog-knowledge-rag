"""比较已存向量与重新计算的向量；仅提供数值诊断，不推断建库模型身份。"""

import math


def compare_vectors(stored, recomputed, *, absolute_tolerance=1e-5, relative_tolerance=1e-4) -> dict:
    """计算两个向量的一致性指标。

    参数：stored为库中向量，recomputed为相同正文的新向量；absolute_tolerance
        和relative_tolerance为逐维接近判断的绝对、相对容差，不是检索相关性阈值。
    返回值：维度、范数、余弦相似度、欧氏距离、最大绝对误差及接近判断组成的字典。
        维度不同时只返回维度不匹配；空向量、非有限数值或负容差直接报错。
    """
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("容差不能为负数。")
    left, right = [float(x) for x in stored], [float(x) for x in recomputed]
    if not left or not right or not all(math.isfinite(x) for x in left + right):
        raise ValueError("向量必须非空且全部为有限数值。")
    result = {"stored_dimension": len(left), "recomputed_dimension": len(right),
              "dimension_matches": len(left) == len(right)}
    if len(left) != len(right):
        return result
    left_norm = math.sqrt(sum(x * x for x in left))
    right_norm = math.sqrt(sum(x * x for x in right))
    errors = [abs(a - b) for a, b in zip(left, right)]
    cosine = (sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
              if left_norm and right_norm else None)
    result.update({
        "stored_norm": left_norm, "recomputed_norm": right_norm,
        "cosine_similarity": max(-1.0, min(1.0, cosine)) if cosine is not None else None,
        "l2_distance": math.sqrt(sum(error * error for error in errors)),
        "max_absolute_error": max(errors),
        "mean_absolute_error": sum(errors) / len(errors),
        "absolute_tolerance": absolute_tolerance, "relative_tolerance": relative_tolerance,
        "numerically_close": all(math.isclose(a, b, abs_tol=absolute_tolerance,
                                               rel_tol=relative_tolerance)
                                  for a, b in zip(left, right)),
    })
    return result
