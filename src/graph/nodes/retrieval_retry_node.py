from typing import Any

from src.graph.states.state import DogState
from src.logger import logger
from src.runtime.context import runtime_ctx
from src.settings import settings


DEFAULT_TOP_K = 5


def build_retrieval_retry_node(
        checkpoint_provider=None,
):
    """
    构建 retrieval_retry_node 节点。

    功能：
        使用闭包方式注入 CheckpointProvider，避免节点内部直接 import container。
        该节点负责根据 retrieval_failure_type 调整下一轮 RAG 检索策略。

    技术名词：
        Retry：
            重试。当前表示 RAG 召回质量不足时，调整状态后重新进入 retrieve_node。

        Retry Policy：
            重试策略。根据失败原因选择不同的重试方式。

        Checkpoint：
            检查点。用于保存图执行过程中的中间状态。

        Provider：
            提供者。用于统一管理服务对象。

    参数：
        checkpoint_provider:
            CheckpointProvider 实例。
            中文释义：用于保存运行时 checkpoint。

    返回值：
        callable:
            返回一个 retrieval_retry_node 函数，供 LangGraph 注册使用。
    """

    def retry_node(
            state: DogState,
    ) -> dict[str, Any]:
        """
        执行检索重试节点。

        功能：
            调用 execute_retrieval_retry_node 执行核心重试逻辑。

        参数：
            state:
                当前 DogState。

        返回值：
            dict[str, Any]:
                返回需要合并进 DogState 的状态更新。
        """

        return execute_retrieval_retry_node(
            state=state,
            checkpoint_provider=checkpoint_provider,
        )

    return retry_node


def retrieval_retry_node(
        state: DogState,
) -> dict[str, Any]:
    """
    旧版兼容 retrieval_retry_node。

    功能：
        保留旧函数名，避免旧代码仍然直接导入 retrieval_retry_node 时立刻报错。

        注意：
            v1.5 新代码推荐使用 build_retrieval_retry_node 注入 checkpoint_provider。
            这个函数只是兼容入口，不建议新代码继续依赖它。

    参数：
        state:
            当前 DogState。

    返回值：
        dict[str, Any]:
            返回 retry_count、filters、top_k 等状态更新。
    """

    from src.runtime.container.init import container

    checkpoint_provider = container.get(
        "checkpoint"
    )

    return execute_retrieval_retry_node(
        state=state,
        checkpoint_provider=checkpoint_provider,
    )


def execute_retrieval_retry_node(
        state: DogState,
        checkpoint_provider=None,
) -> dict[str, Any]:
    """
    执行检索重试核心逻辑。

    功能：
        根据 evaluate_node 产出的 retrieval_failure_type，
        调整下一轮 RAG 检索策略。

    当前重试原则：
        1. empty：
           完全没有召回，优先放宽 filters。

        2. insufficient_context：
           有结果但上下文不足，优先扩大 top_k。

        3. section_mismatch：
           有结果但 section 不匹配，优先扩大 top_k。

        4. metadata_mismatch：
           召回实体不匹配，优先强化 dog_name filter。

        5. score_too_low：
           向量距离偏差，优先放宽 filters。

        6. low_quality / unknown：
           使用通用渐进式放宽策略。

    技术名词：
        Failure Type：
            失败类型。由 evaluate_node 根据召回质量评估得到。

        State Reset：
            状态重置。清空上一轮 rag_context / docs，避免下一轮误用旧结果。

        Core Filter：
            核心过滤条件。当前主要是 dog_name、size。

    参数：
        state:
            当前 DogState。

        checkpoint_provider:
            CheckpointProvider 实例，可选。

    返回值：
        dict[str, Any]:
            返回需要合并进 DogState 的字段。
    """

    runtime = runtime_ctx.get()

    runtime.state().set_node(
        "retrieval_retry_node"
    )

    runtime.timeline().add_event(
        event_type="node",
        name="retrieval_retry_node"
    )

    retry_count = resolve_retry_count(
        state.get(
            "retry_count",
            0
        )
    )

    failure_type = resolve_failure_type(
        state=state
    )

    filters = copy_filters(
        state.get(
            "filters",
            {}
        )
    )

    tags = list(
        state.get(
            "tags",
            []
        )
        or []
    )

    top_k = resolve_top_k(
        state.get(
            "top_k",
            DEFAULT_TOP_K
        )
    )

    logger.info(
        "进入 retrieval_retry_node 重试节点，"
        f"retry_count={retry_count}, "
        f"failure_type={failure_type}, "
        f"filters={filters}, "
        f"tags={tags}, "
        f"top_k={top_k}"
    )

    strategy_result = build_retry_strategy(
        state=state,
        retry_count=retry_count,
        failure_type=failure_type,
        filters=filters,
        tags=tags,
        top_k=top_k,
    )

    output_state = {
        "retry_count": retry_count + 1,
        "filters": strategy_result["filters"],
        "tags": strategy_result["tags"],
        "top_k": strategy_result["top_k"],
        "retrieval_retry_strategy": strategy_result["strategy"],

        # 清空上一轮召回结果，下一轮 retrieve_node 重新写入。
        "rag_query": None,
        "rag_context": None,
        "docs": [],

        # retry 后需要重新 retrieve 和 evaluate。
        "retrieval_ok": False,
        "retrieval_evaluated": False,
        "retrieval_quality": None,
        "retrieval_failure_type": None,
    }

    logger.info(
        "retrieval_retry_node 策略完成，"
        f"strategy={strategy_result['strategy']}, "
        f"next_filters={strategy_result['filters']}, "
        f"next_top_k={strategy_result['top_k']}"
    )

    logger.debug(
        f"retrieval_retry_node 即将返回 output_state={output_state}"
    )

    save_checkpoint_safely(
        checkpoint_provider=checkpoint_provider
    )

    return output_state


def build_retry_strategy(
        state: DogState,
        retry_count: int,
        failure_type: str,
        filters: dict[str, Any],
        tags: list[str],
        top_k: int,
) -> dict[str, Any]:
    """
    根据失败类型构建重试策略。

    功能：
        读取 retrieval_failure_type，并根据失败原因选择下一轮检索参数。

    参数：
        state:
            当前 DogState。

        retry_count:
            当前已经重试的次数。

        failure_type:
            失败类型。

        filters:
            当前 filters。

        tags:
            当前 tags。

        top_k:
            当前 top_k。

    返回值：
        dict[str, Any]:
            包含 strategy、filters、tags、top_k。
    """

    if failure_type == "empty":

        return build_empty_retry_strategy(
            retry_count=retry_count,
            filters=filters,
            tags=tags,
            top_k=top_k,
        )

    if failure_type == "insufficient_context":

        return build_insufficient_context_retry_strategy(
            retry_count=retry_count,
            filters=filters,
            tags=tags,
            top_k=top_k,
        )

    if failure_type == "section_mismatch":

        return build_section_mismatch_retry_strategy(
            retry_count=retry_count,
            filters=filters,
            tags=tags,
            top_k=top_k,
        )

    if failure_type == "metadata_mismatch":

        return build_metadata_mismatch_retry_strategy(
            state=state,
            retry_count=retry_count,
            filters=filters,
            tags=tags,
            top_k=top_k,
        )

    if failure_type == "score_too_low":

        return build_score_too_low_retry_strategy(
            retry_count=retry_count,
            filters=filters,
            tags=tags,
            top_k=top_k,
        )

    return build_low_quality_retry_strategy(
        retry_count=retry_count,
        filters=filters,
        tags=tags,
        top_k=top_k,
    )


def build_empty_retry_strategy(
        retry_count: int,
        filters: dict[str, Any],
        tags: list[str],
        top_k: int,
) -> dict[str, Any]:
    """
    构建 empty 失败类型的重试策略。

    功能：
        empty 表示完全没有召回结果。
        这种情况下单纯扩大 top_k 通常没有意义，
        应该优先放宽 filters。

    参数：
        retry_count:
            当前重试次数。

        filters:
            当前 filters。

        tags:
            当前 tags。

        top_k:
            当前 top_k。

    返回值：
        dict[str, Any]:
            重试策略结果。
    """

    if retry_count == 0:

        next_filters = relax_to_core_filters(
            filters=filters
        )

        if next_filters == filters:
            next_filters = {}

        return {
            "strategy": "empty_relax_filters",
            "filters": next_filters,
            "tags": [],
            "top_k": top_k,
        }

    return {
        "strategy": "empty_clear_filters",
        "filters": {},
        "tags": [],
        "top_k": increase_top_k(
            top_k=top_k,
            target=settings.rag.retry_second_top_k,
        ),
    }


def build_insufficient_context_retry_strategy(
        retry_count: int,
        filters: dict[str, Any],
        tags: list[str],
        top_k: int,
) -> dict[str, Any]:
    """
    构建 insufficient_context 失败类型的重试策略。

    功能：
        insufficient_context 表示有召回结果，但上下文长度不足。
        这种情况下扩大 top_k 是合理的，因为可能后续 chunk 能补足信息。

    参数：
        retry_count:
            当前重试次数。

        filters:
            当前 filters。

        tags:
            当前 tags。

        top_k:
            当前 top_k。

    返回值：
        dict[str, Any]:
            重试策略结果。
    """

    if retry_count == 0:

        return {
            "strategy": "insufficient_context_increase_top_k",
            "filters": filters,
            "tags": tags,
            "top_k": increase_top_k(
                top_k=top_k,
                target=settings.rag.retry_first_top_k,
            ),
        }

    return {
        "strategy": "insufficient_context_relax_core_filters",
        "filters": relax_to_core_filters(
            filters=filters
        ),
        "tags": [],
        "top_k": increase_top_k(
            top_k=top_k,
            target=settings.rag.retry_second_top_k,
        ),
    }


def build_section_mismatch_retry_strategy(
        retry_count: int,
        filters: dict[str, Any],
        tags: list[str],
        top_k: int,
) -> dict[str, Any]:
    """
    构建 section_mismatch 失败类型的重试策略。

    功能：
        section_mismatch 表示召回到了相关犬种，
        但没有召回到和问题主题匹配的 section。
        这种情况优先扩大 top_k，让后续 chunk 有机会包含正确 section。

    参数：
        retry_count:
            当前重试次数。

        filters:
            当前 filters。

        tags:
            当前 tags。

        top_k:
            当前 top_k。

    返回值：
        dict[str, Any]:
            重试策略结果。
    """

    if retry_count == 0:

        return {
            "strategy": "section_mismatch_increase_top_k",
            "filters": filters,
            "tags": tags,
            "top_k": increase_top_k(
                top_k=top_k,
                target=settings.rag.retry_first_top_k,
            ),
        }

    return {
        "strategy": "section_mismatch_relax_filters",
        "filters": relax_to_core_filters(
            filters=filters
        ),
        "tags": [],
        "top_k": increase_top_k(
            top_k=top_k,
            target=settings.rag.retry_second_top_k,
        ),
    }


def build_metadata_mismatch_retry_strategy(
        state: DogState,
        retry_count: int,
        filters: dict[str, Any],
        tags: list[str],
        top_k: int,
) -> dict[str, Any]:
    """
    构建 metadata_mismatch 失败类型的重试策略。

    功能：
        metadata_mismatch 表示召回结果没有命中期望 dog_name。

        v1.5 新版策略：
        1. 如果 state 中存在 dog_name，则构建标准 Chroma dog_name filter。
        2. 将 dog_name filter 和当前 filters 合并。
        3. 如果已经强化过仍失败，则清空 filters 做语义兜底。

    参数：
        state:
            当前 DogState。

        retry_count:
            当前重试次数。

        filters:
            当前 filters。

        tags:
            当前 tags。

        top_k:
            当前 top_k。

    返回值：
        dict[str, Any]:
            重试策略结果。
    """

    dog_name = state.get(
        "dog_name"
    )

    if retry_count == 0 and dog_name:

        dog_name_filter = build_eq_filter(
            field_name="dog_name",
            value=dog_name,
        )

        next_filters = merge_chroma_filters_for_retry(
            filters,
            dog_name_filter,
        )

        return {
            "strategy": "metadata_mismatch_enforce_dog_name",
            "filters": next_filters,
            "tags": tags,
            "top_k": top_k,
        }

    return {
        "strategy": "metadata_mismatch_clear_filters",
        "filters": {},
        "tags": [],
        "top_k": increase_top_k(
            top_k=top_k,
            target=settings.rag.retry_second_top_k,
        ),
    }

def build_eq_filter(
        field_name: str,
        value: Any,
) -> dict[str, Any]:
    """
    构建 Chroma $eq filter。

    功能：
        将字段名和值构建成标准 Chroma 等值过滤条件。

    示例：
        输入：
            field_name="dog_name"
            value="Shih Tzu"

        输出：
            {
                "dog_name": {
                    "$eq": "Shih Tzu"
                }
            }

    参数：
        field_name:
            metadata 字段名。

        value:
            目标值。

    返回值：
        dict[str, Any]:
            Chroma $eq filter。
    """

    return {
        field_name: {
            "$eq": value
        }
    }

def merge_chroma_filters_for_retry(
        left_filter: dict[str, Any],
        right_filter: dict[str, Any],
) -> dict[str, Any]:
    """
    合并两个 Chroma filters。

    功能：
        将两个 filter 合并成一个合法 Chroma where filter。

        当前用于 retry 场景：
        例如把当前 filters 和 dog_name filter 合并。

    规则：
        1. 如果两边都为空，返回 {}。
        2. 如果只有一边有值，返回有值的一边。
        3. 如果两边都有值，则拆成条件列表后用 $and 合并。
        4. 如果相同字段重复出现，后者覆盖前者。

    参数：
        left_filter:
            左侧 filter。

        right_filter:
            右侧 filter。

    返回值：
        dict[str, Any]:
            合并后的 Chroma filter。
    """

    left_conditions = extract_filter_conditions(
        filters=left_filter or {},
    )

    right_conditions = extract_filter_conditions(
        filters=right_filter or {},
    )

    merged_by_field: dict[str, dict[str, Any]] = {}

    for condition in left_conditions + right_conditions:

        field_name = get_filter_condition_field_name(
            condition=condition,
        )

        if not field_name:
            continue

        merged_by_field[
            field_name
        ] = condition

    return build_chroma_filter_from_conditions(
        conditions=list(
            merged_by_field.values()
        )
    )


def build_score_too_low_retry_strategy(
        retry_count: int,
        filters: dict[str, Any],
        tags: list[str],
        top_k: int,
) -> dict[str, Any]:
    """
    构建 score_too_low 失败类型的重试策略。

    功能：
        score_too_low 表示召回结果的向量距离偏大。
        这种情况下扩大 top_k 不一定有用，
        更常见策略是逐步放宽 filters，最后进入纯语义兜底。

    参数：
        retry_count:
            当前重试次数。

        filters:
            当前 filters。

        tags:
            当前 tags。

        top_k:
            当前 top_k。

    返回值：
        dict[str, Any]:
            重试策略结果。
    """

    if retry_count == 0:

        return {
            "strategy": "score_too_low_relax_core_filters",
            "filters": relax_to_core_filters(
                filters=filters
            ),
            "tags": [],
            "top_k": top_k,
        }

    return {
        "strategy": "score_too_low_clear_filters",
        "filters": {},
        "tags": [],
        "top_k": increase_top_k(
            top_k=top_k,
            target=settings.rag.retry_second_top_k,
        ),
    }


def build_low_quality_retry_strategy(
        retry_count: int,
        filters: dict[str, Any],
        tags: list[str],
        top_k: int,
) -> dict[str, Any]:
    """
    构建通用 low_quality / unknown 重试策略。

    功能：
        当失败原因不够明确时，使用保守的渐进式策略：
        1. 第一次扩大 top_k。
        2. 第二次只保留核心 filters。
        3. 后续清空 filters。

    参数：
        retry_count:
            当前重试次数。

        filters:
            当前 filters。

        tags:
            当前 tags。

        top_k:
            当前 top_k。

    返回值：
        dict[str, Any]:
            重试策略结果。
    """

    if retry_count == 0:

        return {
            "strategy": "low_quality_increase_top_k",
            "filters": filters,
            "tags": tags,
            "top_k": increase_top_k(
                top_k=top_k,
                target=settings.rag.retry_first_top_k,
            ),
        }

    if retry_count == 1:

        return {
            "strategy": "low_quality_relax_core_filters",
            "filters": relax_to_core_filters(
                filters=filters
            ),
            "tags": [],
            "top_k": increase_top_k(
                top_k=top_k,
                target=settings.rag.retry_second_top_k,
            ),
        }

    return {
        "strategy": "low_quality_clear_filters",
        "filters": {},
        "tags": [],
        "top_k": increase_top_k(
            top_k=top_k,
            target=settings.rag.retry_third_top_k,
        ),
    }


def relax_to_core_filters(
        filters: dict[str, Any],
) -> dict[str, Any]:
    """
    只保留核心 filters。

    功能：
        根据 settings.rag.retry_core_filter_fields 保留核心字段。

        v1.5 新版支持：
        1. 旧版 flat filter。
        2. 新版 Chroma $and filter。
        3. 自动从 $and 条件中提取 dog_name、size 等核心字段。
        4. 返回合法 Chroma where filter。

    示例 1：
        输入：
            {
                "dog_name": {"$eq": "Shih Tzu"},
                "barking_level": {"$lte": 3}
            }

        输出：
            {
                "dog_name": {"$eq": "Shih Tzu"}
            }

    示例 2：
        输入：
            {
                "$and": [
                    {"size": {"$eq": "small"}},
                    {"barking_level": {"$lte": 3}},
                    {"good_for_apartment": {"$eq": True}}
                ]
            }

        输出：
            {
                "size": {"$eq": "small"}
            }

    技术名词：
        Relax：
            放宽。删除较细的偏好条件，只保留关键条件。

        Core Filter：
            核心过滤条件，例如 dog_name、size。

        Chroma Where Filter：
            Chroma 向量数据库使用的 metadata 过滤条件。

        $and：
            Chroma 逻辑与操作符，表示多个条件必须同时满足。

    参数：
        filters:
            原始 filters。

    返回值：
        dict[str, Any]:
            只保留核心字段后的 Chroma where filter。
    """

    if not isinstance(
            filters,
            dict,
    ) or not filters:
        return {}

    core_fields = resolve_retry_core_filter_fields()

    conditions = extract_filter_conditions(
        filters=filters,
    )

    core_conditions = []

    for condition in conditions:

        field_name = get_filter_condition_field_name(
            condition=condition,
        )

        if field_name in core_fields:
            core_conditions.append(
                condition
            )

    return build_chroma_filter_from_conditions(
        conditions=core_conditions,
    )

def resolve_retry_core_filter_fields() -> set[str]:
    """
    解析 retry 核心 filter 字段。

    功能：
        优先读取 settings.rag.retry_core_filter_fields。
        如果配置不存在或为空，则使用默认核心字段：
            dog_name、size

    参数：
        无。

    返回值：
        set[str]:
            核心 filter 字段集合。
    """

    default_core_fields = {
        "dog_name",
        "size",
    }

    try:
        configured_fields = getattr(
            settings.rag,
            "retry_core_filter_fields",
            None,
        )

        if configured_fields:
            return set(
                configured_fields
            )

    except Exception as e:
        logger.warning(
            f"读取 retry_core_filter_fields 失败，使用默认核心字段: {e}"
        )

    return default_core_fields


def extract_filter_conditions(
        filters: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    从 Chroma filter 中提取条件列表。

    功能：
        将不同结构的 filters 转换成统一的 condition list。

    支持结构 1：flat filter
        {
            "dog_name": {"$eq": "Shih Tzu"},
            "size": {"$eq": "small"}
        }

        转换为：
        [
            {"dog_name": {"$eq": "Shih Tzu"}},
            {"size": {"$eq": "small"}}
        ]

    支持结构 2：$and filter
        {
            "$and": [
                {"dog_name": {"$eq": "Shih Tzu"}},
                {"size": {"$eq": "small"}}
            ]
        }

        转换为：
        [
            {"dog_name": {"$eq": "Shih Tzu"}},
            {"size": {"$eq": "small"}}
        ]

    参数：
        filters:
            原始 Chroma filter。

    返回值：
        list[dict[str, Any]]:
            条件列表。
    """

    if not isinstance(
            filters,
            dict,
    ) or not filters:
        return []

    if "$and" in filters:
        and_conditions = filters.get(
            "$and",
            [],
        )

        if not isinstance(
                and_conditions,
                list,
        ):
            return []

        normalized_conditions = []

        for condition in and_conditions:

            if not isinstance(
                    condition,
                    dict,
            ):
                continue

            # 支持嵌套 $and，虽然当前项目一般不会生成多层。
            if "$and" in condition:
                normalized_conditions.extend(
                    extract_filter_conditions(
                        filters=condition,
                    )
                )

                continue

            if is_single_field_filter_condition(
                    condition=condition,
            ):
                normalized_conditions.append(
                    condition
                )

        return normalized_conditions

    conditions = []

    for field_name, value in filters.items():

        if field_name.startswith(
                "$"
        ):
            continue

        conditions.append(
            {
                field_name: value
            }
        )

    return conditions

def is_single_field_filter_condition(
        condition: dict[str, Any],
) -> bool:
    """
    判断是否是单字段 filter 条件。

    功能：
        Chroma 的一个普通条件通常长这样：
            {"dog_name": {"$eq": "Shih Tzu"}}

        它应该只有一个字段，并且字段名不是 $and / $or 这类操作符。

    参数：
        condition:
            单个 filter 条件。

    返回值：
        bool:
            True 表示是单字段条件。
            False 表示不是。
    """

    if not isinstance(
            condition,
            dict,
    ):
        return False

    if len(
            condition
    ) != 1:
        return False

    field_name = next(
        iter(
            condition.keys()
        )
    )

    if str(
            field_name
    ).startswith(
        "$"
    ):
        return False

    return True

def get_filter_condition_field_name(
        condition: dict[str, Any],
) -> str | None:
    """
    获取单个 filter 条件的字段名。

    功能：
        从：
            {"dog_name": {"$eq": "Shih Tzu"}}

        提取：
            dog_name

    参数：
        condition:
            单个 filter 条件。

    返回值：
        str | None:
            字段名。
            如果无法提取，则返回 None。
    """

    if not is_single_field_filter_condition(
            condition=condition,
    ):
        return None

    return next(
        iter(
            condition.keys()
        )
    )


def build_chroma_filter_from_conditions(
        conditions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    根据条件列表重新构建 Chroma filter。

    功能：
        将 condition list 重新转换成合法 Chroma where filter。

    规则：
        1. 如果没有条件，返回 {}。
        2. 如果只有一个条件，直接返回该条件。
        3. 如果有多个条件，返回 {"$and": conditions}。

    参数：
        conditions:
            单个 filter 条件列表。

    返回值：
        dict[str, Any]:
            Chroma where filter。
    """

    if not conditions:
        return {}

    if len(
            conditions
    ) == 1:
        return conditions[0]

    return {
        "$and": conditions
    }


def increase_top_k(
        top_k: int,
        target: int,
) -> int:
    """
    增加 top_k。

    功能：
        将当前 top_k 提升到目标值。
        如果当前 top_k 已经大于目标值，则保持当前值。

    参数：
        top_k:
            当前 top_k。

        target:
            目标 top_k。

    返回值：
        int:
            新 top_k。
    """

    return max(
        top_k,
        target
    )


def resolve_failure_type(
        state: DogState,
) -> str:
    """
    解析 retrieval_failure_type。

    功能：
        优先读取 state["retrieval_failure_type"]。
        如果没有，则尝试从 state["retrieval_quality"]["failure_type"] 中读取。
        如果仍然没有，则返回 unknown。

    参数：
        state:
            当前 DogState。

    返回值：
        str:
            失败类型。
    """

    failure_type = state.get(
        "retrieval_failure_type"
    )

    if failure_type:

        return str(
            failure_type
        )

    retrieval_quality = state.get(
        "retrieval_quality"
    )

    if isinstance(
            retrieval_quality,
            dict
    ):

        nested_failure_type = retrieval_quality.get(
            "failure_type"
        )

        if nested_failure_type:

            return str(
                nested_failure_type
            )

    return "unknown"


def resolve_retry_count(
        value: Any,
) -> int:
    """
    解析 retry_count。

    功能：
        将 retry_count 转换成合法整数。

    参数：
        value:
            原始 retry_count。

    返回值：
        int:
            合法 retry_count。
    """

    try:
        retry_count = int(
            value
        )
    except (
            TypeError,
            ValueError,
    ):
        return 0

    return max(
        retry_count,
        0
    )


def resolve_top_k(
        value: Any,
        default: int = DEFAULT_TOP_K,
) -> int:
    """
    解析 top_k。

    功能：
        将 top_k 转换成合法整数。

    参数：
        value:
            原始 top_k。

        default:
            默认 top_k。

    返回值：
        int:
            合法 top_k。
    """

    try:
        top_k = int(
            value
        )
    except (
            TypeError,
            ValueError,
    ):
        return default

    if top_k <= 0:

        return default

    return top_k


def copy_filters(
        value: Any,
) -> dict[str, Any]:
    """
    安全复制 filters。

    功能：
        如果 value 是 dict，则复制一份。
        如果不是 dict，则返回空 dict。

    参数：
        value:
            原始 filters。

    返回值：
        dict[str, Any]:
            复制后的 filters。
    """

    if not isinstance(
            value,
            dict
    ):

        return {}

    return dict(
        value
    )


def save_checkpoint_safely(
        checkpoint_provider=None,
) -> None:
    """
    安全保存 checkpoint。

    功能：
        如果 checkpoint_provider 存在，则调用 save_checkpoint。
        如果保存失败，只记录 warning，不中断主流程。

    参数：
        checkpoint_provider:
            CheckpointProvider 实例，可选。

    返回值：
        None。
    """

    if checkpoint_provider is None:

        return

    try:

        checkpoint_provider.manager.save_checkpoint()

    except Exception as e:

        logger.warning(
            f"retrieval_retry_node 保存 checkpoint 失败: {e}"
        )