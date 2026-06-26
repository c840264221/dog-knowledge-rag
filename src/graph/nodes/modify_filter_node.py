from __future__ import annotations

from typing import (
    Any,
    Mapping,
)

from src.graph.states.dog_state import (
    DogState,
)

from src.logger import (
    logger,
)

from src.runtime.context import (
    runtime_ctx,
)

from src.rag.query_builders.rag_query_builder import (
    merge_metadata_filters,
)


DEFAULT_FALLBACK_DOG_NAME = "Golden Retriever"


def build_modify_filter_node(
        checkpoint_provider=None,
):
    """
    构建 modify_filter_node 节点。

    功能：
        使用 Provider Injection（提供者注入）的方式构建 modify_filter_node，
        避免节点内部直接 import container。

        modify_filter_node 主要用于 ask_user_node 之后，
        当用户要求“换一个品种”或“修改过滤条件”时，
        修改当前 state 中的 filters，并清空上一轮 RAG 结果，
        让下一轮 retrieve_node 重新检索。

    技术名词：
        Provider Injection：
            提供者注入。依赖对象从外部传入，而不是在函数内部直接 import container。

        Metadata Filter：
            元数据过滤条件。用于 Chroma 检索时按 dog_name、size、barking_level 等字段过滤。

        Chroma Filter：
            Chroma 向量数据库使用的过滤条件格式，例如：
            {"dog_name": {"$eq": "Golden Retriever"}}

        State Reset：
            状态重置。修改检索条件后，需要清空旧的 rag_context / docs，避免后续误用旧结果。

    参数：
        checkpoint_provider:
            CheckpointProvider 实例。
            中文释义：用于保存运行时 checkpoint。

    返回值：
        callable:
            返回一个 LangGraph 可注册的 modify_filter_node 函数。
    """

    def node(
            state: DogState,
    ) -> dict[str, Any]:
        """
        执行 modify_filter_node。

        功能：
            调用 execute_modify_filter_node 执行核心逻辑。

        参数：
            state:
                当前 DogState。

        返回值：
            dict[str, Any]:
                返回需要合并进 DogState 的状态更新。
        """

        return execute_modify_filter_node(
            state=state,
            checkpoint_provider=checkpoint_provider,
        )

    return node


def modify_filter_node(
        state: DogState,
) -> dict[str, Any]:
    """
    旧版兼容 modify_filter_node。

    功能：
        保留旧函数名，避免旧代码直接导入 modify_filter_node 时报错。

        注意：
            v1.5 新代码推荐使用 build_modify_filter_node 注入 checkpoint_provider。
            当前函数只是兼容入口。

    参数：
        state:
            当前 DogState。

    返回值：
        dict[str, Any]:
            返回修改后的 filters、retry_count，以及清空后的 RAG 状态。
    """

    from src.runtime.container.init import (
        container,
    )

    checkpoint_provider = container.get(
        "checkpoint"
    )

    return execute_modify_filter_node(
        state=state,
        checkpoint_provider=checkpoint_provider,
    )


def execute_modify_filter_node(
        state: DogState,
        checkpoint_provider=None,
) -> dict[str, Any]:
    """
    执行过滤条件修改逻辑。

    功能：
        根据当前 state 修改 filters。

        当前 v1.5 策略：
        1. 优先从 user_feedback 中解析用户想换成的 dog_name。
        2. 如果没有 user_feedback，则使用默认兜底犬种 Golden Retriever。
        3. 使用新版 dog_name 字段，而不是旧版 name 字段。
        4. 使用 Chroma filter 格式：
           {"dog_name": {"$eq": "..."}}
        5. 将新 filter 和旧 filters 合并。
        6. 清空 rag_query、rag_context、docs，让下一轮 retrieve 重新检索。
        7. retry_count + 1。

    参数：
        state:
            当前 DogState。

        checkpoint_provider:
            CheckpointProvider 实例，可选。

    返回值：
        dict[str, Any]:
            状态更新字典。
    """

    runtime = runtime_ctx.get()

    runtime.state().set_node(
        "modify_filter_node"
    )

    runtime.timeline().add_event(
        event_type="node",
        name="modify_filter_node",
    )

    logger.info(
        f"进入 modify_filter_node 节点，state={state}"
    )

    current_filters = resolve_current_filters(
        state=state
    )

    target_dog_name = resolve_target_dog_name(
        state=state
    )

    dog_name_filter = build_dog_name_filter(
        dog_name=target_dog_name
    )

    merged_filters = merge_metadata_filters(
        current_filters,
        dog_name_filter,
    )

    retry_count = resolve_retry_count(
        state=state
    )

    output_state = {
        "filters": merged_filters,
        "dog_name": target_dog_name,
        "retry_count": retry_count + 1,

        # 修改 filter 后，旧 RAG 结果必须清空。
        # 下一轮 retrieve_node 会根据新的 filters 重新构建 RagQuery。
        "rag_query": None,
        "rag_context": None,
        "docs": [],

        # 重置召回评估状态。
        "retrieval_ok": False,
        "retrieval_evaluated": False,
        "retrieval_quality": None,
        "retrieval_failure_type": None,
    }

    logger.info(
        "modify_filter_node 修改 filters 完成，"
        f"target_dog_name={target_dog_name}, "
        f"merged_filters={merged_filters}, "
        f"retry_count={retry_count + 1}"
    )

    save_checkpoint_safely(
        checkpoint_provider=checkpoint_provider
    )

    return output_state


def resolve_current_filters(
        state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从 state 中读取当前 filters。

    功能：
        安全读取 state["filters"]。
        如果不存在或不是 dict，则返回空 dict。

    参数：
        state:
            当前 DogState。

    返回值：
        dict[str, Any]:
            当前 filters。
    """

    filters = state.get(
        "filters",
        {},
    )

    if not isinstance(
            filters,
            dict,
    ):
        return {}

    return dict(
        filters
    )


def resolve_target_dog_name(
        state: Mapping[str, Any],
) -> str:
    """
    解析目标犬种名。

    功能：
        根据当前 state 判断用户希望修改成哪个犬种。

        当前优先级：
        1. user_feedback 中提到的犬种名。
        2. state["dog_name"]。
        3. 默认兜底 Golden Retriever。

    注意：
        当前版本只做简单规则。
        后续可以接入 DogQueryFilterParser，
        从用户反馈中更智能地解析 dog_name 和其他 metadata filters。

    参数：
        state:
            当前 DogState。

    返回值：
        str:
            目标犬种名。
    """

    user_feedback = str(
        state.get(
            "user_feedback",
            "",
        )
        or ""
    ).strip()

    detected_from_feedback = detect_dog_name_from_text(
        text=user_feedback
    )

    if detected_from_feedback:
        return detected_from_feedback

    dog_name = str(
        state.get(
            "dog_name",
            "",
        )
        or ""
    ).strip()

    if dog_name:
        return dog_name

    return DEFAULT_FALLBACK_DOG_NAME


def detect_dog_name_from_text(
        text: str,
) -> str | None:
    """
    从文本中简单识别犬种名。

    功能：
        当前使用轻量规则识别常见犬种名。
        这个函数是临时规则版，后续可以替换成 DogQueryFilterParser。

    参数：
        text:
            用户反馈文本。

    返回值：
        str | None:
            识别到的犬种名。
            如果没有识别到，则返回 None。
    """

    if not text:
        return None

    normalized_text = text.lower()

    dog_name_aliases = {
        "golden retriever": "Golden Retriever",
        "金毛": "Golden Retriever",
        "金毛寻回犬": "Golden Retriever",

        "labrador retriever": "Labrador Retriever",
        "labrador": "Labrador Retriever",
        "拉布拉多": "Labrador Retriever",

        "shih tzu": "Shih Tzu",
        "西施犬": "Shih Tzu",

        "affenpinscher": "Affenpinscher",
        "猴头梗": "Affenpinscher",

        "japanese spitz": "Japanese Spitz",
        "日本狐狸犬": "Japanese Spitz",

        "biewer terrier": "Biewer Terrier",
        "比沃犬": "Biewer Terrier",

        "italian greyhound": "Italian Greyhound",
        "意大利灵缇": "Italian Greyhound",
    }

    for alias, dog_name in dog_name_aliases.items():

        if alias.lower() in normalized_text:
            return dog_name

    return None


def build_dog_name_filter(
        dog_name: str,
) -> dict[str, Any]:
    """
    构建 dog_name Chroma filter。

    功能：
        将犬种名转换成新版 Chroma metadata filter。

    示例：
        输入：
            Golden Retriever

        输出：
            {
                "dog_name": {
                    "$eq": "Golden Retriever"
                }
            }

    参数：
        dog_name:
            犬种名。

    返回值：
        dict[str, Any]:
            Chroma metadata filter。
    """

    return {
        "dog_name": {
            "$eq": dog_name,
        }
    }


def resolve_retry_count(
        state: Mapping[str, Any],
) -> int:
    """
    解析 retry_count。

    功能：
        从 state 中读取 retry_count。
        如果不存在或无法转换成 int，则返回 0。

    参数：
        state:
            当前 DogState。

    返回值：
        int:
            当前重试次数。
    """

    raw_retry_count = state.get(
        "retry_count",
        0,
    )

    try:
        retry_count = int(
            raw_retry_count
        )
    except (
            TypeError,
            ValueError,
    ):
        return 0

    return max(
        retry_count,
        0,
    )


def save_checkpoint_safely(
        checkpoint_provider=None,
) -> None:
    """
    安全保存 checkpoint。

    功能：
        如果 checkpoint_provider 存在，则保存 checkpoint。
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
            f"modify_filter_node 保存 checkpoint 失败: {e}"
        )