import inspect
from typing import Any, Awaitable, Callable

from src.logger import logger
from src.runtime.context import runtime_ctx


def _resolve_memory_retrieval_text(
    state: dict[str, Any],
) -> str:
    """
    选择记忆召回真正使用的业务文本。

    功能：
        优先使用记忆召回专用字段；尚未接入该字段的旧调用方可以继续使用
        RAG 的干净检索问题，最后才回退到完整 question。

    参数含义：
        state:
            当前图状态，可能包含新旧不同版本的问题字段。

    返回值含义：
        str:
            不包含 Skill 执行说明的记忆召回查询；没有输入时返回空字符串。
    """

    return str(
        state.get("memory_retrieval_text")
        or state.get("retrieval_question")
        or state.get("question")
        or ""
    ).strip()


def build_memory_retrieve_node(
    semantic_recall: Any,
    checkpoint_manager: Any = None,
    runtime_context_getter=None,
    pet_profile_service: Any = None,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """
    构建 Memory 召回节点。

    功能：
        接收外部注入的 MemorySemanticRecallService。
        接收外部注入的 checkpoint_manager 检查点管理器。
        接收外部注入的 runtime_context_getter 运行时上下文获取函数。
        返回一个符合 LangGraph 节点签名的 async node 函数。
        node 执行时只使用注入的服务，不直接 import container。
        接入 Runtime Context、Timeline、Checkpoint、Logger。
        避免 Graph Node 与 RuntimeContainer 之间产生循环导入。

    参数：
        semantic_recall：
            MemorySemanticRecallService（记忆语义召回服务）。
            用于根据用户问题召回相关长期记忆。
            需要提供 retrieve(user_id, question, limit) 方法。

        pet_profile_service：
            PetProfileService（宠物档案服务）。用于根据当前宠物标识召回
            结构化档案；为空时保持旧版只召回普通记忆的行为。

        checkpoint_manager：
            CheckpointManager（检查点管理器）。
            用于在关键节点执行后保存运行状态。
            可以为 None，为 None 时不保存 checkpoint。

        runtime_context_getter：
            RuntimeContext Getter（运行时上下文获取函数）。
            用于获取当前请求的 RuntimeContext。
            如果不传，则默认使用 runtime_ctx.get。

    返回值：
        Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]：
            返回一个 LangGraph 可调用的异步节点函数。
            该函数接收 state，返回需要合并进 state 的 dict。

    专业名词：
        Semantic Recall（语义召回）：
            根据语义相似度查找相关长期记忆，而不是只做关键词匹配。

        Runtime Context（运行时上下文）：
            当前请求执行过程中的上下文对象，用于记录状态、时间线、trace 等信息。

        Checkpoint（检查点）：
            用于保存当前运行状态，方便恢复、追踪和调试。
    """

    if runtime_context_getter is None:
        runtime_context_getter = runtime_ctx.get

    async def memory_retrieve_node(
        state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        执行 Memory 语义召回。

        功能：
            1. 设置当前运行节点名称
            2. 写入 Timeline 时间线事件
            3. 从 state 中读取 user_id
            4. 如果没有 user_id，则使用 session_id
            5. 如果 user_id 和 session_id 都没有，则使用 default_user
            6. 从 state 中选择记忆召回专用业务文本
            7. 调用 MemorySemanticRecallService 召回相关长期记忆
            8. 将召回结果写入 memory_context 字段
            9. 保存 checkpoint 检查点
            10. 如果召回失败，则返回“暂无用户记忆”，不阻断主流程

        参数：
            state：
                LangGraph 当前状态。
                Graph 节点之间传递的数据字典。

        返回值：
            dict[str, Any]：
                返回需要合并进 state 的字段。
                包含 memory_context 和 memory_recall_result。
        """

        node_name = "memory_retrieve_node"
        memory_context = "暂无用户记忆"
        memory_recall_result: dict[str, Any] = {
            "status": "empty",
            "candidate_count": 0,
            "threshold_passed_count": 0,
            "selected_count": 0,
            "semantic_threshold": 0.0,
            "max_semantic_score": None,
            "selected_memory_ids": [],
            "reason": "未执行记忆召回。",
        }
        pet_profile_recall_result: dict[str, Any] = {
            "status": "empty",
            "pet_key": "",
            "pet_name": "",
            "selection_source": "none",
            "facts": {},
            "selected_attributes": [],
            "reason": "未配置宠物档案召回服务。",
        }
        active_pet_update: dict[str, str] = {}

        # 这里只执行回答用途的访问计划，不再混入 Skill 补参字段。
        raw_access_decision = state.get(
            "answer_profile_access_decision"
        )
        profile_access_decision = (
            dict(raw_access_decision)
            if isinstance(raw_access_decision, dict)
            else {}
        )
        raw_allowed_attributes = profile_access_decision.get(
            "allowed_attributes",
            [],
        )
        allowed_profile_attributes = (
            list(raw_allowed_attributes)
            if isinstance(raw_allowed_attributes, (list, tuple, set))
            else []
        )

        # 用户标识同时供普通记忆和宠物档案使用，必须在两个独立降级块之前准备。
        user_id = str(
            state.get("user_id")
            or state.get("session_id")
            or "default_user"
        )

        try:
            ctx = runtime_context_getter()

            if ctx is not None:
                ctx.state().set_node(
                    node_name
                )

                ctx.timeline().add_event(
                    event_type="node",
                    name=node_name
                )

            logger.info(
                "开始执行 Memory 召回节点"
            )

            question = _resolve_memory_retrieval_text(state)

            logger.info(
                f"Memory Retrieve 输入: user_id={user_id}, question={question}"
            )

            retrieve_with_details = getattr(
                semantic_recall,
                "retrieve_with_details",
                None,
            )

            if callable(retrieve_with_details):
                retrieved_memory = retrieve_with_details(
                    user_id=user_id,
                    question=question,
                    limit=5,
                )
            else:
                retrieved_memory = semantic_recall.retrieve(
                    user_id=user_id,
                    question=question,
                    limit=5,
                )

            if inspect.isawaitable(
                retrieved_memory
            ):
                retrieved_memory = await retrieved_memory

            if isinstance(retrieved_memory, dict) and (
                "memory_context" in retrieved_memory
            ):
                memory_context = _format_memory_context(
                    retrieved_memory.get("memory_context")
                )
                memory_recall_result = {
                    key: value
                    for key, value in retrieved_memory.items()
                    if key != "memory_context"
                }
            else:
                memory_context = _format_memory_context(
                    retrieved_memory
                )
                has_memory = memory_context != "暂无用户记忆"
                memory_recall_result.update(
                    {
                        "status": "applied" if has_memory else "empty",
                        "selected_count": 1 if has_memory else 0,
                        "reason": (
                            "兼容旧版召回服务，已获取可用记忆文本。"
                            if has_memory
                            else "兼容旧版召回服务，未获取到可用记忆。"
                        ),
                    }
                )

            logger.info(
                "Memory Retrieve 结果: "
                f"memory_context={memory_context}, "
                f"memory_recall_result={memory_recall_result}"
            )

        except Exception as e:
            memory_recall_result.update(
                {
                    "status": "failed",
                    "reason": f"记忆召回异常，已降级为空记忆：{e}",
                }
            )
            logger.warning(
                f"Memory 召回失败，已降级为空记忆: {e}"
            )

        # 回答阶段只复用回答用途、字段集合完全相同的结果。
        cached_profile_result = state.get("pet_profile_recall_result")
        cached_profile_applied = (
            isinstance(cached_profile_result, dict)
            and cached_profile_result.get("status") == "applied"
            and str(cached_profile_result.get("pet_key") or "")
            == str(state.get("active_pet_key") or "")
            and set(cached_profile_result.get("selected_attributes") or [])
            == set(allowed_profile_attributes)
        )
        if cached_profile_applied:
            pet_profile_recall_result = dict(cached_profile_result)
            active_pet_update = {
                "active_pet_key": str(
                    pet_profile_recall_result.get("pet_key") or ""
                ),
                "active_pet_name": str(
                    pet_profile_recall_result.get("pet_name") or ""
                ),
            }

        # 宠物档案与普通语义记忆独立降级，一侧失败不会清空另一侧结果。
        elif pet_profile_service is not None:
            try:
                profile_result = pet_profile_service.recall_profile(
                    user_id=user_id,
                    active_pet_key=state.get("active_pet_key"),
                    active_pet_name=state.get("active_pet_name"),
                    selected_attributes=allowed_profile_attributes,
                )
                pet_profile_recall_result = profile_result.model_dump(
                    mode="python"
                )
                if pet_profile_recall_result.get("status") == "applied":
                    # 单宠物回退也会升级成明确的当前宠物，供后续节点和下一轮复用。
                    active_pet_update = {
                        "active_pet_key": str(
                            pet_profile_recall_result.get("pet_key") or ""
                        ),
                        "active_pet_name": str(
                            pet_profile_recall_result.get("pet_name") or ""
                        ),
                    }
                logger.info(
                    "宠物档案召回完成: "
                    f"status={pet_profile_recall_result.get('status')}, "
                    f"pet_key={pet_profile_recall_result.get('pet_key')}, "
                    "attributes="
                    f"{pet_profile_recall_result.get('selected_attributes')}"
                )
            except Exception as profile_error:
                pet_profile_recall_result.update(
                    {
                        "status": "failed",
                        "reason": (
                            "宠物档案召回异常，已跳过档案注入："
                            f"{profile_error}"
                        ),
                    }
                )
                logger.warning(
                    "宠物档案召回失败，已独立降级: %s",
                    profile_error,
                )

        if checkpoint_manager is not None:
            try:
                checkpoint_manager.save_checkpoint()

            except Exception as checkpoint_error:
                logger.warning(
                    f"Memory 召回节点保存 checkpoint 失败: {checkpoint_error}"
                )

        return {
            "memory_context": memory_context,
            "memory_recall_result": memory_recall_result,
            "pet_profile_recall_result": pet_profile_recall_result,
            "answer_profile_access_decision": profile_access_decision,
            **active_pet_update,
        }

    return memory_retrieve_node


def _format_memory_context(
    retrieved_memory: Any,
) -> str:
    """
    格式化 Memory 召回结果。

    功能：
        将 MemorySemanticRecallService 返回的结果统一转换成字符串。
        如果返回 None 或空值，则使用“暂无用户记忆”。
        如果返回 str，则直接使用。
        如果返回其他类型，则使用 str(...) 转换。

    参数：
        retrieved_memory：
            记忆召回服务返回的结果。
            可能是 str、list、dict、None 或其他对象。

    返回值：
        str：
            可注入 state["memory_context"] 的记忆上下文文本。
    """

    if not retrieved_memory:
        return "暂无用户记忆"

    if isinstance(
        retrieved_memory,
        str
    ):
        return retrieved_memory

    return str(
        retrieved_memory
    )
