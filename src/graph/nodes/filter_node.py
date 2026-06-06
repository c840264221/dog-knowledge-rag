from src.logger import logger

from src.runtime.context import runtime_ctx

def filter_node(state):
    """
    构建数据库可用的 filter_dict
    """

    runtime_ctx.get().state().set_node(
        "filter_node"
    )

    # 记录时间线
    runtime_ctx.get().timeline().add_event(

        event_type="node",

        name="filter_node"
    )

    logger.info(
        f"进入filter_node节点  "
        f"state: "
        f"question:{state['question']}, "
        f"intent:{state['intent']}, "
        f"strategy:{state['strategy']}, "
        f"filters:{state['filters']}, "
        f"tags:{state['tags']}, "
        f"dog_name:{state['dog_name']}, "
    )
    filters = state.get("filters", {})
    dog_name = state.get("dog_name")

    filter_dict = {}

    # 数值过滤（直接透传）
    for k, v in filters.items():
        filter_dict[k] = v

    # 品种过滤（如果识别到了）
    if dog_name:
        filter_dict["name"] = dog_name

    logger.debug(f"filter_node节点执行完毕 filter_dict: {filter_dict}")

    from src.runtime.container.init import container

    container.get("checkpoint").manager.save_checkpoint()


    return {
        "filters": filter_dict
    }