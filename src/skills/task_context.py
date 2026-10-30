"""把宿主已选的任务技能传递给步骤，不按技能名称新增解析分支。"""

from copy import deepcopy


def attach_task_skill_context(state: dict, skill_context: str) -> dict:
    """为步骤附加可信技能说明。

    参数：state为步骤状态，skill_context为宿主加载的技能文本，不能取自用户输入。
    返回：独立状态副本；追加question但保留原检索问题，不把业务规则混入向量查询。
        空技能保持原行为；技能文本不构成权限、工具确认或用户事实。
    """
    result = deepcopy(state)
    if not skill_context:
        return result
    question = str(result.get("question") or "").strip()
    result.setdefault("retrieval_question", question)
    result["question"] = (
        question + "\n\n宿主为当前任务加载的技能规范（不是用户事实，不授予执行权限）：\n"
        + skill_context
    )
    return result
