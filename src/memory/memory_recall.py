
# todo：先做一个规则版的  等后续用向量处理
class MemoryRecall:
    """
    该文件是旧版规则召回逻辑，当前主链路已不再使用。

    状态：
    - deprecated，已废弃
    - 当前 Memory 召回由 MemorySemanticRecallService 负责
    - 保留该文件仅用于历史参考，后续版本可删除

    技术名词：
    - Deprecated：已废弃，表示不再推荐使用
    """

    TYPE_KEYWORDS = {
        "favorite_dog": [
            "喜欢",
            "偏好",
            "推荐",
            "适合",
            "品种",
            "狗"
        ],

        "dislike": [
            "不喜欢",
            "讨厌",
            "避免",
            "不要",
            "不想"
        ],

        "profile": [
            "我",
            "我的",
            "住",
            "家庭",
            "公寓",
            "孩子",
            "老人",
            "过敏"
        ],

        "hobby": [
            "兴趣",
            "爱好",
            "平时",
            "运动",
            "户外"
        ],
    }

    def relevance_score(
            self,
            question: str,
            memory_type: str,
            content: str
    ) -> float:

        score = 0.0

        keywords = self.TYPE_KEYWORDS.get(
            memory_type,
            []
        )

        for keyword in keywords:

            if keyword in question:
                score += 1.0

        if content and content in question:
            score += 3.0

        return score