"""
Dog Query Filter Parser。

Dog Query Filter Parser（狗狗查询过滤解析器）：
负责把用户自然语言问题解析成 Chroma metadata filter。

当前模块职责：
1. 接收用户问题 question
2. 根据规则关键词识别犬种名、体型、吠叫、精力、训练难度等条件
3. 构建 Chroma metadata filter
4. 构建 RagQuery

当前模块不负责：
1. 调用 Chroma 检索
2. 调用 Retriever
3. 调用 LLM
4. 写入向量数据库
5. 做复杂语义理解

注意：
    当前是 Rule-based Parser（规则解析器）。
    后续可以增加 LLM Parser（大模型解析器），
    但规则解析器仍然可以作为 fallback（兜底方案）。
"""

from __future__ import annotations

from typing import Any

import re

from src.rag.schemas import RagQuery

from src.logger import logger

from src.retrieval.alias_loader import (
    get_alias_dict,
)


MetadataFilter = dict[str, Any]


class DogQueryFilterParser:
    """
    Dog Query Filter Parser（狗狗查询过滤解析器）。

    Parser（解析器）：
    负责把用户输入的自然语言问题转换成结构化过滤条件。

    当前支持解析：
    1. dog_name：犬种名
    2. size：体型
    3. barking_level：吠叫等级
    4. energy_level：精力等级
    5. trainability_level：可训练等级
    6. good_for_apartment：是否适合公寓
    7. good_for_beginner：是否适合新手
    8. good_with_young_children_level：是否适合小孩
    9. good_with_other_dogs_level：是否适合多狗家庭
    10. shedding_level：掉毛等级
    11. drooling_level：流口水等级
    12. coat_grooming_frequency_level：打理频率
    """

    PARSER_NAME = "dog_query_filter_parser_v1"

    FALLBACK_BREED_ALIAS_MAP: dict[str, str] = {
        "金毛": "Golden Retriever",
        "金毛寻回犬": "Golden Retriever",
        "黄金猎犬": "Golden Retriever",
        "golden retriever": "Golden Retriever",

        "拉布拉多": "Labrador Retriever",
        "拉布拉多犬": "Labrador Retriever",
        "labrador": "Labrador Retriever",
        "labrador retriever": "Labrador Retriever",

        "哈士奇": "Siberian Husky",
        "西伯利亚雪橇犬": "Siberian Husky",
        "husky": "Siberian Husky",
        "siberian husky": "Siberian Husky",

        "边牧": "Border Collie",
        "边境牧羊犬": "Border Collie",
        "border collie": "Border Collie",

        "柴犬": "Shiba Inu",
        "shiba": "Shiba Inu",
        "shiba inu": "Shiba Inu",

        "柯基": "Pembroke Welsh Corgi",
        "威尔士柯基": "Pembroke Welsh Corgi",
        "corgi": "Pembroke Welsh Corgi",
        "pembroke welsh corgi": "Pembroke Welsh Corgi",

        "贵宾": "Poodle",
        "贵宾犬": "Poodle",
        "泰迪": "Poodle",
        "poodle": "Poodle",

        "博美": "Pomeranian",
        "博美犬": "Pomeranian",
        "pomeranian": "Pomeranian",

        "萨摩耶": "Samoyed",
        "samoyed": "Samoyed",

        "德牧": "German Shepherd Dog",
        "德国牧羊犬": "German Shepherd Dog",
        "german shepherd": "German Shepherd Dog",
        "german shepherd dog": "German Shepherd Dog",

        "斗牛犬": "Bulldog",
        "bulldog": "Bulldog",

        "法斗": "French Bulldog",
        "法国斗牛犬": "French Bulldog",
        "french bulldog": "French Bulldog",

        "比熊": "Bichon Frise",
        "比熊犬": "Bichon Frise",
        "bichon frise": "Bichon Frise",

        "约克夏": "Yorkshire Terrier",
        "约克夏犬": "Yorkshire Terrier",
        "yorkshire terrier": "Yorkshire Terrier",

        "吉娃娃": "Chihuahua",
        "chihuahua": "Chihuahua",

        "秋田": "Akita",
        "秋田犬": "Akita",
        "akita": "Akita",

        "阿拉斯加": "Alaskan Malamute",
        "阿拉斯加雪橇犬": "Alaskan Malamute",
        "alaskan malamute": "Alaskan Malamute",

        "杜宾": "Doberman Pinscher",
        "杜宾犬": "Doberman Pinscher",
        "doberman": "Doberman Pinscher",
        "doberman pinscher": "Doberman Pinscher",

        "罗威纳": "Rottweiler",
        "罗威纳犬": "Rottweiler",
        "rottweiler": "Rottweiler",
    }

    SIZE_KEYWORDS: dict[str, list[str]] = {
        "small": [
            "小型犬",
            "小型",
            "小体型",
            "体型小",
            "小狗",
            "small dog",
            "small breed",
            "small",
        ],
        "medium": [
            "中型犬",
            "中型",
            "中等体型",
            "medium dog",
            "medium breed",
            "medium",
        ],
        "large": [
            "大型犬",
            "大型",
            "大狗",
            "体型大",
            "large dog",
            "large breed",
            "large",
        ],
        "giant": [
            "巨型犬",
            "巨型",
            "超大型",
            "giant dog",
            "giant breed",
            "giant",
        ],
    }

    LOW_BARKING_KEYWORDS: list[str] = [
        "不太爱叫",
        "不爱叫",
        "少叫",
        "少吠",
        "安静",
        "别太吵",
        "不要太吵",
        "不能太吵",
        "低吠叫",
        "low barking",
        "quiet",
        "not bark much",
        "does not bark much",
    ]

    HIGH_BARKING_KEYWORDS: list[str] = [
        "爱叫",
        "会叫",
        "警觉",
        "看门",
        "看家",
        "watchdog",
        "protective",
        "high barking",
    ]

    LOW_ENERGY_KEYWORDS: list[str] = [
        "精力不要太高",
        "精力低",
        "运动量小",
        "不需要太多运动",
        "懒一点",
        "安静一点",
        "low energy",
        "calm",
        "not too active",
        "less exercise",
    ]

    HIGH_ENERGY_KEYWORDS: list[str] = [
        "精力旺盛",
        "精力高",
        "运动量大",
        "活跃",
        "能跑",
        "适合运动",
        "high energy",
        "active",
        "athletic",
    ]

    EASY_TRAINING_KEYWORDS: list[str] = [
        "容易训练",
        "易于训练",
        "是否易于训练",
        "好训练",
        "好训",
        "听话",
        "服从",
        "聪明",
        "trainable",
        "easy to train",
        "obedient",
    ]

    APARTMENT_KEYWORDS: list[str] = [
        "适合公寓",
        "公寓",
        "楼房",
        "室内",
        "小空间",
        "apartment",
        "condo",
        "indoor",
    ]

    BEGINNER_KEYWORDS: list[str] = [
        "适合新手",
        "新手",
        "第一次养狗",
        "第一次养",
        "新手友好",
        "beginner",
        "first time owner",
        "first-time owner",
    ]

    CHILDREN_KEYWORDS: list[str] = [
        "适合小孩",
        "适合孩子",
        "孩子",
        "儿童",
        "家庭犬",
        "young children",
        "children",
        "kids",
        "family dog",
    ]

    OTHER_DOGS_KEYWORDS: list[str] = [
        "适合多狗家庭",
        "和其他狗相处",
        "其他狗",
        "多狗",
        "other dogs",
        "good with dogs",
    ]

    LOW_SHEDDING_KEYWORDS: list[str] = [
        "掉毛少",
        "不掉毛",
        "少掉毛",
        "低掉毛",
        "low shedding",
        "does not shed much",
        "no shedding",
    ]

    LOW_DROOLING_KEYWORDS: list[str] = [
        "不流口水",
        "少流口水",
        "流口水少",
        "low drooling",
        "does not drool much",
    ]

    LOW_GROOMING_KEYWORDS: list[str] = [
        "容易打理",
        "打理简单",
        "美容少",
        "护理简单",
        "low grooming",
        "easy grooming",
    ]

    AMBIGUOUS_ENGLISH_ALIASES: set[str] = {
        "dog",
        "hound",
        "terrier",
        "shepherd",
        "retriever",
        "spaniel",
        "setter",
        "pointer",
        "mastiff",
        "toy",
    }

    SHORT_ENGLISH_ALIAS_MAX_LENGTH = 2

    def __init__(
            self,
            alias_dict: dict[str, list[str]] | None = None,
    ):
        """
        初始化 DogQueryFilterParser。

        功能：
            初始化犬种别名索引。

            当前 alias_dog_name.json 的结构是：
                {
                    "标准犬种英文名": [
                        "中文名",
                        "英文别名",
                        "缩写"
                    ]
                }

            初始化时会将它转换成：
                {
                    "别名": "标准犬种英文名"
                }

        技术名词：
            Alias：
                别名。例如“西施犬”、“shih tzu”、“shihtzu”。

            Canonical Name：
                标准名。也就是 metadata 中 dog_name 字段使用的标准犬种名。

            Alias Index：
                别名索引。用于快速从用户问题中识别犬种名。

        参数：
            alias_dict:
                外部传入的 alias 字典。
                主要用于测试。
                如果不传，则自动调用 get_alias_dict() 读取 alias_dog_name.json。

        返回值：
            None。
        """

        self.breed_alias_map = self._load_breed_alias_map(
            alias_dict=alias_dict,
        )

    def _load_breed_alias_map(
            self,
            alias_dict: dict[str, list[str]] | None = None,
    ) -> dict[str, str]:
        """
        加载犬种别名映射表。

        功能：
            从 alias_dog_name.json 加载数据，
            并转换成 alias -> dog_name 的结构。

        当前 JSON 结构：
            {
                "Affenpinscher": [
                    "猴面梗",
                    "affenpinscher"
                ],
                "Afghan Hound": [
                    "阿富汗猎犬",
                    "afghan",
                    "afghan hound",
                    "hound",
                    "ah"
                ]
            }

        转换后结构：
            {
                "猴面梗": "Affenpinscher",
                "affenpinscher": "Affenpinscher",
                "阿富汗猎犬": "Afghan Hound",
                "afghan": "Afghan Hound",
                "afghan hound": "Afghan Hound",
                "ah": "Afghan Hound"
            }

        参数：
            alias_dict:
                外部传入的 alias 字典。
                如果为 None，则调用 get_alias_dict()。

        返回值：
            dict[str, str]:
                alias -> dog_name 映射。
        """

        try:
            raw_alias_dict = alias_dict

            if raw_alias_dict is None:
                raw_alias_dict = get_alias_dict()

            alias_map = self._build_alias_map_from_canonical_dict(
                raw_alias_dict=raw_alias_dict,
            )

            if alias_map:
                logger.info(
                    "DogQueryFilterParser 加载犬种 alias 成功，"
                    f"alias_count={len(alias_map)}"
                )

                return alias_map

            logger.warning(
                "DogQueryFilterParser 加载 alias 后结果为空，使用 fallback alias map"
            )

        except Exception as e:
            logger.warning(
                f"DogQueryFilterParser 加载 alias_dog_name.json 失败，使用 fallback alias map: {e}"
            )

        return self._normalize_fallback_alias_map()

    def _build_alias_map_from_canonical_dict(
            self,
            raw_alias_dict: dict[str, list[str]],
    ) -> dict[str, str]:
        """
        将 canonical -> aliases 的 JSON 结构转换成 alias -> canonical。

        功能：
            你的 alias_dog_name.json 是这种结构：
                {
                    "标准犬种名": [
                        "别名1",
                        "别名2"
                    ]
                }

            但是解析用户问题时，更适合使用：
                {
                    "别名1": "标准犬种名",
                    "别名2": "标准犬种名"
                }

        额外处理：
            1. 标准犬种名本身也会被加入 alias。
               例如 "Shih Tzu" -> "Shih Tzu"。
            2. 英文 alias 会统一转小写。
            3. 空 alias 会被跳过。
            4. 过于泛化的英文 alias 会被跳过，例如 hound。

        参数：
            raw_alias_dict:
                从 alias_dog_name.json 读取出来的原始字典。

        返回值：
            dict[str, str]:
                alias -> dog_name 映射。
        """

        alias_map: dict[str, str] = {}

        if not isinstance(
                raw_alias_dict,
                dict,
        ):
            return alias_map

        for canonical_name, aliases in raw_alias_dict.items():

            clean_canonical_name = str(
                canonical_name
                or ""
            ).strip()

            if not clean_canonical_name:
                continue

            self._add_alias_mapping(
                target=alias_map,
                alias=clean_canonical_name,
                dog_name=clean_canonical_name,
            )

            if not isinstance(
                    aliases,
                    list,
            ):
                continue

            for alias in aliases:
                self._add_alias_mapping(
                    target=alias_map,
                    alias=alias,
                    dog_name=clean_canonical_name,
                )

        for alias, dog_name in self.FALLBACK_BREED_ALIAS_MAP.items():
            self._add_alias_mapping(
                target=alias_map,
                alias=alias,
                dog_name=dog_name,
            )

        return alias_map

    def _normalize_fallback_alias_map(
            self,
    ) -> dict[str, str]:
        """
        归一化 fallback alias map。

        功能：
            将 FALLBACK_BREED_ALIAS_MAP 中的别名统一清洗成小写 key。

        参数：
            无。

        返回值：
            dict[str, str]:
                alias -> dog_name 映射。
        """

        alias_map: dict[str, str] = {}

        for alias, dog_name in self.FALLBACK_BREED_ALIAS_MAP.items():
            self._add_alias_mapping(
                target=alias_map,
                alias=alias,
                dog_name=dog_name,
            )

        return alias_map

    def _add_alias_mapping(
            self,
            target: dict[str, str],
            alias: Any,
            dog_name: Any,
    ) -> None:
        """
        添加一条 alias -> dog_name 映射。

        功能：
            对 alias 和 dog_name 做清洗后加入 target。

        安全策略：
            1. 空 alias 跳过。
            2. 空 dog_name 跳过。
            3. 过于泛化的英文 alias 跳过。
            4. alias 统一转小写。

        参数：
            target:
                目标 alias map。

            alias:
                犬种别名。

            dog_name:
                标准犬种名。

        返回值：
            None。
        """

        clean_alias = str(
            alias
            or ""
        ).strip().lower()

        clean_dog_name = str(
            dog_name
            or ""
        ).strip()

        if not clean_alias or not clean_dog_name:
            return

        if self._is_ambiguous_english_alias(
                alias=clean_alias,
        ):
            logger.debug(
                f"DogQueryFilterParser 跳过泛化 alias: {clean_alias} -> {clean_dog_name}"
            )

            return

        target[
            clean_alias
        ] = clean_dog_name

    def _is_ambiguous_english_alias(
            self,
            alias: str,
    ) -> bool:
        """
        判断英文 alias 是否过于泛化。

        功能：
            避免把 hound、terrier、retriever 这类泛化词当成具体犬种名。

        参数：
            alias:
                已经小写化的 alias。

        返回值：
            bool:
                True 表示该 alias 过于泛化，应该跳过。
                False 表示可以使用。
        """

        if self._contains_chinese(
                text=alias,
        ):
            return False

        return alias in self.AMBIGUOUS_ENGLISH_ALIASES

    def parse(
            self,
            question: str,
            user_id: str = "default",
            top_k: int = 5,
            intent: str = "dog_info",
    ) -> RagQuery:
        """
        将用户问题解析成 RagQuery。

        功能：
            1. 校验 question
            2. 解析 metadata filters
            3. 构建 RagQuery

        参数：
            question: str
                用户自然语言问题。

            user_id: str
                用户 ID。
                用于多用户隔离或个性化检索。

            top_k: int
                召回数量。

            intent: str
                用户意图。
                狗狗 RAG 默认使用 dog_info。

        返回值：
            RagQuery：
                标准 RAG 检索请求对象。
        """

        clean_question = self._normalize_question_for_output(
            question=question,
        )

        if not clean_question:
            raise ValueError(
                "question 不能为空"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0"
            )

        filters = self.parse_filters(
            question=clean_question,
        )

        return RagQuery(
            question=clean_question,
            user_id=user_id,
            top_k=top_k,
            filters=filters or {},
            intent=intent,
        )

    def parse_filters(
            self,
            question: str,
    ) -> MetadataFilter | None:
        """
        将用户问题解析成 Chroma metadata filter。

        功能：
            根据关键词规则生成 Chroma where filter。
            如果没有识别到任何过滤条件，则返回 None。

        参数：
            question: str
                用户自然语言问题。

        返回值：
            MetadataFilter | None：
                Chroma metadata filter。
        """

        normalized_question = self._normalize_question(
            question=question,
        )

        if not normalized_question:
            return None

        conditions: list[MetadataFilter] = []

        dog_name = self._detect_dog_name(
            normalized_question=normalized_question,
        )

        if dog_name:
            conditions.append(
                self._eq(
                    field_name="dog_name",
                    value=dog_name,
                )
            )

        size = self._detect_size(
            normalized_question=normalized_question,
        )

        if size:
            conditions.append(
                self._eq(
                    field_name="size",
                    value=size,
                )
            )

        if self._contains_any(
                text=normalized_question,
                keywords=self.LOW_BARKING_KEYWORDS,
        ):
            conditions.append(
                self._lte(
                    field_name="barking_level",
                    value=3,
                )
            )
        elif self._contains_any(
                text=normalized_question,
                keywords=self.HIGH_BARKING_KEYWORDS,
        ):
            conditions.append(
                self._gte(
                    field_name="barking_level",
                    value=4,
                )
            )

        if self._contains_any(
                text=normalized_question,
                keywords=self.LOW_ENERGY_KEYWORDS,
        ):
            conditions.append(
                self._lte(
                    field_name="energy_level",
                    value=3,
                )
            )
        elif self._contains_any(
                text=normalized_question,
                keywords=self.HIGH_ENERGY_KEYWORDS,
        ):
            conditions.append(
                self._gte(
                    field_name="energy_level",
                    value=4,
                )
            )

        if self._contains_any(
                text=normalized_question,
                keywords=self.EASY_TRAINING_KEYWORDS,
        ):
            conditions.append(
                self._gte(
                    field_name="trainability_level",
                    value=4,
                )
            )

        if self._contains_any(
                text=normalized_question,
                keywords=self.APARTMENT_KEYWORDS,
        ):
            conditions.append(
                self._eq(
                    field_name="good_for_apartment",
                    value=True,
                )
            )

        if self._contains_any(
                text=normalized_question,
                keywords=self.BEGINNER_KEYWORDS,
        ):
            conditions.append(
                self._eq(
                    field_name="good_for_beginner",
                    value=True,
                )
            )

        if self._contains_any(
                text=normalized_question,
                keywords=self.CHILDREN_KEYWORDS,
        ):
            conditions.append(
                self._gte(
                    field_name="good_with_young_children_level",
                    value=4,
                )
            )

        if self._contains_any(
                text=normalized_question,
                keywords=self.OTHER_DOGS_KEYWORDS,
        ):
            conditions.append(
                self._gte(
                    field_name="good_with_other_dogs_level",
                    value=4,
                )
            )

        if self._contains_any(
                text=normalized_question,
                keywords=self.LOW_SHEDDING_KEYWORDS,
        ):
            conditions.append(
                self._lte(
                    field_name="shedding_level",
                    value=2,
                )
            )

        if self._contains_any(
                text=normalized_question,
                keywords=self.LOW_DROOLING_KEYWORDS,
        ):
            conditions.append(
                self._lte(
                    field_name="drooling_level",
                    value=2,
                )
            )

        if self._contains_any(
                text=normalized_question,
                keywords=self.LOW_GROOMING_KEYWORDS,
        ):
            conditions.append(
                self._lte(
                    field_name="coat_grooming_frequency_level",
                    value=3,
                )
            )

        return self._build_where_filter(
            conditions=conditions,
        )

    def _alias_matches_question(
            self,
            alias: str,
            normalized_question: str,
    ) -> bool:
        """
        判断 alias 是否命中用户问题。

        功能：
            根据 alias 类型选择不同匹配方式。

        匹配策略：
            1. 中文 alias：
               直接 substring 匹配。
            2. 英文短 alias：
               使用 token 边界匹配，避免误命中。
            3. 英文普通 alias：
               使用单词边界匹配，避免局部字符串误命中。

        参数：
            alias:
                已经小写化的犬种别名。

            normalized_question:
                已经小写化的用户问题。

        返回值：
            bool:
                True 表示命中。
                False 表示未命中。
        """

        if not alias or not normalized_question:
            return False

        if self._contains_chinese(
                text=alias,
        ):
            return alias in normalized_question

        if len(
                alias
        ) <= self.SHORT_ENGLISH_ALIAS_MAX_LENGTH:
            return self._match_english_token(
                alias=alias,
                text=normalized_question,
            )

        return self._match_english_phrase(
            alias=alias,
            text=normalized_question,
        )

    def _match_english_token(
            self,
            alias: str,
            text: str,
    ) -> bool:
        """
        匹配英文短 token。

        功能：
            用于匹配 ah、ig 等短缩写。
            必须作为完整 token 出现，不能只是某个单词的一部分。

        示例：
            alias = "ah"

            可以匹配：
                "tell me about ah"

            不应该匹配：
                "what"

        参数：
            alias:
                英文短别名。

            text:
                用户问题。

        返回值：
            bool:
                是否命中。
        """

        pattern = rf"(?<![a-zA-Z]){re.escape(alias)}(?![a-zA-Z])"

        return re.search(
            pattern,
            text,
        ) is not None


    def _match_english_phrase(
            self,
            alias: str,
            text: str,
    ) -> bool:
        """
        匹配英文 alias 短语。

        功能：
            使用单词边界匹配英文别名，避免误匹配单词片段。

        示例：
            alias = "shih tzu"

            可以匹配：
                "shih tzu lifespan"

            不应该匹配：
                "abcshih tzuxyz"

        参数：
            alias:
                英文 alias。

            text:
                用户问题。

        返回值：
            bool:
                是否命中。
        """

        escaped_alias = re.escape(
            alias
        )

        escaped_alias = escaped_alias.replace(
            "\\ ",
            r"\s+",
        )

        pattern = rf"(?<![a-zA-Z]){escaped_alias}(?![a-zA-Z])"

        return re.search(
            pattern,
            text,
        ) is not None

    def _contains_chinese(
            self,
            text: str,
    ) -> bool:
        """
        判断文本中是否包含中文。

        功能：
            用于区分中文 alias 和英文 alias。
            中文 alias 可以直接 substring 匹配。
            英文 alias 需要更严格的 token / phrase 匹配。

        参数：
            text:
                待检查文本。

        返回值：
            bool:
                True 表示包含中文。
                False 表示不包含中文。
        """

        return any(
            "\u4e00" <= char <= "\u9fff"
            for char in text
        )

    def _detect_dog_name(
            self,
            normalized_question: str,
    ) -> str | None:
        """
        识别犬种名。

        功能：
            根据 alias_dog_name.json 加载出来的别名索引，
            将用户问题中的犬种别名映射成 metadata 中的 dog_name。

        匹配策略：
            1. 长 alias 优先匹配。
               例如先匹配 afghan hound，再匹配 afghan。
            2. 中文 alias 使用 substring 匹配。
            3. 英文 alias 使用单词边界匹配。
            4. 短英文缩写使用更严格的 token 匹配。
            5. 过于泛化的 alias，例如 hound，会在加载阶段跳过。

        参数：
            normalized_question:
                已经小写化和去空格的用户问题。

        返回值：
            str | None:
                识别到的犬种标准英文名。
                如果没有识别到，则返回 None。
        """

        sorted_alias_items = sorted(
            self.breed_alias_map.items(),
            key=lambda item: len(
                item[0]
            ),
            reverse=True,
        )

        for alias, dog_name in sorted_alias_items:

            if self._alias_matches_question(
                    alias=alias,
                    normalized_question=normalized_question,
            ):
                logger.debug(
                    "DogQueryFilterParser 识别到 dog_name，"
                    f"alias={alias}, dog_name={dog_name}"
                )

                return dog_name

        return None

    def _detect_size(
            self,
            normalized_question: str,
    ) -> str | None:
        """
        识别犬种体型。

        功能：
            根据 SIZE_KEYWORDS 将用户问题中的体型描述映射成 metadata.size。

        参数：
            normalized_question: str
                已经小写化和去空格的用户问题。

        返回值：
            str | None：
                small / medium / large / giant。
                如果没有识别到，则返回 None。
        """

        for size, keywords in self.SIZE_KEYWORDS.items():

            if self._contains_any(
                    text=normalized_question,
                    keywords=keywords,
            ):
                return size

        return None

    def _build_where_filter(
            self,
            conditions: list[MetadataFilter],
    ) -> MetadataFilter | None:
        """
        构建 Chroma where filter。

        功能：
            如果没有条件，返回 None。
            如果只有一个条件，直接返回该条件。
            如果有多个条件，使用 $and 组合。

        参数：
            conditions: list[MetadataFilter]
                单个 metadata filter 条件列表。

        返回值：
            MetadataFilter | None：
                Chroma where filter。
        """

        if not conditions:
            return None

        if len(
                conditions,
        ) == 1:
            return conditions[0]

        return {
            "$and": conditions,
        }

    def _eq(
            self,
            field_name: str,
            value: Any,
    ) -> MetadataFilter:
        """
        构建 $eq 条件。

        功能：
            表示 metadata[field_name] 必须等于 value。

        参数：
            field_name: str
                metadata 字段名。

            value: Any
                目标值。

        返回值：
            MetadataFilter：
                Chroma $eq 条件。
        """

        return {
            field_name: {
                "$eq": value,
            }
        }

    def _lte(
            self,
            field_name: str,
            value: int | float,
    ) -> MetadataFilter:
        """
        构建 $lte 条件。

        功能：
            表示 metadata[field_name] 必须小于等于 value。

        参数：
            field_name: str
                metadata 字段名。

            value: int | float
                最大值。

        返回值：
            MetadataFilter：
                Chroma $lte 条件。
        """

        return {
            field_name: {
                "$lte": value,
            }
        }

    def _gte(
            self,
            field_name: str,
            value: int | float,
    ) -> MetadataFilter:
        """
        构建 $gte 条件。

        功能：
            表示 metadata[field_name] 必须大于等于 value。

        参数：
            field_name: str
                metadata 字段名。

            value: int | float
                最小值。

        返回值：
            MetadataFilter：
                Chroma $gte 条件。
        """

        return {
            field_name: {
                "$gte": value,
            }
        }

    def _contains_any(
            self,
            text: str,
            keywords: list[str],
    ) -> bool:
        """
        判断文本中是否包含任意关键词。

        功能：
            遍历关键词列表，只要命中一个就返回 True。

        参数：
            text: str
                待检查文本。

            keywords: list[str]
                关键词列表。

        返回值：
            bool：
                True 表示命中；
                False 表示未命中。
        """

        for keyword in keywords:

            normalized_keyword = keyword.lower()

            if normalized_keyword in text:
                return True

        return False

    def _normalize_question(
            self,
            question: str,
    ) -> str:
        """
        规范化用于规则匹配的问题文本。

        功能：
            1. 处理 None 或空文本
            2. 去除首尾空格
            3. 转成小写，方便英文关键词匹配

        参数：
            question: str
                原始用户问题。

        返回值：
            str：
                规范化后的问题文本。
        """

        return str(
            question
            or ""
        ).strip().lower()

    def _normalize_question_for_output(
            self,
            question: str,
    ) -> str:
        """
        规范化用于输出到 RagQuery 的问题文本。

        功能：
            只去除首尾空格，不转小写。
            这样可以保留用户原始问题的展示效果。

        参数：
            question: str
                原始用户问题。

        返回值：
            str：
                清洗后的用户问题。
        """

        return str(
            question
            or ""
        ).strip()