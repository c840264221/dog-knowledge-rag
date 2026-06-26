"""
Dog Breed Metadata Extractor。

Dog Breed Metadata Extractor（狗狗品种元数据提取器）：
用于从当前项目的 AKC Markdown 犬种文档中提取结构化 metadata。

当前 Markdown 文档结构示例：

# Affenpinscher

## 🏷️ 标签
confident / famously funny / fearless

## 📏 基本信息
- 身高: 9-11.5 inches
- 体重: 7-10 pounds
- 寿命: 12-15 years

## 🧬 性格特征
- Trainability Level: 3
- Energy Level: 3
- Barking Level: 3

本模块只负责：
1. 从 RagDocument.content 中解析 metadata
2. 将解析结果写入 RagDocument.metadata
3. 返回新的 RagDocument

本模块不负责：
1. Markdown 文件加载
2. 文档切块
3. Embedding（向量化）
4. 写入 Chroma
5. Retriever（检索）
"""

from __future__ import annotations

import re
from typing import Any

from src.rag.schemas import RagDocument


MetadataValue = str | int | float | bool


class DogBreedMetadataExtractor:
    """
    狗狗品种元数据提取器。

    Metadata Extractor（元数据提取器）：
    用于把 Markdown 正文中的非结构化或半结构化内容，
    转换成 Chroma metadata filter 可以使用的结构化字段。

    当前版本专门适配你的 AKC Markdown 文档结构：
    1. 一级标题提取 dog_name
    2. 标签章节提取 dog_tags
    3. 基本信息章节提取身高、体重、寿命
    4. 性格特征章节提取 AKC 数值评分
    5. 根据体重、精力、吠叫、适应性推断 size、good_for_beginner、good_for_apartment
    """

    EXTRACTOR_NAME = "dog_breed_metadata_extractor_v2"

    METADATA_SCHEMA = "akc_markdown_dog_breed_v1"

    TRAIT_FIELD_MAP = {
        "affectionate with family": "affectionate_with_family_level",
        "good with young children": "good_with_young_children_level",
        "good with other dogs": "good_with_other_dogs_level",
        "shedding level": "shedding_level",
        "coat grooming frequency": "coat_grooming_frequency_level",
        "drooling level": "drooling_level",
        "openness to strangers": "openness_to_strangers_level",
        "playfulness level": "playfulness_level",
        "watchdog/protective nature": "watchdog_protective_nature_level",
        "watchdog protective nature": "watchdog_protective_nature_level",
        "adaptability level": "adaptability_level",
        "trainability level": "trainability_level",
        "energy level": "energy_level",
        "barking level": "barking_level",
        "mental stimulation needs": "mental_stimulation_needs_level",
    }

    TEXT_TRAIT_FIELD_MAP = {
        "coat type": "coat_type",
        "coat length": "coat_length",
    }

    def __init__(
            self,
            overwrite_existing: bool = False,
    ):
        """
        初始化 DogBreedMetadataExtractor。

        参数：
            overwrite_existing: bool
                是否覆盖 RagDocument.metadata 中已经存在的同名字段。

                False：
                    默认值。
                    如果 Loader 或人工标注已经写入某些 metadata，
                    本 extractor 不会覆盖它们。

                True：
                    如果本 extractor 提取到了同名字段，则覆盖旧值。

        返回值：
            None：
                构造函数无返回值。
        """

        self.overwrite_existing = overwrite_existing

    def extract(
            self,
            document: RagDocument,
    ) -> RagDocument:
        """
        从单个 RagDocument 中提取狗狗品种 metadata。

        功能：
            1. 读取 document.content
            2. 按 Markdown 章节解析 metadata
            3. 合并原始 metadata 和新提取的 metadata
            4. 返回一个新的 RagDocument

        参数：
            document: RagDocument
                RAG 文档对象。
                一般由 MarkdownDocumentLoader 加载 .md 文件得到。

        返回值：
            RagDocument：
                metadata 被增强后的新文档对象。
        """

        content = str(
            document.content
            or ""
        )

        original_metadata = dict(
            document.metadata
            or {}
        )

        extracted_metadata = self.extract_metadata_from_content(
            content=content,
        )

        clean_extracted_metadata = self._remove_invalid_metadata_values(
            metadata=extracted_metadata,
        )

        merged_metadata = self._merge_metadata(
            original_metadata=original_metadata,
            extracted_metadata=clean_extracted_metadata,
        )

        return self._copy_document_with_metadata(
            document=document,
            metadata=merged_metadata,
        )

    def extract_many(
            self,
            documents: list[RagDocument],
    ) -> list[RagDocument]:
        """
        批量提取多个 RagDocument 的 metadata。

        功能：
            对文档列表中的每个 RagDocument 调用 extract 方法。

        参数：
            documents: list[RagDocument]
                待增强 metadata 的文档列表。

        返回值：
            list[RagDocument]：
                metadata 被增强后的文档列表。
        """

        return [
            self.extract(
                document=document,
            )
            for document in documents
        ]

    def extract_metadata_from_content(
            self,
            content: str,
    ) -> dict[str, Any]:
        """
        从 Markdown 正文中提取 metadata。

        功能：
            这是核心解析入口。
            会依次提取：
            1. dog_name
            2. dog_tags
            3. 基本信息字段
            4. 性格特征字段
            5. 推断字段 size、good_for_beginner、good_for_apartment

        参数：
            content: str
                Markdown 文档正文。

        返回值：
            dict[str, Any]：
                原始提取结果。
                后续会过滤掉 None、list、dict 等不适合写入 Chroma metadata 的值。
        """

        sections = self._split_sections(
            content=content,
        )

        metadata: dict[str, Any] = {
            "dog_name": self._extract_dog_name(
                content=content,
            ),
            "dog_tags": self._extract_dog_tags(
                tags_section=sections.get(
                    "tags",
                    "",
                ),
            ),
        }

        basic_info_metadata = self._extract_basic_info_metadata(
            basic_info_section=sections.get(
                "basic_info",
                "",
            ),
        )

        trait_metadata = self._extract_trait_metadata(
            trait_section=sections.get(
                "traits",
                "",
            ),
        )

        metadata.update(
            basic_info_metadata,
        )

        metadata.update(
            trait_metadata,
        )

        metadata["size"] = self._infer_size(
            metadata=metadata,
        )

        metadata["good_for_beginner"] = self._infer_good_for_beginner(
            metadata=metadata,
        )

        metadata["good_for_apartment"] = self._infer_good_for_apartment(
            metadata=metadata,
        )

        metadata["metadata_extractor"] = self.EXTRACTOR_NAME
        metadata["metadata_schema"] = self.METADATA_SCHEMA

        return metadata

    def _split_sections(
            self,
            content: str,
    ) -> dict[str, str]:
        """
        按 Markdown 二级标题拆分章节。

        功能：
            识别当前文档中的章节：
            1. 标签
            2. 基本信息
            3. 性格特征
            4. 关于该犬种
            5. 历史
            6. 养护指南

            当前 extractor 主要使用：
            1. tags
            2. basic_info
            3. traits

        参数：
            content: str
                Markdown 正文。

        返回值：
            dict[str, str]：
                key 是标准化后的章节名；
                value 是该章节下面的正文。
        """

        sections: dict[str, list[str]] = {}

        current_section_key: str | None = None

        for line in content.splitlines():

            heading_match = re.match(
                pattern=r"^\s*##\s+(.+?)\s*$",
                string=line,
            )

            if heading_match:
                current_section_key = self._normalize_section_title(
                    title=heading_match.group(
                        1,
                    ),
                )

                if current_section_key:
                    sections.setdefault(
                        current_section_key,
                        [],
                    )

                continue

            if current_section_key:
                sections[
                    current_section_key
                ].append(
                    line,
                )

        return {
            key: "\n".join(
                lines,
            ).strip()
            for key, lines in sections.items()
        }

    def _normalize_section_title(
            self,
            title: str,
    ) -> str | None:
        """
        将 Markdown 章节标题转换成内部标准 key。

        功能：
            把带 emoji 的中文标题转换成稳定英文 key。
            例如：
                🏷️ 标签 -> tags
                📏 基本信息 -> basic_info
                🧬 性格特征 -> traits

        参数：
            title: str
                原始章节标题。

        返回值：
            str | None：
                标准化后的章节 key。
                不关心的章节返回 None。
        """

        normalized_title = str(
            title
            or ""
        ).strip()

        if "标签" in normalized_title:
            return "tags"

        if "基本信息" in normalized_title:
            return "basic_info"

        if "性格特征" in normalized_title:
            return "traits"

        if "关于该犬种" in normalized_title:
            return "about"

        if "历史" in normalized_title:
            return "history"

        if "养护指南" in normalized_title:
            return "care"

        return None

    def _extract_dog_name(
            self,
            content: str,
    ) -> str | None:
        """
        从 Markdown 一级标题中提取犬种名称。

        功能：
            当前文档中犬种名位于一级标题：
                # Affenpinscher

        参数：
            content: str
                Markdown 正文。

        返回值：
            str | None：
                成功时返回犬种名；
                失败时返回 None。
        """

        for line in content.splitlines():

            match = re.match(
                pattern=r"^\s*#\s+(.+?)\s*$",
                string=line,
            )

            if not match:
                continue

            raw_name = match.group(
                1,
            ).strip()

            clean_name = self._clean_heading_text(
                text=raw_name,
            )

            if clean_name:
                return clean_name

        return None

    def _extract_dog_tags(
            self,
            tags_section: str,
    ) -> str | None:
        """
        从标签章节中提取 dog_tags。

        功能：
            当前标签格式是：
                confident / famously funny / fearless

            由于 Chroma metadata 不适合直接保存 list[str]，
            所以这里保存为字符串：
                confident / famously funny / fearless

        参数：
            tags_section: str
                标签章节正文。

        返回值：
            str | None：
                成功时返回标签字符串；
                没有标签时返回 None。
        """

        raw_text = str(
            tags_section
            or ""
        ).strip()

        if not raw_text:
            return None

        lines = [
            line.strip().lstrip(
                "-* "
            ).strip()
            for line in raw_text.splitlines()
            if line.strip()
        ]

        if not lines:
            return None

        tags_text = " / ".join(
            lines,
        )

        tags = [
            tag.strip()
            for tag in re.split(
                pattern=r"\s*/\s*|、|，|,",
                string=tags_text,
            )
            if tag.strip()
        ]

        if not tags:
            return None

        return " / ".join(
            tags,
        )

    def _extract_basic_info_metadata(
            self,
            basic_info_section: str,
    ) -> dict[str, Any]:
        """
        从基本信息章节中提取 metadata。

        功能：
            解析以下字段：
            1. 身高 -> height_min_inches / height_max_inches
            2. 体重 -> weight_min_pounds / weight_max_pounds
            3. 寿命 -> lifespan_min_years / lifespan_max_years

        参数：
            basic_info_section: str
                基本信息章节正文。

        返回值：
            dict[str, Any]：
                基本信息 metadata。
        """

        key_values = self._parse_bullet_key_values(
            section_text=basic_info_section,
        )

        metadata: dict[str, Any] = {}

        height_value = key_values.get(
            "身高",
        )

        if height_value:
            height_min, height_max = self._parse_numeric_range(
                value=height_value,
            )

            metadata[
                "height_min_inches"
            ] = height_min

            metadata[
                "height_max_inches"
            ] = height_max

        weight_value = key_values.get(
            "体重",
        )

        if weight_value:
            weight_min, weight_max = self._parse_numeric_range(
                value=weight_value,
            )

            metadata[
                "weight_min_pounds"
            ] = weight_min

            metadata[
                "weight_max_pounds"
            ] = weight_max

        lifespan_value = key_values.get(
            "寿命",
        )

        if lifespan_value:
            lifespan_min, lifespan_max = self._parse_numeric_range(
                value=lifespan_value,
            )

            metadata[
                "lifespan_min_years"
            ] = lifespan_min

            metadata[
                "lifespan_max_years"
            ] = lifespan_max

        return metadata

    def _extract_trait_metadata(
            self,
            trait_section: str,
    ) -> dict[str, Any]:
        """
        从性格特征章节中提取 metadata。

        功能：
            解析 AKC 风格的评分字段，例如：
                - Trainability Level: 3
                - Energy Level: 3
                - Barking Level: 3

            数值型字段会转换成 int。
            文本型字段会保留 str。

        参数：
            trait_section: str
                性格特征章节正文。

        返回值：
            dict[str, Any]：
                性格特征 metadata。
        """

        key_values = self._parse_bullet_key_values(
            section_text=trait_section,
        )

        metadata: dict[str, Any] = {}

        for raw_key, raw_value in key_values.items():

            normalized_key = self._normalize_field_key(
                key=raw_key,
            )

            if normalized_key in self.TRAIT_FIELD_MAP:

                metadata_key = self.TRAIT_FIELD_MAP[
                    normalized_key
                ]

                metadata[
                    metadata_key
                ] = self._parse_level(
                    value=raw_value,
                )

                continue

            if normalized_key in self.TEXT_TRAIT_FIELD_MAP:

                metadata_key = self.TEXT_TRAIT_FIELD_MAP[
                    normalized_key
                ]

                metadata[
                    metadata_key
                ] = self._clean_text_value(
                    value=raw_value,
                )

        return metadata

    def _parse_bullet_key_values(
            self,
            section_text: str,
    ) -> dict[str, str]:
        """
        解析 Markdown 列表中的 key-value 字段。

        功能：
            支持当前文档中的格式：
                - 身高: 9-11.5 inches
                - Trainability Level: 3

        参数：
            section_text: str
                某个 Markdown 章节正文。

        返回值：
            dict[str, str]：
                key 是字段名；
                value 是字段值。
        """

        key_values: dict[str, str] = {}

        for line in section_text.splitlines():

            match = re.match(
                pattern=r"^\s*[-*]\s*(.+?)\s*[:：]\s*(.+?)\s*$",
                string=line,
            )

            if not match:
                continue

            key = match.group(
                1,
            ).strip()

            value = match.group(
                2,
            ).strip()

            if key:
                key_values[
                    key
                ] = value

        return key_values

    def _parse_numeric_range(
            self,
            value: str,
    ) -> tuple[int | float | None, int | float | None]:
        """
        从文本中解析数字范围。

        功能：
            支持：
                9-11.5 inches
                7-10 pounds
                12-15 years
                10 pounds

        参数：
            value: str
                原始字段值。

        返回值：
            tuple[int | float | None, int | float | None]：
                第一个值是最小值；
                第二个值是最大值。
                如果只解析到一个数字，则 min 和 max 相同。
        """

        numbers = re.findall(
            pattern=r"\d+(?:\.\d+)?",
            string=str(
                value
                or ""
            ),
        )

        if not numbers:
            return None, None

        parsed_numbers = [
            self._to_int_if_possible(
                value=float(
                    number,
                ),
            )
            for number in numbers
        ]

        if len(
                parsed_numbers,
        ) == 1:
            return parsed_numbers[0], parsed_numbers[0]

        return parsed_numbers[0], parsed_numbers[1]

    def _parse_level(
            self,
            value: str,
    ) -> int | None:
        """
        解析 AKC 评分等级。

        功能：
            当前文档中的评分一般是 1-5 的整数。
            例如：
                Trainability Level: 3

        参数：
            value: str
                原始评分文本。

        返回值：
            int | None：
                成功时返回 1-5；
                失败时返回 None。
        """

        match = re.search(
            pattern=r"\b([1-5])\b",
            string=str(
                value
                or ""
            ),
        )

        if not match:
            return None

        return int(
            match.group(
                1,
            )
        )

    def _infer_size(
            self,
            metadata: dict[str, Any],
    ) -> str | None:
        """
        根据体重推断犬种体型 size。

        功能：
            使用 weight_max_pounds 粗略推断：
                <= 25 pounds -> small
                <= 60 pounds -> medium
                <= 100 pounds -> large
                > 100 pounds -> giant

        参数：
            metadata: dict[str, Any]
                已经提取到的 metadata。

        返回值：
            str | None：
                small、medium、large、giant 之一；
                无法判断时返回 None。
        """

        weight_max = metadata.get(
            "weight_max_pounds",
        )

        if not isinstance(
                weight_max,
                (
                    int,
                    float,
                ),
        ):
            return None

        if weight_max <= 25:
            return "small"

        if weight_max <= 60:
            return "medium"

        if weight_max <= 100:
            return "large"

        return "giant"

    def _infer_good_for_beginner(
            self,
            metadata: dict[str, Any],
    ) -> bool | None:
        """
        推断是否适合新手饲养。

        功能：
            当前 Markdown 没有直接提供 good_for_beginner 字段，
            所以这里根据以下字段保守推断：
            1. trainability_level
            2. energy_level
            3. mental_stimulation_needs_level
            4. adaptability_level

            简化规则：
            - 训练难度不高、精力不极端、精神刺激需求不极端、适应性较好 -> True
            - 可训练性很低，或精力极高，或精神刺激需求极高 -> False
            - 信息不足 -> None

        参数：
            metadata: dict[str, Any]
                已经提取到的 metadata。

        返回值：
            bool | None：
                True 表示适合新手；
                False 表示不适合新手；
                None 表示无法判断。
        """

        trainability_level = metadata.get(
            "trainability_level",
        )

        energy_level = metadata.get(
            "energy_level",
        )

        mental_stimulation_needs_level = metadata.get(
            "mental_stimulation_needs_level",
        )

        adaptability_level = metadata.get(
            "adaptability_level",
        )

        if (
                isinstance(
                    trainability_level,
                    int,
                )
                and trainability_level <= 2
        ):
            return False

        if (
                isinstance(
                    energy_level,
                    int,
                )
                and energy_level >= 5
        ):
            return False

        if (
                isinstance(
                    mental_stimulation_needs_level,
                    int,
                )
                and mental_stimulation_needs_level >= 5
        ):
            return False

        if (
                isinstance(
                    trainability_level,
                    int,
                )
                and isinstance(
                    energy_level,
                    int,
                )
                and isinstance(
                    mental_stimulation_needs_level,
                    int,
                )
                and isinstance(
                    adaptability_level,
                    int,
                )
        ):

            if (
                    trainability_level >= 3
                    and energy_level <= 3
                    and mental_stimulation_needs_level <= 3
                    and adaptability_level >= 3
            ):
                return True

        return None

    def _infer_good_for_apartment(
            self,
            metadata: dict[str, Any],
    ) -> bool | None:
        """
        推断是否适合公寓饲养。

        功能：
            当前 Markdown 没有直接提供 good_for_apartment 字段，
            所以这里根据以下字段保守推断：
            1. size
            2. energy_level
            3. barking_level
            4. adaptability_level

            简化规则：
            - small / medium 且精力不高、吠叫不高、适应性较好 -> True
            - 吠叫很高或精力极高 -> False
            - 信息不足 -> None

        参数：
            metadata: dict[str, Any]
                已经提取到的 metadata。

        返回值：
            bool | None：
                True 表示适合公寓；
                False 表示不适合公寓；
                None 表示无法判断。
        """

        size = metadata.get(
            "size",
        )

        energy_level = metadata.get(
            "energy_level",
        )

        barking_level = metadata.get(
            "barking_level",
        )

        adaptability_level = metadata.get(
            "adaptability_level",
        )

        if (
                isinstance(
                    barking_level,
                    int,
                )
                and barking_level >= 5
        ):
            return False

        if (
                isinstance(
                    energy_level,
                    int,
                )
                and energy_level >= 5
        ):
            return False

        if (
                size in {
                    "small",
                    "medium",
                }
                and isinstance(
                    energy_level,
                    int,
                )
                and isinstance(
                    barking_level,
                    int,
                )
                and isinstance(
                    adaptability_level,
                    int,
                )
        ):

            if (
                    energy_level <= 3
                    and barking_level <= 3
                    and adaptability_level >= 3
            ):
                return True

        return None

    def _merge_metadata(
            self,
            original_metadata: dict[str, Any],
            extracted_metadata: dict[str, MetadataValue],
    ) -> dict[str, MetadataValue]:
        """
        合并原始 metadata 和新提取的 metadata。

        功能：
            1. 保留原始 metadata
            2. 写入新提取 metadata
            3. 根据 overwrite_existing 决定是否覆盖同名字段

        参数：
            original_metadata: dict[str, Any]
                RagDocument 原始 metadata。

            extracted_metadata: dict[str, MetadataValue]
                本 extractor 新提取的 metadata。

        返回值：
            dict[str, MetadataValue]：
                合并后的 metadata。
        """

        merged_metadata = self._remove_invalid_metadata_values(
            metadata=original_metadata,
        )

        for key, value in extracted_metadata.items():

            if (
                    key in merged_metadata
                    and not self.overwrite_existing
            ):
                continue

            merged_metadata[
                key
            ] = value

        return merged_metadata

    def _remove_invalid_metadata_values(
            self,
            metadata: dict[str, Any],
    ) -> dict[str, MetadataValue]:
        """
        删除不适合写入 Chroma metadata 的值。

        功能：
            Chroma metadata 推荐使用扁平基础类型：
                str、int、float、bool

            因此这里会过滤掉：
                None、list、dict、tuple、set 等复杂类型。

        参数：
            metadata: dict[str, Any]
                原始 metadata。

        返回值：
            dict[str, MetadataValue]：
                清洗后的 metadata。
        """

        clean_metadata: dict[str, MetadataValue] = {}

        for key, value in metadata.items():

            if value is None:
                continue

            if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
            ):
                clean_metadata[
                    key
                ] = value

        return clean_metadata

    def _copy_document_with_metadata(
            self,
            document: RagDocument,
            metadata: dict[str, MetadataValue],
    ) -> RagDocument:
        """
        复制 RagDocument 并替换 metadata。

        功能：
            使用 Pydantic v2 的 model_copy 创建新对象，
            避免直接修改原始 document。

            这样做可以保证：
            1. Loader 输出的原始文档不被污染
            2. Extractor 是纯转换模块
            3. 后续测试更容易判断输入和输出

        参数：
            document: RagDocument
                原始 RAG 文档。

            metadata: dict[str, MetadataValue]
                新 metadata。

        返回值：
            RagDocument：
                metadata 被替换后的新文档。
        """

        if hasattr(
                document,
                "model_copy",
        ):
            return document.model_copy(
                update={
                    "metadata": metadata,
                }
            )

        if hasattr(
                document,
                "copy",
        ):
            return document.copy(
                update={
                    "metadata": metadata,
                }
            )

        raise TypeError(
            "RagDocument 必须支持 Pydantic model_copy 或 copy 方法"
        )

    def _normalize_field_key(
            self,
            key: str,
    ) -> str:
        """
        标准化字段名。

        功能：
            将字段名转换成小写、去除多余空格，
            方便和 TRAIT_FIELD_MAP 做匹配。

        参数：
            key: str
                原始字段名。

        返回值：
            str：
                标准化后的字段名。
        """

        normalized_key = str(
            key
            or ""
        ).strip().lower()

        normalized_key = normalized_key.replace(
            "/",
            "/",
        )

        normalized_key = re.sub(
            pattern=r"\s+",
            repl=" ",
            string=normalized_key,
        )

        return normalized_key

    def _clean_heading_text(
            self,
            text: str,
    ) -> str | None:
        """
        清洗 Markdown 标题文本。

        功能：
            去掉标题两侧空格和多余的 Markdown 符号。

        参数：
            text: str
                原始标题文本。

        返回值：
            str | None：
                清洗后的标题；
                如果为空则返回 None。
        """

        clean_text = str(
            text
            or ""
        ).strip()

        clean_text = clean_text.strip(
            "# "
        ).strip()

        if not clean_text:
            return None

        return clean_text

    def _clean_text_value(
            self,
            value: str,
    ) -> str | None:
        """
        清洗普通文本字段值。

        功能：
            用于清洗 Coat Type、Coat Length 等字段。

        参数：
            value: str
                原始字段值。

        返回值：
            str | None：
                清洗后的字符串；
                如果为空则返回 None。
        """

        clean_value = str(
            value
            or ""
        ).strip()

        clean_value = re.sub(
            pattern=r"\s+",
            repl=" ",
            string=clean_value,
        )

        if not clean_value:
            return None

        return clean_value

    def _to_int_if_possible(
            self,
            value: float,
    ) -> int | float:
        """
        将 float 尽量转换成 int。

        功能：
            例如：
                9.0 -> 9
                11.5 -> 11.5

            这样 metadata 更干净。

        参数：
            value: float
                原始数字。

        返回值：
            int | float：
                如果没有小数部分，返回 int；
                否则返回 float。
        """

        if value.is_integer():
            return int(
                value,
            )

        return value