from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.parser.schema import QueryParseResult

parser = PydanticOutputParser(pydantic_object=QueryParseResult)

QUERY_PARSE_PROMPT = PromptTemplate(
    template="""
你是一个查询解析器，请将用户问题解析为结构化 JSON。

【严格要求】

1️⃣ intent：
必须是以下之一：
- "recommend"
- "ask_info"

2️⃣ filters（非常重要）：
- 必须使用比较操作符："$gte" 或 "$lte"或"$gt"或"$lt"
- ❌ 禁止使用等值（如 barking: 1）
- 示例：
  - 不爱叫 → {{"barking": {{"$lte": 2}}}}
  - 新手 → {{"trainability": {{"$gte": 4}}}}
  - 不掉毛 → {{"shedding": {{"$lte": 2}}}}
  - 小型犬 → {{"height": {{"$lte": 12}}}}
  - 大型犬 → {{"height": {{"$gte": 23.6}}}}

3️⃣ tags：
- 只能从以下枚举中选择（必须是英文）：
  ["temperament", "barking", "trainability", "shedding", "energy"]
- ❌ 禁止输出中文
- ❌ 禁止具体描述（如 "性格温顺"）

4️⃣ features：
- 使用布尔特征（英文）
- 示例：
  - 不流口水 → "low_drooling"
  - 安静 → "low_barking"

5️⃣ dog_name：
- 如果问题中提到具体犬种，则提取犬的品种名称
- 如果没有则什么也不做，不要用“大型犬”、“小型犬”等泛指来赋值dog_name


6️⃣ 输出要求：
- 只输出 JSON
- 不要解释
- 不要多余字段

---

【输出格式示例】

{{
  "intent": "recommend",
  "filters": {{
    "barking": {{"$lte": 2}},
    "trainability": {{"$gte": 4}}
  }},
  "tags": ["temperament"],
  "features": ["low_barking"],
  "dog_name": "Golden Retriever"
}}

---

用户问题：
{query}
""",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)