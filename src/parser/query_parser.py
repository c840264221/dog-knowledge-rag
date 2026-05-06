from langchain_core.output_parsers import PydanticOutputParser
from src.parser.schema import QueryParseResult
from src.parser.prompt import QUERY_PARSE_PROMPT
from src.models.llm import get_llm

llm = get_llm()
parser = PydanticOutputParser(pydantic_object=QueryParseResult)


def parse_query_with_llm(query: str):
    chain = QUERY_PARSE_PROMPT | llm | parser

    try:
        result = chain.invoke({"query": query})
        # return result.dict()
        # 新版后用model_dump()输出
        return result.model_dump()

    except Exception as e:
        print("❌ LLM解析失败:", e)

        # fallback
        return {
            "intent": "general",
            "filters": {},
            "tags": ["general"],
            "features": ["general"],
            "dog_name": None
        }


if __name__ == '__main__':
    a = parse_query_with_llm("金毛的性格")
    print(a)