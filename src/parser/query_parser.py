from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import  RunnableLambda
from src.parser.schema import QueryParseResult
from src.parser.prompt import QUERY_PARSE_PROMPT
from src.models.llm import get_llm, safe_llm_invoke
from src.parser.schema import Intent

llm = get_llm()
parser = PydanticOutputParser(pydantic_object=QueryParseResult)


def parse_query_with_llm(query: str):
    safe_llm = RunnableLambda(
        lambda x: safe_llm_invoke(
            llm=llm,
            prompt=x,
            fallback_response="调用LLM失败"
        )
    )
    chain = QUERY_PARSE_PROMPT | safe_llm | parser

    try:
        result = chain.invoke({"query": query})
        # return result.dict()
        # 新版后用model_dump()输出
        return result

    except Exception as e:
        print("❌ LLM解析失败:", e)

        # fallback
        return QueryParseResult(
            intent=Intent.GENERAL,
            filters={},
            tags=["general"],
            features=["general"],
            dog_name=None
        )


if __name__ == '__main__':
    a = parse_query_with_llm("金毛的性格")
    print(a)