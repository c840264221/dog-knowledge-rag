from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import  RunnableLambda
from src.parser.schema import QueryParseResult
from src.parser.prompt import QUERY_PARSE_PROMPT
# from src.models.llm import get_instance_llm, safe_llm_ainvoke, get_chinese_llm
from src.parser.schema import Intent


parser = PydanticOutputParser(pydantic_object=QueryParseResult)


async def parse_query_with_llm(query: str):

    def get_llm_provider():
        from src.runtime.container.init import container
        return container.get("llm")

    llm_provider = get_llm_provider()

    # chinese_llm = llm_provider.chinese_llm
    chinese_llm = llm_provider.main_llm

    async def create_async_safe_llm_ainvoke(x):
        return await llm_provider.safe_ainvoke(
            llm=chinese_llm,
            prompt=x,
            fallback_response="调用LLM失败"
        )


    # safe_llm = RunnableLambda(
    #         afunc=lambda x: safe_llm_ainvoke(
    #         llm=llm,
    #         prompt=x,
    #         fallback_response="调用LLM失败"
    #     )
    # )
    safe_llm = RunnableLambda(create_async_safe_llm_ainvoke)

    chain = QUERY_PARSE_PROMPT | safe_llm | parser

    try:
        result = await chain.ainvoke({"query": query})
        # return result.dict()
        # 新版后用model_dump()输出
        return result

    except Exception as e:
        print("❌ LLM解析失败:", e)

        # fallback
        return QueryParseResult(
            intent=Intent.GENERAL.value,
            filters={},
            tags=["general"],
            features=["general"],
            dog_name=None
        )


if __name__ == '__main__':
    a = parse_query_with_llm("金毛的性格")
    print(a)