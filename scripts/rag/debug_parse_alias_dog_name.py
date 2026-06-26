from src.rag.query_parsers.dog_query_filter_parser import DogQueryFilterParser

parser = DogQueryFilterParser()

questions = [
    "shih tzu 的寿命是多少",
    "Shih Tzu 适合新手吗",
    "西施犬掉毛严重吗",
    "affenpinscher 的性格怎么样",
    "猴面梗适合公寓吗",
    "afghan hound 的寿命是多少",
    "阿富汗猎犬掉毛严重吗",
]

for question in questions:
    rag_query = parser.parse(
        question=question,
        user_id="test",
        top_k=5,
        intent="ask_info",
    )

    print(question)
    print(rag_query.filters)
    print("-" * 80)