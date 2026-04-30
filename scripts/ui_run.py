import gradio as gr

from src.service.recommender import parse_preferences, recommend
from src.service.dog_data_loader import load_dog_data  # 你需要自己实现


dogs = load_dog_data()


def recommend_fn(query):
    prefs = parse_preferences(query)
    results = recommend(dogs, prefs)

    if not results:
        return "没有找到合适的狗狗 😢"

    output = ""
    for name, score in results:
        output += f"{name} ⭐ {score}\n"

    return output


demo = gr.Interface(
    fn=recommend_fn,
    inputs=gr.Textbox(
        placeholder="例如：适合新手 + 不掉毛 + 小型犬"
    ),
    outputs=gr.Textbox(),
    title="🐶 Dog Recommender",
    description="根据你的需求推荐适合的狗狗",
)

demo.launch()