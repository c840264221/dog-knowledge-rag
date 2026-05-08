from src.graph.build_graph import build_graph
from src.vectorstore.vector_store import load_vector_store
from src.embedding.embedder import get_embedding
from src.config import CHROMA_DB_DIR
from src.graph.sub_graphs.main_graph import build_main_graph


# embedding = get_embedding()
# db = load_vector_store(embedding, CHROMA_DB_DIR)
# app = build_graph(db)
app_2 = build_main_graph()

# def run(question: str):
#     result = app.invoke({
#         "question": question
#     })
#
#     return result["answer"]

def run_main_graph(question: str):
    result = app_2.invoke({
        "question": question
    })

    return result["answer"]