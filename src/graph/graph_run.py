from src.graph.build_graph import build_graph
from src.vectorstore.vector_store import load_vector_store
from src.embedding.embedder import get_embedding
from src.config import CHROMA_DB_DIR


embedding = get_embedding()
db = load_vector_store(embedding, CHROMA_DB_DIR)
app = build_graph(db)


def run(question: str):
    result = app.invoke({
        "question": question
    })

    return result["answer"]