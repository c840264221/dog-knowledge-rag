from src.ingestion.markdown_file_loader import load_markdown_files
from src.ingestion.splitter import split_markdown
from src.embedding.embedder import get_embedding
from src.vectorstore.vector_store import build_vector_store, reset_chroma
from src.config import CHROMA_DB_DIR, DOG_MD_DATA_DIR

reset_chroma(CHROMA_DB_DIR)
docs = load_markdown_files(DOG_MD_DATA_DIR)
# chunks = split_docs(docs)
chunks = split_markdown(docs)

embeddings = get_embedding()

build_vector_store(chunks, embeddings)