import os
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOG_MD_DATA_DIR = os.path.join(BASE_DIR,os.getenv("DOG_MD_DATA_DIR", "data/dog_markdown"))
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen:1.8b")
CHROMA_DB_DIR = os.path.join(BASE_DIR,os.getenv("CHROMA_DB_DIR", "chroma_db"))
HF_TOKEN = os.getenv("HF_TOKEN", "")
EMBEDDING_MODEL = "BAAI/bge-small-zh"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CACHE_DIR = os.path.join(BASE_DIR,"models_cache")

DOG_NAME_JSON_PATH = os.path.join(BASE_DIR, "data", "dog_names.json")
DOG_NAME_ALIAS_JSON_PATH = os.path.join(BASE_DIR, "data", "alias_dog_name.json")
DOG_DATA_JSON_PATH = os.path.join(BASE_DIR, "data", "dogs.json")
FILTER_RULES_PATH = os.path.join(BASE_DIR, "data", "filter_rules.json")
INTENT_RULES_PATH = os.path.join(BASE_DIR, "data", "intent_rules.json")
TAG_RULES_PATH = os.path.join(BASE_DIR, "data", "tag_rules.json")

if __name__ == "__main__":
    print(DOG_MD_DATA_DIR)
    print(CHROMA_DB_DIR)
    print(CACHE_DIR)