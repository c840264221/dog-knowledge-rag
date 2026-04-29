import json
from src.config import DOG_NAME_ALIAS_JSON_PATH

_alias_cache = None

def get_alias_dict():
    global _alias_cache

    if _alias_cache is None:
        print("📦 加载 alias_dict...")
        with open(DOG_NAME_ALIAS_JSON_PATH, "r", encoding="utf-8") as f:
            _alias_cache = json.load(f)

    return _alias_cache