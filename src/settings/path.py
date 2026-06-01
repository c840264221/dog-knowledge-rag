from pathlib import Path

from src.settings.base import BaseAppSettings


class PathSettings(BaseAppSettings):

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    DATA_DIR: Path = BASE_DIR / "data"

    LOG_DIR: Path = BASE_DIR / "logs"

    CACHE_DIR: Path = BASE_DIR / "models_cache"

    # CHROMA_DB_DIR: Path = BASE_DIR / "chroma_db"

    CHROMA_DB_DIR: Path = BASE_DIR / "chroma_db"

    CHECKPOINT_DIR: Path = DATA_DIR / "checkpoints_db"

    MEMORY_DIR: Path = DATA_DIR / "memory_db"

    USER_DIR: Path = DATA_DIR / "user"

    DOG_MD_DATA_DIR: Path = DATA_DIR / "dog_markdown"

    DOG_NAME_JSON_PATH: Path = DATA_DIR / "dog_names.json"

    DOG_NAME_ALIAS_JSON_PATH: Path = DATA_DIR / "alias_dog_name.json"

    DOG_DATA_JSON_PATH: Path = DATA_DIR / "dogs.json"

    FILTER_RULES_PATH: Path = DATA_DIR / "filter_rules.json"

    INTENT_RULES_PATH: Path = DATA_DIR / "intent_rules.json"

    TAG_RULES_PATH: Path = DATA_DIR / "tag_rules.json"

    CHECKPOINTS_DB_PATH: Path = (
        CHECKPOINT_DIR / "checkpoints.db"
    )

    MEMORY_DB_PATH: Path = (
        MEMORY_DIR / "memory.db"
    )

    USER_FILE: Path = (
        USER_DIR / "user.json"
    )
