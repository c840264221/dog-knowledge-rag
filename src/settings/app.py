from src.settings.base import BaseAppSettings


class AppSettings(BaseAppSettings):

    app_name: str = "Dog Knowledge RAG"

    debug: bool = True

    log_level: str = "INFO"