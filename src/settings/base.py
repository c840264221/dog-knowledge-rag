from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv


load_dotenv()

class BaseAppSettings(BaseSettings):

    model_config = SettingsConfigDict(
        # env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )