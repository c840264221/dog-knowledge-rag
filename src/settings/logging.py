from src.settings.base import BaseAppSettings


class LoggingSettings(BaseAppSettings):

    level: str = "INFO"

    log_dir: str = "logs"

    enable_file_log: bool = True

    enable_console_log: bool = True

    rotation: str = "10 MB"

    retention: str = "7 days"