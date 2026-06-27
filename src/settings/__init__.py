from src.settings.app import AppSettings
from src.settings.llm import LLMSettings
from src.settings.path import PathSettings
from src.settings.runtime import RuntimeSettings
from src.settings.vectorstore import VectorStoreSettings

from src.settings.logging import LoggingSettings
from src.settings.trace import TraceSettings
from src.settings.tool import ToolSettings
from src.settings.embedding import EmbeddingSettings
from src.settings.reranker import RerankerSettings
from src.settings.memory import MemorySettings
from src.settings.rag import RagSettings
from src.settings.observability import ObservabilitySettings


class Settings:

    def __init__(self):
        self.app = AppSettings()

        self.llm = LLMSettings()

        self.runtime = RuntimeSettings()

        self.vectorstore = VectorStoreSettings()

        self.path = PathSettings()

        self.logging = LoggingSettings()

        self.trace = TraceSettings()

        self.tool = ToolSettings()

        self.embedding = EmbeddingSettings()

        self.reranker = RerankerSettings()

        self.memory = MemorySettings()

        self.rag = RagSettings()

        self.observability = ObservabilitySettings()


settings = Settings()

