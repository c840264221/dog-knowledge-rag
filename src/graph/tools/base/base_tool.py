from abc import ABC, abstractmethod
from src.graph.tools.schemas.tool_metadata import ToolMetadata


class BaseTool(ABC):

    metadata: ToolMetadata

    @abstractmethod
    def run(self, args: dict):
        pass