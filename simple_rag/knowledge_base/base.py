import abc
from typing import Optional


class KnowledgeBaseModel(abc.ABC):
    @abc.abstractmethod
    def query(self, query: str) -> Optional[str]:
        pass
