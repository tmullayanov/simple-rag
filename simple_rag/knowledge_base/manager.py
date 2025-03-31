from math import e
from typing import Callable
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from loguru import logger

from .base import KnowledgeBaseModel


class KnowledgeBaseManager:
    """
    Provides access to registered knowledge base models
    """

    _llm: BaseChatModel
    _embeddings: Embeddings

    def __init__(self, llm: BaseChatModel, embeddings: Embeddings):
        self._llm = llm
        self._embeddings = embeddings

    models: dict[str, KnowledgeBaseModel] = {}
    builders: dict[str, Callable[[], KnowledgeBaseModel]] = {}

    @staticmethod
    def register_model(key: str, model: Callable[[], KnowledgeBaseModel]):
        logger.debug(f"Registering model by {key=}")
        KnowledgeBaseManager.builders.update({key: model})

    def get_model(self, key: str):
        try:
            if key not in self.models:
                logger.debug(f"Model {key=} not found, creating from builders...")
                self.models[key] = self.builders[key]()

            return self.models[key]
        except KeyError as e:
            logger.exception(f"Model {key=} not found")
            raise NoSuchModelError(f"{key=}")


class NoSuchModelError(Exception):
    pass
