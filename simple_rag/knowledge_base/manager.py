from math import e
from typing import Callable
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from loguru import logger

from simple_rag.models.classic_static_file_rag import (
    build_classic_rag_knowledgebase_model,
)

from .base import KnowledgeBaseModel


class KnowledgeBaseManager:
    """
    Provides access to registered knowledge base models
    """

    _llm: BaseChatModel
    _embeddings: Embeddings
    _app_config: dict

    builders: dict[
        str, Callable[[BaseChatModel, Embeddings, dict], KnowledgeBaseModel]
    ] = {"classic": build_classic_rag_knowledgebase_model}

    def __init__(self, llm: BaseChatModel, embeddings: Embeddings, app_config: dict):
        self._llm = llm
        self._embeddings = embeddings
        self._app_config = app_config

    models: dict[str, KnowledgeBaseModel] = {}

    def available_models(self) -> list[str]:
        built_models = set(self.models.keys())
        models_from_builders = set(self.builders.keys())
        
        all_models = built_models.union(models_from_builders)

        return list(all_models)

    @staticmethod
    def register_model(key: str, model: Callable[[], KnowledgeBaseModel]):
        logger.debug(f"Registering model by {key=}")
        KnowledgeBaseManager.builders.update({key: model})

    def get_model(self, key: str):
        try:
            if key not in self.models:
                logger.debug(f"Model {key=} not found, creating from builders...")
                self.models[key] = KnowledgeBaseManager.builders[key](
                    self._llm, self._embeddings, self._app_config
                )

            return self.models[key]
        except KeyError as e:
            logger.exception(f"Model {key=} not found")
            raise NoSuchModelError(f"{key=}")


class NoSuchModelError(Exception):
    pass
