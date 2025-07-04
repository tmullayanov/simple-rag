from typing import Callable
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from loguru import logger

from simple_rag.chats.chat import ChatModel
from simple_rag.models.classic_static_file_rag import build_classic_rag_model
from simple_rag.models.qna_rag.model import QnAServiceConfig, build_static_file_model
from simple_rag.models.simple_chat_models.raw_chat import build_raw_chat_model
from simple_rag.models.stub_model import StubModel


class ModelCreator:
    _llm: BaseChatModel
    _embeddings: Embeddings
    _models: dict[str, Callable[[BaseChatModel, Embeddings, dict], ChatModel]]
    _config: dict[str, str]

    # FIXME: config should be generalized or the whole approach to models should be changed
    def __init__(
        self, llm: BaseChatModel, embeddings: Embeddings, config: QnAServiceConfig
    ):
        self._llm = llm
        self._embeddings = embeddings
        self._config = config

        self._models = {
            "rag_question_vector": build_static_file_model,
            "classic_rag": build_classic_rag_model,
            "echo": lambda *args: ChatModel(),
            "raw": build_raw_chat_model,
            "stub_model": lambda *args: StubModel(),
        }

    def build(self, name: str) -> ChatModel:
        try:
            logger.debug(f"Building model: {name}")

            builder = self._models[name]
            logger.debug(f"Found builder. Initializing model")

            return builder(self._llm, self._embeddings, self._config)
        except KeyError:
            raise ValueError(f"Unknown model name: {name}")

    def models(self) -> list[str]:
        return list(self._models.keys())
