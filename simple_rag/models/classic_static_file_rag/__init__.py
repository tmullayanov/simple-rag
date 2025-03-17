from typing import Optional

from langchain.chat_models.base import BaseChatModel
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore

from loguru import logger

from .csv_parser import parse_csv
from .model import ClassicRagModel
from .store import populate_vector_store, embeddings

_store: Optional[VectorStore] = None


def get_store(filename: str) -> VectorStore:
    global _store

    if _store is None:
        logger.debug("Populating VectorStore for classic rag.")
        _store = InMemoryVectorStore(embeddings)
        docs = parse_csv(filename)
        _store.add_documents(docs)
        logger.debug("VectorStore populated successfully.")

    return _store


def build_classic_rag_model(llm: BaseChatModel, config: dict[str, str]):
    # FIXME: use different key from config
    store = get_store(config["qna_path"])
    return ClassicRagModel(llm=llm, store=store)
