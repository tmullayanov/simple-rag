from typing import Optional

from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

from loguru import logger

from simple_rag.models.classic_static_file_rag.kb_model import ClassicRagKnowledgeBase

from .csv_parser import parse_csv
from .chat_model import ClassicRagModel

_store: Optional[VectorStore] = None


def get_store(embeddings: Embeddings, filename: str) -> VectorStore:
    global _store

    if _store is None:
        logger.debug("Populating VectorStore for classic rag.")
        _store = make_store(embeddings)
        docs = parse_csv(filename)
        _store.add_documents(docs)
        logger.debug("VectorStore populated successfully.")

    return _store


def make_store(embeddings: Embeddings) -> VectorStore:
    return Chroma(
        collection_name="classis_rag",
        embedding_function=embeddings,
        persist_directory="./chroma_store",
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_classic_rag_model(
    llm: BaseChatModel, embeddings: Embeddings, config: dict[str, str]
):
    # FIXME: use different key from config
    store = get_store(embeddings, config["qna_path"])
    return ClassicRagModel(llm=llm, store=store)


def build_classic_rag_knowledgebase_model(
    llm: BaseChatModel, embeddings: Embeddings, config: dict[str, str]
):
    store = get_store(embeddings, config["qna_path"])
    return ClassicRagKnowledgeBase(llm=llm, store=store)
