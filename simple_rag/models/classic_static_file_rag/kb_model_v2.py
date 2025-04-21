from loguru import logger
import pandas as pd
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from simple_rag.knowledge_base.base import KnowledgeBaseModel
from simple_rag.knowledge_base.store.default_store import Store


def init_support_kb_model(
    llm: BaseChatModel, embeddings: Embeddings, app_cfg: dict, *args, **kwargs
):
    db_link = "sqlite:///support.kb.db"
    model_name = "sample_kbase"
    store = Store(db_cfg=app_cfg["db_cfg"], vectorstore_cfg=app_cfg["vectorstore_cfg"])

    if store.is_empty:
        logger.warning("Store is empty, populating it with data...")
        df = pd.read_csv("assets/support_kbase.csv")
        store.store_dataframe(df)
        logger.info("Store populated successfully.")

    logger.info("Initializing Support KB model...")
    return ClassicV2RagKBModel(store=store)


class ClassicV2RagKBModel(KnowledgeBaseModel):
    store: Store

    def __init__(self, store: Store):
        self.store = store

    def _format_doc(self, doc: dict) -> str:
        return (
            f"Q:{doc['Question']};Desc:{doc['Description']};Sol.:{doc['Solution']}\n\n"
        )

    def _retrieve(self, query: str) -> list[str]:
        docs = self.store.get_entries_similar_to_problem(query)
        logger.debug("Found {} documents", len(docs))

        fmt_docs = [self._format_doc(d) for d in docs]
        return fmt_docs

    def query(self, query: str) -> list[str]:
        return self._retrieve(query)
