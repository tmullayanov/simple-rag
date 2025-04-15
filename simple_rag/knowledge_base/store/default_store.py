from langchain.vectorstores.base import VectorStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from typing import Callable, Optional
from langchain_chroma import Chroma
from loguru import logger
import pandas as pd


from simple_rag.embeddings import embeddings
from .db_engine import DBEngine



def default_doc_transform(row: dict) -> Document:
    return Document(
        page_content="\n".join(f"{col}: {val}" for (col, val) in row.items())
    )


class Store:
    _is_empty = True
    df: pd.DataFrame = None
    vectorStore: VectorStore
    engine: DBEngine = None

    def __init__(self, db_cfg: dict = {}, vectorstore_cfg: dict = {}, *args, **kwargs):
        self.engine = Store.build_db_manager(db_cfg)
        self.vectorStore = Store.build_vector_store(vectorstore_cfg)

        self.df = self.engine.load_dataframe()

    @staticmethod
    def build_vector_store(cfg: dict):
        if cfg.get("type", None) == "chroma":
            return Chroma(
                collection_name=cfg["collection_name"],
                embedding_function=embeddings,
                persist_directory=cfg["persist_directory"],
                collection_metadata={"hnsw:space": "cosine"},
            )

        return InMemoryVectorStore(embeddings)
    
    @staticmethod
    def build_db_manager(cfg: dict = {}):
        return DBEngine(cfg)


    @property
    def is_empty(self):
        return self.df is None

    def store_dataframe(
        self,
        df,
        doc_transform: Callable[[pd.Series], list[Document]] = default_doc_transform,
    ):
        self.df = df        

        try:
            # Step 1: Saving DataFrame to relational DB - potentially with exception
            self.engine.store_dataframe(df)
            logger.debug('DataFrame saved to relational DB')

            # Step 2: Vectorizing data
            docs = self.df.apply(lambda x: doc_transform(x.to_dict()), axis=1).tolist()
            logger.debug('docs created', docs_len=len(docs))
            
            self.vectorStore.add_documents(docs)
            logger.debug('docs added to vectorStore')

        except DBEngine.StoreDFError as db_error:
            # FIXME: Replace with DBEngine dedicated exception
            logger.error(f"Failed to save DataFrame to relational DB: {db_error}")
            raise db_error

        except Exception as vectorization_error:
            # If vectorization fails, attempt to roll back DB changes
            logger.error(f"Vectorization failed: {vectorization_error}. Rolling back DB changes.")
            try:
                self.engine.clear_table()
                logger.info("Rolled back DB changes due to vectorization failure")
            except DBEngine.RollbackDBError as rollback_error:
                logger.error("Failed to roll back DB changes: {rollback_error}")
            

    def get(self, column_name, value) -> list[dict]:
        if self.df is None:
            return []
        return (
            self.df[self.df[column_name] == value]
            .apply(lambda x: x.to_dict(), axis=1)
            .tolist()
        )

    def similarity_search(self, query, config: dict = {}) -> list[Document]:
        return self.vectorStore.similarity_search(query, **config)
