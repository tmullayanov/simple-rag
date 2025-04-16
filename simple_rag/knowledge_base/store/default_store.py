from langchain.vectorstores.base import VectorStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from typing import Callable
from langchain_chroma import Chroma
from loguru import logger
import pandas as pd


from simple_rag.embeddings import embeddings
from .db_engine import DBEngine, PseudoDBEngine, RollbackDBError, StoreDFError


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
        if not cfg or 'db_link' not in cfg or 'model_name' not in cfg:
            logger.debug('Create InMemDBEngine()')
            return PseudoDBEngine()
        logger.debug('Create default DBEngine()')
        return DBEngine(cfg)

    @property
    def is_empty(self):
        return self.df is None

    def store_dataframe(
        self,
        df: pd.DataFrame,
        doc_transform: Callable[[dict], Document] = default_doc_transform,
    ):
        if not self.engine or not self.vectorStore:
            logger.warning(
                "DB engine or VectorStore not configured, skip store_dataframe"
            )
            return

        try:
            df = df.copy()
            # Шаг 1: Сохраняем DataFrame в БД
            new_version, new_ids = self.engine.store_dataframe(df)
            logger.debug("DataFrame saved to DB")
            df['_id'] = new_ids

            # Шаг 2: Векторизация данных
            docs = []
            for i, (_, row) in enumerate(df.iterrows()):
                # Преобразуем строку в документ
                doc = doc_transform(row.to_dict())
                
                # Добавляем метаданные: версия и ID из БД
                if not hasattr(doc, "metadata"):
                    doc.metadata = {}
                doc.metadata["_version"] = new_version
                doc.metadata["_db_id"] = new_ids[i]  # Используем ID из БД

                docs.append(doc)

            logger.debug("docs created", docs_len=len(docs))

            self.vectorStore.add_documents(docs)
            logger.debug("docs added to vectorStore")

            # Шаг 3: Обновляем DataFrame в памяти
            self.df = df
        
        except StoreDFError as store_df_error:
            logger.error(f"Failed to store DataFrame: {store_df_error}")
            raise store_df_error

        except Exception as vectorization_error:
            # Если векторизация завершилась ошибкой, откатываем изменения в БД
            logger.error(
                f"Vectorization failed: {vectorization_error}. Rolling back DB changes."
            )
            try:
                self.engine.rollback_version(new_version)
            except RollbackDBError as rollback_error:
                logger.error(f"Failed to roll back DB changes: {rollback_error}")
                raise rollback_error from vectorization_error

            raise vectorization_error

    def get(self, column_name, value) -> list[dict]:
        if self.df is None:
            return []
        return (
            self.df[self.df[column_name] == value]
            .drop(columns=['_id'], axis=1)
            .apply(lambda x: x.to_dict(), axis=1)
            .tolist()
        )

    def similarity_search(self, query, config: dict = {}) -> list[Document]:
        return self.vectorStore.similarity_search(query, **config)
    
    def get_entries_similar_to_problem(self, problem: str, search_config: dict = {}, *arg, **kwargs) -> list[dict]:
        docs = self.vectorStore.similarity_search_with_relevance_scores(
            problem,
            **search_config,
            filter={"_version": self.engine.version},
        )
        logger.debug("docs retrieved {docs_len}", docs_len=len(docs))
        
        doc_ids = (doc.metadata["_db_id"] for (doc, _) in docs)
        records = self.df[self.df['_id'].isin(doc_ids)].to_dict(orient='records')

        return records



