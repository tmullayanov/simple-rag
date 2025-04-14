from langchain.vectorstores.base import VectorStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from typing import Callable
from langchain_chroma import Chroma
import pandas as pd
from sqlalchemy import MetaData, create_engine, Table
from sqlalchemy.exc import SQLAlchemyError
from structlog import get_logger

from simple_rag.embeddings import embeddings


logger = get_logger()


def default_doc_transform(row: dict) -> Document:
    return Document(
        page_content="\n".join(f"{col}: {val}" for (col, val) in row.items())
    )



class DBEngine:
    db_link: str = None
    table_name: str = None

    def __init__(self, db_cfg: dict = {}):
        self.db_link = db_cfg.get("db_link", None)
        self.table_name = db_cfg.get("tbl_name", None)

    def store_dataframe(self, df: pd.DataFrame, *args, **kwargs):
        if not self.db_link or not self.table_name:
            logger.warn("DB engine not configured, skip store_dataframe")
            return
        
        engine = create_engine(self.db_link)
        with engine.connect() as connection:

            if not engine.dialect.has_table(connection, self.table_name):
                logger.debug('creating table', table_name=self.table_name)
                df.to_sql(self.table_name, con=engine, index=False, if_exists='replace')
            else:
                logger.debug('adding to table')
                df.to_sql(self.table_name, con=engine, index=False, if_exists='append')
        
        logger.info('store_dataframe DONE')

class Store:
    _is_empty = True
    df: pd.DataFrame = None
    vectorStore: VectorStore
    engine: DBEngine = None

    def __init__(self, db_cfg: dict = {}, vectorstore_cfg: dict = {}, *args, **kwargs):
        self.engine = Store.build_db_manager(db_cfg)
        self.vectorStore = Store.build_vector_store(vectorstore_cfg)

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

        except SQLAlchemyError as db_error:
            # FIXME: Replace with DBEngine dedicated exception
            logger.error(f"Failed to save DataFrame to relational DB: {db_error}")
            raise db_error

        except Exception as vectorization_error:
            # If vectorization fails, attempt to roll back DB changes
            logger.error(f"Vectorization failed: {vectorization_error}. Rolling back DB changes.")
            
            try:
                db_url = self.engine.db_link
                table_name = self.engine.table_name
                engine = create_engine(db_url)
                metadata = MetaData()
                with engine.connect() as connection:
                    # Удаляем только что добавленные строки из таблицы
                    table = Table(table_name, metadata, autoload_with=engine)
                    delete_stmt = table.delete()
                    connection.execute(delete_stmt)
                    connection.commit()
                    logger.debug("Rolled back DB changes due to vectorization failure")
            except Exception as rollback_error:
                logger.error(f"Failed to roll back DB changes: {rollback_error}")
            
            raise vectorization_error

    def get(self, column_name, value) -> list[dict]:
        if self.df is None:
            return []
        return (
            self.df[self.df[column_name] == value]
            .apply(lambda x: x.to_dict(), axis=1)
            .tolist()
        )

    def similarity_search(self, query, config: dict = {}):
        return self.vectorStore.similarity_search(query, **config)
