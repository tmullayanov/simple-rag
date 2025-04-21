from typing import Optional
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_chroma import Chroma
from loguru import logger
import pandas as pd
from pydantic import BaseModel


from simple_rag.embeddings import embeddings
from simple_rag.knowledge_base.store.vectorizer import Vectorizer
from .db_engine import DBEngine, PseudoDBEngine, RollbackDBError, StoreDFError

class DbConfig(BaseModel):
    db_link: Optional[str] = None
    model_name: Optional[str] = None


class Store:
    _is_empty = True
    df: pd.DataFrame = None
    engine: DBEngine = None

    def __init__(self, db_cfg: dict = {}, vectorstore_cfg: dict = {}, *args, **kwargs):
        self.engine = Store.build_db_manager(db_cfg)
        vectorStore = Store.build_vector_store(vectorstore_cfg)
        self.vectorizer = Vectorizer(vectorStore)

        self.df = self.engine.load_dataframe()

        # sync and cleanup
        self.clear_old_versions()
        self.check_and_vectorize_unprocessed()

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
    
    def check_and_vectorize_unprocessed(self):
        """
        Проверяет, есть ли невекторизованные записи в БД, и выполняет их векторизацию.
        """
        if not self.engine:
            logger.warning("DB engine not configured, skip check_and_vectorize_unprocessed")
            return
        
        try:
            gen = self.engine.process_unvectorized_rows()
            entity = next(gen)

            while True:
                df_row = {
                    'Question': entity.question,
                    'Description': entity.description,
                    'Solution': entity.solution
                }
                
                logger.debug(f'transforming row to doc')
                doc = self.vectorizer.transform_row_to_document(df_row, entity.version, entity.id)
                logger.debug(f'transformed row to {doc=}')

                try:
                    ids = self.vectorizer.vectorize_documents([doc])
                    success = True
                    logger.debug(f'Vectorized. {ids=}')
                except Exception as e:
                    logger.error("Failed to add doc to vectorstore!")
                    success = False
                finally:
                    entity = gen.send(success)

        except StopIteration:
            logger.info("All unvectorized rows have been processed")
        except Exception as e:
            logger.error(f"Failed to vectorize: {e}")
            raise


    def store_dataframe(
        self,
        df: pd.DataFrame,
    ):
        if not self.engine:
            logger.warning(
                "DB engine not configured, skip store_dataframe"
            )
            return

        try:
            df = df.copy()
            # Step 1: Store DataFrame in the database
            new_version, new_ids = self.engine.store_dataframe(df)
            logger.info("DataFrame saved to DB")
            df['_id'] = new_ids

            # Step 2: Vectorize the data
            docs = []
            for (_, row) in df.iterrows():
                doc = self.vectorizer.transform_row_to_document(row.to_dict(), version=new_version, db_id=row['_id'])
                docs.append(doc)

            logger.debug("docs created", docs_len=len(docs))

            self.vectorizer.vectorize_documents(docs)
            logger.info("docs added to vectorStore")

            # Step 2.5: Update vectorized attr in DB
            self.engine._update_vectorized_flag(new_version)

            # Step 3: Update the DataFrame in memory
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
        
    def clear_old_versions(self):
        self.engine.clear_old_versions()
        self.vectorizer.delete_old_vectors(self.engine.version)

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
        return self.vectorizer.similarity_search(query, **config)
    
    def get_entries_similar_to_problem(self, problem: str, search_config: dict = {}, *args, **kwargs) -> list[dict]:
        docs = self.vectorizer.similarity_search_with_relevance_scores(
            problem,
            **search_config,
            filter={"_version": self.engine.version},
        )
        logger.debug("GET_ENTRIES docs retrieved {docs_len}", docs_len=len(docs))
        logger.debug("GET_ENTRIES {docs}", docs=docs)
        
        doc_ids = {doc.metadata["_db_id"] for (doc, _) in docs}
        logger.debug("GET_ENTRIES unique doc_ids retrieved {doc_ids_len}", doc_ids_len=len(doc_ids))
        records = self.df[self.df['_id'].isin(doc_ids)].to_dict(orient='records')

        return records
