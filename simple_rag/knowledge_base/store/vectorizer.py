from langchain.vectorstores.base import VectorStore
from langchain_core.documents import Document
from loguru import logger


def default_doc_transform(row: dict) -> Document:
    return Document(
        page_content="\n".join(f"{col}: {val}" for (col, val) in row.items())
    )


class Vectorizer:
    vector_store: VectorStore

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def transform_row_to_document(
        self, row: dict, version: int, db_id: int
    ) -> Document:
        """
        Преобразует строку из БД в документ с метаданными.
        """
        doc = default_doc_transform(row)
        if not hasattr(doc, "metadata"):
            doc.metadata = {}
        doc.metadata["_version"] = version
        doc.metadata["_db_id"] = db_id

        return doc

    def vectorize_documents(self, docs: list[Document]) -> list[str]:
        """
        Векторизует список документов и возвращает их идентификаторы.
        """
        try:
            ids = self.vector_store.add_documents(docs)
            logger.debug(f"Vectorized {len(docs)} documents. IDs: {ids}")
            return ids
        except Exception as e:
            logger.error(f"Failed to vectorize documents: {e}")
            raise

    def delete_old_vectors(self, current_version: int):
        """
        Deletes all documents from the VectorStore where the "_version" metadata field is less than the current_version.
        """
        try:
            self.vector_store.delete(where={"_version": {"$lt": current_version}})
            logger.info(
                f"Cleared old vectors from VectorStore (less than {current_version}) versions)"
            )
        except Exception as e:
            logger.error(f"Failed to clear old vectors from VectorStore: {e}")
            raise

    def similarity_search(self, query, config: dict = {}) -> list[Document]:
        return self.vector_store.similarity_search(query, **config)

    def similarity_search_with_relevance_scores(
        self, query: str, search_config: dict = {}, *args, **kwargs
    ) -> list[tuple[Document, float]]:
        return self.vector_store.similarity_search_with_relevance_scores(
            query,
            **search_config,
        )
