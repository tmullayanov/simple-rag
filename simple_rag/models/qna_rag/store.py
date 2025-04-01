from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document

from langchain_chroma import Chroma
from loguru import logger

from .qna import AbstractQnA


class QuestionVectorStore:
    vector_store: Chroma
    doc_ids: list[str]
    qna: AbstractQnA

    def __init__(self, embeddings: Embeddings, qna: Optional[AbstractQnA] = None):
        self.vector_store = QuestionVectorStore.make_vector_store(embeddings)

        if qna:
            self.store_qna(qna)

    @classmethod
    def make_vector_store(cls, embeddings: Embeddings):
        return Chroma(
            collection_name="qna_question_store",
            embedding_function=embeddings,
            persist_directory="./chroma_store",
            collection_metadata={"hnsw:space": "cosine"},
        )

    def store_qna(self, qna: AbstractQnA):
        logger.debug(f"Storing qna entries...")
        self.qna = qna
        docs = self.__build_docs__()

        if self.check_empty():
            logger.debug("Empty store, populating...")
            self.doc_ids = self.vector_store.add_documents(docs)
        else:
            logger.debug("Store's not empty, skipping...")
        logger.debug("Storing done")

    def __build_docs__(self):
        docs = [
            Document(page_content=s, metadata={"source": "local_qna"})
            for s in self.qna.get_questions()
        ]

        return docs

    def check_empty(self):
        query = self.vector_store.get(include=[])
        logger.debug(f"Ids in store: {query['ids']}")
        return len(query["ids"]) == 0

    def similarity_search(self, question: str):
        return self.vector_store.similarity_search(question)

    def lookup_answers(self, question: str):
        return self.qna.lookup_answer(question)

    def as_retriever(self):
        return self.vector_store.as_retriever()
