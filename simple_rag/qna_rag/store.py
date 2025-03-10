from typing import Optional
from typing_extensions import TypedDict

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from simple_rag.qna.pseudo_db import AbstractQnA, SimpleQna


class SimpleVectorStore:
    vector_store: InMemoryVectorStore
    doc_ids: list[str]
    qna: AbstractQnA

    def __init__(self, qna: Optional[AbstractQnA] = None):
        self.vector_store = SimpleVectorStore.make_vector_store()

        if qna:
            self.store_qna(qna)

    @classmethod
    def make_vector_store(cls):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        return InMemoryVectorStore(embeddings)

    def store_qna(self, qna: AbstractQnA):
        self.qna = qna
        docs = self.__build_docs__()
        self.doc_ids = self.vector_store.add_documents(docs)

    def __build_docs__(self):
        docs = [
            Document(page_content=s, metadata={"source": "local_qna"})
            for s in self.qna.get_questions()
        ]

        return docs

    def similarity_search(self, question: str):
        return self.vector_store.similarity_search(question)

    def lookup_answers(self, question: str):
        return self.qna.lookup_answer(question)

    def as_retriever(self):
        return self.vector_store.as_retriever()
