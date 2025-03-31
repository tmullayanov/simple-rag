from loguru import logger
from simple_rag.knowledge_base.base import KnowledgeBaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain.docstore.document import Document

from .prompt import default_rag_prompt

SIMILARITY_THRESHOLD = 0.81


class ClassicRagKnowledgeBase(KnowledgeBaseModel):
    llm: BaseChatModel
    store: VectorStore
    prompt: PromptTemplate = default_rag_prompt

    def __init__(self, llm, store: VectorStore):
        self.llm = llm
        self.store = store

    def _retrieve(self, query: str):
        retrieved_docs = self.store.similarity_search(query)

        logger.debug(f"{len(retrieved_docs)} found")
        return retrieved_docs

    def _generate(self, docs: list[Document], query: str):
        docs_content = "\n\n".join(doc.page_content for doc in docs)
        messages = self.prompt.invoke({"question": query, "context": docs_content})
        response = self.llm.invoke(messages)

        return response.content

    def query(self, query) -> str:
        docs = self._retrieve(query)
        return self._generate(docs, query)
