from typing import TypedDict
from langchain.docstore.document import Document


class RagState(TypedDict):
    question: str
    context: list[Document]
    answer: str
