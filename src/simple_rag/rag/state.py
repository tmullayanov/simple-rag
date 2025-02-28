from typing_extensions import List, TypedDict
from langchain.docstore.document import Document

class RagState(TypedDict):
    raw_input: str
    questions: list[Document]
    qna_context: List[str]
    answer: str