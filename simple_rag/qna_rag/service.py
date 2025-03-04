from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models.base import BaseChatModel

from simple_rag.llm.groq import llm
from simple_rag.qna.csv_parser import QnAFileParser
from simple_rag.qna_rag.engine import build_rag_graph
from typing_extensions import TypedDict

from simple_rag.qna_rag.store import SimpleVectorStore


class QnAServiceConfig(TypedDict):
    qna_path: str
    qna_delimiter: str = ";"


class QnaStaticFileService:
    
    store: SimpleVectorStore
    llm: BaseChatModel
    graph: CompiledStateGraph
    
    def __init__(self, config: QnAServiceConfig):
        
        parser = QnAFileParser(config["qna_path"], config["qna_delimiter"])
        qna = parser.build_qna()

        self.store = SimpleVectorStore()
        self.store.store_qna(qna)
        self.llm = llm

        self.graph = build_rag_graph(
            self.llm, self.store
        )

    def ask(self, question: str):
        return self.graph.invoke({"raw_input": question})

    
