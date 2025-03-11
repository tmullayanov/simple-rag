import logging
import uuid
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models.base import BaseChatModel

from simple_rag.chats.chat import ChatModel
from simple_rag.logger import GLOBAL_LOGGER_NAME
from simple_rag.models.qna_rag.parser.csv_parser import QnAFileParser
from typing_extensions import TypedDict

from .store import SimpleVectorStore
from .engine import RagEngineDynamicPrompt


from loguru import logger


class QnAServiceConfig(TypedDict):
    qna_path: str
    qna_delimiter: str


class QnaStaticFileModel(ChatModel):
    store: SimpleVectorStore
    llm: BaseChatModel
    engine: RagEngineDynamicPrompt
    rag: CompiledStateGraph

    def __init__(self, config: QnAServiceConfig, llm: BaseChatModel):
        logger.debug("QnaStaticFileService::init()")

        parser = QnAFileParser(**config)
        qna = parser.parse_qna()

        self.store = SimpleVectorStore(qna)
        self.llm = llm

        logger.debug("QnaStaticFileService:: building rag graph...")
        self.engine = RagEngineDynamicPrompt(self.llm, self.store)
        self.rag = self.engine.build_rag()

        logger.debug("QnaStaticFileService:: init() DONE")

    def send(self, id: uuid.UUID, question: str) -> str:
        config = {"configurable": {"thread_id": str(id)}}
        answer = self.rag.invoke({"raw_input": question}, config=config)

        return answer["answer"]

    def update(self, new_cfg: dict[str, str]):
        logger.debug("QnaStaticFileService::update()")

        if "prompt" in new_cfg:
            self.engine.change_prompt(new_cfg["prompt"])
            logger.debug("Updated prompt for QnaStaticFileService")
