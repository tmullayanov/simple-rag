from logging import Logger
from langchain.chat_models.base import BaseChatModel
from langchain_core.embeddings import Embeddings

from simple_rag.knowledge_base.manager import KnowledgeBaseManager
from simple_rag.web.config import APP_SETTINGS, AppSettings
from simple_rag.chats import ChatManager
from simple_rag.logger import setup_logger
from simple_rag.llm import llm
from simple_rag.embeddings import embeddings
from simple_rag.models import ModelCreator


class AppContext:
    logger: Logger
    chatManager: ChatManager
    modelCreator: ModelCreator
    llm: BaseChatModel
    knowledge_base_mgr: KnowledgeBaseManager
    embeddings: Embeddings

    settings: AppSettings

    def __init__(self, settings: AppSettings):
        self.logger = setup_logger(settings.model_dump())
        self.chatManager = ChatManager()
        self.settings = settings
        self.embeddings = embeddings

        # XXX: NOTE THAT LLM, KB_MANAGER AND MODEL_CREATOR ARE NOT INITIALIZED HERE!

    async def on_startup(self):
        self.logger.debug("AppContext STARTUP")
        # FIXME: init llm here based on config
        self.llm = llm
        self.modelCreator = ModelCreator(llm=llm, config=self.settings.model_dump())
        self.knowledge_base_mgr = KnowledgeBaseManager(
            llm=llm, embeddings=self.embeddings
        )

    async def on_shutdown(self):
        self.logger.debug("AppContext SHUTDOWN")


APP_CTX = AppContext(APP_SETTINGS)


def get_chat_manager():
    return APP_CTX.chatManager


def get_model_creator():
    return APP_CTX.modelCreator


def get_default_llm():
    return APP_CTX.llm


__all__ = ["AppContext", "APP_CTX"]
