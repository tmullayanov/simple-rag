from logging import Logger
from langchain.chat_models.base import BaseChatModel

from simple_rag.llm import llm
from simple_rag.chats import ChatManager
from simple_rag.logger import setup_logger
from simple_rag.web.config import APP_SETTINGS, AppSettings


class AppContext:
    logger: Logger
    chatManager: ChatManager
    llm: BaseChatModel

    def __init__(self, settings: AppSettings):
        self.logger = setup_logger(settings.model_dump())
        self.chatManager = ChatManager()

    async def on_startup(self):
        self.logger.debug("AppContext STARTUP")
        # FIXME: init llm here based on config
        self.llm = llm

    async def on_shutdown(self):
        self.logger.debug("AppContext SHUTDOWN")


APP_CTX = AppContext(APP_SETTINGS)


def get_chat_manager():
    return APP_CTX.chatManager


__all__ = ["AppContext", "APP_CTX"]
