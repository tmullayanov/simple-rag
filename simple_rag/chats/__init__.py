from typing import Optional
from uuid import UUID
from simple_rag.chats.chat import Chat, ChatModel
import logging

from simple_rag.logger import GLOBAL_LOGGER_NAME


from loguru import logger


class ChatManager:
    _chats: dict[UUID, Chat]

    def __init__(self):
        logger.info("Initializing ChatManager")
        self._chats = {}

    def total_chats(self):
        return 0

    def create_chat(self, model: ChatModel):
        chat = Chat(model)
        self._chats[chat.id] = chat
        return chat

    def send_message(self, chat_id: UUID, message: str) -> str:
        # if chat does not exist, raises KeyError automatically.
        return self._chats[chat_id].send(message)

    def get_chat(self, id: UUID) -> Optional[Chat]:
        return self._chats.get(id)

    def remove_chat(self, id: UUID):
        self._chats.pop(id, None)


def get_chat_manager() -> ChatManager:
    return ChatManager()
