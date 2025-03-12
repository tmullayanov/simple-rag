import asyncio
from datetime import timedelta, datetime
from typing import Optional
from uuid import UUID
from simple_rag.chats.chat import Chat, ChatModel

from loguru import logger


class ChatManager:
    INACTIVITY_TIMEOUT = timedelta(minutes=5)  # FIXME: use config

    _chats: dict[UUID, Chat]
    _check_interval: int # in seconds
    _running: bool # indicates whether the background task is running

    def __init__(self, check_interval: int = 0):
        logger.info("Initializing ChatManager")
        self._chats = {}
        self._check_interval = check_interval

        has_interval = self._check_interval > 0

        self._running = has_interval
        if has_interval:
            asyncio.create_task(self.__cleanup_task())

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
        # FIXME: possibly cleanup model too.
        self._chats.pop(id, None)

    async def __cleanup_task(self):
        """Background task for cleaning up inactive chats"""
        while self._running:
            await asyncio.sleep(self._check_interval)
            logger.debug("Running cleanup task")
            current_time = datetime.now()
            inactive_chats = [
                chat_id for chat_id, chat in self._chats.items()
                if current_time - chat.last_active > self.INACTIVITY_TIMEOUT
            ]
            for chat_id in inactive_chats:
                logger.info(f"Removing inactive chat {chat_id}")
                self.remove_chat(chat_id)

    def stop(self):
        """Method for stopping the background task"""
        logger.debug("Stopping ChatManager background task")
        self._running = False


def get_chat_manager() -> ChatManager:
    return ChatManager()
