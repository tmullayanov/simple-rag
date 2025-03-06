from typing import Optional
from uuid import UUID
from simple_rag.chats.chat import Chat, ChatModel


class ChatManager:

    _chats: dict[UUID, Chat]

    def __init__(self):
        self._chats = {}
    

    def total_chats(self):
        return 0
    
    def create_chat(self, model: ChatModel):
        chat = Chat(model)
        self._chats[chat.id] = chat
        return chat
    
    def get_chat(self, id: UUID) -> Optional[Chat]:
        return self._chats.get(id)
    
    def remove_chat(self, id: UUID):
        self._chats.pop(id, None)