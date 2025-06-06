import asyncio
from datetime import timedelta
import uuid
from simple_rag.chats import ChatManager
import pytest
from unittest.mock import MagicMock

from simple_rag.chats.chat import Chat, ChatModel


@pytest.fixture
async def chat_mgr():
    yield ChatManager()


def test_create_chat_manager(chat_mgr):
    assert chat_mgr is not None


def test_fresh_chat_manager_has_no_chats(chat_mgr):
    assert chat_mgr.total_chats() == 0


def test_create_chat(chat_mgr):
    chat = chat_mgr.create_chat(model=ParrotModel())

    assert isinstance(chat, Chat)


def test_can_access_chat_by_id(chat_mgr):
    chat = chat_mgr.create_chat(model=ParrotModel())

    assert chat_mgr.get_chat(chat.id) == chat


def test_get_none_if_chat_does_not_exist(chat_mgr):
    assert chat_mgr.get_chat(uuid.uuid4()) is None


def test_can_remove_chat_manually(chat_mgr):
    chat = chat_mgr.create_chat(model=ParrotModel())
    assert chat_mgr.get_chat(chat.id) == chat

    chat_mgr.remove_chat(chat.id)
    assert chat_mgr.get_chat(chat.id) is None


def test_chat_mgr_proxies_to_chat(chat_mgr):
    chat = chat_mgr.create_chat(model=ParrotModel())

    response = chat_mgr.send_message(chat.id, "Hello")

    # actually we break the abstraction a bit here, but it's ok for now.
    assert chat_mgr.get_chat(chat.id).history == [
        {"role": "user", "msg": "Hello"},
        {"role": "model", "msg": response},
    ]


@pytest.mark.asyncio
async def test_auto_delete_inactive_chats(chat_mgr):
    chat_mgr = ChatManager(check_interval=1.0)
    chat_mgr.INACTIVITY_TIMEOUT = timedelta(
        seconds=2
    )  # Override the default inactivity timeout for faster testing

    chat = chat_mgr.create_chat(model=MagicMock())
    assert chat_mgr.get_chat(chat.id) == chat

    for _ in range(3):
        await asyncio.sleep(1)
        if chat_mgr.get_chat(chat.id) is None:
            break

    assert chat_mgr.get_chat(chat.id) is None
    chat_mgr.stop()


class ParrotModel(ChatModel):
    def send(self, id, message):
        return message

    def update(self, new_cfg: dict[str, str]):
        pass
