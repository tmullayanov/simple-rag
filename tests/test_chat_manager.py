from simple_rag.chats import ChatManager
import pytest

@pytest.fixture
def chat_mgr():
    yield ChatManager()


def test_create_chat_manager(chat_mgr):
    assert chat_mgr is not None

def test_fresh_chat_manager_has_no_chats(chat_mgr):

    assert chat_mgr.total_chats() == 0


def test_create_chat(chat_mgr):

    chat = chat_mgr.create_chat()

