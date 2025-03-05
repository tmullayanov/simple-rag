from uuid import UUID
import pytest

from simple_rag.chats.chat import Chat, ChatModel


@pytest.fixture
def dummy_chat():
    model = EmptyModel()
    chat = Chat(model)

    yield (chat, model)


def test_chat_init(dummy_chat):
    chat, _ = dummy_chat
    assert chat is not None

def test_chat_has_unique_id():

    model = EmptyModel()
    chat1 = Chat(model)
    chat2 = Chat(model)

    assert type(chat1.id) is UUID
    assert type(chat2.id) is UUID

    assert chat1.id != chat2.id

def test_chat_has_model_attached(dummy_chat):

    chat, _ = dummy_chat

    assert hasattr(chat, 'model')
    assert isinstance(chat.model, ChatModel)


def test_chat_can_send_message_to_model_no_raises(dummy_chat):
    chat, _ = dummy_chat

    chat.send('Hello')

def test_chat_actually_uses_model():
    model = MockModel()
    chat = Chat(model)

    msg = 'Hello'
    chat.send(msg)

    assert model.messages == [msg]

def test_chat_keep_track_of_messages():
    parrot = ParrotModel()
    chat = Chat(parrot)

    msg = 'Hello'
    chat.send(msg)

    au_history = [{
        'role': 'user',
        'msg': msg
    }, {
        'role': 'model',
        'msg': msg
    }]

    assert chat.history == au_history


class EmptyModel(ChatModel):
    
    def send(self, message: str):
        pass

class MockModel(ChatModel):

    messages: list[str]

    def __init__(self):
        self.messages = []

    def send(self, message: str):
        self.messages.append(message)

class ParrotModel(ChatModel):

    def send(self, message):
        return message