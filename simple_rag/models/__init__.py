
from simple_rag.chats.chat import ChatModel
from simple_rag.models.stub_model import StubModel


class ModelCreator:
    

    def build(self, name: str) -> ChatModel:
        if name == 'stub_model':
            return StubModel()
        else:
            raise ValueError(f"Unknown model name: {name}")