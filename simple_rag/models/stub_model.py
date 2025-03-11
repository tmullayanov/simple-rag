from simple_rag.chats.chat import ChatModel


class StubModel(ChatModel):
    def send(self, id, message):
        raise NotImplementedError

    def update(self, new_cfg):
        raise NotImplementedError
