import abc
from typing import TypedDict
import uuid
from datetime import timedelta, datetime


class ChatModel(abc.ABC):
    @abc.abstractmethod
    def send(self, id: uuid.UUID, message: str) -> str:
        raise NotImplementedError("Must be implemented")

    @abc.abstractmethod
    def update(self, new_cfg: dict[str, str]):
        raise NotImplementedError("Must be implemented")


class HistoryMessage(TypedDict):
    role: str
    msg: str


class Chat:
    id: uuid.UUID
    model: ChatModel
    last_active: timedelta

    # TODO: history might be a DB table in the future
    # TODO: so this is point of possible abstraction
    history: list[HistoryMessage]

    def __init__(self, model: ChatModel):
        self.id = uuid.uuid4()
        self.model = model
        self.last_active = datetime.now()
        self.history = []

    def send(self, message: str) -> str:
        self.last_active = datetime.now()
        self.history.append({"role": "user", "msg": message})
        response = self.model.send(self.id, message)
        self.history.append({"role": "model", "msg": response})

        return response
