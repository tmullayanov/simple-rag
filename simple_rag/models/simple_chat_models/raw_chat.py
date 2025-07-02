from simple_rag.chats.chat import ChatModel
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings

from loguru import logger

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


history = get_by_session_id("1")


class RawChat(ChatModel):
    llm: BaseChatModel
    embeddings: Embeddings
    chain: RunnableWithMessageHistory

    def __init__(self, llm: BaseChatModel, embeddings: Embeddings):
        super().__init__()
        self.llm = llm
        self.embeddings = embeddings

        prompt = ChatPromptTemplate.from_messages(
            [
                # ("system", "You're an assistant who's good at {ability}"),
                ("system", "You're a helpful assistant"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = prompt | self.llm

        self.chain = RunnableWithMessageHistory(
            chain,
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        )


    def send(self, id, message):
        logger.info("RawChat.send()")
        res = self.chain.invoke({
            "question": message
        }, config={"configurable": {"session_id": id}})
        
        return res.content

    def update(self, new_cfg):
        raise NotImplementedError


def build_raw_chat_model(llm: BaseChatModel, embeddings: Embeddings, cfg: dict) -> RawChat:
    return RawChat(llm=llm, embeddings=embeddings)