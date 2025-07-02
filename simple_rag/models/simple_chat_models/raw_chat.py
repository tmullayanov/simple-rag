from simple_rag.chats.chat import ChatModel
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings

from loguru import logger

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from simple_rag.chats.history import get_by_session_id


class RawChat(ChatModel):
    llm: BaseChatModel
    embeddings: Embeddings
    chain: RunnableWithMessageHistory

    def __init__(self, llm: BaseChatModel, embeddings: Embeddings):
        self.llm = llm
        self.embeddings = embeddings

        prompt = ChatPromptTemplate.from_messages(
            [
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
        res = self.chain.invoke(
            {"question": message}, config={"configurable": {"session_id": id}}
        )

        return res.content

    def update(self, new_cfg):
        raise NotImplementedError


def build_raw_chat_model(
    llm: BaseChatModel, embeddings: Embeddings, cfg: dict
) -> RawChat:
    return RawChat(llm=llm, embeddings=embeddings)
