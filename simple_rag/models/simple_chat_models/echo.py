from loguru import logger
from simple_rag.chats.chat import ChatModel


class EchoChatModel(ChatModel):
    def send(self, id, message):
        logger.info("EchoChatModel.send()")
        return message

    def update(self, new_cfg):
        logger.info("EchoChatModel.update() NOT SUPPORTED")
        raise NotImplementedError()
