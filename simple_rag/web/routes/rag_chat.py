import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from simple_rag.chats.chat import ChatModel
from simple_rag.logger import GLOBAL_LOGGER_NAME
from simple_rag.chats import ChatManager
from simple_rag.qna_rag.service import QnAServiceConfig, QnaStaticFileService
from simple_rag.web.config import APP_SETTINGS
from simple_rag.web.context import get_chat_manager

router = APIRouter(prefix="/rag_chat")

logger = logging.getLogger(GLOBAL_LOGGER_NAME)


def get_qna_service(cfg: QnAServiceConfig):
    return QnaStaticFileService(cfg)


# Модели данных для API
class ChatResponse(BaseModel):
    chat_id: UUID

class MessageRequest(BaseModel):
    chat_id: UUID
    message: str

class MessageResponse(BaseModel):
    response: str

class UpdateModelRequest(BaseModel):
    chat_id: UUID
    prompt: str


@router.post("/create", response_model=ChatResponse)
async def create_chat(
    chat_manager: ChatManager = Depends(get_chat_manager),
    model: ChatModel = Depends(lambda: get_qna_service(APP_SETTINGS.model_dump()))):
    """Создание нового чата"""
    chat = chat_manager.create_chat(model)
    logger.info("Chat created with id: %s", chat.id)
    return ChatResponse(chat_id=chat.id)

@router.post("/message", response_model=MessageResponse)
async def send_message(
    request: MessageRequest, 
    chat_manager: ChatManager = Depends(get_chat_manager)
):
    """Отправка сообщения в существующий чат"""
    try:
        logger.info("Sending message to chat %s", request.chat_id)
        response = chat_manager.send_message(request.chat_id, request.message)
        return MessageResponse(response=response)
    except KeyError:
        raise HTTPException(status_code=404, detail="Chat not found")


@router.post("/update_model")
async def update_model(
    request: UpdateModelRequest,
    chat_manager: ChatManager = Depends(get_chat_manager),
):
    try:
        logger.info("Updating model for chat %s", request.chat_id)
        chat = chat_manager.get_chat(request.chat_id)
        if not chat:
            raise ValueError("Chat not found")
        
        model = chat.model
        model.update({
            "prompt": request.prompt
        })

        logger.info("Updated model for chat %s successfully", request.chat_id)

        return {"success": True}
        
    except Exception as ex:
        logger.error("Error updating model: %s", ex)
        return HTTPException(status_code=502, detail="Internal server error")


@router.delete("/chat/{chat_id}")
async def delete_chat(
    chat_id: UUID, 
    chat_manager: ChatManager = Depends(get_chat_manager)
):
    """Удаление существующего чата"""
    try:
        chat_manager.remove_chat(chat_id)
        logger.info("Deleted chat with id: %s", chat_id)
        return {"status": "success", "message": "Chat deleted"}
    except KeyError:
        raise HTTPException(status_code=404, detail="Chat not found") 