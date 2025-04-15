from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from simple_rag import embeddings
from simple_rag.chats import ChatManager
from simple_rag.models import ModelCreator
from simple_rag.models.qna_rag.model import QnAServiceConfig, build_static_file_model
from simple_rag.web.context import get_chat_manager, get_model_creator
from loguru import logger

router = APIRouter(prefix="/chat")


def get_qna_service(cfg: QnAServiceConfig, llm):
    return build_static_file_model(llm=llm, config=cfg, embeddings=embeddings)


# Модели данных для API
class ChatCreateRequest(BaseModel):
    model: str


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
    chat_create_request: ChatCreateRequest,
    chat_manager: ChatManager = Depends(get_chat_manager),
    model_creator: ModelCreator = Depends(get_model_creator),
):
    """Создание нового чата"""
    try:
        model = model_creator.build(chat_create_request.model)
        chat = chat_manager.create_chat(model)
        logger.info("Chat created with id: %s" % chat.id)

        return ChatResponse(chat_id=chat.id)
    except ValueError as ex:
        logger.error("Could not create chat: %s" % ex)
        raise HTTPException(status_code=400, detail=ex.args)


@router.post("/message", response_model=MessageResponse)
async def send_message(
    request: MessageRequest, chat_manager: ChatManager = Depends(get_chat_manager)
):
    """Отправка сообщения в существующий чат"""
    try:
        logger.info("Sending message to chat %s" % request.chat_id)
        response = chat_manager.send_message(request.chat_id, request.message)
        return MessageResponse(response=response)
    except KeyError:
        logger.error("Error while sending message! Chat not found!")
        raise HTTPException(status_code=404, detail="Chat not found")
    except Exception as ex:
        logger.error("Error while sending message: %s" % ex)
        raise HTTPException(status_code=502, detail="Internal server error")


@router.post("/update_model")
async def update_model(
    request: UpdateModelRequest,
    chat_manager: ChatManager = Depends(get_chat_manager),
):
    try:
        logger.info("Updating model for chat %s" % request.chat_id)
        chat = chat_manager.get_chat(request.chat_id)
        if not chat:
            raise ValueError("Chat not found")

        model = chat.model
        model.update({"prompt": request.prompt})

        logger.info("Updated model for chat %s successfully" % request.chat_id)

        return JSONResponse(content={"success": True})

    except Exception as ex:
        logger.error("Error updating model: %s" % ex)
        return HTTPException(status_code=502, detail="Internal server error")


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: UUID, chat_manager: ChatManager = Depends(get_chat_manager)
):
    """Удаление существующего чата"""
    try:
        chat_manager.remove_chat(chat_id)
        logger.info("Deleted chat with id: %s" % chat_id)
        return {"status": "success", "message": "Chat deleted"}
    except KeyError:
        # TODO: discuss whether it is better to return HTTP 204 (No Content) here and warn.
        logger.error("Error deleting chat! Chat not found!")
        raise HTTPException(status_code=404, detail="Chat not found")
