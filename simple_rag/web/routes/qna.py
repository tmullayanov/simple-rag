from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

import logging

from simple_rag.chats.chat import Chat
from simple_rag.logger import GLOBAL_LOGGER_NAME
from simple_rag.qna_rag.service import QnAServiceConfig, QnaStaticFileService
from simple_rag.web.config import APP_SETTINGS

router = APIRouter(prefix="/qna")

logger = logging.getLogger(GLOBAL_LOGGER_NAME)


def get_qna_service(cfg: QnAServiceConfig):
    return QnaStaticFileService(cfg)


@router.get("/")
async def create_qna_rag(
    service: QnaStaticFileService = Depends(
        lambda: get_qna_service(APP_SETTINGS.model_dump())
    ),
):
    logger.info(service.store.doc_ids)
    chat = Chat(model=service)
    answer = chat.send("Чем машинное обучение отличается от глубокого?")

    return JSONResponse(content={"message": answer})
