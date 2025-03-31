from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from simple_rag.knowledge_base.manager import KnowledgeBaseManager
from simple_rag.web.context import get_knowledge_base_manager


router = APIRouter(prefix="/kbase")


class QuestionRequest(BaseModel):
    question: str
    model: str

class MessageResponse(BaseModel):
    response: str

@router.post('/')
def ask_question(
    request: QuestionRequest,
    knowledge_base_manager: KnowledgeBaseManager = Depends(get_knowledge_base_manager),
):
    try:
        logger.info(f"Got request to knowledge base model={request.model}")
        model = knowledge_base_manager.get_model(request.model)
        logger.info(f"Found model={request.model}")

        response = model.query(request.question)
        return MessageResponse(response=response)
    except Exception as e:
        logger.exception(
            f"Got exception while using knowledge base (model={request.model}, details={e})"
        )
        return HTTPException(status_code=500, detail=str(e))
