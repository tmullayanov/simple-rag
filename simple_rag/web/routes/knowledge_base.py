from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from simple_rag.knowledge_base.manager import KnowledgeBaseManager, NoSuchModelError
from simple_rag.metrics import get_metrics, increment_metric
from simple_rag.metrics.kbase_metric import get_db
from simple_rag.web.context import get_knowledge_base_manager

from sqlalchemy.orm import Session

router = APIRouter(prefix="/kbase")


class QuestionRequest(BaseModel):
    question: str
    model: str


class MessageResponse(BaseModel):
    response: list[str]

class MetricsReport(BaseModel):
    metrics: dict[str, dict[str, int]]
    totals: int


@router.post("/")
def ask_question(
    request: QuestionRequest,
    knowledge_base_manager: KnowledgeBaseManager = Depends(get_knowledge_base_manager),
    # it breaks the layers isolation but for simple case we follow the fastapi docs
    db: Session = Depends(get_db),
) -> MessageResponse:
    try:
        logger.info(f"Got request to knowledge base model={request.model}")
        model = knowledge_base_manager.get_model(request.model)
        increment_metric(db, endpoint='/kbase/', model_name=request.model)
        logger.info(f"Found model={request.model}")

        response = model.query(request.question)
        return MessageResponse(response=response)
    except NoSuchModelError as e:
        logger.exception(f"No such model, details={e}")
        raise HTTPException(status_code=404, detail="No such model")
    except Exception as e:
        logger.exception(
            f"Got exception while using knowledge base (model={request.model}, details={e})"
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/")
def available_models(
    knowledge_base_manager: KnowledgeBaseManager = Depends(get_knowledge_base_manager),
):
    try:
        logger.info(f"Got request to knowledge base available models")

        return JSONResponse(
            content={"models": knowledge_base_manager.available_models()}
        )
    except Exception as e:
        logger.exception(f"Got exception while retrieving available models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get('/metrics/')
def metrics(db: Session = Depends(get_db)) -> MetricsReport:
    logger.info(f"Got request to knowledge base metrics")
    metrics, totals = get_metrics(db)

    return MetricsReport(metrics=metrics, totals=totals)
