from fastapi import APIRouter, Depends
from starlette.responses import JSONResponse

from simple_rag.models import ModelCreator
from simple_rag.web.context import get_model_creator

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/")
def models(model_creator: ModelCreator = Depends(get_model_creator)):
    return JSONResponse(content={"models": model_creator.models()})
