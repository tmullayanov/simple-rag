from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from simple_rag.qna_rag.service import QnAServiceConfig, QnaStaticFileService
from simple_rag.web.config import APP_SETTINGS

router = APIRouter(prefix="/qna")


def get_qna_service(cfg: QnAServiceConfig):
    return QnaStaticFileService(cfg)

@router.get("/")
async def create_qna_rag(service: QnaStaticFileService = Depends(lambda: get_qna_service(APP_SETTINGS.model_dump()))):
    
    print(service.store.doc_ids)
    return JSONResponse(content={"message": "Hello from the router!"})
