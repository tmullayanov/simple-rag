from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from simple_rag.web.context import get_default_llm
import simple_rag.models.summarizer as summarizer

router = APIRouter(prefix='/summary')


class BaseRequest(BaseModel):
    prompt_id: Optional[str] = None
    prompt: Optional[str] = None
    content: str

@router.post('/')
async def summary(
    request: BaseRequest,
    llm = Depends(get_default_llm)):

    return summarizer.summary(llm=llm, content=request.content, prompt_id=request.prompt_id, custom_prompt=request.prompt)