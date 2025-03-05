from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

import logging

from simple_rag.logger import GLOBAL_LOGGER_NAME

router = APIRouter(prefix="/example")

logger = logging.getLogger(GLOBAL_LOGGER_NAME)

@router.get("/")
async def read_root():
    logger.info('GET read_root')
    return JSONResponse(content={"message": "Hello from the router!"})


@router.get("/hello/{name}")
async def greet_user(name: str):
    return JSONResponse(content={"message": f"Hello, {name}!"})
