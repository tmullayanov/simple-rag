from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/example")


@router.get("/")
async def read_root():
    return JSONResponse(content={"message": "Hello from the router!"})


@router.get("/hello/{name}")
async def greet_user(name: str):
    return JSONResponse(content={"message": f"Hello, {name}!"})
