import uvicorn
from fastapi import FastAPI

from .contextmanager import startup_and_shutdown
from .routes import example_router, qna_router, rag_assistant_router
from .config import APP_SETTINGS


app = FastAPI(lifespan=startup_and_shutdown)


app.include_router(example_router, tags=["test"])
app.include_router(qna_router, tags=['rag', 'qna'])
app.include_router(rag_assistant_router, tags=['rag', 'assistant'])


def run():
    uvicorn.run(app, host=APP_SETTINGS.host, port=APP_SETTINGS.port)
