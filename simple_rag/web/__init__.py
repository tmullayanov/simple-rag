import uvicorn
from fastapi import FastAPI

from .contextmanager import startup_and_shutdown
from .routes import rag_assistant_router, summarizer_router, models_router
from .config import APP_SETTINGS


app = FastAPI(
    lifespan=startup_and_shutdown,
    )


app.include_router(rag_assistant_router, tags=["rag", "assistant"])
app.include_router(summarizer_router, tags=["summarizer"])
app.include_router(models_router, tags=['models'])


def run():
    uvicorn.run(app, host=APP_SETTINGS.host, port=APP_SETTINGS.port)
