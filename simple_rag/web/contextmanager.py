import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def startup_and_shutdown(app: FastAPI):
    """
    Context manager.
    Here we initialize connections before the start of the application
    and close resources after the end of the application.

    ------

    Контекстный менеджер.
    Здесь инициализируем подключения до начала работы приложения
    и закрываем ресурсы после окончания работы приложения.
    """
    from .context import APP_CTX

    # Making initialization (e.g., connecting to the database)
    await APP_CTX.on_startup()

    # Pass control to the main code
    yield

    # Making shutdown (e.g., closing connections)
    await APP_CTX.on_shutdown()
