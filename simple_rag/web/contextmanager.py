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
    print("Starting up the application...")
    # Faking initialization (e.g., connecting to the database)
    await asyncio.sleep(1)

    # Pass control to the main code
    yield

    # Faking shutdown (e.g., closing connections)
    await asyncio.sleep(1)
    print("Shutting down the application...")
