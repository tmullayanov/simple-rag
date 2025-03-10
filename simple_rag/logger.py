import logging
from logging import Logger
from typing import Optional, TypedDict
import os
from pathlib import Path

GLOBAL_LOGGER_NAME = "simple_rag_logger"


class LogConfig(TypedDict):
    log_level: str
    log_file: Optional[str]
    file_log_level: Optional[str]  # Сделаем необязательным


def setup_logger(cfg: LogConfig) -> Logger:
    logger = logging.getLogger(GLOBAL_LOGGER_NAME)
    logger.setLevel(
        getattr(logging, cfg["log_level"].upper())
    )  # Устанавливаем уровень из конфига

    formatter = logging.Formatter("%(asctime)s: %(name)s - %(levelname)s - %(message)s")

    # Логирование в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Явно задаём уровень для консоли
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Логирование в файл, если указан log_file
    if "log_file" in cfg and cfg["log_file"]:
        Path(os.path.dirname(cfg["log_file"])).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(cfg["log_file"])
        file_handler.setLevel(
            getattr(
                logging, cfg.get("file_log_level", "INFO").upper()
            )  # Уровень по умолчанию DEBUG
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug(f"Set up logger with configuration: {cfg}")
    return logger


def get_logger():
    return logging.getLogger(GLOBAL_LOGGER_NAME)
