import logging
from logging import Logger
from typing import Optional, TypedDict


GLOBAL_LOGGER_NAME = 'simple_rag_logger'

class LogConfig(TypedDict):
    log_level: str
    log_file: Optional[str]
    file_log_level: str


def setup_logger(cfg: LogConfig) -> Logger:
    logger = logging.getLogger(GLOBAL_LOGGER_NAME)

    logger.setLevel(logging.DEBUG)  # Уровень логирования

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Логирование в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Логирование в файл, если LOG_TO_FILE=True
    if "log_file" in cfg:
        file_handler = logging.FileHandler(cfg.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


    logger.debug(f'Set up logger with configuration: {cfg}')
    return logger

def get_logger():
    return logging.getLogger(GLOBAL_LOGGER_NAME)