import sys
from loguru import logger
from typing import Optional, TypedDict
import os
from pathlib import Path


class LogConfig(TypedDict):
    log_level: str
    log_file: Optional[str]
    file_log_level: Optional[str]


def setup_logger(cfg: LogConfig):
    level = cfg.get("log_level", "INFO")
    logger.remove()
    logger.add(sys.stdout, level=cfg["log_level"].upper())

    if "log_file" in cfg and cfg["log_file"]:
        Path(os.path.dirname(cfg["log_file"])).mkdir(parents=True, exist_ok=True)
        logger.add(
            cfg["log_file"],
            rotation="2 MB",
            compression="zip",
            level=cfg.get("file_log_level", level).upper(),
        )

    logger.debug(f"Set up logger with configuration: {cfg}")
    return logger


def get_logger():
    return logger
