from logging import Logger

from simple_rag.logger import LogConfig, setup_logger
from simple_rag.web.config import APP_SETTINGS


class AppContext:

    logger: Logger

    def __init__(self, log_cfg: LogConfig):
        self.logger = setup_logger(log_cfg)

    async def on_startup(self):
        self.logger.debug('AppContext STARTUP')

    async def on_shutdown(self):
        self.logger.debug('AppContext SHUTDOWN')


APP_CTX = AppContext(APP_SETTINGS)

__all__ = [
    'AppContext',
    'APP_CTX'
]