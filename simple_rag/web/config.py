from typing import Literal, Optional, Union
from pydantic import Field
from pydantic_settings import BaseSettings

from dotenv import load_dotenv


load_dotenv(override=True)


class GroqSettings(BaseSettings):
    model_name: str = Field(validation_alias="GROQ_MODEL_NAME")


class HttpSettings(BaseSettings):
    host: str = Field(validation_alias="HOST", default="127.0.0.1")
    port: int = Field(validation_alias="PORT", cast=int, default=8000)


class QnaFileSettings(BaseSettings):
    qna_path: str = Field(validation_alias="QNA_FILE_PATH")
    qna_delimiter: str = Field(validation_alias="QNA_DELIMITER", default=";")


class LoggerSettings(BaseSettings):
    log_level: str = Field(validation_alias="CONSOLE_LOG_LEVEL", default="info")
    log_file: Optional[str] = Field(validation_alias="FILE_LOG", default=None)
    file_log_level: Optional[str] = Field(
        validation_alias="FILE_LOG_LEVEL", default="info"
    )


class DbConfig(BaseSettings):
    db_link: Optional[str] = None
    model_name: Optional[str] = None

    class Config:
        env_prefix = "DB_"  # Префикс для переменных окружения


class ChromaVectorStoreConfig(BaseSettings):
    type: Literal['chroma'] = 'chroma'
    collection_name: str
    persist_directory: str

    class Config:
        env_prefix = "VECTORSTORE_CHROMA_"

VectorStoreConfig = Union[ChromaVectorStoreConfig]

class StoreConfig(BaseSettings):
    db_cfg: DbConfig = Field(default_factory=DbConfig)
    vectorstore_cfg: VectorStoreConfig = Field(default_factory=VectorStoreConfig, discriminator='type')
    csv_fallback_path: str = Field(validation_alias='STORE_CSV_FALLBACK', default=None)


class AppSettings(HttpSettings, GroqSettings, QnaFileSettings, LoggerSettings, StoreConfig):
    pass


APP_SETTINGS = AppSettings()

__all__ = ["AppSettings", "APP_SETTINGS"]
