from pydantic import Field
from pydantic_settings import BaseSettings

from dotenv import load_dotenv


load_dotenv()


class GroqSettings(BaseSettings):
    model_name: str = Field(validation_alias="GROQ_MODEL_NAME")


class HttpSettings(BaseSettings):
    host: str = Field(validation_alias="HOST", default="127.0.0.1")
    port: int = Field(validation_alias="PORT", cast=int, default=8000)


class QnaFileSettings(BaseSettings):
    qna_path: str = Field(validation_alias="QNA_FILE_PATH")
    qna_delimiter: str = Field(validation_alias="QNA_DELIMITER", default=";")


class AppSettings(HttpSettings, GroqSettings, QnaFileSettings):
    pass


APP_SETTINGS = AppSettings()

__all__ = ["AppSettings", "APP_SETTINGS"]
