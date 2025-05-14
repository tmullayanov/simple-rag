import pandas as pd
from sqlalchemy import Column, String
from .base import BaseEntity


class SampleKBase(BaseEntity):
    __tablename__ = "sample_kbase"
    question = Column(String)
    description = Column(String)
    solution = Column(String)

    @classmethod
    def from_row(cls, row: pd.Series, version: int):
        return cls(
            question=row["Question"],
            description=row["Description"],
            solution=row["Solution"],
            version=version
        )
