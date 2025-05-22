from typing import Self
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
    
    @classmethod
    def make_df(cls, batch: list[Self]):
        return pd.DataFrame(
            [(row.id, row.question, row.description, row.solution) for row in batch],
            columns=["_id", "Question", "Description", "Solution"],
        )
    
    def to_vector_document(self) -> dict:
        return {
            "Question": self.question,
            "Description": self.description,
            "Solution": self.solution,
        }
