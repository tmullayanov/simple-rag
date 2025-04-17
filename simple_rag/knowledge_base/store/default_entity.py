from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class SampleKBase(Base):
    __tablename__ = "sample_kbase"
    id = Column(Integer, primary_key=True)
    question = Column(String)
    description = Column(String)
    solution = Column(String)
    version = Column(Integer, default=0)
    vectorized = Column(Boolean, default=False)
