"""
base.py - Base module for knowledge base models.

This module defines the foundational classes and protocols for ORM entities that are designed to support:
1. **Versioning**: Each entity tracks its version using the `version` field.
2. **Vectorization**: Entities can be marked as vectorized using the `vectorized` field.
3. **Interoperability with Pandas DataFrame**: Entities can be constructed from DataFrame rows and converted back if needed.

To maintain a clean separation of concerns, the logic for DataFrame interoperability and vectorization is encapsulated in mixins and protocols rather than being directly embedded in the SQLAlchemy base class.

### Key Components:

- **BaseEntity**:
  An abstract SQLAlchemy base class for all entities. It provides common fields (`id`, `version`, `vectorized`) and serves as the foundation for all ORM models. Use this class if you only need basic ORM functionality without additional features like DataFrame or vector store support.

- **DataframeMixin**:
  A protocol defining methods for interoperability with Pandas DataFrames. It includes a `from_row` method for creating instances from DataFrame rows.

- **BaseEntityProtocol**:
  A protocol combining `BaseEntity` and `DataframeMixin`. Use this protocol as the base for entities that require both ORM functionality and DataFrame interoperability. If you need to work with vector stores or DataFrames, inherit from this protocol instead of `BaseEntity`.

### Usage:

- Extend **`BaseEntity`** to create new ORM models if you only need basic ORM functionality.
- Use mixins (e.g., `DataframeMixin`) to add specific functionality without enforcing inheritance if your entity needs to support both ORM functionality and interoperability with DataFrames or vector stores.

### Design Philosophy:

- **Modularity**: Business logic (e.g., vectorization, DataFrame handling) is decoupled from the core ORM functionality to improve maintainability and testability.
- **Flexibility**: Protocols are used to define clear interfaces without enforcing inheritance, allowing for reusable and adaptable designs.
- **Guidance for Developers**: If you need advanced features like DataFrame or vector store support, always inherit from **`BaseEntityProtocol`** instead of `BaseEntity`. This ensures that your entity adheres to the required interface.

Example:

```python
from sqlalchemy import Column, String

class SampleKBase(BaseEntity, DataframeMixin):
    __tablename__ = "sample_kbase"

    id = Column(Integer, primary_key=True)
    description = Column(String)

    @classmethod
    def from_row(cls, row):
        return cls(**dict(row))

    def to_vector_document(self):
        return {"id": self.id, "data": self.description}
"""        
from typing import Protocol, Self
import pandas as pd
from sqlalchemy import Column, Integer, Boolean
from sqlalchemy.orm import declarative_base
from abc import abstractmethod

Base = declarative_base()

class BaseEntity(Base):
    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Integer, default=0)
    vectorized = Column(Boolean, default=False)

    @classmethod
    @abstractmethod
    def from_row(cls, row: pd.Series, version: int):
        '''Create instance from a DataFrame row'''
        raise NotImplementedError("Subclasses must implement this method")
    
    @classmethod
    @abstractmethod
    def make_df(cls, batch: list[Self]) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def to_vector_document(self) -> dict:
        raise NotImplementedError("Subclasses must implement this method")

