# database.py
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create the database engine
# NOTE: extract into parameters
engine = create_engine('sqlite:///metrics.db', connect_args={"check_same_thread": False})
Base = declarative_base()

# Storing accesses to kbase models
class KBaseMetric(Base):
    __tablename__ = 'metrics'
    id = Column(Integer, primary_key=True)
    endpoint = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    count = Column(Integer, default=0)

Base.metadata.create_all(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Function for getting a new session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()