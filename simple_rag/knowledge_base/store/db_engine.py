from typing import Optional
from loguru import logger
import pandas as pd
from sqlalchemy import Engine, MetaData, create_engine, Table
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

from simple_rag.knowledge_base.store.default_entity import Base, SampleKBase


class DBEngine:
    class StoreDFError(Exception):
        pass

    class RollbackDBError(Exception):
        pass

    db_link: str = None
    model_name: str = None
    engine: Engine = None

    def __init__(self, db_cfg: dict = {}):
        self.db_link = db_cfg.get("db_link", None)
        self.model_name = db_cfg.get("model_name", None)

        if self.db_link:
            self.engine = create_engine(self.db_link)

    def load_dataframe(self) -> Optional[pd.DataFrame]:
        no_db = self.engine is None
        if no_db:
            logger.info("LOAD_DF skip", no_db=no_db)
            return None

        with self.engine.connect() as connection:
            has_table = self.engine.dialect.has_table(connection, self.model_name)
            if not has_table:
                return None

        Session = sessionmaker(bind=self.engine)
        session = Session()

        try:
            max_version = (
                session.query(SampleKBase)
                .order_by(SampleKBase.version.desc())
                .first()
                .version
                if session.query(SampleKBase).first()
                else 0
            )
        except OperationalError:
            max_version = 0

        data = (
            session.query(SampleKBase).filter(SampleKBase.version == max_version).all()
        )

        df = pd.DataFrame(
            [(row.question, row.description, row.solution) for row in data],
            columns=["Question", "Description", "Solution"],
        )

        return df

    def store_dataframe(self, df: pd.DataFrame, *args, **kwargs):
        if not self.db_link or not self.model_name:
            logger.warning("DB engine not configured, skip store_dataframe")
            return

        Session = sessionmaker(bind=self.engine)
        session = Session()

        with self.engine.connect() as connection:
            if not self.engine.dialect.has_table(connection, self.model_name):
                Base.metadata.create_all(self.engine)

        # trying to find max version
        try:
            max_version = (
                session.query(SampleKBase)
                .order_by(SampleKBase.version.desc())
                .first()
                .version
                if session.query(SampleKBase).first()
                else 0
            )
        except OperationalError:
            logger.info("")
            max_version = 0

        new_version = max_version + 1
        for _, row in df.iterrows():
            new_row = SampleKBase(
                question=row["Question"],
                description=row["Description"],
                solution=row["Solution"],
                version=new_version,
            )
            session.add(new_row)

        session.commit()

        logger.info("store_dataframe DONE")

    def clear_table(self):
        logger.error("DBEngine.clear_table NOT IMPLEMENTED")
