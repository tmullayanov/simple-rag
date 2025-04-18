from typing import Optional, Tuple
from loguru import logger
import pandas as pd
from sqlalchemy import Engine, create_engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

from simple_rag.knowledge_base.store.default_entity import Base, SampleKBase

from sqlalchemy.orm import sessionmaker
from sqlalchemy import func

class StoreDFError(Exception):
    pass

class RollbackDBError(Exception):
    pass

class PseudoDBEngine():
    def __init__(self):
        self._df = None
        self.version = 0

    def load_dataframe(self) -> Optional[pd.DataFrame]:
        return self._df

    def store_dataframe(self, df: pd.DataFrame) -> Tuple[int, list[int]]:
        self._df = df
        self.version += 1
        return self.version, list(range(self._df.shape[0]))
    
    def rollback_version(self, version: int):
        self.version -= 1

    def process_unvectorized_rows(self):
        logger.warning("PseudoDBEngine: process_unvectorized_rows() not implemented")
        return
        

class DBEngine:
    db_link: str = None
    model_name: str = None
    engine: Engine | None = None
    version: int = 0

    def __init__(self, db_cfg: dict = {}):
        self.db_link = db_cfg["db_link"]
        self.model_name = db_cfg["model_name"]

        if self.db_link:
            self.engine = create_engine(self.db_link)

    @property
    def is_configured(self) -> bool:
        return self.engine is not None

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
            max_version = session.query(func.max(SampleKBase.version)).scalar() or 0
        except OperationalError:
            max_version = 0
        
        logger.debug(f"max_version: {max_version}")
        self.version = max_version

        data = (
            session.query(SampleKBase).filter(SampleKBase.version == max_version).all()
        )

        df = pd.DataFrame(
            [(row.id, row.question, row.description, row.solution) for row in data],
            columns=["_id", "Question", "Description", "Solution"],
        )

        return df

    def store_dataframe(self, df: pd.DataFrame) -> Tuple[int, list[int]]:
        """
        Store the DataFrame in the relational database and return the new version number.
        If something goes wrong, raise an exception.

        Args:
            df (pd.DataFrame): The DataFrame to be stored.

        Returns:
            int: The new version number.

        Raises:
            StoreDFError: If something goes wrong during the storage process.
        """
        if not self.engine:
            logger.warning("DB engine not configured, skip store_dataframe")
            raise StoreDFError("DB engine not configured")
        
        with self.engine.connect() as connection:
            if not self.engine.dialect.has_table(connection, self.model_name):
                Base.metadata.create_all(connection)
                logger.info(f"Table '{self.model_name}' created in relational DB")


        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            # Find the latest version and increment it
            max_version = session.query(func.max(SampleKBase.version)).scalar() or 0
            new_version = max_version + 1
            self.version = new_version

            new_ids = []

            for _, row in df.iterrows():
                new_row = SampleKBase(
                    question=row["Question"],
                    description=row["Description"],
                    solution=row["Solution"],
                    version=new_version,
                )
                session.add(new_row)
                session.flush()
                new_ids.append(new_row.id)

            session.commit()
            logger.debug(f"DataFrame saved to DB with version {new_version}")
            
            return new_version, new_ids

        except Exception as e:
            logger.error(f"Failed to save DataFrame to relational DB: {e}")
            logger.exception(e)
            session.rollback()
            raise StoreDFError("Failed to save DataFrame") from e
        finally:
            session.close()

    def rollback_version(self, version: int):
        """
        Удаляет все строки с указанной версией из таблицы.
        """
        if not self.engine:
            logger.warning("DB engine not configured, skip rollback_version")
            raise RollbackDBError("DB engine not configured")

        Session = sessionmaker(bind=self.engine)
        session = Session()

        try:
            with session.begin():
                session.query(SampleKBase).filter(
                    SampleKBase.version == version
                ).delete()
                logger.info(f"Rolled back DB changes for version {version}")
        except Exception as e:
            logger.error(f"Failed to roll back DB changes for version {version}: {e}")
            logger.exception(e)
            raise RollbackDBError("Failed to roll back DB changes") from e

    def process_unvectorized_rows(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()

        try:
            unprocessed_rows = (
                session.query(SampleKBase)
                .filter(SampleKBase.vectorized == False)
                .all()
            )

            for row in unprocessed_rows:
                logger.debug(f'processing {row=}')
                success = yield row
                logger.debug(f'row processed. status={success}')

                if success:
                    row.vectorized = True
                else:
                    # NOTE: maybe we need to raise and abort the rest processing.
                    logger.warning(f"Vectorization failed for row with id={row.id}")

            logger.debug('Loop finished, going to commit...')
            session.commit()
            logger.info("Processed all unvectorized rows")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to process unvectorized rows: {e}")
            raise


    def clear_old_versions(self):
        if not self.engine:
            logger.warning("DB engine not configured, skip clear_old_versions")
            return

        Session = sessionmaker(bind=self.engine)
        session = Session()

        try:
            # Находим максимальную версию
            max_version = (
                session.query(func.max(SampleKBase.version)).scalar() or 0
            )

            session.query(SampleKBase).filter(SampleKBase.version < max_version).delete()

            session.commit()
            logger.info(f"Cleared old versions from DB (less than {max_version}) versions)")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to clear old versions from DB: {e}")
            raise