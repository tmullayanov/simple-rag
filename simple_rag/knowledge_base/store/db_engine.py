from typing import Optional, Tuple, Type, TypedDict
from loguru import logger
import pandas as pd
from sqlalchemy import Engine, create_engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

from simple_rag.knowledge_base.store.entity.base import Base, BaseEntity
from simple_rag.knowledge_base.store.entity.default import SampleKBase

from sqlalchemy.orm import sessionmaker
from sqlalchemy import func


class StoreDFError(Exception):
    pass


class RollbackDBError(Exception):
    pass

class DBEngineConf(TypedDict):
    db_link: str
    model_name: str
    entity_class: Type[BaseEntity] | None


class PseudoDBEngine:
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
        # actually this is how you make empty generator
        return
        yield

    def clear_old_versions(self):
        logger.warning("PseudoDBEngine: clear_old_versions() not implemented")
        return

    def _update_vectorized_flag(self, *args, **kwargs):
        logger.warning("PseudoDBEngine: update_vectorized_flag() not implemented")
        return


class DBEngine:
    db_link: str = None
    model_name: str = None
    engine: Engine | None = None
    version: int = 0
    entity_class: Type[BaseEntity]

    def __init__(self, db_cfg: DBEngineConf = {}):
        self.db_link = db_cfg["db_link"]
        self.model_name = db_cfg["model_name"]
        self.entity_class = db_cfg.get("entity_class", None) or SampleKBase

        if self.db_link:
            logger.debug(f"DBEngine: db_link={self.db_link}")
            self.engine = create_engine(self.db_link)
            self.Session = sessionmaker(bind=self.engine)

    @property
    def is_configured(self) -> bool:
        return self.engine is not None

    def load_dataframe(self) -> Optional[pd.DataFrame]:
        no_db = self.engine is None
        if no_db:
            logger.info("LOAD_DF skip", no_db=no_db)
            return None

        has_table = self._check_if_table_exists(self.entity_class.__tablename__)
        if not has_table:
            logger.info("LOAD_DF: No table in DB")
            return None

        session = self.Session()

        try:
            max_version = (
                session.query(func.max(self.entity_class.version)).scalar() or 0
            )
        except OperationalError:
            max_version = 0

        logger.debug(f"LOAD_DF max_version: {max_version}")
        self.version = max_version

        data = (
            session.query(self.entity_class)
            .filter(self.entity_class.version == max_version)
            .all()
        )

        df = pd.DataFrame(
            [(row.id, row.question, row.description, row.solution) for row in data],
            columns=["_id", "Question", "Description", "Solution"],
        )
        logger.info(f"LOAD_DF, {df.empty=}")

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
            if not self.engine.dialect.has_table(
                connection, self.entity_class.__tablename__
            ):
                Base.metadata.create_all(connection)
                logger.info(f"Table '{self.model_name}' created in relational DB")


        session = self.Session()
        try:
            # Find the latest version and increment it
            max_version = (
                session.query(func.max(self.entity_class.version)).scalar() or 0
            )
            new_version = max_version + 1
            self.version = new_version

            new_ids = []

            for _, row in df.iterrows():
                # NOTE: candidate for static ctor
                new_row = self.entity_class.from_row(row, new_version)
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

        session = self.Session()

        try:
            with session.begin():
                session.query(self.entity_class).filter(
                    self.entity_class.version == version
                ).delete()
                logger.info(f"Rolled back DB changes for version {version}")
        except Exception as e:
            logger.error(f"Failed to roll back DB changes for version {version}: {e}")
            logger.exception(e)
            raise RollbackDBError("Failed to roll back DB changes") from e

    def process_unvectorized_rows(self):
        has_table = self._check_if_table_exists(self.model_name)
        if not has_table:
            logger.info("PROCE_UNVEC: No table in DB, return")
            return
            
        session = self.Session()

        try:
            unprocessed_rows = (
                session.query(self.entity_class)
                .filter(self.entity_class.vectorized == False)
                .all()
            )

            for row in unprocessed_rows:
                logger.debug(f"processing {row=}")
                success = yield row
                logger.debug(f"row processed. status={success}")

                if success:
                    row.vectorized = True
                else:
                    # NOTE: maybe we need to raise and abort the rest processing.
                    logger.warning(f"Vectorization failed for row with id={row.id}")

            logger.debug("Loop finished, going to commit...")
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

        has_table = self._check_if_table_exists(self.model_name)
        if not has_table:
            logger.info("CLEAR_OLD_VERS: No table in DB, return")
            return None

        session = self.Session()

        try:
            # Находим максимальную версию
            max_version = (
                session.query(func.max(self.entity_class.version)).scalar() or 0
            )

            session.query(self.entity_class).filter(
                self.entity_class.version < max_version
            ).delete()

            session.commit()
            logger.info(
                f"Cleared old versions from DB (less than {max_version}) versions)"
            )

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to clear old versions from DB: {e}")
            raise

    def _check_if_table_exists(self, table_name):
        with self.engine.connect() as connection:
            has_table = self.engine.dialect.has_table(connection, table_name)
            logger.debug(f"_check if table exists: {has_table=}")
            return has_table

    def _update_vectorized_flag(self, version: int, new_status: bool = True):
        if not self.engine:
            logger.warning("DB engine not configured, skip clear_old_versions")
            return

        has_table = self._check_if_table_exists(self.model_name)
        if not has_table:
            logger.info("CLEAR_OLD_VERS: No table in DB, return")
            return None

        session = self.Session()

        try:
            unprocessed_rows = (
                session.query(self.entity_class)
                .filter(self.entity_class.version == version)
                .all()
            )

            for row in unprocessed_rows:
                logger.debug(f"processing {row=}")
                row.vectorized = new_status

            session.commit()
            logger.info(f"Updated vectorized flag for version {version}")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update vectorized flag for version {version}: {e}")
