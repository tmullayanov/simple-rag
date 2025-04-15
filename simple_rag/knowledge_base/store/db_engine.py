from typing import Optional
from loguru import logger
import pandas as pd
from sqlalchemy import Engine, MetaData, create_engine, Table


class DBEngine:
    class StoreDFError(Exception): pass
    class RollbackDBError(Exception): pass

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
            logger.info("LOAD_DF skip", no_db=no_db, no_table=has_table)
            return None

        logger.info('LOAD_DF load', db_link=self.db_link, model_name=self.model_name)
        return pd.read_sql_table(self.model_name, self.engine)


    def store_dataframe(self, df: pd.DataFrame, *args, **kwargs):
        try:
            if not self.db_link or not self.model_name:
                logger.warning("DB engine not configured, skip store_dataframe")
                return
            
            with self.engine.connect() as connection:
                if not self.engine.dialect.has_table(connection, self.model_name):
                    logger.debug('creating table', table_name=self.model_name)
                    df.to_sql(self.model_name, con=self.engine, index=False, if_exists='replace')
                else:
                    logger.debug('adding to table')
                    df.to_sql(self.model_name, con=self.engine, index=False, if_exists='append')
            
            logger.info('store_dataframe DONE')
        except Exception as ex:
            logger.error("ERR_STORE_DATAFRAME", error=ex)
            raise DBEngine.StoreDFError(ex)
    
    def clear_table(self):
        try:
            table_name = self.engine.table_name
            metadata = MetaData()
            with self.engine.connect() as connection:
                # Удаляем только что добавленные строки из таблицы
                table = Table(table_name, metadata, autoload_with=self.engine)
                delete_stmt = table.delete()
                connection.execute(delete_stmt)
                connection.commit()
                logger.debug("Rolled back DB changes due to vectorization failure")
        except Exception as rollback_error:
            logger.error(f"Failed to roll back DB changes: {rollback_error}")
            raise DBEngine.RollbackDBError(rollback_error)
            
