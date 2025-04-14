import pytest

from langchain_chroma import Chroma
import pandas as pd
import tempfile
from structlog import get_logger
from sqlalchemy import create_engine

from simple_rag.knowledge_base.default_store import Store

logger = get_logger()


@pytest.fixture(scope="session")
def store():
    return Store()


@pytest.fixture(scope="session")
def df():
    return pd.read_csv("assets/support_kbase.csv")


def test_store_can_be_created():
    store = Store()


def test_store_is_empty_by_default(store):
    assert store.is_empty


def test_store_can_be_populated_with_dataframe(store, df):
    store.store_dataframe(df)


def test_store_is_not_empty_after_dataframe_is_stored(store, df):
    store.store_dataframe(df)
    assert not store.is_empty


def test_store_can_access_row(store, df):
    store.store_dataframe(df)

    fst_row = df.iloc[0]

    matches = store.get("Question", fst_row["Question"])

    assert len(matches) == 1
    assert matches[0] == fst_row.to_dict()


def test_store_get_empty_list_when_no_df(df):
    store = Store()
    fst_row = df.iloc[0]
    matches = store.get("Question", fst_row["Question"])

    assert isinstance(matches, list)
    assert len(matches) == 0


def test_store_can_perform_fuzzy_search(store, df):
    store.store_dataframe(df)
    matches = store.similarity_search("what is the proper way to connect to VM?")
    logger.info("matches", matches=matches)
    assert len(matches) != 0


def test_store_can_use_chroma_store():
    store = Store(
        vectorstore_cfg={
            "type": "chroma",
            "collection_name": "qna_question_store",
            "persist_directory": tempfile.mkdtemp(),
        }
    )

    # not the best way to test this but I think it will suffice
    assert isinstance(store.vectorStore, Chroma)

@pytest.fixture
def sample_dataframe():
    """
    Фикстура для создания тестового DataFrame.
    """
    return pd.DataFrame(
        {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
    )


def test_save_dataframe_to_tempfile_db(sample_dataframe):
    """
    Тест проверяет, что DataFrame успешно сохраняется в in-memory SQLite базу данных.
    """
    _, db_fname = tempfile.mkstemp()
    db_link = f"sqlite:///{db_fname}"
    tbl_name = "df_table"

    # Create store and save dataframe
    store = Store(
        db_cfg={
            "db_link": db_link,
            "tbl_name": tbl_name,
        }
    )
    store.store_dataframe(sample_dataframe)

    # Connect to DB and check that the data is written correctly
    engine = create_engine(db_link)
    df = pd.read_sql_table(tbl_name, engine)
    pd.testing.assert_frame_equal(df, sample_dataframe)
