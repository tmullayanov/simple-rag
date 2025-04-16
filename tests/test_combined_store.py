import pytest

from langchain_chroma import Chroma
import pandas as pd
import tempfile
from loguru import logger
from sqlalchemy import MetaData, create_engine

from simple_rag.knowledge_base.store.default_store import Store


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
        {
            "Question": ["q1", "q2", "q3"],
            "Description": ["d1", "d2", "d3"],
            "Solution": ["s1", "s2", "s3"],
        }
    )


def test_save_dataframe_to_tempfile_db(sample_dataframe):
    """
    Тест проверяет, что DataFrame успешно сохраняется в in-memory SQLite базу данных.
    """
    _, db_fname = tempfile.mkstemp()
    db_link = f"sqlite:///{db_fname}"
    tbl_name = "sample_kbase"

    # Create store and save dataframe
    store = Store(
        db_cfg={
            "db_link": db_link,
            "model_name": tbl_name,
        }
    )
    store.store_dataframe(sample_dataframe)

    # Connect to DB and check that the data is written correctly
    engine = create_engine(db_link)
    metadata = MetaData()
    metadata.reflect(engine)

    df = pd.read_sql_table(tbl_name, engine)
    df.drop(labels=["id", "version"], axis=1, inplace=True)

    logger.info(df)
    df.columns = df.columns.str.lower()
    sample_dataframe.columns = sample_dataframe.columns.str.lower()
    assert set(["question", "description", "solution"]).issubset(df.columns)
    pd.testing.assert_frame_equal(df, sample_dataframe)


def test_store_loads_df_from_db(sample_dataframe):
    _, db_fname = tempfile.mkstemp()
    db_link = f"sqlite:///{db_fname}"
    tbl_name = "sample_kbase"

    # Create store and save dataframe
    save_store = Store(
        db_cfg={
            "db_link": db_link,
            "model_name": tbl_name,
        }
    )
    save_store.store_dataframe(sample_dataframe)

    # Create another store and load dataframe
    load_store = Store(
        db_cfg={
            "db_link": db_link,
            "model_name": tbl_name,
        }
    )
    val = load_store.get("Question", "q1")

    assert val == sample_dataframe[sample_dataframe["Question"] == "q1"].to_dict(
        orient="records"
    )
    pd.testing.assert_frame_equal(sample_dataframe, load_store.df)


def test_store_keeps_only_latest_version(sample_dataframe):
    _, db_fname = tempfile.mkstemp()
    db_link = f"sqlite:///{db_fname}"
    tbl_name = "sample_kbase"

    # Create store and save dataframe
    store = Store(db_cfg={
        "db_link": db_link,
        "model_name": tbl_name,
    })

    store.store_dataframe(sample_dataframe)
    lower_val = store.get("Question", "q1")
    logger.info(lower_val)
    assert len(lower_val) == 1

    df = sample_dataframe.apply(lambda x: x.str.upper() if x.dtype == "object" else x)

    store.store_dataframe(df)
    upper_val = store.get("Question", "Q1")
    logger.info(upper_val)
    assert len(upper_val) == 1

    lower_val = store.get("Question", "q1")
    logger.info(lower_val)
    assert len(lower_val) == 0

    
def test_store_keeps_latest_version_after_restart(sample_dataframe):
    _, db_fname = tempfile.mkstemp()
    db_link = f"sqlite:///{db_fname}"
    tbl_name = "sample_kbase"

    # Create store and save dataframe
    store = Store(db_cfg={
        "db_link": db_link,
        "model_name": tbl_name,
    })

    store.store_dataframe(sample_dataframe)
    
    df = sample_dataframe.apply(lambda x: x.str.upper() if x.dtype == "object" else x)
    store.store_dataframe(df)
    
    store_2 = Store(db_cfg={
        "db_link": db_link,
        "model_name": tbl_name,
    })

    upper_val = store_2.get("Question", "Q1")
    logger.info(upper_val)
    assert len(upper_val) == 1

    lower_val = store_2.get("Question", "q1")
    logger.info(lower_val)
    assert len(lower_val) == 0

def test_store_rolls_back_on_vectorization_error(sample_dataframe):
    _, db_fname = tempfile.mkstemp()
    db_link = f"sqlite:///{db_fname}"
    tbl_name = "sample_kbase"

    # Create store and save dataframe
    store = Store(db_cfg={
        "db_link": db_link,
        "model_name": tbl_name,
    })

    assert store.is_empty

    # break vectorizer on purpose in a white-box manner
    store.vectorStore = 10 # ugly but this will cause exception on every attr call.

    with pytest.raises(Exception):
        store.store_dataframe(sample_dataframe)
        
    assert store.is_empty

    engine = create_engine(db_link)
    metadata = MetaData()
    metadata.reflect(engine)

    df = pd.read_sql_table(tbl_name, engine)
    assert df.empty