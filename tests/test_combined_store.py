from xml.etree.ElementInclude import include
import pytest

from langchain_chroma import Chroma
import pandas as pd
import tempfile
from loguru import logger
from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import sessionmaker

from simple_rag.knowledge_base.store.db_engine import DBEngine
from simple_rag.knowledge_base.store.entity.default import SampleKBase
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
    assert isinstance(store.vectorizer.vector_store, Chroma)


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
    df = pd.read_sql_table(tbl_name, engine)
    df.drop(labels=["id", "version", "vectorized"], axis=1, inplace=True)

    logger.info("df={df}", df=df)
    df.columns = df.columns.str.lower()

    sample_dataframe.columns = sample_dataframe.columns.str.lower()
    
    assert set(["question", "description", "solution"]).issubset(df.columns)
    pd.testing.assert_frame_equal(df, sample_dataframe, check_like=True)


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
    
    # we need to drop _id from store.df
    df = load_store.df.drop(columns=["_id"], axis=1)
    pd.testing.assert_frame_equal(sample_dataframe, df, check_like=True)


def test_store_keeps_only_latest_version(sample_dataframe):
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

    # we get what we saved
    store.store_dataframe(sample_dataframe)
    lower_val = store.get("Question", "q1")
    logger.info(lower_val)
    assert len(lower_val) == 1

    # change df contents and save
    df = sample_dataframe.apply(lambda x: x.str.upper() if x.dtype == "object" else x)
    store.store_dataframe(df)
    upper_val = store.get("Question", "Q1")
    logger.info(upper_val)
    assert len(upper_val) == 1

    # check that we overwrote previous version
    lower_val = store.get("Question", "q1")
    logger.info(lower_val)
    assert len(lower_val) == 0


def test_store_keeps_latest_version_after_restart(sample_dataframe):
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

    df = sample_dataframe.apply(lambda x: x.str.upper() if x.dtype == "object" else x)
    store.store_dataframe(df)

    # emulate restart with new instance
    store_2 = Store(
        db_cfg={
            "db_link": db_link,
            "model_name": tbl_name,
        }
    )

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
    store = Store(
        db_cfg={
            "db_link": db_link,
            "model_name": tbl_name,
        }
    )

    assert store.is_empty

    # break vectorizer on purpose in a white-box manner
    store.vectorizer = 10  # ugly but this will cause exception on every attr call.

    with pytest.raises(Exception):
        store.store_dataframe(sample_dataframe)

    assert store.is_empty

    engine = create_engine(db_link)
    df = pd.read_sql_table(tbl_name, engine)
    assert df.empty


def test_similar_search_returns_original_kbase_row(df):
    _, db_fname = tempfile.mkstemp()
    db_link = f"sqlite:///{db_fname}"
    tbl_name = "sample_kbase"

    # Create store and save dataframe
    store = Store(
        db_cfg={
            "db_link": db_link,
            "model_name": tbl_name,
        },
        vectorstore_cfg={
            "type": "chroma",
            "collection_name": "support_knowledge_base",
            "persist_directory": tempfile.mkdtemp(),
        },
    )

    store.store_dataframe(df)
    matches = store.get_entries_similar_to_problem(
        problem="what is the proper way to connect to VM?"
    )
    logger.info("matches {matches}", matches=matches)

    assert len(matches) > 0

    # checks below are tightly coupled with sample dataframe contents
    assert(all(
        'Question' in entry for entry in matches
    ))
    assert(any(
        'connect to my VM' in entry['Question'] for entry in matches
    ))

def test_unvectored_rows_are_processed_at_startup(df):
    _, db_fname = tempfile.mkstemp()
    db_link = f"sqlite:///{db_fname}"
    tbl_name = "sample_kbase"

    engine = DBEngine({
        'db_link': db_link,
        'model_name': tbl_name
    })
    assert engine.is_configured

    version, ids = engine.store_dataframe(df)
    assert version == 1, "Incorrect version after saving DataFrame"
    assert len(ids) == len(df), "Length of IDs doesn't match DataFrame size"

    store = Store(
        db_cfg={
            "db_link": db_link,
            "model_name": tbl_name,
        },
        vectorstore_cfg={
            "type": "chroma",
            "collection_name": "support_knowledge_base",
            "persist_directory": tempfile.mkdtemp(),
        },
    )

    # store.check_and_vectorize_unprocessed() - this should be called automatically

    # the following works because ChromaStore provides a convenient method to get all docs and ids
    ids = store.vectorizer.vector_store.get(include=[])['ids']
    assert len(df) == len(ids), "Length of IDs doesn't match DataFrame size"

    matches = store.get_entries_similar_to_problem(
        problem="what is the proper way to connect to VM?"
    )
    logger.info("matches {matches}", matches=matches)

    assert len(matches) > 0

def test_db_keeps_only_latest_version(sample_dataframe):
    _, db_fname = tempfile.mkstemp()
    db_link = f"sqlite:///{db_fname}"
    tbl_name = "sample_kbase"
    vector_store_persistence_dir = tempfile.mkdtemp()

    # Create store and save dataframe
    store = Store(
        db_cfg={
            "db_link": db_link,
            "model_name": tbl_name,
        },
        vectorstore_cfg={
            "type": "chroma",
            "collection_name": "support_knowledge_base",
            "persist_directory": vector_store_persistence_dir,
        }
    )

    store.store_dataframe(sample_dataframe)

    df = sample_dataframe.apply(lambda x: x.str.upper() if x.dtype == "object" else x)
    store.store_dataframe(df)

    engine = create_engine(db_link)

    # emulate restart and make sure we keep only the latest version
    store = Store(
        db_cfg={
            "db_link": db_link,
            "model_name": tbl_name,
        },
        vectorstore_cfg={
            "type": "chroma",
            "collection_name": "support_knowledge_base",
            "persist_directory": vector_store_persistence_dir,
        }
    )

    # check that version is unique in db.
    Session = sessionmaker(bind=engine)
    with Session() as session:
        versions = session.query(SampleKBase.version).distinct().all()
        assert len(versions) == 1

    # check that vector store also has only 1 version
    # once again, this trick works only for ChromaStore
    # and it's tightly coupled to metadata structure
    metadatas = store.vectorizer.vector_store.get()['metadatas']
    versions = set(m['_version'] for m in metadatas)
    assert len(versions) == 1