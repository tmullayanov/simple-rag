from typing import Callable
from langchain_chroma import Chroma
import pandas as pd
import pytest
from langchain.vectorstores.base import VectorStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from structlog import get_logger
import tempfile

from simple_rag.embeddings import embeddings

logger = get_logger()

def default_doc_transform(row: dict) -> Document:
    return Document(page_content='\n'.join(f'{col}: {val}' for (col, val) in row.items()))

class Store:

    _is_empty = True
    df: pd.DataFrame = None
    vectorStore: VectorStore

    def __init__(self, vectorstore_cfg: dict = {}, *args, **kwargs):
        self.vectorStore = Store.build_vector_store(vectorstore_cfg)

    @staticmethod
    def build_vector_store(cfg: dict):
        if cfg.get('type', None) == 'chroma':
            return Chroma(
                collection_name=cfg['collection_name'],
                embedding_function=embeddings,
                persist_directory=cfg['persist_directory'],
                collection_metadata={"hnsw:space": "cosine"}
            )

        return InMemoryVectorStore(embeddings)

    @property
    def is_empty(self):
        return self.df is None
    
    def store_dataframe(self, 
                        df, 
                        doc_transform: Callable[[pd.Series], list[Document]] = default_doc_transform):
        self.df = df
        
        logger.debug('vectorStore created')
        docs = self.df.apply(lambda x: doc_transform(x.to_dict()), axis=1).tolist()
        logger.debug('docs created', docs_len=len(docs))
        self.vectorStore.add_documents(docs)
        logger.debug('docs added to vectorStore')

    def get(self, column_name, value) -> list[dict]:
        if self.df is None:
            return []
        return self.df[self.df[column_name] == value].apply(lambda x: x.to_dict(), axis=1).tolist()
    
    def similarity_search(self, query, config: dict = {}):
        return self.vectorStore.similarity_search(query, **config)


@pytest.fixture(scope='session')
def store():
    return Store()

@pytest.fixture
def df(scope='session'):
    return pd.read_csv('assets/support_kbase.csv')

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
    
    matches = store.get('Question', fst_row['Question'])

    assert(len(matches) == 1)
    assert(matches[0] == fst_row.to_dict())

def test_store_get_empty_list_when_no_df(df):
    store = Store()
    fst_row = df.iloc[0]
    matches = store.get('Question', fst_row['Question'])

    assert(isinstance(matches, list))
    assert(len(matches) == 0)


def test_store_can_perform_fuzzy_search(store, df):
    store.store_dataframe(df)
    matches = store.similarity_search('what is the proper way to connect to VM?')
    logger.info('matches', matches=matches)
    assert(len(matches) != 0)
    

def test_store_can_use_chroma_store():
    store = Store(vectorstore_cfg={
        'type': 'chroma',
        'collection_name': "qna_question_store",
        'persist_directory': tempfile.mkdtemp()
    })

    # not the best way to test this but I think it will suffice
    assert(isinstance(store.vectorStore, Chroma))