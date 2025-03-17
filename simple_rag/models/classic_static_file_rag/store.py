from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = InMemoryVectorStore(embeddings)


def populate_vector_store(docs: list):
    vector_store.add_documents(docs)
