from langchain_huggingface import HuggingFaceEmbeddings

# FIXME: move to context
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")