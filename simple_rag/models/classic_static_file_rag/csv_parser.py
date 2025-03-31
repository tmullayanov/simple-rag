from langchain.docstore.document import Document
from langchain_community.document_loaders import CSVLoader


def parse_csv(filename) -> list[Document]:
    loader = CSVLoader(
        file_path=filename,
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["Question, Answer"],
        },
        encoding="utf-8",
    )

    return loader.load()
