import pandas as pd

from simple_rag.qna.pseudo_db import AbstractQnA, SimpleQna


class QnAFileParser:
    def __init__(self, qna_path: str, delimiter: str = ";"):
        self.qna_path = qna_path
        self.delimiter = delimiter
        self.tag_question = "Вопрос"
        self.tag_answer = "Ответ"
        self.qna_df = None
        self.qna_db = None

    def load_data(self) -> None:
        """Загружает данные из CSV файла."""
        self.qna_df = pd.read_csv(self.qna_path, delimiter=self.delimiter)

    def build_qna(self) -> AbstractQnA:
        """Создает словарь соответствий вопрос-ответы из DataFrame."""
        if self.qna_df is None:
            self.load_data()

        self.qna_db = {}
        for _, row in self.qna_df.iterrows():
            question = row[self.tag_question]
            answer = row[self.tag_answer]
            self.qna_db.setdefault(question, []).append(answer)

        return SimpleQna(self.qna_db)
