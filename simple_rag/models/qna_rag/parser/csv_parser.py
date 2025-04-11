import pandas as pd

from simple_rag.models.qna_rag.qna import AbstractQnA, SimpleQna


class QnAFileParser:
    def __init__(
        self,
        qna_path: str,
        delimiter: str = ";",
        tag_question: str = "Вопрос",
        tag_answer: str = "Ответ",
        **kwargs,
    ):
        self.qna_path = qna_path
        self.delimiter = delimiter
        self.tag_question = tag_question
        self.tag_answer = tag_answer
        self.qna_df = None
        self.qna_db = None

    def load_data(self) -> None:
        """Загружает данные из CSV файла."""
        self.qna_df = pd.read_csv(self.qna_path, delimiter=self.delimiter)

    def parse_qna(self) -> SimpleQna:
        """Создает словарь соответствий вопрос-ответы из DataFrame."""
        if self.qna_df is None:
            self.load_data()

        self.qna_db = {}
        for _, row in self.qna_df.iterrows():
            question = row[self.tag_question]
            answer = row[self.tag_answer]
            self.qna_db.setdefault(question, []).append(answer)

        return SimpleQna(self.qna_db)
