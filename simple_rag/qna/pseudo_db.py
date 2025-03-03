from typing import Optional, Iterator
import abc

from simple_rag.qna.csv_parser import QnAFileParser


class AbstractQnA(abc.ABC):
    @abc.abstractmethod
    def get_questions() -> Iterator[str]:
        """
        Get questions as an iterator of strings
        """
        pass

    @abc.abstractmethod
    def get_questions_with_answers() -> Iterator[tuple[str, list[str]]]:
        """
        Get questions with answers as an iterator of tuples(question, answers).
        """
        pass

    @abc.abstractmethod
    def lookup_answer(question: str) -> Optional[list[str]]:
        """
        Get answers for a question from our Q&A database as a list of strings.
        If there is no such question, return None.
        """
        pass


class SimpleQna(AbstractQnA):
    """
    Simple Q&A database.

    The database is a dictionary {question: [answers]} where answers are a list of strings.
    """

    db: dict[str, list[str]]
    parser: QnAFileParser

    def __init__(self, parser: QnAFileParser):
        parser.load_data()
        self.parser = parser
        self.db = parser.build_qna_dict()

    def get_questions(self) -> Iterator[str]:
        return self.db.keys()

    def get_questions_with_answers(self) -> Iterator[tuple[str, list[str]]]:
        return self.db.items()

    def lookup_answer(self, question: str) -> Optional[list[str]]:
        return self.db.get(question, None)
