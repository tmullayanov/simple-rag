from typing import Iterable, Optional
import abc


class AbstractQnA(abc.ABC):
    @abc.abstractmethod
    def get_questions(self) -> Iterable[str]:
        """
        Get questions as an iterator of strings
        """
        pass

    @abc.abstractmethod
    def get_questions_with_answers(self) -> Iterable[tuple[str, list[str]]]:
        """
        Get questions with answers as an iterator of tuples(question, answers).
        """
        pass

    @abc.abstractmethod
    def lookup_answer(self, question: str) -> Optional[list[str]]:
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

    def __init__(self, db: dict[str, list[str]]):
        self.db = db

    def get_questions(self) -> Iterable[str]:
        return list(self.db.keys())

    def get_questions_with_answers(self) -> Iterable[tuple[str, list[str]]]:
        return self.db.items()

    def lookup_answer(self, question: str) -> Optional[list[str]]:
        return self.db.get(question, None)
