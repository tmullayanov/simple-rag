from simple_rag.qna.qna_db import SimpleQna

import unittest


class TestSimpleQnaDB(unittest.TestCase):
    def setUp(self):
        # Подготовка тестовых данных
        self.test_db = {
            "What is Python?": ["A programming language", "A snake"],
            "What is 2 + 2?": ["4"],
            "What is the capital of France?": ["Paris"],
        }
        self.qna_db = SimpleQna(self.test_db)

    def test_get_questions(self):
        # Проверка метода get_questions
        questions = list(self.qna_db.get_questions())
        expected_questions = list(self.test_db.keys())
        self.assertEqual(sorted(questions), sorted(expected_questions))

    def test_get_questions_with_answers(self):
        # Проверка метода get_questions_with_answers
        questions_with_answers = list(self.qna_db.get_questions_with_answers())
        expected_questions_with_answers = list(self.test_db.items())
        self.assertEqual(
            sorted(questions_with_answers), sorted(expected_questions_with_answers)
        )

    def test_lookup_answer_existing_question(self):
        # Проверка метода lookup_answer для существующего вопроса
        question = "What is Python?"
        answers = self.qna_db.lookup_answer(question)
        expected_answers = self.test_db[question]
        self.assertEqual(answers, expected_answers)

    def test_lookup_answer_non_existing_question(self):
        # Проверка метода lookup_answer для несуществующего вопроса
        question = "What is the meaning of life?"
        answers = self.qna_db.lookup_answer(question)
        self.assertIsNone(answers)

    def test_lookup_answer_case_sensitivity(self):
        # Проверка чувствительности к регистру
        question = "what is python?"
        answers = self.qna_db.lookup_answer(question)
        self.assertIsNone(answers)


if __name__ == "__main__":
    unittest.main()
