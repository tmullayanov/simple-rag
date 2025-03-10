import io
import sys
import unittest
import tempfile
import logging

from simple_rag.logger import LogConfig, setup_logger


class TestLogger(unittest.TestCase):
    """
    It's hard to check if the log is written to stdout and file at the same time.
    self.assertLogs intercepts all logs and prevents them from being wirtten to file.
    If we check the file, it's impossible to check if the log is written to stdout.

    So, there are two test-cases with identical configuration. It's simpler than possible alternatives.
    """

    def test_log_writes_to_file(self):
        log_message = "Test log message"

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            log_cfg: LogConfig = {
                "log_level": "DEBUG",
                "log_file": temp_file.name,
                "file_log_level": "DEBUG",
            }

            logger = setup_logger(log_cfg)
            logger.info(log_message)

            with open(temp_file.name, "r") as f:
                file_content = f.read().strip()
                self.assertIn(log_message, file_content)

    def test_log_writes_to_stdout(self):
        log_message = "Test log message"

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            log_cfg: LogConfig = {
                "log_level": "DEBUG",
                "log_file": temp_file.name,
                "file_log_level": "DEBUG",
            }

            logger = setup_logger(log_cfg)
            logger.info(log_message)

            with self.assertLogs(logger, level=logging.DEBUG) as cm:
                logger.info(log_message)

            self.assertTrue(any(log_message in log for log in cm.output))
