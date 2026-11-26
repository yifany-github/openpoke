import logging

from server.logging_config import _ExtraFormatter


def test_formatter_includes_extra_fields():
    """Errors are logged as logger.error(msg, extra={"error": ...}); the reason must show."""
    record = logging.LogRecord("openpoke.server", logging.ERROR, __file__, 1, "Interaction agent failed", (), None)
    record.error = "Reached tool iteration limit without final response"
    line = _ExtraFormatter("%(levelname)s - %(message)s").format(record)
    assert line == "ERROR - Interaction agent failed | error='Reached tool iteration limit without final response'"


def test_formatter_unchanged_without_extras():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", (), None)
    assert _ExtraFormatter("%(message)s").format(record) == "hello"
