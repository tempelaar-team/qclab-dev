import logging
from io import StringIO

_log_stream = StringIO()

class QCDataHandler(logging.Handler):
    """Logging handler that stores logs in memory."""

    def emit(self, record):
        msg = self.format(record)
        _log_stream.write(msg + "\n")


def configure_memory_logger(level=logging.INFO):
    """Configure root logger to store logs without printing."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Remove existing handlers to avoid duplicate output
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    handler = QCDataHandler()
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    return handler


def get_log_output() -> str:
    """Return all collected log messages."""
    return _log_stream.getvalue()
