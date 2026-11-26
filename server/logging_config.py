from __future__ import annotations

import logging

logger = logging.getLogger("openpoke.server")

# Attributes every LogRecord has; anything else came from `extra=`.
_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class _ExtraFormatter(logging.Formatter):
    """Append `extra=` fields to the line.

    The codebase logs failures as ``logger.error("... failed", extra={"error": str(exc)})``,
    but the plain format string dropped every extra field, so error logs never said *why*.
    """

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = {k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS}
        if extras:
            line += " | " + " ".join(f"{k}={v!r}" for k, v in extras.items())
        return line


def configure_logging() -> None:
    """Configure logging with a fixed log level."""
    if logger.handlers:
        return

    logging.basicConfig(level=logging.INFO)
    formatter = _ExtraFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
