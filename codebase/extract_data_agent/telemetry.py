import logging

from rich.logging import RichHandler


def get_logger():
    """Create or reuse the shared workflow logger."""
    logger = logging.getLogger("extract_data_agent")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = RichHandler(rich_tracebacks=True, markup=True)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


logger = get_logger()
