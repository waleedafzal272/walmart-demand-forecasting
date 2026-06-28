import logging
import sys
from pathlib import Path


def get_logger(name, log_level="INFO", log_file="logs/pipeline.log"):
    """
    Creates a logger that prints to both the screen AND saves to a log file.
    Use it in every file like this:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Format: 2024-01-01 12:00:00 | INFO | module_name | Your message
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Print to screen
    screen_handler = logging.StreamHandler(sys.stdout)
    screen_handler.setFormatter(fmt)
    logger.addHandler(screen_handler)

    # Save to file
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger