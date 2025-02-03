import logging
import sys
from typing import Optional


def setup_logger(
    name: str = 'ml_dev_bench', level: Optional[int] = None
) -> logging.Logger:
    """Setup and configure logger.

    Args:
        name: Logger name (default: 'ml_dev_bench' to catch all ml_dev_bench.* and calipers.* loggers)
        level: Optional logging level. If None, uses INFO

    Returns:
        Configured logger instance
    """
    # Configure the root logger for the entire ml_dev_bench namespace
    logger = logging.getLogger(name)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    # Only configure if not already configured and no handlers exist
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Only set level and propagation if we're configuring for the first time
        logger.propagate = True
        logger.setLevel(level or logging.INFO)

    # If level is explicitly provided, update it
    elif level is not None:
        logger.setLevel(level)

    return logger


# Create default logger instance
logger = setup_logger()
