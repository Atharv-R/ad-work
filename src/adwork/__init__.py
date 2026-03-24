# src/adwork/__init__.py

from loguru import logger
import sys

from adwork.config import settings


# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
)

__version__ = "0.1.0"