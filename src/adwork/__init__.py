# src/adwork/__init__.py

"""Ad-Work: Agentic Ad Optimization Pipeline"""

__version__ = "0.1.0"

try:
    import sys

    from loguru import logger

    from adwork.config import settings

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    )
except Exception:
    # If config fails (e.g., missing env vars on first import),
    # fall back to basic logging so the app can still show errors
    import logging
    logging.basicConfig(level=logging.INFO)