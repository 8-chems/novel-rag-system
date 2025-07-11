import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_level: str = "INFO", log_file: str = "rag_system.log") -> logging.Logger:
    """Configure logging for the RAG system."""
    logger = logging.getLogger("RAGSystem")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)

        # File handler with rotation (max 5MB, keep 3 backups)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    # Example usage
    logger = setup_logging(log_level="DEBUG")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")