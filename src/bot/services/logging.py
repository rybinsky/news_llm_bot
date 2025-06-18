import logging
from pathlib import Path
from typing import Optional


class CustomLogger:
    def __init__(self, name: str, log_level: str = "INFO", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        return self.logger


def setup_logging(config: dict) -> logging.Logger:
    """Setup application-wide logging."""
    logger = CustomLogger(
        name=config.get("name", "news_classifier"), log_level=config.get("level", "INFO"), log_file=config.get("file")
    ).get_logger()

    return logger
