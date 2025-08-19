from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from config import BCIConfig
from utils.data_loader import BCIDataLoader
from utils.logger import get_logger, setup_file_logging


class ExecutableMode(Protocol):
    def execute(self, subject: int, run: int, dataset_dir: str) -> None: ...


class ModeBase(ABC):
    """Abstract base for modes providing common logger setup."""
    LOG_NAME = 'Mode'
    LOG_FILE = 'mode.log'

    def __init__(self, config: BCIConfig, data_loader: BCIDataLoader):
        self.config = config
        self.data_loader = data_loader
        self.logger = get_logger(self.LOG_NAME)
        setup_file_logging(Path('logs'), self.LOG_FILE)

    @abstractmethod
    def execute(self, subject: int, run: int, dataset_dir: str) -> None:
        raise NotImplementedError

    def _finish(self):
        try:
            self.logger.stop()
        except Exception:
            pass
