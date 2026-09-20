"""Abstract base trainer class for MARL algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch


class BaseMARLTrainer(ABC):
    """Abstract interface defining the contract for MARL training algorithms."""

    @abstractmethod
    def train_step(self, rollouts: Any) -> Dict[str, float]:
        pass

    @abstractmethod
    def save_checkpoint(self, filepath: str) -> None:
        pass

    @abstractmethod
    def load_checkpoint(self, filepath: str) -> None:
        pass
