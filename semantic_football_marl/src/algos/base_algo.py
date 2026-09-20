"""Abstract base trainer class for MARL algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch


class BaseMARLTrainer(ABC):
    """Abstract interface defining the contract for MARL training algorithms."""

    @abstractmethod
    def train_step(self, rollouts: Any) -> Dict[str, float]:
        """Performs an optimization step on collected experience rollouts.

        Args:
            rollouts: Replay buffer or rollout buffer holding environment transitions.

        Returns:
            Dictionary of training loss components and gradient diagnostics.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, filepath: str) -> None:
        """Saves trainer neural network weights and optimizer states.

        Args:
            filepath: Destination file path for saved state.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, filepath: str) -> None:
        """Loads weights and optimizer states from a saved checkpoint file.

        Args:
            filepath: Path to checkpoint file.
        """
        pass
