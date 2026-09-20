"""Abstract interface for modular reward shaping."""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class BaseRewardShaper(ABC):
    """Abstract base class for reward transformation and domain shaping."""

    @abstractmethod
    def compute_reward(
        self,
        agent_id: str,
        env_reward: float,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: int,
        done: bool,
        info: Dict[str, Any],
    ) -> float:
        """Computes the shaped scalar reward for a given agent transition.

        Args:
            agent_id: Identifier of the agent.
            env_reward: Raw scalar reward emitted by the underlying simulator.
            obs: Pre-transition observation vector.
            next_obs: Post-transition observation vector.
            action: Action executed by the agent.
            done: Termination/truncation flag.
            info: Environment auxiliary information dictionary.

        Returns:
            Shaped float reward.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets any internal episode-level state tracking (e.g. past positions)."""
        pass
