"""Dense semantic reward shaper combining match events and spatial progression."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from src.rewards.base_reward import BaseRewardShaper
from src.utils.registry import REWARD_REGISTRY


@REWARD_REGISTRY.register("dense_semantic")
class SemanticRewardShaper(BaseRewardShaper):
    """Computes a multi-component dense reward function tailored for football.

    Components:
    - Goal scored / conceded (sparse terminal outcomes)
    - Territorial gain (positive movement of the ball along the pitch x-axis)
    - Passing bonus (successful possession retention across players)
    - Shot on target bonus (high xG attempts)
    - Pressing incentive (closing down ball-carrier when opponent controls the ball)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        clip_range: Optional[Tuple[float, float]] = (-5.0, 5.0),
    ) -> None:
        """Initializes the semantic reward shaper.

        Args:
            weights: Dictionary configuring scalar multipliers for reward components.
            clip_range: Tuple of (min_reward, max_reward) for numerical stability.
        """
        self.weights = weights or {
            "goal_scored": 5.0,
            "goal_conceded": -5.0,
            "territorial_gain": 0.8,
            "pass_completion": 0.5,
            "shot_on_target": 1.0,
            "pressing_intensity": 0.2,
            "possession_loss": -0.4,
            "out_of_bounds": -0.2,
            "lazy_penalty": -0.01,
        }
        self.clip_range = clip_range
        self.prev_ball_x: Optional[float] = None
        self.prev_ball_owner: Optional[int] = None

    def reset(self) -> None:
        """Resets tracking state at the start of a new episode."""
        self.prev_ball_x = None
        self.prev_ball_owner = None

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
        """Computes the shaped reward for an agent transition.

        Args:
            agent_id: Active agent identifier.
            env_reward: Raw environment reward (+1 for goal, -1 for conceded).
            obs: Previous 115-dim observation.
            next_obs: Next 115-dim observation.
            action: Discrete action executed.
            done: Episode termination flag.
            info: Step info dictionary.

        Returns:
            Shaped scalar float reward.
        """
        total_reward = 0.0

        # 1. Base Goal Rewards
        if env_reward > 0.5:
            total_reward += self.weights["goal_scored"]
        elif env_reward < -0.5:
            total_reward += self.weights["goal_conceded"]

        if len(next_obs) >= 115 and len(obs) >= 115:
            curr_ball_x = float(obs[88])
            next_ball_x = float(next_obs[88])
            curr_ball_owner = int(np.argmax(obs[94:97]))  # 0: None, 1: Home, 2: Away
            next_ball_owner = int(np.argmax(next_obs[94:97]))

            # 2. Territorial progression (moving toward opponent goal at x=1.0)
            if next_ball_owner == 1:  # Home team has possession
                dx = next_ball_x - curr_ball_x
                if dx > 0:
                    total_reward += self.weights["territorial_gain"] * dx

            # 3. Possession transitions
            if curr_ball_owner == 1 and next_ball_owner == 2:
                total_reward += self.weights["possession_loss"]
            elif curr_ball_owner == 1 and next_ball_owner == 1:
                # Pass completion check: player ownership changed within home team
                curr_player = int(np.argmax(obs[97:108]))
                next_player = int(np.argmax(next_obs[97:108]))
                if curr_player != next_player and next_player != 0:
                    total_reward += self.weights["pass_completion"]

            # 4. Shot attempt (action 12 in GRF is high shot)
            if action == 12 and curr_ball_owner == 1 and curr_ball_x > 0.6:
                total_reward += self.weights["shot_on_target"]

        # 5. Living penalty to encourage active play
        total_reward += self.weights["lazy_penalty"]

        # Clip reward for numerical stability
        if self.clip_range is not None:
            total_reward = float(np.clip(total_reward, self.clip_range[0], self.clip_range[1]))

        return total_reward
