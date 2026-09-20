"""Unit tests for dense semantic reward shaping."""

import numpy as np
from src.rewards.semantic_reward import SemanticRewardShaper


def test_semantic_reward_shaper() -> None:
    """Verifies reward calculation across match transitions."""
    shaper = SemanticRewardShaper(
        weights={
            "goal_scored": 5.0,
            "goal_conceded": -5.0,
            "territorial_gain": 1.0,
            "pass_completion": 0.5,
            "shot_on_target": 1.0,
            "pressing_intensity": 0.2,
            "possession_loss": -0.5,
            "out_of_bounds": -0.2,
            "lazy_penalty": 0.0,
        },
        clip_range=(-5.0, 5.0),
    )

    obs = np.zeros(115, dtype=np.float32)
    next_obs = np.zeros(115, dtype=np.float32)

    # 1. Goal Scored
    r_goal = shaper.compute_reward(
        agent_id="agent_0",
        env_reward=1.0,
        obs=obs,
        next_obs=next_obs,
        action=0,
        done=True,
        info={},
    )
    assert r_goal == 5.0

    # 2. Territorial advancement: ball moves from x=0.2 to x=0.5 under home possession
    obs[88] = 0.2
    obs[95] = 1.0  # home owns ball
    next_obs[88] = 0.5
    next_obs[95] = 1.0  # home maintains ownership

    r_progression = shaper.compute_reward(
        agent_id="agent_0",
        env_reward=0.0,
        obs=obs,
        next_obs=next_obs,
        action=1,
        done=False,
        info={},
    )
    assert r_progression > 0.0

    # 3. Possession loss: home owns in obs, away owns in next_obs
    obs_loss = np.zeros(115, dtype=np.float32)
    next_obs_loss = np.zeros(115, dtype=np.float32)
    obs_loss[95] = 1.0  # home
    next_obs_loss[96] = 1.0  # away

    r_loss = shaper.compute_reward(
        agent_id="agent_0",
        env_reward=0.0,
        obs=obs_loss,
        next_obs=next_obs_loss,
        action=0,
        done=False,
        info={},
    )
    assert r_loss < 0.0
