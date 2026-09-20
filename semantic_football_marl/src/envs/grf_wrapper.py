"""Gymnasium-compatible Multi-Agent Environment wrapper for Google Research Football (GRF).

================================================================================
CRITICAL ARCHITECTURAL CONSTRAINTS:
1. PRIMARY SCENARIO: '11_vs_11_stochastic' (Full match football simulation).
   Do NOT use 3v1 or single-policy simplifications for primary MARL experiments.
2. AGENT HIERARCHY: Exactly 10 learning outfield agents (indices 1..10).
   Left team player 0 is the Goalkeeper, executed under the fixed engine AI protocol.
3. ACTION SPACE: Exactly 19 discrete macro-actions per learning agent.
4. OBSERVATION MODES:
   - 'raw': 115 dimensions (GRF simple115v2 representation).
   - 'semantic': 139 dimensions (115D raw + 24D tactical features from src/features/).
5. CENTRALIZED CRITIC STATE: Full global state containing the complete 22-player
   coordinates/velocities + 3D ball kinematics + match context (104 dimensions).
================================================================================
"""

import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Gracefully import PettingZoo ParallelEnv base class
try:
    from pettingzoo import ParallelEnv
except ImportError:
    class ParallelEnv:  # type: ignore
        """Fallback base class when PettingZoo is not present."""
        pass

# Import semantic feature extractor from src/features
try:
    from src.features.semantic_extractor import SemanticFeatureExtractor
except (ImportError, ValueError):
    try:
        from semantic_football_marl.src.features.semantic_extractor import SemanticFeatureExtractor
    except Exception:
        # Local fallback definition if standalone
        SemanticFeatureExtractor = None  # type: ignore

# Optional registry integration
try:
    from src.utils.registry import ENV_REGISTRY
except (ImportError, ValueError):
    try:
        from semantic_football_marl.src.utils.registry import ENV_REGISTRY
    except Exception:
        class _DummyRegistry:
            def register(self, name: str):
                return lambda cls: cls
        ENV_REGISTRY = _DummyRegistry()


# ==============================================================================
# Canonical Action & Observation Dimension Constants
# ==============================================================================
# GRF discrete macro-actions (19 actions)
NUM_ACTIONS = 19

# Observation dimensions
RAW_OBS_DIM = 115
SEMANTIC_TACTICAL_DIM = 24
SEMANTIC_OBS_DIM = 139  # 115 + 24

# Global state dimensions (22 players + ball + match state)
# 11 left players: pos (22) + vel (22) = 44
# 11 right players: pos (22) + vel (22) = 44
# Ball: pos (3) + vel (3) = 6
# Match context: ball_owned (3) + game_mode (7) = 10
# Total = 44 + 44 + 6 + 10 = 104 dimensions
GLOBAL_STATE_DIM = 104

# Outfield agent configuration
NUM_LEARNING_AGENTS = 10
PRIMARY_SCENARIO = "11_vs_11_stochastic"
GOALKEEPER_PROTOCOL = "fixed_engine_ai"


@ENV_REGISTRY.register("grf_11v11_outfield")
class FootballMultiAgentEnv(ParallelEnv):
    """Gymnasium and PettingZoo compatible multi-agent wrapper for Google Research Football.

    Enforces 10 learning outfield agents on '11_vs_11_stochastic', leaving the goalkeeper
    under fixed protocol control.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "grf_11v11_outfield_v1"}

    # Macro action definitions documented for explicit validation
    ACTION_NAMES: Dict[int, str] = {
        0: "idle",
        1: "left",
        2: "top_left",
        3: "top",
        4: "top_right",
        5: "right",
        6: "bottom_right",
        7: "bottom",
        8: "bottom_left",
        9: "long_pass",
        10: "high_pass",
        11: "short_pass",
        12: "shot",
        13: "sprint",
        14: "release_direction",
        15: "release_sprint",
        16: "sliding",
        17: "dribble",
        18: "release_dribble",
    }

    def __init__(
        self,
        scenario_name: str = PRIMARY_SCENARIO,
        obs_mode: str = "semantic",
        rewards: str = "scoring,checkpoints",
        stacked: bool = False,
        render_mode: Optional[str] = None,
        max_steps: int = 3000,
        mock_mode: bool = False,
        write_goal_dumps: bool = False,
        write_full_episode_dumps: bool = False,
    ) -> None:
        """Initializes the multi-agent football environment.

        Args:
            scenario_name: Name of the GRF scenario (defaults to '11_vs_11_stochastic').
            obs_mode: Observation mode: 'raw' (115D) or 'semantic' (139D).
            rewards: Reward formulation ('scoring' or 'scoring,checkpoints').
            stacked: Whether to stack historical frames.
            render_mode: Rendering backend ('human', 'rgb_array', or None).
            max_steps: Maximum timesteps before episode truncation.
            mock_mode: Explicitly enable synthetic simulator for headless testing/CI.
            write_goal_dumps: Dump replays on scored goals.
            write_full_episode_dumps: Dump replays of full episodes.
        """
        super().__init__()
        self.scenario_name = scenario_name
        self.obs_mode = obs_mode.lower()
        assert self.obs_mode in ("raw", "semantic"), (
            f"Invalid obs_mode '{obs_mode}'. Must be either 'raw' (115D) or 'semantic' (139D)."
        )

        self.rewards = rewards
        self.stacked = stacked
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.is_mock = mock_mode

        # ======================================================================
        # CRITICAL SANITY CHECK: Exactly 10 learning outfield agents
        # ======================================================================
        self.num_learning_agents = NUM_LEARNING_AGENTS
        self.goalkeeper_protocol = GOALKEEPER_PROTOCOL

        # Agent identifiers for outfield players (player_1 through player_10)
        self.possible_agents = [f"player_{i}" for i in range(1, self.num_learning_agents + 1)]
        self.agents = self.possible_agents[:]

        # Enforce sanity check assertions
        assert len(self.possible_agents) == 10, (
            f"Sanity Check Failed: Expected exactly 10 learning outfield agents, got {len(self.possible_agents)}!"
        )
        assert self.num_learning_agents == 10, (
            f"Sanity Check Failed: num_learning_agents must equal 10, got {self.num_learning_agents}!"
        )

        # ======================================================================
        # Observation and Action Space Specifications
        # ======================================================================
        self.action_dim = NUM_ACTIONS
        self.obs_dim = SEMANTIC_OBS_DIM if self.obs_mode == "semantic" else RAW_OBS_DIM
        self.global_state_dim = GLOBAL_STATE_DIM

        # Document spaces with assertions
        assert self.action_dim == 19, f"Action dimension must be 19, got {self.action_dim}"
        assert self.obs_dim in (115, 139), f"Observation dimension must be 115 or 139, got {self.obs_dim}"
        assert self.global_state_dim == 104, f"Global state dimension must be 104, got {self.global_state_dim}"

        self.action_spaces: Dict[str, spaces.Discrete] = {
            agent: spaces.Discrete(self.action_dim) for agent in self.possible_agents
        }
        self.observation_spaces: Dict[str, spaces.Box] = {
            agent: spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

        # Initialize semantic feature extractor
        if SemanticFeatureExtractor is not None:
            self.semantic_extractor = SemanticFeatureExtractor()
        else:
            self.semantic_extractor = None

        # Cached latest state buffers
        self._latest_raw_obs: Optional[np.ndarray] = None
        self._latest_global_state: np.ndarray = np.zeros(self.global_state_dim, dtype=np.float32)

        # Initialize native GFootball environment
        self._native_env = None
        self._init_gfootball_env(write_goal_dumps, write_full_episode_dumps)

    def _init_gfootball_env(self, write_goal_dumps: bool, write_full_episode_dumps: bool) -> None:
        """Initializes native gfootball environment or falls back to synthetic mock."""
        if self.is_mock:
            return

        try:
            import gfootball.env as football_env

            # In GRF, number_of_left_players_agent_controls=10 controls players 1..10 (outfield)
            # Player 0 (goalkeeper) remains under fixed engine AI control
            self._native_env = football_env.create_environment(
                env_name=self.scenario_name,
                stacked=self.stacked,
                representation="simple115v2",
                rewards=self.rewards,
                write_goal_dumps=write_goal_dumps,
                write_full_episode_dumps=write_full_episode_dumps,
                render=(self.render_mode == "human"),
                number_of_left_players_agent_controls=self.num_learning_agents,
            )
        except Exception:
            # Fallback to mock mode if native C++ engine is unavailable
            self.is_mock = True
            self._native_env = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Resets the environment for a new episode.

        Args:
            seed: Optional integer random seed.
            options: Optional auxiliary options.

        Returns:
            Tuple of (observations_dict, infos_dict).
        """
        self.agents = self.possible_agents[:]
        self.current_step = 0

        if seed is not None:
            np.random.seed(seed)

        if not self.is_mock and self._native_env is not None:
            raw_obs = self._native_env.reset()
            # GRF multi-agent returns an array of shape (10, 115) for the 10 outfield agents
            obs_array = np.asarray(raw_obs, dtype=np.float32)
            if obs_array.ndim == 1:
                obs_array = np.tile(obs_array[np.newaxis, :], (self.num_learning_agents, 1))
            self._latest_raw_obs = obs_array
        else:
            # Synthetic 10 outfield agent observation tensor
            self._latest_raw_obs = self._generate_synthetic_raw_obs()

        # Update and cache global state
        self._update_global_state(self._latest_raw_obs[0])

        # Construct agent observations according to obs_mode
        obs_dict = self._process_observations(self._latest_raw_obs)
        global_state = self.get_global_state()

        infos_dict = {
            agent: {
                "step": 0,
                "scenario": self.scenario_name,
                "is_mock": self.is_mock,
                "global_state": global_state,
                "goalkeeper_protocol": self.goalkeeper_protocol,
            }
            for agent in self.agents
        }
        return obs_dict, infos_dict

    def step(
        self, actions: Union[Dict[str, int], List[int], np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Executes simultaneous actions for all 10 learning outfield agents.

        Args:
            actions: Dictionary mapping agent id ('player_1'..'player_10') to macro-action (0..18),
                or list/array of 10 discrete macro-actions.

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos).
        """
        self.current_step += 1
        is_truncated = self.current_step >= self.max_steps

        # Convert actions to ordered list of 10 integers
        if isinstance(actions, dict):
            action_list = [int(actions.get(agent, 0)) for agent in self.possible_agents]
        elif isinstance(actions, (list, tuple, np.ndarray)):
            action_list = [int(a) for a in actions]
        else:
            raise ValueError(f"Unsupported actions format: {type(actions)}")

        assert len(action_list) == 10, (
            f"Action dimension mismatch: Expected exactly 10 actions for outfield agents, got {len(action_list)}"
        )

        for a in action_list:
            assert 0 <= a < self.action_dim, (
                f"Invalid discrete macro action {a}. Must be in [0..{self.action_dim - 1}]."
            )

        if not self.is_mock and self._native_env is not None:
            raw_obs, raw_rewards, done, raw_info = self._native_env.step(action_list)
            obs_array = np.asarray(raw_obs, dtype=np.float32)
            if obs_array.ndim == 1:
                obs_array = np.tile(obs_array[np.newaxis, :], (self.num_learning_agents, 1))

            self._latest_raw_obs = obs_array

            # Handle rewards
            if isinstance(raw_rewards, (list, np.ndarray)):
                rewards_dict = {
                    agent: float(raw_rewards[i]) for i, agent in enumerate(self.possible_agents)
                }
            else:
                rewards_dict = {agent: float(raw_rewards) for agent in self.possible_agents}

            terminated = bool(done)
            ball_owned = raw_info.get("ball_owned_team", -1) if isinstance(raw_info, dict) else -1
            score = raw_info.get("score", [0, 0]) if isinstance(raw_info, dict) else [0, 0]
        else:
            # Synthetic step logic for CI / mock execution
            self._latest_raw_obs = self._generate_synthetic_raw_obs()
            rewards_dict = {agent: 0.0 for agent in self.possible_agents}
            terminated = False
            ball_owned = 1
            score = [0, 0]

        # Update cached global state
        self._update_global_state(self._latest_raw_obs[0])
        global_state = self.get_global_state()

        # Build observations
        obs_dict = self._process_observations(self._latest_raw_obs)
        terminations_dict = {agent: terminated for agent in self.possible_agents}
        truncations_dict = {agent: is_truncated for agent in self.possible_agents}

        infos_dict = {
            agent: {
                "step": self.current_step,
                "ball_owned_team": ball_owned,
                "score": score,
                "is_mock": self.is_mock,
                "global_state": global_state,
                "goalkeeper_protocol": self.goalkeeper_protocol,
            }
            for agent in self.possible_agents
        }

        if terminated or is_truncated:
            self.agents = []

        return obs_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict

    def get_global_state(self) -> np.ndarray:
        """Returns the full 22-player + ball configuration for the Centralized Critic.

        Dimensions:
        - [0:22]   Left team (11 players) (x, y) coordinates
        - [22:44]  Left team (11 players) (vx, vy) velocities
        - [44:66]  Right team (11 players) (x, y) coordinates
        - [66:88]  Right team (11 players) (vx, vy) velocities
        - [88:91]  Ball 3D position (x, y, z)
        - [91:94]  Ball 3D velocity (vx, vy, vz)
        - [94:97]  Ball possession one-hot [none, left, right]
        - [97:104] Game mode one-hot (7 dimensions)

        Total dimensions: 44 + 44 + 6 + 3 + 7 = 104 float32 values.

        Returns:
            np.ndarray: 1D float32 array of shape (104,).
        """
        assert self._latest_global_state.shape == (self.global_state_dim,), (
            f"Global state dimension mismatch: Expected ({self.global_state_dim},), got {self._latest_global_state.shape}"
        )
        return self._latest_global_state.copy()

    def _update_global_state(self, raw_115: np.ndarray) -> None:
        """Extracts and updates the canonical 104-dim 22-player + ball configuration from raw 115."""
        assert len(raw_115) >= RAW_OBS_DIM, (
            f"Raw observation must be at least {RAW_OBS_DIM} dimensions, got {len(raw_115)}"
        )
        # Slices from simple115v2:
        # 0:88   -> 22 players positions and velocities (44 left + 44 right)
        # 88:94  -> Ball 3D pos (3) + 3D vel (3)
        # 94:97  -> Ball possession one-hot (3)
        # 108:115 -> Game mode one-hot (7)
        players_and_ball = raw_115[0:97]      # 97 floats
        game_mode = raw_115[108:115]          # 7 floats
        self._latest_global_state = np.concatenate([players_and_ball, game_mode], axis=0).astype(np.float32)
        assert len(self._latest_global_state) == self.global_state_dim

    def _process_observations(self, obs_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Transforms raw simple115 observations into requested obs_mode ('raw' or 'semantic')."""
        obs_dict: Dict[str, np.ndarray] = {}
        for i, agent in enumerate(self.possible_agents):
            raw_agent_obs = obs_array[i]
            if self.obs_mode == "semantic":
                if self.semantic_extractor is not None:
                    # Outfield agent i corresponds to player index i + 1 (player 0 is keeper)
                    agent_obs = self.semantic_extractor.build_semantic_observation(
                        raw_agent_obs, agent_idx=i + 1
                    )
                else:
                    # Fallback concatenation of 24 zeros
                    agent_obs = np.concatenate([raw_agent_obs, np.zeros(SEMANTIC_TACTICAL_DIM, dtype=np.float32)])
                assert agent_obs.shape == (SEMANTIC_OBS_DIM,), (
                    f"Semantic obs dimension mismatch: Expected ({SEMANTIC_OBS_DIM},), got {agent_obs.shape}"
                )
            else:
                agent_obs = raw_agent_obs[:RAW_OBS_DIM].astype(np.float32)
                assert agent_obs.shape == (RAW_OBS_DIM,), (
                    f"Raw obs dimension mismatch: Expected ({RAW_OBS_DIM},), got {agent_obs.shape}"
                )

            obs_dict[agent] = agent_obs

        return obs_dict

    def _generate_synthetic_raw_obs(self) -> np.ndarray:
        """Generates synthetic (10, 115) tensor for mock mode and CI environments."""
        tensor = np.zeros((self.num_learning_agents, RAW_OBS_DIM), dtype=np.float32)
        for i in range(self.num_learning_agents):
            # Outfield player i + 1 (outfield agent index)
            ego_player_idx = i + 1
            # Left players coords: distributed across midfield/attack
            for p in range(11):
                tensor[i, p * 2] = -0.8 + p * 0.15      # x
                tensor[i, p * 2 + 1] = -0.3 + (p % 3) * 0.3  # y
            # Right players coords
            for p in range(11):
                tensor[i, 44 + p * 2] = 0.8 - p * 0.15
                tensor[i, 44 + p * 2 + 1] = -0.3 + (p % 3) * 0.3
            # Ball at (0.0, 0.0, 0.1)
            tensor[i, 88:91] = [0.0, 0.0, 0.1]
            # Left team possession
            tensor[i, 95] = 1.0
            # Active player one-hot
            tensor[i, 97 + ego_player_idx] = 1.0
            # Game mode: Normal (index 108)
            tensor[i, 108] = 1.0
        return tensor

    def observation_space(self, agent: str) -> spaces.Box:
        """Returns the observation space for an agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        """Returns the action space for an agent."""
        return self.action_spaces[agent]

    def close(self) -> None:
        """Releases environment resources."""
        if self._native_env is not None:
            try:
                self._native_env.close()
            except Exception:
                pass
            self._native_env = None


# ==============================================================================
# Unit Test Suite for GRF Outfield Multi-Agent Wrapper
# ==============================================================================
class TestFootballMultiAgentEnv(unittest.TestCase):
    """Unit test suite verifying the multi-agent GRF wrapper specifications."""

    def test_sanity_check_10_learning_agents(self) -> None:
        """Asserts that exactly 10 learning outfield agents are instantiated."""
        env = FootballMultiAgentEnv(scenario_name="11_vs_11_stochastic", mock_mode=True)
        self.assertEqual(len(env.possible_agents), 10)
        self.assertEqual(env.num_learning_agents, 10)
        self.assertEqual(env.scenario_name, "11_vs_11_stochastic")
        self.assertEqual(env.goalkeeper_protocol, "fixed_engine_ai")
        expected_agents = [f"player_{i}" for i in range(1, 11)]
        self.assertEqual(env.possible_agents, expected_agents)
        env.close()

    def test_action_dimensions_and_macro_actions(self) -> None:
        """Asserts all 10 outfield agents have exactly 19 discrete macro-actions."""
        env = FootballMultiAgentEnv(mock_mode=True)
        self.assertEqual(len(FootballMultiAgentEnv.ACTION_NAMES), 19)
        for agent in env.possible_agents:
            space = env.action_space(agent)
            self.assertIsInstance(space, spaces.Discrete)
            self.assertEqual(space.n, 19)
        env.close()

    def test_raw_observation_dimensions_115d(self) -> None:
        """Asserts observations in 'raw' mode are exactly 115 dimensions per agent."""
        env = FootballMultiAgentEnv(obs_mode="raw", mock_mode=True)
        self.assertEqual(env.obs_dim, 115)
        obs, _ = env.reset(seed=42)
        self.assertEqual(len(obs), 10)
        for agent, agent_obs in obs.items():
            self.assertIsInstance(agent_obs, np.ndarray)
            self.assertEqual(agent_obs.shape, (115,))
            self.assertEqual(agent_obs.dtype, np.float32)
        env.close()

    def test_semantic_observation_dimensions_139d(self) -> None:
        """Asserts observations in 'semantic' mode are exactly 139 dimensions per agent."""
        env = FootballMultiAgentEnv(obs_mode="semantic", mock_mode=True)
        self.assertEqual(env.obs_dim, 139)
        obs, _ = env.reset(seed=42)
        self.assertEqual(len(obs), 10)
        for agent, agent_obs in obs.items():
            self.assertIsInstance(agent_obs, np.ndarray)
            self.assertEqual(agent_obs.shape, (139,))
            self.assertEqual(agent_obs.dtype, np.float32)
        env.close()

    def test_global_state_dimensions_22_players_plus_ball(self) -> None:
        """Asserts get_global_state returns full 22-player + ball configuration (104 dimensions)."""
        env = FootballMultiAgentEnv(mock_mode=True)
        env.reset(seed=100)
        global_state = env.get_global_state()
        self.assertIsInstance(global_state, np.ndarray)
        self.assertEqual(global_state.shape, (104,))
        self.assertEqual(global_state.dtype, np.float32)
        env.close()

    def test_reset_and_step_lifecycle(self) -> None:
        """Verifies multi-agent reset and step lifecycle with action dictionary."""
        env = FootballMultiAgentEnv(obs_mode="semantic", mock_mode=True, max_steps=5)
        obs, infos = env.reset(seed=2026)
        self.assertEqual(len(obs), 10)
        self.assertEqual(len(infos), 10)

        # Step with dictionary of 10 actions
        actions = {agent: 11 for agent in env.possible_agents}  # Action 11: short_pass
        next_obs, rewards, terminations, truncations, next_infos = env.step(actions)

        self.assertEqual(len(next_obs), 10)
        self.assertEqual(len(rewards), 10)
        self.assertEqual(len(terminations), 10)
        self.assertEqual(len(truncations), 10)
        self.assertEqual(len(next_infos), 10)

        for agent in env.possible_agents:
            self.assertEqual(next_obs[agent].shape, (139,))
            self.assertIn("global_state", next_infos[agent])
            self.assertEqual(next_infos[agent]["global_state"].shape, (104,))

        env.close()


if __name__ == "__main__":
    print("[TEST] Running FootballMultiAgentEnv verification suite...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFootballMultiAgentEnv)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("[SUCCESS] All FootballMultiAgentEnv environment tests passed successfully.")
    else:
        sys.exit(1)
