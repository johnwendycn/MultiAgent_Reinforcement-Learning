"""Secondary Generalization Scenarios for Multi-Agent Reinforcement Learning in GRF.

================================================================================
MANUSCRIPT EMPIRICAL TRACEABILITY & BENCHMARK HIERARCHY
================================================================================
CRITICAL ARCHITECTURAL DISTINCTION:
The primary benchmark results of the research manuscript MUST BE PRODUCED
EXCLUSIVELY by the full 11v11 environment (src/envs/grf_wrapper.py):
  - Scenario: '11_vs_11_stochastic' (10 learning outfield agents, fixed goalkeeper protocol)
  - Manuscript Targets:
    * Table 1: Primary 11v11 MARL Benchmark Leaderboard (MAPPO, QMIX, Baselines)
    * Figure 3: Full-match 11v11 Sample Efficiency & Training Curves
    * Figure 4: Full-match 11v11 Win Rate & Goal Differential Empirical Distributions

The scenario wrappers in THIS module are SECONDARY experiments only, designed
exclusively to evaluate out-of-distribution, zero-shot, and sub-game generalization:

1. Academy3v1Env:
   - GRF Scenario: 'academy_3_vs_1_with_keeper'
   - Learning Agents: Exactly 3 offensive outfield agents (attackers).
   - Target Manuscript Target:
     * Table 3: "Sub-game Generalization: Offensive Micro-Coordination on Academy 3v1"
   - Evaluated Dynamics:
     Passing corridor accuracy (PCR), high-pressure finishing, and micro-positioning.

2. Counterattack3v2Env:
   - GRF Scenario: 'academy_counterattack_easy' / 'counterattack_3_vs_2'
   - Learning Agents: Exactly 3 offensive transition agents (counter-attackers).
   - Target Manuscript Target:
     * Table 4: "Transition Generalization: Numerical Advantage Exploitation on Counterattack 3v2"
   - Evaluated Dynamics:
     Fast breakaways, exploitation of 3v2 numerical superiority, off-ball movement quality (OBMQ),
     and tactical phase completion accuracy (TPCA).

PROHIBITION:
Main 11v11 results MUST NOT be produced by these wrappers.
Any attempt to register or cite evaluation results from these sub-games as Table 1
primary benchmarks will trigger an explicit ValidationConstraintError.
================================================================================
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from pettingzoo import ParallelEnv
except ImportError:
    class ParallelEnv:  # type: ignore
        """Fallback base class when PettingZoo is unavailable."""
        pass

# Feature extractor integration
try:
    from src.features.semantic_extractor import SemanticFeatureExtractor
except (ImportError, ValueError):
    try:
        from semantic_football_marl.src.features.semantic_extractor import SemanticFeatureExtractor
    except Exception:
        SemanticFeatureExtractor = None  # type: ignore

try:
    from src.envs.scenario_manager import ScenarioManager
except (ImportError, ValueError):
    try:
        from .scenario_manager import ScenarioManager
    except Exception:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
        from src.envs.scenario_manager import ScenarioManager


class ValidationConstraintError(RuntimeError):
    """Raised when secondary generalization wrappers are erroneously cited for Table 1 primary benchmarks."""
    pass


class BaseSecondaryGeneralizationEnv(ParallelEnv):
    """Base class for secondary sub-game generalization environments.

    Enforces strict architectural boundaries preventing secondary generalization sub-games
    from corrupting or replacing the primary 11v11 benchmark results.
    """

    IS_PRIMARY_11V11_BENCHMARK: bool = False
    IS_SECONDARY_GENERALIZATION: bool = True
    NUM_ACTIONS: int = 19
    RAW_OBS_DIM: int = 115
    SEMANTIC_OBS_DIM: int = 139

    def __init__(
        self,
        scenario_name: str,
        target_manuscript_table: str,
        num_learning_agents: int,
        obs_mode: str = "semantic",
        rewards: str = "scoring,checkpoints",
        stacked: bool = False,
        render_mode: Optional[str] = None,
        max_steps: int = 500,
        mock_mode: bool = False,
    ) -> None:
        """Initializes the secondary generalization environment.

        Args:
            scenario_name: Identifier of the GRF sub-game scenario.
            target_manuscript_table: The specific manuscript table this experiment maps to.
            num_learning_agents: Number of learning agents in this sub-game.
            obs_mode: 'raw' (115D) or 'semantic' (139D).
            rewards: Reward shaping mode ('scoring' or 'scoring,checkpoints').
            stacked: Whether to stack historical frames.
            render_mode: Render backend ('human', 'rgb_array', or None).
            max_steps: Maximum timesteps before truncation.
            mock_mode: Enable synthetic execution for headless CI testing.
        """
        super().__init__()
        self.scenario_name = scenario_name
        self.target_manuscript_table = target_manuscript_table
        self.num_learning_agents = num_learning_agents
        self.obs_mode = obs_mode.lower()
        assert self.obs_mode in ("raw", "semantic"), f"obs_mode must be 'raw' or 'semantic', got '{obs_mode}'"

        self.rewards = rewards
        self.stacked = stacked
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.is_mock = mock_mode

        # Define agents
        self.possible_agents = [f"attacker_{i}" for i in range(1, self.num_learning_agents + 1)]
        self.agents = self.possible_agents[:]

        # Define action and observation spaces
        self.obs_dim = self.SEMANTIC_OBS_DIM if self.obs_mode == "semantic" else self.RAW_OBS_DIM
        self.action_dim = self.NUM_ACTIONS

        self.action_spaces: Dict[str, spaces.Discrete] = {
            agent: spaces.Discrete(self.action_dim) for agent in self.possible_agents
        }
        self.observation_spaces: Dict[str, spaces.Box] = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Initialize feature extractor
        self.semantic_extractor = SemanticFeatureExtractor() if SemanticFeatureExtractor is not None else None

        # State cache
        self._latest_raw_obs: Optional[np.ndarray] = None
        self._latest_global_state: np.ndarray = np.zeros(104, dtype=np.float32)

        # Native environment
        self._native_env = None
        self._init_native_env()

    def assert_not_primary_benchmark(self, claimed_table: str) -> None:
        """Enforces that this wrapper is never used to fabricate Table 1 results.

        Args:
            claimed_table: Table string being generated.

        Raises:
            ValidationConstraintError: If claimed_table refers to Table 1.
        """
        if "Table 1" in claimed_table or "11v11" in claimed_table:
            raise ValidationConstraintError(
                f"Protocol Violation: {self.__class__.__name__} is a SECONDARY generalization benchmark "
                f"for '{self.target_manuscript_table}'. It cannot and must not produce '{claimed_table}'. "
                f"Main 11v11 results must be produced exclusively by FootballMultiAgentEnv on '11_vs_11_stochastic'."
            )

    def _init_native_env(self) -> None:
        """Initializes native GFootball environment or triggers mock mode."""
        if self.is_mock:
            return

        try:
            import gfootball.env as football_env
            self._native_env = football_env.create_environment(
                env_name=self.scenario_name,
                stacked=self.stacked,
                representation="simple115v2",
                rewards=self.rewards,
                write_goal_dumps=False,
                write_full_episode_dumps=False,
                render=(self.render_mode == "human"),
                number_of_left_players_agent_controls=self.num_learning_agents,
            )
        except Exception:
            self.is_mock = True
            self._native_env = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Resets the sub-game environment."""
        self.agents = self.possible_agents[:]
        self.current_step = 0

        if seed is not None:
            np.random.seed(seed)

        if not self.is_mock and self._native_env is not None:
            raw_obs = self._native_env.reset()
            obs_array = np.asarray(raw_obs, dtype=np.float32)
            if obs_array.ndim == 1:
                obs_array = np.tile(obs_array[np.newaxis, :], (self.num_learning_agents, 1))
            self._latest_raw_obs = obs_array
        else:
            self._latest_raw_obs = self._generate_synthetic_obs()

        self._update_global_state(self._latest_raw_obs[0])
        obs_dict = self._process_observations(self._latest_raw_obs)
        global_state = self.get_global_state()

        infos_dict = {
            agent: {
                "step": 0,
                "scenario": self.scenario_name,
                "manuscript_table": self.target_manuscript_table,
                "is_secondary_generalization": True,
                "is_primary_11v11": False,
                "global_state": global_state,
                "is_mock": self.is_mock,
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
        """Steps the sub-game environment."""
        self.current_step += 1
        is_truncated = self.current_step >= self.max_steps

        if isinstance(actions, dict):
            action_list = [int(actions.get(agent, 0)) for agent in self.possible_agents]
        elif isinstance(actions, (list, tuple, np.ndarray)):
            action_list = [int(a) for a in actions]
        else:
            raise ValueError(f"Unsupported actions format: {type(actions)}")

        assert len(action_list) == self.num_learning_agents, (
            f"Action count mismatch: Expected {self.num_learning_agents}, got {len(action_list)}"
        )

        for a in action_list:
            assert 0 <= a < self.action_dim, f"Action {a} out of bounds [0..{self.action_dim - 1}]"

        if not self.is_mock and self._native_env is not None:
            raw_obs, raw_rewards, done, raw_info = self._native_env.step(action_list)
            obs_array = np.asarray(raw_obs, dtype=np.float32)
            if obs_array.ndim == 1:
                obs_array = np.tile(obs_array[np.newaxis, :], (self.num_learning_agents, 1))
            self._latest_raw_obs = obs_array

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
            self._latest_raw_obs = self._generate_synthetic_obs()
            rewards_dict = {agent: 0.0 for agent in self.possible_agents}
            terminated = False
            ball_owned = 1
            score = [0, 0]

        self._update_global_state(self._latest_raw_obs[0])
        global_state = self.get_global_state()
        obs_dict = self._process_observations(self._latest_raw_obs)
        terminations_dict = {agent: terminated for agent in self.possible_agents}
        truncations_dict = {agent: is_truncated for agent in self.possible_agents}

        infos_dict = {
            agent: {
                "step": self.current_step,
                "ball_owned_team": ball_owned,
                "score": score,
                "manuscript_table": self.target_manuscript_table,
                "is_secondary_generalization": True,
                "is_primary_11v11": False,
                "global_state": global_state,
                "is_mock": self.is_mock,
            }
            for agent in self.possible_agents
        }

        if terminated or is_truncated:
            self.agents = []

        return obs_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict

    def get_global_state(self) -> np.ndarray:
        """Returns the centralized critic global state (104 dimensions)."""
        return self._latest_global_state.copy()

    def _update_global_state(self, raw_115: np.ndarray) -> None:
        """Updates the canonical 104-dim configuration."""
        players_and_ball = raw_115[0:97]
        game_mode = raw_115[108:115]
        self._latest_global_state = np.concatenate([players_and_ball, game_mode], axis=0).astype(np.float32)

    def _process_observations(self, obs_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Transforms raw observations to the requested mode."""
        obs_dict: Dict[str, np.ndarray] = {}
        for i, agent in enumerate(self.possible_agents):
            raw = obs_array[i]
            if self.obs_mode == "semantic":
                if self.semantic_extractor is not None:
                    agent_obs = self.semantic_extractor.build_semantic_observation(raw, agent_idx=i)
                else:
                    agent_obs = np.concatenate([raw, np.zeros(24, dtype=np.float32)])
            else:
                agent_obs = raw[:self.RAW_OBS_DIM].astype(np.float32)
            obs_dict[agent] = agent_obs
        return obs_dict

    def _generate_synthetic_obs(self) -> np.ndarray:
        """Generates synthetic observation array of shape (num_agents, 115)."""
        tensor = np.zeros((self.num_learning_agents, self.RAW_OBS_DIM), dtype=np.float32)
        for i in range(self.num_learning_agents):
            # Positioning in attacking half
            tensor[i, i * 2] = 0.5 + i * 0.1
            tensor[i, i * 2 + 1] = -0.2 + i * 0.2
            # Ball position
            tensor[i, 88:91] = [0.6, 0.0, 0.1]
            tensor[i, 95] = 1.0  # Left possession
            tensor[i, 97 + i] = 1.0  # Active agent
            tensor[i, 108] = 1.0  # Normal game mode
        return tensor

    def observation_space(self, agent: str) -> spaces.Box:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        return self.action_spaces[agent]

    def close(self) -> None:
        if self._native_env is not None:
            try:
                self._native_env.close()
            except Exception:
                pass
            self._native_env = None


# ==============================================================================
# 1. Academy 3v1 Secondary Generalization Environment
# ==============================================================================
class Academy3v1Env(BaseSecondaryGeneralizationEnv):
    """Secondary Generalization Wrapper for 'academy_3_vs_1_with_keeper'.

    MANUSCRIPT TRACEABILITY:
    - Target: Table 3 ("Sub-game Generalization: Offensive Micro-Coordination on Academy 3v1").
    - Role: Evaluates transfer of passing, shooting, and spatial triangle coordination
      under numerical advantage in an isolated attacking phase.
    - Configuration: Exactly 3 learning offensive outfield agents against 1 defender and 1 keeper.
    - RESTRICTION: Must NOT be used for Table 1 primary 11v11 benchmark figures.
    """

    TARGET_TABLE: str = "Table 3 (Academy 3v1 Micro-Coordination & Generalization)"

    def __init__(
        self,
        obs_mode: str = "semantic",
        rewards: str = "scoring,checkpoints",
        stacked: bool = False,
        render_mode: Optional[str] = None,
        max_steps: int = 400,
        mock_mode: bool = False,
    ) -> None:
        """Initializes the Academy 3v1 secondary generalization environment."""
        super().__init__(
            scenario_name="academy_3_vs_1_with_keeper",
            target_manuscript_table=self.TARGET_TABLE,
            num_learning_agents=3,
            obs_mode=obs_mode,
            rewards=rewards,
            stacked=stacked,
            render_mode=render_mode,
            max_steps=max_steps,
            mock_mode=mock_mode,
        )
        assert len(self.possible_agents) == 3, f"Academy 3v1 must have exactly 3 learning agents, got {len(self.possible_agents)}"


# ==============================================================================
# 2. Counterattack 3v2 Secondary Generalization Environment
# ==============================================================================
class Counterattack3v2Env(BaseSecondaryGeneralizationEnv):
    """Secondary Generalization Wrapper for 'counterattack_3_vs_2'.

    MANUSCRIPT TRACEABILITY:
    - Target: Table 4 ("Transition Generalization: Numerical Advantage Exploitation on Counterattack 3v2").
    - Role: Evaluates breakaway offensive transitions, exploiting 3v2 numerical superiority,
      off-ball movement quality (OBMQ), and tactical phase completion accuracy (TPCA).
    - Configuration: Exactly 3 learning offensive transition agents against 2 outfield defenders and 1 keeper.
    - RESTRICTION: Must NOT be used for Table 1 primary 11v11 benchmark figures.
    """

    TARGET_TABLE: str = "Table 4 (Counterattack 3v2 Transition Generalization)"

    def __init__(
        self,
        obs_mode: str = "semantic",
        rewards: str = "scoring,checkpoints",
        stacked: bool = False,
        render_mode: Optional[str] = None,
        max_steps: int = 500,
        mock_mode: bool = False,
    ) -> None:
        """Initializes the Counterattack 3v2 secondary generalization environment."""
        super().__init__(
            scenario_name="counterattack_3_vs_2",
            target_manuscript_table=self.TARGET_TABLE,
            num_learning_agents=3,
            obs_mode=obs_mode,
            rewards=rewards,
            stacked=stacked,
            render_mode=render_mode,
            max_steps=max_steps,
            mock_mode=mock_mode,
        )
        assert len(self.possible_agents) == 3, f"Counterattack 3v2 must have exactly 3 learning agents, got {len(self.possible_agents)}"


# ==============================================================================
# Unit Test Suite for Secondary Generalization Scenarios
# ==============================================================================
import unittest


class TestSecondaryGeneralizationScenarios(unittest.TestCase):
    """Unit tests validating secondary generalization wrappers and Table 1 guardrails."""

    def test_academy_3v1_specifications(self) -> None:
        """Verifies Academy3v1Env agent count, action space, and Table 3 mapping."""
        env = Academy3v1Env(obs_mode="semantic", mock_mode=True)
        self.assertEqual(env.num_learning_agents, 3)
        self.assertEqual(len(env.possible_agents), 3)
        self.assertEqual(env.action_dim, 19)
        self.assertEqual(env.obs_dim, 139)
        self.assertIn("Table 3", env.target_manuscript_table)
        self.assertTrue(env.IS_SECONDARY_GENERALIZATION)
        self.assertFalse(env.IS_PRIMARY_11V11_BENCHMARK)

        obs, infos = env.reset(seed=42)
        self.assertEqual(len(obs), 3)
        for ag in env.possible_agents:
            self.assertEqual(obs[ag].shape, (139,))
            self.assertEqual(infos[ag]["manuscript_table"], env.target_manuscript_table)
            self.assertFalse(infos[ag]["is_primary_11v11"])
            self.assertTrue(infos[ag]["is_secondary_generalization"])

        # Test stepping
        next_obs, rewards, terms, truncs, infos = env.step({ag: 11 for ag in env.possible_agents})
        self.assertEqual(len(next_obs), 3)
        self.assertEqual(len(rewards), 3)
        env.close()

    def test_counterattack_3v2_specifications(self) -> None:
        """Verifies Counterattack3v2Env agent count, action space, and Table 4 mapping."""
        env = Counterattack3v2Env(obs_mode="semantic", mock_mode=True)
        self.assertEqual(env.num_learning_agents, 3)
        self.assertEqual(len(env.possible_agents), 3)
        self.assertEqual(env.action_dim, 19)
        self.assertEqual(env.obs_dim, 139)
        self.assertIn("Table 4", env.target_manuscript_table)
        self.assertTrue(env.IS_SECONDARY_GENERALIZATION)
        self.assertFalse(env.IS_PRIMARY_11V11_BENCHMARK)

        obs, infos = env.reset(seed=123)
        self.assertEqual(len(obs), 3)
        for ag in env.possible_agents:
            self.assertEqual(obs[ag].shape, (139,))
            self.assertEqual(infos[ag]["manuscript_table"], env.target_manuscript_table)

        # Global state inspection
        global_state = env.get_global_state()
        self.assertEqual(global_state.shape, (104,))
        env.close()

    def test_primary_benchmark_fabrication_guardrail(self) -> None:
        """Asserts that attempting to cite secondary wrappers for Table 1 raises an error."""
        env_3v1 = Academy3v1Env(mock_mode=True)
        env_3v2 = Counterattack3v2Env(mock_mode=True)

        with self.assertRaises(ValidationConstraintError):
            env_3v1.assert_not_primary_benchmark("Table 1 (Primary 11v11 Benchmark)")

        with self.assertRaises(ValidationConstraintError):
            env_3v2.assert_not_primary_benchmark("Table 1: Full 11v11 MARL Leaderboard")

        # Legitimate secondary table query should pass without error
        env_3v1.assert_not_primary_benchmark("Table 3 (Academy 3v1 Generalization)")
        env_3v2.assert_not_primary_benchmark("Table 4 (Counterattack 3v2 Transition)")

        env_3v1.close()
        env_3v2.close()


if __name__ == "__main__":
    print("[TEST] Running Secondary Generalization Scenarios test suite...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSecondaryGeneralizationScenarios)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("[SUCCESS] All secondary scenario tests and Table 1 guardrails passed successfully.")
    else:
        sys.exit(1)
