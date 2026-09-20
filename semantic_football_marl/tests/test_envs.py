"""Unit tests for football multi-agent simulation environment."""

import numpy as np
from src.envs.scenario_manager import ScenarioManager
from src.envs.grf_wrapper import FootballMultiAgentEnv


def test_scenario_manager() -> None:
    """Verifies that scenario metadata is correctly retrieved."""
    scenarios = ScenarioManager.list_scenarios()
    assert "11_vs_11_stochastic" in scenarios
    assert "11_vs_11_kaggle" in scenarios

    info_11v11 = ScenarioManager.get_info("11_vs_11_stochastic")
    assert info_11v11.num_controlled == 10
    assert info_11v11.num_left_players == 11
    assert info_11v11.has_keeper is True

    try:
        ScenarioManager.get_info("non_existent_scenario")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_football_env_11v11_outfield_reset_and_step() -> None:
    """Verifies reset and stepping logic of FootballMultiAgentEnv on 11_vs_11_stochastic."""
    env = FootballMultiAgentEnv(
        scenario_name="11_vs_11_stochastic",
        obs_mode="semantic",
        mock_mode=True,
    )
    # Sanity check: exactly 10 learning outfield agents
    assert env.num_learning_agents == 10
    assert len(env.possible_agents) == 10
    assert env.obs_dim == 139
    assert env.action_dim == 19

    obs_dict, infos = env.reset(seed=42)
    assert len(obs_dict) == 10
    assert len(infos) == 10

    for ag in env.possible_agents:
        assert ag in obs_dict
        assert obs_dict[ag].shape == (139,)
        assert ag in infos
        assert "global_state" in infos[ag]
        assert infos[ag]["global_state"].shape == (104,)

    # Global state check
    global_state = env.get_global_state()
    assert global_state.shape == (104,)

    # Step with 10 discrete macro-actions
    actions = {ag: int(np.random.randint(0, 19)) for ag in env.possible_agents}
    next_obs, rewards, terms, truncs, infos = env.step(actions)

    assert len(next_obs) == 10
    assert len(rewards) == 10
    assert len(terms) == 10
    assert len(truncs) == 10

    for ag in env.possible_agents:
        assert next_obs[ag].shape == (139,)
        assert isinstance(rewards[ag], float)
        assert isinstance(terms[ag], bool)
        assert isinstance(truncs[ag], bool)

    env.close()


def test_football_env_raw_mode_115d() -> None:
    """Verifies that raw observation mode returns exactly 115 dimensions."""
    env = FootballMultiAgentEnv(
        scenario_name="11_vs_11_stochastic",
        obs_mode="raw",
        mock_mode=True,
    )
    assert env.obs_dim == 115
    obs_dict, _ = env.reset(seed=123)
    assert len(obs_dict) == 10
    for ag in env.possible_agents:
        assert obs_dict[ag].shape == (115,)
    env.close()
