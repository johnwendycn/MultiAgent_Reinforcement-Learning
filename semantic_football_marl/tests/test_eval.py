"""Unit tests for evaluation pipeline and raw CSV generation."""

import tempfile
import csv
from pathlib import Path
from src.envs.grf_wrapper import FootballMultiAgentEnv
from src.models.actor_critic import MAPPOPolicy
from src.eval.evaluator import PolicyEvaluator


def test_evaluator_produces_valid_raw_csv() -> None:
    """Verifies that PolicyEvaluator outputs immutable CSV conforming to schema."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        env = FootballMultiAgentEnv(
            scenario_name="academy_3_vs_1_with_keeper",
            mock_mode=True,
        )
        policy = MAPPOPolicy(
            obs_dim=env.OBS_DIM,
            state_dim=env.OBS_DIM * env.num_agents,
            action_dim=env.ACTION_DIM,
        )
        evaluator = PolicyEvaluator(
            env=env,
            raw_output_dir=tmp_dir,
        )

        csv_path, summary = evaluator.evaluate(
            policy=policy,
            num_episodes=3,
            run_id="test_run",
            checkpoint_step=100,
        )

        assert Path(csv_path).exists()
        assert "win_rate" in summary
        assert "avg_goals_scored" in summary

        # Check CSV header and row count
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        assert header == PolicyEvaluator.CSV_HEADER
        assert len(rows) == 3
        env.close()
