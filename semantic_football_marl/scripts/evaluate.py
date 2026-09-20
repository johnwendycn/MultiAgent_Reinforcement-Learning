#!/usr/bin/env python3
"""Standalone evaluation script generating immutable raw evaluation CSVs."""

import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.seed import seed_everything
from src.envs.grf_wrapper import FootballMultiAgentEnv
from src.models.actor_critic import MAPPOPolicy
from src.eval.evaluator import PolicyEvaluator


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def evaluate_cli(cfg: DictConfig) -> None:
    """Evaluates a policy on a football scenario and records raw CSV results.

    Args:
        cfg: Hydra configuration dictionary.
    """
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

    env = FootballMultiAgentEnv(
        scenario_name=cfg.env.scenario,
        representation=cfg.env.representation,
        rewards=cfg.env.rewards,
        max_steps=cfg.env.max_episode_steps,
    )

    policy = MAPPOPolicy(
        obs_dim=env.OBS_DIM,
        state_dim=env.OBS_DIM * env.num_agents,
        action_dim=env.ACTION_DIM,
        actor_hidden_dims=list(cfg.model.actor.hidden_dims),
        critic_hidden_dims=list(cfg.model.critic.hidden_dims),
    ).to(device)

    # Load checkpoint if specified
    checkpoint_path = getattr(cfg, "checkpoint_path", None)
    checkpoint_step = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        print(f"[OK] Loaded weights from: {checkpoint_path}")
        # Extract step number from filename if available
        try:
            checkpoint_step = int(Path(checkpoint_path).stem.split("_step_")[-1])
        except Exception:
            checkpoint_step = 1000

    evaluator = PolicyEvaluator(
        env=env,
        raw_output_dir=cfg.paths.raw_eval_dir,
    )

    num_episodes = 5 if cfg.dry_run else int(cfg.training.num_eval_episodes)
    print(f"[*] Commencing evaluation over {num_episodes} episodes...")

    csv_path, summary = evaluator.evaluate(
        policy=policy,
        num_episodes=num_episodes,
        run_id=cfg.run_name,
        checkpoint_step=checkpoint_step,
        device=device,
    )

    print("=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Immutable Raw CSV: {csv_path}")
    print(f"Win Rate:          {summary['win_rate'] * 100:.1f}%")
    print(f"Draw Rate:         {summary['draw_rate'] * 100:.1f}%")
    print(f"Loss Rate:         {summary['loss_rate'] * 100:.1f}%")
    print(f"Avg Goals Scored:  {summary['avg_goals_scored']:.2f}")
    print(f"Avg Goals Conceded:{summary['avg_goals_conceded']:.2f}")
    print(f"Pass Completion:   {summary['pass_completion_rate'] * 100:.1f}%")
    print(f"Home Possession:   {summary['possession_share_home'] * 100:.1f}%")
    print("=" * 80)
    env.close()


if __name__ == "__main__":
    evaluate_cli()
