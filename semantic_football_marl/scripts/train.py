#!/usr/bin/env python3
"""Main training script for Semantic Football MARL using Hydra configuration."""

import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Ensure repository root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.seed import seed_everything
from src.utils.logger import RunLogger
from src.envs.grf_wrapper import FootballMultiAgentEnv
from src.rewards.semantic_reward import SemanticRewardShaper
from src.models.actor_critic import MAPPOPolicy
from src.algos.mappo import MAPPOTrainer
from src.eval.evaluator import PolicyEvaluator


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Executes the MARL training and evaluation workflow.

    Args:
        cfg: Composed Hydra configuration dictionary.
    """
    print("=" * 80)
    print("SEMANTIC FOOTBALL MARL - TRAINING RUN")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))

    # 1. Reproducibility
    seed_everything(cfg.seed)

    # 2. Logger initialization
    logger = RunLogger(run_id=cfg.run_name, logs_dir=cfg.paths.logs_dir)
    logger.set_config(OmegaConf.to_container(cfg, resolve=True))
    logger.set_status("RUNNING")

    # 3. Environment & Reward Shaper
    env = FootballMultiAgentEnv(
        scenario_name=cfg.env.scenario,
        representation=cfg.env.representation,
        rewards=cfg.env.rewards,
        max_steps=cfg.env.max_episode_steps,
    )
    obs_dim = env.OBS_DIM
    action_dim = env.ACTION_DIM
    num_agents = env.num_agents
    state_dim = obs_dim * num_agents

    reward_shaper = SemanticRewardShaper(
        weights=OmegaConf.to_container(cfg.rewards.weights, resolve=True)
    )

    # 4. Neural Policy & Trainer
    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    policy = MAPPOPolicy(
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        actor_hidden_dims=list(cfg.model.actor.hidden_dims),
        critic_hidden_dims=list(cfg.model.critic.hidden_dims),
        use_layer_norm=cfg.model.actor.use_layer_norm,
    )

    trainer = MAPPOTrainer(
        policy=policy,
        lr_actor=float(cfg.algo.lr_actor),
        lr_critic=float(cfg.algo.lr_critic),
        clip_param=float(cfg.algo.clip_param),
        ppo_epoch=int(cfg.algo.ppo_epoch),
        num_mini_batch=int(cfg.algo.num_mini_batch),
        value_loss_coef=float(cfg.algo.value_loss_coef),
        entropy_coef=float(cfg.algo.entropy_coef),
        max_grad_norm=float(cfg.algo.max_grad_norm),
        device=device,
    )

    # 5. Evaluator
    evaluator = PolicyEvaluator(
        env=env,
        raw_output_dir=cfg.paths.raw_eval_dir,
    )

    print(f"\n[OK] Environment: {cfg.env.scenario} ({num_agents} controlled agents)")
    print(f"[OK] Policy & Trainer initialized on device: {device}")
    print(f"[OK] Raw evaluation CSVs targeted to: {cfg.paths.raw_eval_dir}")

    total_steps = 100 if cfg.dry_run else int(cfg.training.total_timesteps)
    eval_interval = 50 if cfg.dry_run else int(cfg.training.eval_interval_steps)
    ckpt_interval = 50 if cfg.dry_run else int(cfg.training.checkpoint_interval_steps)

    print(f"[*] Beginning training loop for {total_steps} steps...\n")

    # Initial zero-shot evaluation
    raw_csv, initial_metrics = evaluator.evaluate(
        policy=policy,
        num_episodes=min(5, int(cfg.training.num_eval_episodes)),
        run_id=cfg.run_name,
        checkpoint_step=0,
        device=device,
    )
    logger.register_raw_csv(raw_csv, evaluation_step=0, num_episodes=5)
    logger.log_metrics(0, initial_metrics)
    print(f"  Initial Eval (Step 0): Win Rate = {initial_metrics['win_rate']:.2f} | Raw CSV: {raw_csv}")

    # Simulated/actual training rollout
    step = 0
    obs_dict, _ = env.reset()
    reward_shaper.reset()

    while step < total_steps:
        actions_dict = {}
        with torch.no_grad():
            for ag in env.agents:
                obs_t = torch.tensor(obs_dict[ag], dtype=torch.float32, device=device).unsqueeze(0)
                act, _ = policy.act(obs_t)
                actions_dict[ag] = int(act.item())

        next_obs_dict, raw_rewards, terms, truncs, infos = env.step(actions_dict)
        step += 1

        # Checkpoints & evaluation intervals
        if step % eval_interval == 0:
            csv_path, metrics = evaluator.evaluate(
                policy=policy,
                num_episodes=int(cfg.training.num_eval_episodes) if not cfg.dry_run else 3,
                run_id=cfg.run_name,
                checkpoint_step=step,
                device=device,
            )
            logger.register_raw_csv(csv_path, evaluation_step=step, num_episodes=3)
            logger.log_metrics(step, metrics)
            print(f"  Step {step}/{total_steps}: Win Rate={metrics['win_rate']:.2f}, Avg Rew={metrics['goal_difference']:.2f}")

        if step % ckpt_interval == 0 and cfg.training.save_checkpoints:
            ckpt_path = Path(cfg.paths.checkpoints_dir) / f"{cfg.run_name}_step_{step}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(ckpt_path))
            logger.register_checkpoint(str(ckpt_path), step=step)
            print(f"  [Checkpoint Saved]: {ckpt_path}")

        done = any(terms.values()) or any(truncs.values())
        if done:
            obs_dict, _ = env.reset()
            reward_shaper.reset()
        else:
            obs_dict = next_obs_dict

    # Final checkpoint
    final_ckpt = Path(cfg.paths.checkpoints_dir) / f"{cfg.run_name}_final.pt"
    final_ckpt.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(final_ckpt))
    logger.register_checkpoint(str(final_ckpt), step=step)

    logger.set_status("COMPLETED")
    env.close()
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"Final model: {final_ckpt}")
    print(f"Run Log:     {logger.log_file_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
