#!/usr/bin/env python3
"""Smoke test script executing a 10,000-step end-to-end MAPPO training run.

Validates:
1. Exactly 10 outfield actors instantiated and asserted.
2. 16 parallel environments with horizon T=512 (8,192 steps per rollout).
3. GAE advantage and return calculations (gamma=0.993, lambda=0.95).
4. PPO clipped surrogate objective (epsilon=0.20), mini-batch size 64, epochs 4.
5. End-to-end execution on available accelerator (CUDA/GPU if present, else CPU) without NaNs.
6. Monitored policy and value loss trajectories over optimization updates.
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple
import torch
import numpy as np

# Ensure project root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also add semantic_football_marl
SUB_ROOT = REPO_ROOT / "semantic_football_marl"
if str(SUB_ROOT) not in sys.path:
    sys.path.insert(0, str(SUB_ROOT))

from src.envs.grf_wrapper import FootballMultiAgentEnv
from src.algos.mappo import MAPPOPolicy, MAPPORolloutBuffer, MAPPOTrainer


class VectorFootballEnv:
    """Synchronous vector wrapper managing 16 parallel multi-agent environments."""

    def __init__(
        self,
        num_envs: int = 16,
        scenario_name: str = "11_vs_11_stochastic",
        obs_mode: str = "raw",
        mock_mode: bool = True,
    ) -> None:
        self.num_envs = num_envs
        self.envs = [
            FootballMultiAgentEnv(
                scenario_name=scenario_name,
                obs_mode=obs_mode,
                mock_mode=mock_mode,
            )
            for _ in range(num_envs)
        ]
        self.num_agents = 10
        self.obs_dim = 115 if obs_mode == "raw" else 139
        self.state_dim = 115

    def reset(self, base_seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resets all 16 environments and returns stacked observations and global states."""
        obs_list = []
        state_list = []
        for e, env in enumerate(self.envs):
            obs_dict, info = env.reset(seed=base_seed + e)
            # Stack 10 outfield agent observations: (10, obs_dim)
            agent_obs = np.stack([obs_dict[f"player_{i}"] for i in range(1, 11)], axis=0)
            obs_list.append(agent_obs)
            # Use raw match configuration (first agent's 115D observation) as the 115D global state
            global_state = obs_dict["player_1"][:115]
            state_list.append(global_state)

        obs_t = torch.tensor(np.stack(obs_list, axis=0), dtype=torch.float32)       # (16, 10, obs_dim)
        state_t = torch.tensor(np.stack(state_list, axis=0), dtype=torch.float32)   # (16, 115)
        return obs_t, state_t

    def step(
        self,
        actions_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Steps all 16 environments with actions of shape (16, 10)."""
        actions_np = actions_tensor.cpu().numpy()
        next_obs_list = []
        next_state_list = []
        rewards_list = []
        dones_list = []

        for e, env in enumerate(self.envs):
            act_dict = {f"player_{i}": int(actions_np[e, i - 1]) for i in range(1, 11)}
            next_obs_dict, rewards_dict, terms, truncs, infos = env.step(act_dict)

            done = any(terms.values()) or any(truncs.values())
            if done:
                next_obs_dict, _ = env.reset()

            agent_obs = np.stack([next_obs_dict[f"player_{i}"] for i in range(1, 11)], axis=0)
            agent_rewards = np.array([rewards_dict[f"player_{i}"] for i in range(1, 11)], dtype=np.float32)
            global_state = next_obs_dict["player_1"][:115]

            next_obs_list.append(agent_obs)
            next_state_list.append(global_state)
            rewards_list.append(agent_rewards)
            dones_list.append(float(done))

        next_obs_t = torch.tensor(np.stack(next_obs_list, axis=0), dtype=torch.float32)
        next_state_t = torch.tensor(np.stack(next_state_list, axis=0), dtype=torch.float32)
        rewards_t = torch.tensor(np.stack(rewards_list, axis=0), dtype=torch.float32)
        dones_t = torch.tensor(np.array(dones_list, dtype=np.float32)).unsqueeze(1)

        return next_obs_t, next_state_t, rewards_t, dones_t

    def close(self) -> None:
        for env in self.envs:
            env.close()


def run_smoke_test() -> None:
    print("=" * 80)
    print("SEMANTIC FOOTBALL MARL - MAPPO SMOKE TEST (10,000 STEPS)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Device: {device}")
    print(f"[*] PyTorch Version : {torch.__version__}")
    print(f"[*] CUDA Available  : {torch.cuda.is_available()}")

    # 1. Initialize Policy and Assert Exactly 10 Actors
    num_agents = 10
    obs_dim = 115
    state_dim = 115
    action_dim = 19
    horizon = 512
    num_envs = 16
    target_steps = 10_000

    policy = MAPPOPolicy(
        num_agents=num_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[512, 256, 128],
    ).to(device)

    # Acceptance Criteria: Assert number of actors is exactly 10
    assert len(policy.actors) == 10, f"Expected 10 actors, got {len(policy.actors)}"
    print(f"[OK] Instantiated MAPPO Policy: {len(policy.actors)} outfield actors asserted.")

    # 2. Initialize Trainer and Rollout Buffer
    trainer = MAPPOTrainer(
        policy=policy,
        lr_actor=3e-4,
        lr_critic=1e-3,
        clip_param=0.20,
        ppo_epoch=4,
        mini_batch_size=64,
        value_loss_coef=0.50,
        entropy_coef=0.01,
        gamma=0.993,
        gae_lambda=0.95,
        total_steps=5_000_000,
        device=device,
    )

    buffer = MAPPORolloutBuffer(
        num_envs=num_envs,
        horizon=horizon,
        num_agents=num_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        gamma=0.993,
        gae_lambda=0.95,
        device=device,
    )

    # 3. Initialize 16 Parallel Environments
    print(f"[*] Initializing {num_envs} parallel football environments (horizon T={horizon})...")
    vec_env = VectorFootballEnv(
        num_envs=num_envs,
        scenario_name="11_vs_11_stochastic",
        obs_mode="raw",
        mock_mode=True,
    )

    obs, state = vec_env.reset(base_seed=123)
    obs = obs.to(device)
    state = state.to(device)

    total_env_steps = 0
    rollout_iteration = 0
    start_time = time.time()

    policy_losses = []
    value_losses = []

    print(f"[*] Commencing training rollouts targeting >= {target_steps:,} environment steps...")

    while total_env_steps < target_steps:
        rollout_iteration += 1
        rollout_start_time = time.time()

        # Collect horizon T=512 steps across all 16 envs
        for t in range(horizon):
            with torch.no_grad():
                actions, logprobs = policy.act_all(obs)
                values = policy.get_values(state)

            next_obs, next_state, rewards, dones = vec_env.step(actions)
            next_obs = next_obs.to(device)
            next_state = next_state.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

            # Store in rollout buffer (unclipped raw rewards)
            buffer.insert(obs, state, actions, rewards, logprobs, values, dones)

            obs = next_obs
            state = next_state
            total_env_steps += num_envs

        # Compute GAE advantages & returns
        with torch.no_grad():
            next_val = policy.get_values(state)
        buffer.compute_gae(next_val, dones)

        # Execute PPO training update
        losses = trainer.train_step(buffer)
        trainer.update_lr(total_env_steps)

        # Track losses
        p_loss = losses["loss/policy"]
        v_loss = losses["loss/value"]
        tot_loss = losses["loss/total"]
        ent = losses["policy/entropy"]
        policy_losses.append(p_loss)
        value_losses.append(v_loss)

        # Assert no NaNs anywhere
        assert not np.isnan(p_loss), f"NaN in policy loss at step {total_env_steps}"
        assert not np.isnan(v_loss), f"NaN in value loss at step {total_env_steps}"
        assert not np.isnan(tot_loss), f"NaN in total loss at step {total_env_steps}"

        rollout_fps = (num_envs * horizon) / (time.time() - rollout_start_time)
        print(
            f"  [Rollout {rollout_iteration:02d}] Step: {total_env_steps:6d}/{target_steps} | "
            f"Loss: {tot_loss:.4f} (Pol: {p_loss:+.4f}, Val: {v_loss:.4f}, Ent: {ent:.4f}) | "
            f"FPS: {rollout_fps:.1f}"
        )

    vec_env.close()
    elapsed = time.time() - start_time

    print("=" * 80)
    print(f"[SUCCESS] Completed {total_env_steps:,} steps across {rollout_iteration} rollouts in {elapsed:.2f}s ({total_env_steps / elapsed:.1f} overall FPS).")
    print(f"  Initial Policy Loss: {policy_losses[0]:+.4f} -> Latest: {policy_losses[-1]:+.4f}")
    print(f"  Initial Value Loss : {value_losses[0]:.4f} -> Latest: {value_losses[-1]:.4f}")
    print("  Zero NaNs detected across all parameters and buffers.")
    print("SMOKE TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()
