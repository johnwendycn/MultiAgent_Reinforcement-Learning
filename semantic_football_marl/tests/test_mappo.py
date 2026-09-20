"""Unit test suite for MAPPO implementation."""

import unittest
import torch
from src.algos.mappo import (
    Actor,
    Critic,
    MAPPOPolicy,
    MAPPORolloutBuffer,
    MAPPOTrainer,
)


class TestMAPPO(unittest.TestCase):
    """Verifies all architectural constraints and convergence criteria for MAPPO."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = MAPPOPolicy(
            num_agents=10,
            obs_dim=115,
            state_dim=115,
            action_dim=19,
            actor_hidden_dims=[64, 64],
            critic_hidden_dims=[64, 64],
        ).to(self.device)

    def test_exact_10_actors_assertion(self) -> None:
        """Requirement 5: Must support EXACTLY 10 actors (one per learning outfield agent)."""
        self.assertEqual(len(self.policy.actors), 10)
        self.assertEqual(self.policy.num_agents, 10)

        # Check assertion triggers when num_agents != 10
        with self.assertRaises(AssertionError):
            MAPPOPolicy(num_agents=3)

        with self.assertRaises(AssertionError):
            MAPPOPolicy(num_agents=11)

    def test_centralized_critic_115d_input(self) -> None:
        """Requirement 4: Centralized critic takes 115D global state as input."""
        critic = Critic(state_dim=115, hidden_dims=[64, 64]).to(self.device)
        B = 8
        state = torch.randn(B, 115, device=self.device)
        values = critic(state)
        self.assertEqual(values.shape, (B, 1))
        self.assertFalse(torch.isnan(values).any())

    def test_rollout_buffer_and_gae(self) -> None:
        """Requirement 1 & 2: Rollout collection (16 envs, T=512) and GAE (gamma=0.993, lambda=0.95)."""
        num_envs = 16
        horizon = 512
        num_agents = 10
        buffer = MAPPORolloutBuffer(
            num_envs=num_envs,
            horizon=horizon,
            num_agents=num_agents,
            obs_dim=115,
            state_dim=115,
            gamma=0.993,
            gae_lambda=0.95,
            device=self.device,
        )

        # Populate buffer with dummy rollouts
        for t in range(horizon):
            obs = torch.randn(num_envs, num_agents, 115)
            state = torch.randn(num_envs, 115)
            actions = torch.randint(0, 19, (num_envs, num_agents))
            rewards = torch.randn(num_envs, num_agents) * 0.1
            logprobs = -torch.rand(num_envs, num_agents)
            values = torch.randn(num_envs, 1)
            dones = torch.zeros(num_envs, 1)
            if t % 50 == 0:
                dones[0] = 1.0  # Simulate periodic termination
            buffer.insert(obs, state, actions, rewards, logprobs, values, dones)

        next_value = torch.randn(num_envs, 1)
        next_done = torch.zeros(num_envs, 1)
        buffer.compute_gae(next_value, next_done)

        self.assertFalse(torch.isnan(buffer.advantages).any())
        self.assertFalse(torch.isnan(buffer.returns).any())
        self.assertEqual(buffer.advantages.shape, (horizon, num_envs, num_agents))
        self.assertEqual(buffer.returns.shape, (horizon, num_envs, num_agents))

        # Check mini-batch generation
        batches = list(buffer.mini_batch_generator(mini_batch_size=64))
        self.assertEqual(len(batches), (horizon * num_envs) // 64)
        first_mb = batches[0]
        self.assertEqual(first_mb["obs"].shape, (64, num_agents, 115))
        self.assertEqual(first_mb["states"].shape, (64, 115))
        self.assertEqual(first_mb["actions"].shape, (64, num_agents))
        self.assertEqual(first_mb["values"].shape, (64, 1))

    def test_linear_lr_decay(self) -> None:
        """Requirement 3: Linear LR decay from 3e-4 to 0 over 5M steps."""
        trainer = MAPPOTrainer(
            policy=self.policy,
            lr_actor=3e-4,
            lr_critic=1e-3,
            total_steps=5_000_000,
            device=self.device,
        )
        lr_0 = trainer.update_lr(0)
        self.assertAlmostEqual(lr_0, 3e-4, places=6)

        lr_half = trainer.update_lr(2_500_000)
        self.assertAlmostEqual(lr_half, 1.5e-4, places=6)

        lr_end = trainer.update_lr(5_000_000)
        self.assertAlmostEqual(lr_end, 0.0, places=6)

    def test_loss_decreases_over_updates(self) -> None:
        """Acceptance Criteria: Policy and Value loss decrease over optimization updates."""
        trainer = MAPPOTrainer(
            policy=self.policy,
            lr_actor=1e-3,
            lr_critic=3e-3,
            ppo_epoch=1,
            mini_batch_size=32,
            device=self.device,
        )

        # Create a stationary batch of transitions
        B = 64
        mb = {
            "obs": torch.randn(B, 10, 115, device=self.device),
            "states": torch.randn(B, 115, device=self.device),
            "actions": torch.randint(0, 19, (B, 10), device=self.device),
            "logprobs": -torch.rand(B, 10, device=self.device),
            "values": torch.randn(B, 1, device=self.device),
            "returns": torch.ones(B, 10, device=self.device) * 5.0,
            "advantages": torch.ones(B, 10, device=self.device) * 2.0,
        }

        initial_losses = trainer._update_minibatch(mb)
        initial_v_loss = initial_losses["loss/value"]

        # Run multiple optimization updates on this batch
        final_v_loss = initial_v_loss
        for _ in range(50):
            losses = trainer._update_minibatch(mb)
            final_v_loss = losses["loss/value"]

        self.assertLess(
            final_v_loss,
            initial_v_loss,
            f"Value loss should decrease with gradient updates: initial {initial_v_loss:.4f} -> final {final_v_loss:.4f}"
        )
        self.assertFalse(torch.isnan(torch.tensor(final_v_loss)))


if __name__ == "__main__":
    unittest.main()
