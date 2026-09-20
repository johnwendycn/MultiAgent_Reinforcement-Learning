"""Unit tests for MARL algorithm trainers."""

import tempfile
import torch
from src.models.actor_critic import MAPPOPolicy
from src.models.q_mixer import AgentQNetwork, QMixer
from src.algos.mappo import MAPPOTrainer
from src.algos.qmix import QMIXTrainer


def test_mappo_trainer_step_and_checkpoint() -> None:
    """Verifies MAPPO optimization step and checkpoint serialization."""
    policy = MAPPOPolicy(obs_dim=115, state_dim=115 * 3, action_dim=19)
    trainer = MAPPOTrainer(
        policy=policy,
        ppo_epoch=2,
        num_mini_batch=2,
        device="cpu",
    )

    batch_size = 16
    rollouts = {
        "obs": torch.randn(batch_size, 115),
        "actions": torch.randint(0, 19, (batch_size,)),
        "states": torch.randn(batch_size, 115 * 3),
        "log_probs_old": torch.zeros(batch_size),
        "returns": torch.randn(batch_size, 1),
        "advantages": torch.randn(batch_size, 1),
    }

    metrics = trainer.train_step(rollouts)
    assert "loss/total" in metrics
    assert "loss/policy" in metrics
    assert "loss/value" in metrics

    # Test checkpoint save and load
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = f"{tmp_dir}/model.pt"
        trainer.save_checkpoint(ckpt_path)
        trainer.load_checkpoint(ckpt_path)


def test_qmix_trainer_step() -> None:
    """Verifies QMIX TD-error update step."""
    num_agents = 3
    obs_dim = 115
    state_dim = 115 * num_agents

    agent_q = AgentQNetwork(obs_dim=obs_dim, action_dim=19)
    mixer = QMixer(num_agents=num_agents, state_dim=state_dim, embed_dim=16)

    trainer = QMIXTrainer(
        agent_q=agent_q,
        mixer=mixer,
        device="cpu",
    )

    batch_size = 4
    batch = {
        "obs": torch.randn(batch_size, num_agents, obs_dim),
        "actions": torch.randint(0, 19, (batch_size, num_agents)),
        "rewards": torch.randn(batch_size, 1),
        "next_obs": torch.randn(batch_size, num_agents, obs_dim),
        "dones": torch.zeros(batch_size, 1),
        "states": torch.randn(batch_size, state_dim),
        "next_states": torch.randn(batch_size, state_dim),
    }

    metrics = trainer.train_step(batch)
    assert "loss/qmix_td" in metrics
    assert "q_values/q_tot_mean" in metrics
