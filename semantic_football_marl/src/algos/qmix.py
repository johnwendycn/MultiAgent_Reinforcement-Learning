"""QMIX value-based multi-agent reinforcement learning algorithm."""

from typing import Any, Dict, List, Optional
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from src.algos.base_algo import BaseMARLTrainer
from src.models.q_mixer import AgentQNetwork, QMixer
from src.utils.registry import ALGO_REGISTRY


@ALGO_REGISTRY.register("qmix")
class QMIXTrainer(BaseMARLTrainer):
    """Trainer implementation for the QMIX monotonic value factorization algorithm."""

    def __init__(
        self,
        agent_q: AgentQNetwork,
        mixer: QMixer,
        lr: float = 5e-4,
        gamma: float = 0.99,
        max_grad_norm: float = 10.0,
        device: str = "cpu",
    ) -> None:
        """Initializes the QMIX trainer.

        Args:
            agent_q: Individual agent Q-network instance.
            mixer: Centralized monotonic mixing network.
            lr: Learning rate for joint optimizer.
            gamma: Temporal discount factor.
            max_grad_norm: Gradient clipping threshold.
            device: Target computing device ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        self.agent_q = agent_q.to(self.device)
        self.mixer = mixer.to(self.device)

        # Target networks
        self.target_agent_q = copy.deepcopy(self.agent_q).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)

        parameters = list(self.agent_q.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(parameters, lr=lr)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Executes a single TD-error gradient update on a replay batch.

        Expected batch keys:
        - 'obs': (batch, num_agents, obs_dim)
        - 'actions': (batch, num_agents)
        - 'rewards': (batch, 1)
        - 'next_obs': (batch, num_agents, obs_dim)
        - 'dones': (batch, 1)
        - 'states': (batch, state_dim)
        - 'next_states': (batch, state_dim)
        """
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)
        states = batch["states"].to(self.device)
        next_states = batch["next_states"].to(self.device)

        batch_size, num_agents, obs_dim = obs.shape

        # 1. Compute current Q values: Q_i(o_i, a_i)
        q_vals_all = self.agent_q(obs.view(-1, obs_dim)).view(batch_size, num_agents, -1)
        chosen_action_q = torch.gather(q_vals_all, dim=2, index=actions.unsqueeze(2)).squeeze(2)
        q_tot = self.mixer(chosen_action_q, states)

        # 2. Compute target Q values: max_{a'} Q'_i(o'_i, a')
        with torch.no_grad():
            next_q_vals = self.target_agent_q(next_obs.view(-1, obs_dim)).view(batch_size, num_agents, -1)
            max_next_q = next_q_vals.max(dim=2)[0]
            target_q_tot = self.target_mixer(max_next_q, next_states)
            y = rewards + self.gamma * (1.0 - dones.float()) * target_q_tot

        td_error = q_tot - y
        loss = 0.5 * (td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent_q.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.mixer.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss/qmix_td": float(loss.item()),
            "q_values/q_tot_mean": float(q_tot.mean().item()),
            "q_values/target_mean": float(y.mean().item()),
        }

    def update_targets(self) -> None:
        """Copies online network parameters into target networks."""
        self.target_agent_q.load_state_dict(self.agent_q.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save_checkpoint(self, filepath: str) -> None:
        """Persists QMIX model weights and optimizer state."""
        torch.save(
            {
                "agent_q": self.agent_q.state_dict(),
                "mixer": self.mixer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str) -> None:
        """Restores QMIX model weights from file."""
        ckpt = torch.load(filepath, map_location=self.device)
        self.agent_q.load_state_dict(ckpt["agent_q"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.update_targets()
