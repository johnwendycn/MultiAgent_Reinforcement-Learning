"""Monotonic QMIX mixing network and individual agent Q-networks."""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("agent_q_network")
class AgentQNetwork(nn.Module):
    """Individual agent Q-network estimating action-values Q_i(o_i, a)."""

    def __init__(
        self,
        obs_dim: int = 115,
        action_dim: int = 19,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        """Initializes the agent Q-network.

        Args:
            obs_dim: Local observation dimension.
            action_dim: Number of discrete actions.
            hidden_dims: List of hidden layer dimensions.
        """
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all discrete actions. Shape: (batch, action_dim)."""
        return self.net(obs)


@MODEL_REGISTRY.register("q_mixer")
class QMixer(nn.Module):
    """Monotonic Q-Mixer network parameterized by hypernetworks conditioned on global state.

    Guarantees monotonicity: dQ_tot / dQ_i >= 0 via absolute weights on mixer layers.
    """

    def __init__(
        self,
        num_agents: int = 3,
        state_dim: int = 115 * 3,
        embed_dim: int = 32,
        hypernet_embed: int = 64,
    ) -> None:
        """Initializes the QMixer.

        Args:
            num_agents: Number of cooperating agents.
            state_dim: Global state dimension.
            embed_dim: Latent mixing dimension.
            hypernet_embed: Hidden dimension for hypernetworks.
        """
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernetwork 1: generates weights of shape (num_agents, embed_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, num_agents * embed_dim)
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # Hypernetwork 2: generates weights of shape (embed_dim, 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim * 1)
        )
        # State-conditioned scalar bias for Q_tot
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Mixes individual agent Q-values into a centralized Q_tot.

        Args:
            agent_qs: Tensor of shape (batch, num_agents) containing chosen action Q-values.
            states: Global state tensor of shape (batch, state_dim).

        Returns:
            Centralized Q_tot tensor of shape (batch, 1).
        """
        batch_size = agent_qs.shape[0]
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)

        # Layer 1
        w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(batch_size, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Layer 2
        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.view(batch_size, 1)
