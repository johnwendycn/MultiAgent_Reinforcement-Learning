"""Decentralized Actor and Centralized Critic architectures for MAPPO."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.distributions import Categorical
from src.utils.registry import MODEL_REGISTRY


def init_orthogonal(module: nn.Module, gain: float = 1.0) -> nn.Module:
    """Applies orthogonal parameter initialization with uniform bias zeros."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module


@MODEL_REGISTRY.register("decentralized_actor")
class DecentralizedActor(nn.Module):
    """Decentralized policy network taking ego observations and outputting action logits."""

    def __init__(
        self,
        obs_dim: int = 115,
        action_dim: int = 19,
        hidden_dims: Optional[List[int]] = None,
        use_layer_norm: bool = True,
    ) -> None:
        """Initializes the actor network.

        Args:
            obs_dim: Dimension of local observation (default 115 for simple115v2).
            action_dim: Number of discrete actions (19 for GRF).
            hidden_dims: List of layer hidden dimensions.
            use_layer_norm: Whether to apply LayerNorm between linear layers.
        """
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        layers: List[nn.Module] = []
        prev_dim = obs_dim

        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            init_orthogonal(linear, gain=np_gain if (np_gain := 1.414) else 1.0)
            layers.append(linear)
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, action_dim)
        init_orthogonal(self.action_head, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Computes action logits given local agent observation."""
        features = self.backbone(obs)
        return self.action_head(features)

    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        """Returns the categorical action distribution for observation tensor."""
        logits = self.forward(obs)
        return Categorical(logits=logits)


@MODEL_REGISTRY.register("centralized_critic")
class CentralizedCritic(nn.Module):
    """Centralized value network taking global state or joint team observations."""

    def __init__(
        self,
        state_dim: int = 115 * 3,
        hidden_dims: Optional[List[int]] = None,
        use_layer_norm: bool = True,
    ) -> None:
        """Initializes the centralized critic network.

        Args:
            state_dim: Dimension of global state or concatenated team observations.
            hidden_dims: Hidden dimensions list.
            use_layer_norm: Whether to apply LayerNorm.
        """
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128]
        layers: List[nn.Module] = []
        prev_dim = state_dim

        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            init_orthogonal(linear, gain=1.414)
            layers.append(linear)
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.val_head = nn.Linear(prev_dim, 1)
        init_orthogonal(self.val_head, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Computes estimated scalar state-value V(s)."""
        features = self.backbone(state)
        return self.val_head(features)


@MODEL_REGISTRY.register("mappo_policy")
class MAPPOPolicy(nn.Module):
    """Joint container bundling Decentralized Actor and Centralized Critic."""

    def __init__(
        self,
        obs_dim: int = 115,
        state_dim: int = 115 * 3,
        action_dim: int = 19,
        actor_hidden_dims: Optional[List[int]] = None,
        critic_hidden_dims: Optional[List[int]] = None,
        use_layer_norm: bool = True,
    ) -> None:
        """Initializes the MAPPO policy bundle."""
        super().__init__()
        self.actor = DecentralizedActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims,
            use_layer_norm=use_layer_norm,
        )
        self.critic = CentralizedCritic(
            state_dim=state_dim,
            hidden_dims=critic_hidden_dims,
            use_layer_norm=use_layer_norm,
        )

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples an action and its log probability.

        Args:
            obs: Agent observation tensor of shape (batch, obs_dim).
            deterministic: If True, selects argmax greedy action.

        Returns:
            Tuple of (action_tensor, log_prob_tensor).
        """
        dist = self.actor.get_distribution(obs)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates actions during PPO updates.

        Returns:
            Tuple of (values, log_probs, entropy).
        """
        dist = self.actor.get_distribution(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        values = self.critic(state)
        return values, log_probs, entropy
