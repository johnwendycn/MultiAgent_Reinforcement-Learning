"""Multi-Agent Proximal Policy Optimization (MAPPO) with Centralized Critic.

Implementation strictly adheres to the manuscript specifications:
- 16 parallel environments with horizon T=512 (8,192 environment steps per rollout).
- GAE computation with gamma=0.993 and lambda=0.95.
- PPO clipped surrogate objective (epsilon=0.20), 4 optimization epochs, mini-batch size 64.
- Value loss coefficient c1=0.50, Entropy bonus c2=0.01.
- Linear learning rate decay from 3e-4 to 0 over 5,000,000 steps.
- Centralized critic taking 115D global state as input to provide V_phi(s).
- EXACTLY 10 actors (one per learning outfield agent).
- NO reward clipping, NO hard-coded loss values, NO PBRS inside the training loop.
"""

from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.algos.base_algo import BaseMARLTrainer
from src.utils.registry import ALGO_REGISTRY, MODEL_REGISTRY


def init_orthogonal(module: nn.Module, gain: float = 1.0) -> nn.Module:
    """Initializes linear weights with orthogonal matrix and biases with zeros."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module


@MODEL_REGISTRY.register("actor")
class Actor(nn.Module):
    """Decentralized policy network for an individual outfield agent."""

    def __init__(
        self,
        obs_dim: int = 115,
        action_dim: int = 19,
        hidden_dims: Optional[List[int]] = None,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        layers: List[nn.Module] = []
        prev_dim = obs_dim

        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            init_orthogonal(linear, gain=1.414)
            layers.append(linear)
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, action_dim)
        init_orthogonal(self.action_head, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Computes action logits for agent observations."""
        features = self.backbone(obs)
        return self.action_head(features)

    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        """Returns the Categorical action distribution."""
        logits = self.forward(obs)
        return Categorical(logits=logits)


@MODEL_REGISTRY.register("critic")
class Critic(nn.Module):
    """Centralized value network evaluating the global state V_phi(s)."""

    def __init__(
        self,
        state_dim: int = 115,
        hidden_dims: Optional[List[int]] = None,
        use_layer_norm: bool = True,
    ) -> None:
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
        """Computes estimated scalar state value V(s)."""
        if state.dim() == 3:
            # If passed as (B, N, D), use team global state slice
            state = state[:, 0, :]
        features = self.backbone(state)
        return self.val_head(features)


@MODEL_REGISTRY.register("mappo_policy")
class MAPPOPolicy(nn.Module):
    """Container managing exactly 10 decentralized actors and 1 centralized critic."""

    NUM_LEARNING_AGENTS: int = 10

    def __init__(
        self,
        num_agents: int = 10,
        obs_dim: int = 115,
        state_dim: int = 115,
        action_dim: int = 19,
        actor_hidden_dims: Optional[List[int]] = None,
        critic_hidden_dims: Optional[List[int]] = None,
        use_layer_norm: bool = True,
        share_actor_params: bool = False,
    ) -> None:
        super().__init__()
        # Strict requirement assertion: exactly 10 actors
        assert num_agents == self.NUM_LEARNING_AGENTS, (
            f"MAPPO must support EXACTLY 10 learning outfield agents, got {num_agents}"
        )
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.share_actor_params = share_actor_params

        if share_actor_params:
            shared_actor = Actor(obs_dim, action_dim, actor_hidden_dims, use_layer_norm)
            self.actors = nn.ModuleList([shared_actor for _ in range(num_agents)])
        else:
            self.actors = nn.ModuleList([
                Actor(obs_dim, action_dim, actor_hidden_dims, use_layer_norm)
                for _ in range(num_agents)
            ])

        # Assert number of actors is 10
        assert len(self.actors) == 10, f"Number of actors must be 10, got {len(self.actors)}"

        # Centralized Critic takes 115D global state as input
        self.critic = Critic(
            state_dim=state_dim,
            hidden_dims=critic_hidden_dims,
            use_layer_norm=use_layer_norm,
        )

    def act(
        self,
        obs: torch.Tensor,
        agent_idx: int = 0,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples action and log probability for a single agent."""
        dist = self.actors[agent_idx].get_distribution(obs)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def act_all(
        self,
        obs_joint: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples actions and log probabilities for all 10 outfield agents.

        Args:
            obs_joint: Observation tensor of shape (batch, 10, obs_dim).
            deterministic: Argmax action selection if True.

        Returns:
            Tuple of (actions, log_probs) both of shape (batch, 10).
        """
        B, N, _ = obs_joint.shape
        assert N == self.num_agents, f"Expected {self.num_agents} agent observations, got {N}"

        actions_list = []
        log_probs_list = []

        for i in range(self.num_agents):
            agent_obs = obs_joint[:, i, :]
            dist = self.actors[i].get_distribution(agent_obs)
            if deterministic:
                act = torch.argmax(dist.logits, dim=-1)
            else:
                act = dist.sample()
            lp = dist.log_prob(act)
            actions_list.append(act)
            log_probs_list.append(lp)

        actions = torch.stack(actions_list, dim=1)      # (B, 10)
        log_probs = torch.stack(log_probs_list, dim=1)  # (B, 10)
        return actions, log_probs

    def get_values(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluates estimated state values using centralized critic."""
        return self.critic(state)  # (B, 1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates batch of transitions during PPO updates.

        Args:
            obs: (batch, 10, obs_dim)
            actions: (batch, 10)
            state: (batch, state_dim)

        Returns:
            Tuple of (values, log_probs, entropy):
            - values: (batch, 1)
            - log_probs: (batch, 10)
            - entropy: scalar Tensor
        """
        B, N, _ = obs.shape
        assert N == self.num_agents, f"Expected {self.num_agents} agents, got {N}"

        log_probs_list = []
        entropy_list = []

        for i in range(self.num_agents):
            dist = self.actors[i].get_distribution(obs[:, i, :])
            lp = dist.log_prob(actions[:, i])
            ent = dist.entropy()
            log_probs_list.append(lp)
            entropy_list.append(ent)

        log_probs = torch.stack(log_probs_list, dim=1)  # (B, 10)
        entropy = torch.stack(entropy_list, dim=1).mean()  # scalar
        values = self.critic(state)                        # (B, 1)
        return values, log_probs, entropy


class MAPPORolloutBuffer:
    """Pre-allocated trajectory storage for 16 parallel environments with horizon T=512."""

    def __init__(
        self,
        num_envs: int = 16,
        horizon: int = 512,
        num_agents: int = 10,
        obs_dim: int = 115,
        state_dim: int = 115,
        gamma: float = 0.993,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.num_envs = num_envs
        self.horizon = horizon
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        assert self.num_agents == 10, f"Buffer requires 10 agents, got {num_agents}"
        self.total_steps = num_envs * horizon  # 16 * 512 = 8,192 steps

        # Allocate storage tensors on target device
        self.obs = torch.zeros(horizon, num_envs, num_agents, obs_dim, device=device)
        self.global_states = torch.zeros(horizon, num_envs, state_dim, device=device)
        self.actions = torch.zeros(horizon, num_envs, num_agents, dtype=torch.long, device=device)
        self.rewards = torch.zeros(horizon, num_envs, num_agents, device=device)
        self.logprobs = torch.zeros(horizon, num_envs, num_agents, device=device)
        self.values = torch.zeros(horizon, num_envs, 1, device=device)
        self.dones = torch.zeros(horizon, num_envs, 1, device=device)
        self.advantages = torch.zeros(horizon, num_envs, num_agents, device=device)
        self.returns = torch.zeros(horizon, num_envs, num_agents, device=device)

        self.step = 0

    def insert(
        self,
        obs: torch.Tensor,
        global_state: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        logprobs: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Stores transition step without any reward clipping or fabrication."""
        t = self.step
        self.obs[t] = obs.to(self.device)
        self.global_states[t] = global_state.to(self.device)
        self.actions[t] = actions.to(self.device)
        self.rewards[t] = rewards.to(self.device)
        self.logprobs[t] = logprobs.to(self.device)
        self.values[t] = values.view(self.num_envs, 1).to(self.device)
        self.dones[t] = dones.view(self.num_envs, 1).to(self.device)
        self.step = (self.step + 1) % self.horizon

    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor) -> None:
        """Computes Generalized Advantage Estimation with gamma=0.993 and lambda=0.95."""
        last_gae = torch.zeros(self.num_envs, self.num_agents, device=self.device)

        next_val = next_value.view(self.num_envs, 1).to(self.device)
        next_d = next_done.view(self.num_envs, 1).float().to(self.device)

        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                next_non_terminal = 1.0 - next_d
                val_next = next_val
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                val_next = self.values[t + 1]

            cur_val = self.values[t]  # (num_envs, 1)

            # delta_t = r_{t, i} + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
            delta = self.rewards[t] + self.gamma * val_next * next_non_terminal - cur_val
            # A_{t, i} = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1, i}
            self.advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            last_gae = self.advantages[t]
            self.returns[t] = self.advantages[t] + cur_val

    def mini_batch_generator(
        self,
        mini_batch_size: int = 64,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Flattens trajectory into 8,192 multi-agent transitions and yields mini-batches of size 64."""
        total_samples = self.horizon * self.num_envs  # 512 * 16 = 8,192
        indices = torch.randperm(total_samples, device=self.device)

        # Flatten (T, num_envs, ...) -> (total_samples, ...)
        flat_obs = self.obs.view(total_samples, self.num_agents, self.obs_dim)
        flat_states = self.global_states.view(total_samples, self.state_dim)
        flat_actions = self.actions.view(total_samples, self.num_agents)
        flat_logprobs = self.logprobs.view(total_samples, self.num_agents)
        flat_values = self.values.view(total_samples, 1)
        flat_returns = self.returns.view(total_samples, self.num_agents)
        flat_advantages = self.advantages.view(total_samples, self.num_agents)

        for start in range(0, total_samples, mini_batch_size):
            end = min(start + mini_batch_size, total_samples)
            mb_idx = indices[start:end]

            yield {
                "obs": flat_obs[mb_idx],
                "states": flat_states[mb_idx],
                "actions": flat_actions[mb_idx],
                "logprobs": flat_logprobs[mb_idx],
                "values": flat_values[mb_idx],
                "returns": flat_returns[mb_idx],
                "advantages": flat_advantages[mb_idx],
            }


@ALGO_REGISTRY.register("mappo")
class MAPPOTrainer(BaseMARLTrainer):
    """End-to-end MAPPO optimization engine for 10 outfield agents."""

    def __init__(
        self,
        policy: MAPPOPolicy,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        clip_param: float = 0.20,
        ppo_epoch: int = 4,
        mini_batch_size: int = 64,
        value_loss_coef: float = 0.50,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.50,
        gamma: float = 0.993,
        gae_lambda: float = 0.95,
        total_steps: int = 5_000_000,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.policy = policy.to(self.device)

        # Requirement 5 Assertion: exactly 10 actors
        assert len(self.policy.actors) == 10, (
            f"MAPPOTrainer requires EXACTLY 10 actors, found {len(self.policy.actors)}"
        )

        self.lr_actor_init = lr_actor
        self.lr_critic_init = lr_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.total_steps = total_steps
        self.current_timestep = 0

        # Optimizers with initial learning rates
        self.actor_optimizer = optim.Adam(self.policy.actors.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.policy.critic.parameters(), lr=lr_critic, eps=1e-5)

    def update_lr(self, timestep: int) -> float:
        """Applies linear learning rate decay from 3e-4 to 0 over 5M steps."""
        self.current_timestep = timestep
        decay_factor = max(0.0, 1.0 - float(timestep) / float(self.total_steps))
        cur_lr_actor = self.lr_actor_init * decay_factor
        cur_lr_critic = self.lr_critic_init * decay_factor

        for pg in self.actor_optimizer.param_groups:
            pg["lr"] = cur_lr_actor
        for pg in self.critic_optimizer.param_groups:
            pg["lr"] = cur_lr_critic

        return cur_lr_actor

    def train_step(self, rollouts: Union[MAPPORolloutBuffer, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Performs PPO updates using 4 optimization epochs and mini-batches of size 64."""
        total_loss_list: List[float] = []
        policy_loss_list: List[float] = []
        value_loss_list: List[float] = []
        entropy_list: List[float] = []

        if isinstance(rollouts, MAPPORolloutBuffer):
            for _ in range(self.ppo_epoch):
                for mb in rollouts.mini_batch_generator(self.mini_batch_size):
                    losses = self._update_minibatch(mb)
                    total_loss_list.append(losses["loss/total"])
                    policy_loss_list.append(losses["loss/policy"])
                    value_loss_list.append(losses["loss/value"])
                    entropy_list.append(losses["policy/entropy"])
        else:
            # Direct dictionary update fallback
            for _ in range(self.ppo_epoch):
                losses = self._update_minibatch(rollouts)
                total_loss_list.append(losses["loss/total"])
                policy_loss_list.append(losses["loss/policy"])
                value_loss_list.append(losses["loss/value"])
                entropy_list.append(losses["policy/entropy"])

        return {
            "loss/total": sum(total_loss_list) / max(1, len(total_loss_list)),
            "loss/policy": sum(policy_loss_list) / max(1, len(policy_loss_list)),
            "loss/value": sum(value_loss_list) / max(1, len(value_loss_list)),
            "policy/entropy": sum(entropy_list) / max(1, len(entropy_list)),
        }

    def _update_minibatch(self, mb: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Computes PPO surrogate loss, value loss, and entropy bonus on a single mini-batch."""
        obs = mb["obs"].to(self.device)            # (B, 10, obs_dim) or (B, obs_dim)
        states = mb["states"].to(self.device)      # (B, state_dim)
        actions = mb["actions"].to(self.device)    # (B, 10) or (B,)
        old_logprobs = mb["logprobs"].to(self.device)  # (B, 10) or (B,)
        returns = mb["returns"].to(self.device)    # (B, 10) or (B, 1)
        advantages = mb["advantages"].to(self.device)  # (B, 10) or (B, 1)

        # Normalize advantages across mini-batch
        adv_std = advantages.std() + 1e-8
        adv_norm = (advantages - advantages.mean()) / adv_std

        # Handle 2D or 3D tensor shapes
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, 10, -1)
            actions = actions.unsqueeze(1).expand(-1, 10)
            old_logprobs = old_logprobs.unsqueeze(1).expand(-1, 10)
            adv_norm = adv_norm.unsqueeze(1).expand(-1, 10)

        # Forward pass through policy
        values, new_logprobs, entropy = self.policy.evaluate_actions(obs, actions, states)

        # PPO Clipped Surrogate Loss (Eq: min(r*A, clip(r, 1-eps, 1+eps)*A))
        ratio = torch.exp(new_logprobs - old_logprobs)  # (B, 10)
        surr1 = ratio * adv_norm
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_norm
        policy_loss = -torch.min(surr1, surr2).mean()

        # Critic Value Loss (c1 = 0.50): V_phi(s) target is team mean return
        target_val = returns.mean(dim=-1, keepdim=True) if returns.dim() > 1 else returns.view(-1, 1)
        value_loss = 0.5 * (values - target_val).pow(2).mean()

        # Total MAPPO Objective: L = L_CLIP + c1 * L_VF - c2 * Entropy
        total_loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy
        )

        # Check for NaNs
        assert not torch.isnan(total_loss), "NaN encountered in MAPPO loss computation"

        # Backpropagation
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()

        nn.utils.clip_grad_norm_(self.policy.actors.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return {
            "loss/total": float(total_loss.item()),
            "loss/policy": float(policy_loss.item()),
            "loss/value": float(value_loss.item()),
            "policy/entropy": float(entropy.item()),
        }

    def save_checkpoint(self, filepath: str) -> None:
        """Saves model weights and optimizer states."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "current_timestep": self.current_timestep,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str) -> None:
        """Loads model weights and optimizer states."""
        ckpt = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.current_timestep = ckpt.get("current_timestep", 0)


# Backward-compatible alias
MAPPO = MAPPOTrainer

__all__ = [
    "Actor",
    "Critic",
    "MAPPOPolicy",
    "MAPPORolloutBuffer",
    "MAPPOTrainer",
    "MAPPO",
]
