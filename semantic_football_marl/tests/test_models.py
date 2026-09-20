"""Unit tests for multi-agent neural network architectures."""

import torch
from src.models.actor_critic import DecentralizedActor, CentralizedCritic, MAPPOPolicy
from src.models.q_mixer import AgentQNetwork, QMixer


def test_actor_critic_shapes() -> None:
    """Verifies output dimensions of Actor and Critic networks."""
    batch_size = 8
    obs_dim = 115
    action_dim = 19
    state_dim = 115 * 3

    actor = DecentralizedActor(obs_dim=obs_dim, action_dim=action_dim)
    critic = CentralizedCritic(state_dim=state_dim)

    obs = torch.randn(batch_size, obs_dim)
    state = torch.randn(batch_size, state_dim)

    logits = actor(obs)
    assert logits.shape == (batch_size, action_dim)

    values = critic(state)
    assert values.shape == (batch_size, 1)


def test_mappo_policy_act_and_eval() -> None:
    """Verifies policy action sampling and action evaluation."""
    policy = MAPPOPolicy(obs_dim=115, state_dim=115 * 3, action_dim=19)

    obs = torch.randn(4, 115)
    state = torch.randn(4, 115 * 3)

    actions, log_probs = policy.act(obs, deterministic=False)
    assert actions.shape == (4,)
    assert log_probs.shape == (4,)

    values, eval_log_probs, entropy = policy.evaluate_actions(obs, actions, state)
    assert values.shape == (4, 1)
    assert eval_log_probs.shape == (4,)
    assert entropy.numel() == 1


def test_qmixer_monotonicity() -> None:
    """Verifies that QMixer satisfies monotonicity: dQ_tot / dQ_i >= 0."""
    num_agents = 3
    state_dim = 115 * num_agents
    batch_size = 2

    mixer = QMixer(num_agents=num_agents, state_dim=state_dim, embed_dim=16)

    agent_qs = torch.randn(batch_size, num_agents, requires_grad=True)
    states = torch.randn(batch_size, state_dim)

    q_tot = mixer(agent_qs, states)
    assert q_tot.shape == (batch_size, 1)

    # Compute gradient of sum(q_tot) with respect to agent_qs
    q_tot.sum().backward()
    assert agent_qs.grad is not None
    # Monotonicity requirement: gradients must be strictly non-negative
    assert (agent_qs.grad >= -1e-6).all(), "QMIX monotonicity violation detected!"
