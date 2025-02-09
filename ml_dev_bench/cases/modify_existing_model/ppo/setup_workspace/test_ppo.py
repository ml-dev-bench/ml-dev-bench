import torch
import pytest
import numpy as np
from policy_network import PolicyNetwork
from ppo import PPOAgent


@pytest.fixture
def discrete_policy():
    return PolicyNetwork(
        state_dim=4, action_dim=2, hidden_dim=64, continuous_actions=False
    )


@pytest.fixture
def continuous_policy():
    return PolicyNetwork(
        state_dim=4, action_dim=2, hidden_dim=64, continuous_actions=True
    )


@pytest.fixture
def discrete_agent(discrete_policy):
    return PPOAgent(
        policy=discrete_policy,
        learning_rate=3e-4,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )


@pytest.fixture
def continuous_agent(continuous_policy):
    return PPOAgent(
        policy=continuous_policy,
        learning_rate=3e-4,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )


def test_policy_network_discrete(discrete_policy):
    batch_size = 32
    states = torch.randn(batch_size, 4)

    # Test forward pass
    dist, values = discrete_policy(states)
    assert isinstance(dist, torch.distributions.Categorical)
    assert values.shape == (batch_size, 1)

    # Test action sampling
    actions, log_probs, _ = discrete_policy.get_action_and_value(states)
    assert actions.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)

    # Test deterministic actions
    det_actions, _, _ = discrete_policy.get_action_and_value(states, deterministic=True)
    assert torch.all(det_actions == det_actions[0])


def test_policy_network_continuous(continuous_policy):
    batch_size = 32
    states = torch.randn(batch_size, 4)

    # Test forward pass
    dist, values = continuous_policy(states)
    assert isinstance(dist, torch.distributions.Normal)
    assert values.shape == (batch_size, 1)

    # Test action sampling
    actions, log_probs, _ = continuous_policy.get_action_and_value(states)
    assert actions.shape == (batch_size, 2)
    assert log_probs.shape == (batch_size,)

    # Test deterministic actions
    det_actions, _, _ = continuous_policy.get_action_and_value(
        states, deterministic=True
    )
    assert torch.allclose(det_actions, dist.mean, atol=1e-6)


def test_advantage_computation(discrete_agent):
    batch_size = 32
    rewards = torch.randn(batch_size)
    values = torch.randn(batch_size)
    dones = torch.zeros(batch_size, dtype=torch.bool)

    advantages = discrete_agent.compute_advantages(rewards, values, dones)
    assert advantages.shape == (batch_size,)

    # Test advantage normalization
    if discrete_agent.normalize_advantages:
        assert torch.abs(advantages.mean()) < 1e-6
        assert torch.abs(advantages.std() - 1.0) < 1e-6


def test_ppo_loss_computation(discrete_agent):
    batch_size = 32
    states = torch.randn(batch_size, 4)
    actions = torch.randint(0, 2, (batch_size,))
    old_log_probs = torch.randn(batch_size)
    advantages = torch.randn(batch_size)
    returns = torch.randn(batch_size)

    loss, metrics = discrete_agent.calculate_ppo_loss(
        states, actions, old_log_probs, advantages, returns
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert isinstance(metrics, dict)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics


def test_policy_update(discrete_agent):
    batch_size = 32
    states = torch.randn(batch_size, 4)
    actions = torch.randint(0, 2, (batch_size,))
    old_log_probs = torch.randn(batch_size)
    advantages = torch.randn(batch_size)
    returns = torch.randn(batch_size)

    metrics = discrete_agent.update_policy(
        states, actions, old_log_probs, advantages, returns
    )

    assert isinstance(metrics, list)
    assert len(metrics) > 0
    assert all(isinstance(m, dict) for m in metrics)


def test_clipping_behavior(discrete_agent):
    batch_size = 32
    states = torch.randn(batch_size, 4)
    actions = torch.randint(0, 2, (batch_size,))
    advantages = torch.ones(batch_size)  # Use positive advantages
    returns = torch.randn(batch_size)

    # Create old_log_probs that would cause large ratios
    old_log_probs = -10.0 * torch.ones(batch_size)

    loss, metrics = discrete_agent.calculate_ppo_loss(
        states, actions, old_log_probs, advantages, returns
    )

    # The clipped loss should be less than the unclipped loss
    assert metrics["clipped_loss"] <= metrics["unclipped_loss"]


def test_entropy_bonus(discrete_agent):
    batch_size = 32
    states = torch.randn(batch_size, 4)
    actions = torch.randint(0, 2, (batch_size,))
    old_log_probs = torch.randn(batch_size)
    advantages = torch.randn(batch_size)
    returns = torch.randn(batch_size)

    # Test with zero entropy coefficient
    discrete_agent.entropy_coef = 0.0
    loss_no_entropy, metrics_no_entropy = discrete_agent.calculate_ppo_loss(
        states, actions, old_log_probs, advantages, returns
    )

    # Test with positive entropy coefficient
    discrete_agent.entropy_coef = 0.01
    loss_with_entropy, metrics_with_entropy = discrete_agent.calculate_ppo_loss(
        states, actions, old_log_probs, advantages, returns
    )

    # Entropy should affect the total loss
    assert torch.abs(loss_no_entropy - loss_with_entropy) > 1e-6
