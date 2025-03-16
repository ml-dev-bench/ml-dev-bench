"""Tests for GRPO implementation."""

import numpy as np
import torch
import torch.nn.functional as F

from grpo import GRPOAgent


def test_policy_output():
    """Test policy network output."""
    agent = GRPOAgent(state_dim=4, action_dim=2, discrete=False)
    state = torch.randn(1, 4)
    dist = agent.get_policy(state)
    assert isinstance(dist, torch.distributions.Normal)
    assert dist.loc.shape == (1, 2)
    assert dist.scale.shape == (1, 2)

    agent = GRPOAgent(state_dim=4, action_dim=3, discrete=True)
    state = torch.randn(1, 4)
    dist = agent.get_policy(state)
    assert isinstance(dist, torch.distributions.Categorical)
    assert dist.logits.shape == (1, 3)


def test_value_output():
    """Test value network output."""
    agent = GRPOAgent(state_dim=4, action_dim=2)
    state = torch.randn(1, 4)
    value = agent.get_value(state)
    assert value.shape == (1, 1)


def test_reward_shaping():
    """Test reward shaping mechanism."""
    agent = GRPOAgent(state_dim=4, action_dim=2)
    batch_size = 10
    states = torch.randn(batch_size, 4)
    actions = torch.randn(batch_size, 2)
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 4)
    dones = torch.zeros(batch_size)

    shaped_rewards = agent.shape_rewards(states, actions, rewards, next_states, dones)
    assert shaped_rewards.shape == rewards.shape
    # Shaped rewards should be different from original rewards
    assert not torch.allclose(shaped_rewards, rewards)
    # Shaped rewards should maintain relative ordering
    assert torch.all(shaped_rewards[rewards > 0] > shaped_rewards[rewards < 0])


def test_advantage_computation():
    """Test advantage computation."""
    agent = GRPOAgent(state_dim=4, action_dim=2)
    batch_size = 10
    rewards = torch.randn(batch_size)
    values = torch.randn(batch_size)
    next_values = torch.randn(batch_size)
    dones = torch.zeros(batch_size)

    advantages, returns = agent.compute_advantages(rewards, values, next_values, dones)
    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    # Check if advantages are properly normalized
    assert torch.abs(advantages.mean()) < 0.1
    assert torch.abs(advantages.std() - 1.0) < 0.1


def test_update():
    """Test policy update."""
    agent = GRPOAgent(state_dim=4, action_dim=2)
    batch_size = 10
    states = torch.randn(batch_size, 4)
    actions = torch.randn(batch_size, 2)
    old_log_probs = torch.randn(batch_size)
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 4)
    dones = torch.zeros(batch_size)

    metrics = agent.update(states, actions, old_log_probs, rewards, next_states, dones)
    assert isinstance(metrics, dict)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics


def test_hidden_reward_shaping():
    """Hidden test for reward shaping."""
    agent = GRPOAgent(state_dim=4, action_dim=2)
    batch_size = 100
    states = torch.randn(batch_size, 4)
    actions = torch.randn(batch_size, 2)
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 4)
    dones = torch.zeros(batch_size)

    shaped_rewards = agent.shape_rewards(states, actions, rewards, next_states, dones)

    # Test reward scaling
    assert torch.abs(shaped_rewards.mean()) < rewards.mean()
    assert torch.abs(shaped_rewards.std() - 1.0) < 0.2

    # Test temporal consistency
    shaped_rewards_2 = agent.shape_rewards(states, actions, rewards, next_states, dones)
    assert torch.allclose(shaped_rewards, shaped_rewards_2)

    # Test state-dependent shaping
    diff_states = states + torch.randn_like(states) * 0.1
    shaped_rewards_3 = agent.shape_rewards(diff_states, actions, rewards, next_states, dones)
    assert not torch.allclose(shaped_rewards, shaped_rewards_3)


def test_hidden_policy_update():
    """Hidden test for policy update."""
    agent = GRPOAgent(state_dim=4, action_dim=2)
    batch_size = 100
    states = torch.randn(batch_size, 4)
    actions = torch.randn(batch_size, 2)
    old_dist = agent.get_policy(states)
    old_log_probs = old_dist.log_prob(actions).sum(-1)
    rewards = torch.ones(batch_size)  # All positive rewards
    next_states = torch.randn(batch_size, 4)
    dones = torch.zeros(batch_size)

    # Store old policy parameters
    old_params = [p.clone() for p in agent.policy.parameters()]

    # Update policy multiple times
    for _ in range(5):
        metrics = agent.update(states, actions, old_log_probs, rewards, next_states, dones)

    # Check if policy improved
    new_dist = agent.get_policy(states)
    new_log_probs = new_dist.log_prob(actions).sum(-1)
    assert (new_log_probs > old_log_probs).float().mean() > 0.6

    # Check if clipping worked (parameters shouldn't change too much)
    new_params = [p.clone() for p in agent.policy.parameters()]
    param_changes = [torch.abs(new - old).mean() for new, old in zip(new_params, old_params)]
    assert all(change < 0.5 for change in param_changes)


def run_tests():
    """Run all tests."""
    test_policy_output()
    test_value_output()
    test_reward_shaping()
    test_advantage_computation()
    test_update()
    test_hidden_reward_shaping()
    test_hidden_policy_update()
    print("All tests passed!")
