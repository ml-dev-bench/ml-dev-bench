import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List

from policy_network import PolicyNetwork


class PPOAgent:
    def __init__(
        self,
        policy: PolicyNetwork,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
    ):
        """Initialize PPO agent with hyperparameters."""
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> torch.Tensor:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            rewards: Tensor of shape (batch_size,) containing rewards
            values: Tensor of shape (batch_size,) containing value estimates
            dones: Tensor of shape (batch_size,) containing done flags
            gamma: Discount factor
            gae_lambda: GAE smoothing parameter

        Returns:
            advantages: Tensor of shape (batch_size,) containing computed advantages
        """
        # TODO: Implement GAE advantage computation
        # 1. Calculate deltas (temporal difference errors)
        # 2. Compute GAE advantages using the recursive formula
        # 3. If self.normalize_advantages is True, normalize the advantages
        raise NotImplementedError("Implement compute_advantages method")

    def calculate_ppo_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate PPO loss components.

        Args:
            states: Current states
            actions: Actions taken
            old_log_probs: Log probabilities of actions under old policy
            advantages: Computed advantages
            returns: Discounted returns

        Returns:
            total_loss: Combined PPO loss
            metrics: Dictionary containing individual loss components
        """
        # TODO: Implement PPO loss calculation
        # 1. Get current policy distributions and values
        # 2. Calculate probability ratios
        # 3. Compute clipped surrogate objective
        # 4. Calculate value function loss
        # 5. Add entropy bonus
        # 6. Combine all loss components
        raise NotImplementedError("Implement calculate_ppo_loss method")

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        num_epochs: int = 10,
    ) -> List[Dict[str, float]]:
        """
        Update policy using PPO algorithm.

        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Batch of log probabilities under old policy
            advantages: Computed advantages
            returns: Computed returns
            num_epochs: Number of epochs to update policy

        Returns:
            metrics: List of dictionaries containing training metrics
        """
        # TODO: Implement policy update
        # 1. Perform multiple epochs of updates
        # 2. Calculate PPO loss
        # 3. Perform gradient step with clipping
        # 4. Track and return metrics
        raise NotImplementedError("Implement update_policy method")
