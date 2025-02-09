import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Tuple, Union, Dict


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        continuous_actions: bool = False,
    ):
        """
        Initialize policy network for PPO.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            continuous_actions: Whether action space is continuous
        """
        super().__init__()
        self.continuous_actions = continuous_actions

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Policy head
        if continuous_actions:
            self.action_mean = nn.Linear(hidden_dim, action_dim)
            self.action_logstd = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_probs = nn.Linear(hidden_dim, action_dim)

        # Value head
        self.value = nn.Linear(hidden_dim, 1)

    def forward(
        self, states: torch.Tensor
    ) -> Tuple[Union[Normal, Categorical], torch.Tensor]:
        """
        Forward pass to compute action distributions and values.

        Args:
            states: Batch of states

        Returns:
            distribution: Action distribution (Normal for continuous, Categorical for discrete)
            values: Value function estimates
        """
        # TODO: Implement forward pass
        # 1. Extract features from states
        # 2. Compute action distribution parameters
        # 3. Create appropriate distribution object
        # 4. Compute state values
        # 5. Add entropy calculation
        raise NotImplementedError("Implement forward method")

    def get_action_and_value(
        self, states: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions and compute their log probabilities and values.

        Args:
            states: Batch of states
            deterministic: Whether to sample deterministically

        Returns:
            actions: Sampled actions
            log_probs: Log probabilities of sampled actions
            values: Value function estimates
        """
        # TODO: Implement action sampling and value computation
        # 1. Get distributions and values from forward pass
        # 2. Sample actions (or take mean for deterministic case)
        # 3. Compute log probabilities
        # 4. Return actions, log_probs, and values
        raise NotImplementedError("Implement get_action_and_value method")

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions to get log probabilities, values, and entropy.

        Args:
            states: Batch of states
            actions: Batch of actions to evaluate

        Returns:
            log_probs: Log probabilities of actions
            values: Value function estimates
            entropy: Entropy of policy distribution
        """
        # TODO: Implement action evaluation
        # 1. Get distributions and values from forward pass
        # 2. Compute log probabilities of given actions
        # 3. Compute distribution entropy
        # 4. Return log_probs, values, and entropy
        raise NotImplementedError("Implement evaluate_actions method")
