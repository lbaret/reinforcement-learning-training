from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network architecture for PPO.
    
    This class implements both the policy (actor) and the value function (critic)
    networks for discrete action spaces.
    """

    def __init__(self, state_dim: int, action_dim: int, device: Union[torch.device, str]) -> None:
        """
        Initialize the ActorCritic networks.

        Args:
            state_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            device (Union[torch.device, str]): Device to perform computations on.
        """
        super(ActorCritic, self).__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self) -> None:
        """
        Forward pass is not explicitly used in PPO as act() and evaluate() 
        handle the logic.
        """
        raise NotImplementedError
    
    def act(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Sample an action based on the current state.

        Args:
            state (np.ndarray): The current environment observation.

        Returns:
            Tuple[int, float]: A tuple containing the sampled action and its log probability.
        """
        state_tensor = torch.from_numpy(state).float().to(self.device)
        action_probs = self.actor(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob.item()
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training updates.

        Args:
            state (torch.Tensor): Batch of states.
            action (torch.Tensor): Batch of actions taken.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                A tuple containing (log_probs, state_values, dist_entropy).
        """
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy