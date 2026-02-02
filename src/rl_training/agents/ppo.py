from typing import Any, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_training.models.actor_critic import ActorCritic


class PPO:
    """
    Proximal Policy Optimization (PPO) agent implementation.
    
    This class handles the policy networks, data collection, and the update mechanism
    for the PPO algorithm.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.002,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """
        Initialize the PPO agent.

        Args:
            state_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            eps_clip (float): Clipping parameter for the PPO objective.
            k_epochs (int): Number of optimization epochs per update.
            device (Union[torch.device, str]): Device to perform computations on.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(state_dim, action_dim, self.device).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()
        self.data: List[Any] = []

    def put_data(self, item: Tuple[np.ndarray, int, float, np.ndarray, float, bool]) -> None:
        """
        Store a single transition in the agent's memory.

        Args:
            item (Tuple[np.ndarray, int, float, np.ndarray, float, bool]): 
                A transition tuple containing (state, action, reward, next_state, action_prob, done).
        """
        self.data.append(item)

    def make_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the collected transitions into tensors for a training update.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Tensors representing (states, actions, rewards, next_states, done_masks, old_log_probs).
        """
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s = torch.tensor(np.array(s_lst), dtype=torch.float).to(self.device)
        a = torch.tensor(np.array(a_lst), dtype=torch.float).to(self.device)
        r = torch.tensor(np.array(r_lst), dtype=torch.float).to(self.device)
        s_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(self.device)
        done_mask = torch.tensor(np.array(done_lst), dtype=torch.float).to(self.device)
        prob_a = torch.tensor(np.array(prob_a_lst), dtype=torch.float).to(self.device)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def update(self) -> None:
        """
        Update the policy and value networks using the collected experience.
        """
        s, a, r, s_prime, done_mask, old_log_prob = self.make_batch()

        with torch.no_grad():
            target = r + self.gamma * self.policy_old.critic(s_prime) * done_mask
            advantage = target - self.policy_old.critic(s)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

        for _ in range(self.k_epochs):
            log_probs, state_values, dist_entropy = self.policy.evaluate(s, a.squeeze())
            ratios = torch.exp(log_probs - old_log_prob.squeeze())
            surr1 = ratios * advantage.squeeze()
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantage.squeeze()
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, target) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
