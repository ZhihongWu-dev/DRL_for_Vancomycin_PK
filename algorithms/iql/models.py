"""Neural network models for IQL: Q, V, Policy (simple MLPs)
"""
from typing import Tuple
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: Tuple[int, ...] = (256, 256), activation: nn.Module = nn.ReLU()):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(dim, h))
            layers.append(activation)
            dim = h
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1, hidden=(256, 256)):
        super().__init__()
        self.model = MLP(state_dim + action_dim, 1, hidden=hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.model(x)


class VNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden=(256, 256)):
        super().__init__()
        self.model = MLP(state_dim, 1, hidden=hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.model(s)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1, hidden=(256, 256), 
                 action_range=None):
        """
        Gaussian policy for continuous actions.
        
        Args:
            state_dim: dimension of state
            action_dim: dimension of action (default=1)
            hidden: hidden layer sizes
            action_range: tuple (min, max) for action bounds in normalized space.
                         If None, no bounds are applied. Common: (-3, 3) for normalized actions
        """
        super().__init__()
        self.net = MLP(state_dim, action_dim * 2, hidden=hidden)  # mean and logstd
        self.action_range = action_range

    def forward(self, s: torch.Tensor):
        out = self.net(s)
        mean, logstd = out.chunk(2, dim=-1)
        
        # Optional: bound mean to valid action range (in normalized space)
        if self.action_range is not None:
            mean = torch.clamp(mean, self.action_range[0], self.action_range[1])
        
        # Stabilize std: limit logstd range and enforce minimum std
        logstd = logstd.clamp(-5, 2)
        std = torch.clamp(logstd.exp(), min=0.1)
        return mean, std

    def sample(self, s: torch.Tensor):
        mean, std = self.forward(s)
        eps = torch.randn_like(mean)
        action = mean + eps * std
        
        # Clip sampled action to valid range if specified
        if self.action_range is not None:
            action = torch.clamp(action, self.action_range[0], self.action_range[1])
        
        return action

# convenience helper to initialize model weights using utilities
from algorithms.iql.utils import init_weights

def init_model_weights(model: nn.Module):
    """Apply standard initializations to a model in-place."""
    model.apply(init_weights)
    return model