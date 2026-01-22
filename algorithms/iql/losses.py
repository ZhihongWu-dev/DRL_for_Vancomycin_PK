"""Loss functions for IQL: expectile loss, q loss, policy loss (AW-regression)
"""
import torch
import torch.nn.functional as F


def expectile_loss(v: torch.Tensor, q: torch.Tensor, tau: float) -> torch.Tensor:
    """Expectile loss for IQL: V should be tau-quantile of Q.
    
    Implements: V = argmin E[ ρ_tau(Q - V) ]
    where ρ_tau(u) = |tau - I(u<0)| * u^2
    
    For tau=0.7:
    - When Q > V (diff > 0): weight = tau = 0.7 (penalize MORE)
    - When Q < V (diff < 0): weight = 1-tau = 0.3 (penalize LESS)
    → This pushes V upward to ~70% quantile of Q
    → Result: ~30% of Q > V, ~70% of Q < V
    
    Args:
        v: V network values (shape: [batch, 1])
        q: Q network values, must be detached (shape: [batch, 1])
        tau: expectile parameter (0.7 → V ≈ 70th percentile of Q)
    
    Returns:
        Scalar loss
    """
    diff = q - v  # Compute diff inside function (safer)
    # CRITICAL: diff > 0 means Q > V, should use weight = tau
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


def q_mse_loss(q_values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(q_values, targets)


def policy_aw_loss(pi_actions: torch.Tensor, behavior_actions: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted MSE between policy action and behavior action with advantage weights.
    weight should be non-negative and shaped (batch, 1)
    """
    loss = (weight * ((pi_actions - behavior_actions) ** 2)).mean()
    return loss
