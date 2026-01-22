"""Training utilities for IQL: single-step update function.

Provides:
- iql_update_step: runs one Q/V/Policy update given a batch and optimizers
"""
from typing import Dict
import torch
import torch.nn.functional as F
from algorithms.iql.losses import expectile_loss


def iql_update_step(batch: Dict[str, torch.Tensor],
                    q_net: torch.nn.Module,
                    v_net: torch.nn.Module,
                    pi_net: torch.nn.Module,
                    q_opt: torch.optim.Optimizer,
                    v_opt: torch.optim.Optimizer,
                    pi_opt: torch.optim.Optimizer,
                    gamma: float = 0.99,
                    tau: float = 0.7,
                    beta: float = 3.0,
                    weight_clip: float = 1e2,
                    q_target: torch.nn.Module = None,
                    target_update_rate: float = 0.005,
                    reward_scale: float = 1.0,
                    train_policy: bool = True,
                    train_value: bool = True,
                    train_q: bool = True) -> Dict[str, torch.Tensor]:
    """Perform a single optimization step for IQL.

    Args:
        batch: dict with keys 's','a','r','s_next','done' (tensors on same device)
        networks and optimizers
        gamma, tau, beta as algorithm hyperparameters
        weight_clip: upper clamp for advantage weights
        q_target: target Q network for stabilization (EMA of q_net)
        target_update_rate: polyak averaging rate for target update
        reward_scale: scale rewards to prevent Q explosion
        train_policy: if False, skip policy update (for Q/V pretraining)

    Returns:
        dict of losses and metrics
    """
    s = batch['s']
    a = batch['a']
    r = batch['r'] * reward_scale  # 🔧 Scale rewards to prevent Q explosion
    s_next = batch['s_next']
    done = batch['done']

    # Q update (with optional target network)
    if train_q:
        q_opt.zero_grad()
        q_vals = q_net(s, a)
        with torch.no_grad():
            v_next = v_net(s_next)
            q_targets = r + gamma * v_next * (1 - done)
        q_loss = F.mse_loss(q_vals, q_targets)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        q_opt.step()
        
        # 🔧 Update target Q network if provided (Polyak averaging)
        if q_target is not None:
            with torch.no_grad():
                for param, target_param in zip(q_net.parameters(), q_target.parameters()):
                    target_param.data.copy_(
                        target_update_rate * param.data + (1 - target_update_rate) * target_param.data
                    )
    else:
        q_loss = torch.tensor(0.0, device=s.device)

    # V expectile update
    # 🔧 CRITICAL: No clipping on (Q-V) diff!
    # V must satisfy expectile constraint: P(Q>V) ≈ 1-tau
    if train_value:
        v_opt.zero_grad()
        with torch.no_grad():
            q_vals_detach = q_net(s, a).detach()  # Pure, unclipped Q values
        v_vals = v_net(s)
        # ✅ Pass (V, Q, tau) - NOT (Q-V, tau)!
        # This avoids "double sign flip" bugs
        v_loss = expectile_loss(v_vals, q_vals_detach, tau)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), 10.0)
        v_opt.step()
    else:
        v_loss = torch.tensor(0.0, device=s.device)

    # Policy update: advantage-weighted regression (AWR)
    # 🔧 Skip if train_policy=False (for Q/V pretraining)
    if not train_policy:
        pi_loss = torch.tensor(0.0, device=s.device)
        with torch.no_grad():
            adv = q_net(s, a) - v_net(s)
            weights = torch.zeros_like(adv)
    else:
        pi_opt.zero_grad()
        with torch.no_grad():
            adv = (q_net(s, a).detach() - v_net(s).detach())
            
            # 🔧 FIX 1: Clip raw advantages FIRST (critical!)
            adv = torch.clamp(adv, min=-10.0, max=10.0)
            
            # Compute advantage weights: w = exp(A/beta)
            weights = torch.exp(adv / beta)
            
            # 🔧 FIX 2: Clip weights (prevent explosion)
            weights = torch.clamp(weights, max=20.0)
    
        # Get policy distribution parameters
        mean, std = pi_net(s)
        
        # Compute log probability with numerical stability
        log_prob = -0.5 * ((a - mean) / (std + 1e-8)) ** 2 - torch.log(std + 1e-8) - 0.5 * torch.log(torch.tensor(2 * 3.14159265359))
        # Clamp to prevent extreme negative values
        log_prob = torch.clamp(log_prob, min=-50.0)
        
        # Weighted negative log-likelihood loss
        pi_loss = -(weights * log_prob).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(pi_net.parameters(), 10.0)
        pi_opt.step()

    # 🔧 监控指标：验证 expectile 约束是否满足
    with torch.no_grad():
        q_vals_monitor = q_net(s, a)
        v_vals_monitor = v_net(s)
        adv_monitor = q_vals_monitor - v_vals_monitor  # Unclipped raw (Q-V)
        
        # Expectile constraint validation
        positive_rate = (adv_monitor > 0).float().mean()
        # For tau=0.7, expect positive_rate ≈ 30% (1-tau)
        
        # Q-V distribution quantiles
        sorted_adv = torch.sort(adv_monitor.flatten())[0]
        q25 = sorted_adv[int(len(sorted_adv) * 0.25)]
        q50 = sorted_adv[int(len(sorted_adv) * 0.50)]
        q75 = sorted_adv[int(len(sorted_adv) * 0.75)]

    return {
        "q_loss": q_loss.detach(), 
        "v_loss": v_loss.detach(), 
        "pi_loss": pi_loss.detach(),
        "positive_rate": positive_rate.detach(),  # Target: ~1-tau = 30%
        "mean_adv": adv_monitor.mean().detach(),
        "std_adv": adv_monitor.std().detach(),
        "max_adv": adv_monitor.max().detach(),
        "min_adv": adv_monitor.min().detach(),
        "q25_adv": q25.detach(),
        "q50_adv": q50.detach(),
        "q75_adv": q75.detach(),
        "mean_weight": weights.mean().detach(),
        "max_weight": weights.max().detach()
    }
