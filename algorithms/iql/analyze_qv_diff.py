"""分析Q-V差异（即advantage）的统计量

这些指标反映IQL训练质量：
- mean(Q-V): 平均advantage，应接近正值（表示Q>V）
- std(Q-V): advantage的标准差，反映值函数的稳定性
- max(Q-V): 最大advantage，表示最优动作的优势
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from algorithms.iql.dataset import ReadyDataset
from algorithms.iql.models import QNetwork, VNetwork, GaussianPolicy


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(ckpt_path: str, state_dim: int, action_dim: int, hidden: list):
    """加载检查点并返回Q/V/Policy网络"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    q_net = QNetwork(state_dim, action_dim, hidden)
    v_net = VNetwork(state_dim, hidden)
    pi_net = GaussianPolicy(state_dim, action_dim, hidden)
    
    q_net.load_state_dict(ckpt['q_state'])
    v_net.load_state_dict(ckpt['v_state'])
    pi_net.load_state_dict(ckpt['pi_state'])
    
    q_net.eval()
    v_net.eval()
    pi_net.eval()
    
    return q_net, v_net, pi_net


def analyze_qv_difference(dataset: ReadyDataset, q_net, v_net):
    """计算Q-V统计量"""
    trans_df = dataset.to_transitions()
    
    # 转换为张量
    states = torch.FloatTensor(np.stack(trans_df['s'].values))
    actions = torch.FloatTensor(trans_df['a'].values).unsqueeze(1)
    
    with torch.no_grad():
        # 计算Q和V
        q_values = q_net(states, actions).squeeze()
        v_values = v_net(states).squeeze()
        
        # 计算差异
        advantage = q_values - v_values
        
        # 统计量
        mean_adv = advantage.mean().item()
        std_adv = advantage.std().item()
        max_adv = advantage.max().item()
        min_adv = advantage.min().item()
        median_adv = advantage.median().item()
        
        # 分位数
        q25 = torch.quantile(advantage, 0.25).item()
        q75 = torch.quantile(advantage, 0.75).item()
        
        # 正负占比
        positive_ratio = (advantage > 0).float().mean().item()
        
    return {
        'mean_Q_minus_V': mean_adv,
        'std_Q_minus_V': std_adv,
        'max_Q_minus_V': max_adv,
        'min_Q_minus_V': min_adv,
        'median_Q_minus_V': median_adv,
        'Q25_Q_minus_V': q25,
        'Q75_Q_minus_V': q75,
        'positive_ratio': positive_ratio,
        'n_samples': len(advantage)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='检查点文件路径')
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 加载数据集
    import pandas as pd
    df = pd.read_csv(cfg['data']['path'])
    dataset = ReadyDataset(
        df=df,
        state_cols=cfg['data']['state_cols']
    )
    
    state_dim = len(cfg['data']['state_cols'])
    action_dim = 1
    hidden = cfg['model']['hidden']
    
    # 加载模型
    print(f"Loading checkpoint: {args.checkpoint}")
    q_net, v_net, pi_net = load_checkpoint(args.checkpoint, state_dim, action_dim, hidden)
    
    # 分析Q-V差异
    print("\n=== Analyzing Q-V Statistics ===")
    stats = analyze_qv_difference(dataset, q_net, v_net)
    
    print(f"\nAdvantage (Q-V) Statistics:")
    print(f"  Mean(Q-V):     {stats['mean_Q_minus_V']:.4f}")
    print(f"  Std(Q-V):      {stats['std_Q_minus_V']:.4f}")
    print(f"  Max(Q-V):      {stats['max_Q_minus_V']:.4f}")
    print(f"  Min(Q-V):      {stats['min_Q_minus_V']:.4f}")
    print(f"  Median(Q-V):   {stats['median_Q_minus_V']:.4f}")
    print(f"  Q25(Q-V):      {stats['Q25_Q_minus_V']:.4f}")
    print(f"  Q75(Q-V):      {stats['Q75_Q_minus_V']:.4f}")
    print(f"  Positive Rate: {stats['positive_ratio']*100:.1f}%")
    print(f"  N Samples:     {stats['n_samples']}")
    
    # 解释
    print("\n=== Interpretation ===")
    if stats['mean_Q_minus_V'] > 0:
        print(f"✓ Mean(Q-V) > 0: Q值平均高于V值，符合预期")
    else:
        print(f"✗ Mean(Q-V) ≤ 0: Q值平均低于V值，训练可能有问题")
    
    if stats['std_Q_minus_V'] < 10:
        print(f"✓ Std(Q-V) < 10: Advantage分布较稳定")
    else:
        print(f"! Std(Q-V) = {stats['std_Q_minus_V']:.2f}: Advantage波动较大")
    
    if stats['max_Q_minus_V'] > 0:
        print(f"✓ Max(Q-V) = {stats['max_Q_minus_V']:.2f}: 存在正优势动作")
    
    print(f"\n对于IQL算法 (tau={cfg['model']['tau']}):")
    print(f"  - V ≈ {cfg['model']['tau']}-分位数 of Q")
    print(f"  - 期望 {cfg['model']['tau']*100:.0f}% 的(Q-V) > 0")
    print(f"  - 实际 {stats['positive_ratio']*100:.1f}% 的(Q-V) > 0")


if __name__ == '__main__':
    main()
