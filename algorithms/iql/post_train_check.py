"""Post-training checks for IQL.

Outputs:
- Positive% (Q-V > 0)
- Q-V distribution stats (mean/std/min/max/quantiles)
- Policy return metrics (greedy Q, MC return, mean reward)

Usage:
  python algorithms/iql/post_train_check.py --config configs/iql_fix_expectile.yaml
  python algorithms/iql/post_train_check.py --config configs/iql_fix_expectile.yaml --checkpoint path/to/ckpt.pt
  python algorithms/iql/post_train_check.py --config configs/iql_fix_expectile.yaml --workdir algorithms/iql/runs/exp_fix_expectile
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml

from algorithms.iql.dataset import ReadyDataset
from algorithms.iql.models import QNetwork, VNetwork, GaussianPolicy
from algorithms.iql.evaluate_iql import evaluate_offline


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(ckpt_path: str, state_dim: int, action_dim: int, hidden: list):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    q_net = QNetwork(state_dim, action_dim, hidden)
    v_net = VNetwork(state_dim, hidden)
    pi_net = GaussianPolicy(state_dim, action_dim, hidden)

    q_net.load_state_dict(ckpt["q_state"])
    v_net.load_state_dict(ckpt["v_state"])
    pi_net.load_state_dict(ckpt["pi_state"])

    q_net.eval()
    v_net.eval()
    pi_net.eval()

    return q_net, v_net, pi_net


def find_latest_checkpoint(workdir: Path) -> Path:
    ckpts = list(workdir.glob("ckpt_step*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in: {workdir}")

    def step_num(p: Path) -> int:
        name = p.stem
        try:
            return int(name.replace("ckpt_step", ""))
        except ValueError:
            return -1

    ckpts.sort(key=step_num)
    return ckpts[-1]


def compute_qv_stats(dataset: ReadyDataset, q_net, v_net, normalize_states: bool = True) -> Dict[str, float]:
    trans_df = dataset.to_transitions(normalize=normalize_states)

    states = torch.FloatTensor(np.stack(trans_df["s"].values))
    actions = torch.FloatTensor(trans_df["a"].values).unsqueeze(1)

    with torch.no_grad():
        q_values = q_net(states, actions).squeeze()
        v_values = v_net(states).squeeze()
        advantage = q_values - v_values

        mean_adv = advantage.mean().item()
        std_adv = advantage.std().item()
        min_adv = advantage.min().item()
        max_adv = advantage.max().item()
        q25 = torch.quantile(advantage, 0.25).item()
        q50 = torch.quantile(advantage, 0.50).item()
        q75 = torch.quantile(advantage, 0.75).item()
        positive_rate = (advantage > 0).float().mean().item()

    return {
        "mean_q_minus_v": mean_adv,
        "std_q_minus_v": std_adv,
        "min_q_minus_v": min_adv,
        "max_q_minus_v": max_adv,
        "q25_q_minus_v": q25,
        "q50_q_minus_v": q50,
        "q75_q_minus_v": q75,
        "positive_rate": positive_rate,
        "num_transitions": len(trans_df),
    }


def print_report(ckpt_path: Path, qv_stats: Dict[str, float], eval_results: Dict[str, object], tau: float) -> None:
    print("=" * 72)
    print("IQL Post-Training Check")
    print("=" * 72)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Transitions: {qv_stats['num_transitions']}")

    print("\nQ-V distribution:")
    print(f"  Positive%: {qv_stats['positive_rate'] * 100:.1f}% (target ~{(1 - tau) * 100:.1f}%)")
    print(f"  Mean(Q-V): {qv_stats['mean_q_minus_v']:.4f}")
    print(f"  Std(Q-V):  {qv_stats['std_q_minus_v']:.4f}")
    print(f"  Min/Max:   {qv_stats['min_q_minus_v']:.4f} / {qv_stats['max_q_minus_v']:.4f}")
    print(f"  Q25/Q50/Q75: {qv_stats['q25_q_minus_v']:.4f} / {qv_stats['q50_q_minus_v']:.4f} / {qv_stats['q75_q_minus_v']:.4f}")

    print("\nPolicy return (offline):")
    print(f"  Greedy Q:  {eval_results['greedy_q']:.4f}")
    print(f"  MC Return: {eval_results['mc_return']:.4f}")
    print(f"  Mean reward: {eval_results['mean_reward']:.4f}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training check for IQL")
    parser.add_argument("--config", type=str, required=True, help="训练配置文件")
    parser.add_argument("--checkpoint", type=str, default=None, help="指定checkpoint路径")
    parser.add_argument("--workdir", type=str, default=None, help="训练输出目录（包含ckpt_step*.pt）")
    parser.add_argument("--output", type=str, default=None, help="可选，保存结果为JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)

    workdir = Path(args.workdir) if args.workdir else Path(cfg.get("workdir", ""))
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        if not workdir:
            raise ValueError("Must provide --checkpoint or a config with workdir")
        ckpt_path = find_latest_checkpoint(workdir)

    import pandas as pd
    df = pd.read_csv(cfg["data"]["path"])
    dataset = ReadyDataset(df=df, state_cols=cfg["data"]["state_cols"])
    dataset.fit_normalizer()

    state_dim = len(cfg["data"]["state_cols"])
    action_dim = 1
    hidden = cfg["model"]["hidden"]

    q_net, v_net, pi_net = load_checkpoint(str(ckpt_path), state_dim, action_dim, hidden)

    qv_stats = compute_qv_stats(dataset, q_net, v_net, normalize_states=True)
    eval_results = evaluate_offline(dataset, q_net, v_net, pi_net, cfg["model"]["gamma"], normalize_states=True)

    print_report(ckpt_path, qv_stats, eval_results, cfg["model"]["tau"])

    if args.output:
        payload = {
            "checkpoint": str(ckpt_path),
            "qv_stats": qv_stats,
            "eval_results": eval_results,
            "tau": cfg["model"]["tau"],
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
