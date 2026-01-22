"""Training scaffold for IQL (skeleton)

Usage: python train_iql.py --config configs/iql_base.yaml
"""
import argparse
import yaml
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


import os
import time
import json
from algorithms.iql.dataset import ReadyDataset, ReplayBuffer
from algorithms.iql.models import QNetwork, VNetwork, GaussianPolicy, init_model_weights
from algorithms.iql.train_utils import iql_update_step
from algorithms.iql.utils import set_seed, get_device


class SimpleLogger:
    """Simple JSON logger to replace TensorBoard"""
    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.metrics = []
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, "training_log.json")
        else:
            self.log_file = None
    
    def add_scalar(self, tag, value, step):
        self.metrics.append({"step": step, "tag": tag, "value": float(value)})
    
    def close(self):
        if self.log_file:
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            print(f"Training log saved to: {self.log_file}")


def run_training(cfg: dict, workdir: str = None) -> str:
    """Run a short training job using cfg dict. Returns checkpoint path."""
    # config parsing with defaults
    seed = cfg.get("train", {}).get("seed", 0)
    set_seed(seed)
    device = get_device()

    # data
    data_cfg = cfg.get("data", {})
    if "dataframe" in data_cfg and data_cfg["dataframe"] is not None:
        df = data_cfg["dataframe"]
        state_cols = data_cfg.get("state_cols")
        ds = ReadyDataset.from_dataframe(df, state_cols=state_cols)
    else:
        path = data_cfg.get("path")
        sheet = data_cfg.get("sheet", 0)
        state_cols = data_cfg.get("state_cols")
        if path is None:
            raise ValueError("No data.path or data.dataframe provided in cfg")
        import pandas as pd
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            ds = ReadyDataset.from_dataframe(df, state_cols=state_cols)
        elif path.endswith(".xlsx") or path.endswith(".xls"):
            df = pd.read_excel(path, sheet_name=sheet)
            ds = ReadyDataset.from_dataframe(df, state_cols=state_cols)
        else:
            raise ValueError("Unsupported data file type: must be .csv or .xlsx")

    ds.fit_normalizer()
    trans = ds.to_transitions(normalize=True)

    # replay buffer
    buf = ReplayBuffer(capacity=cfg.get("train", {}).get("buffer_capacity", max(1000, len(trans) * 10)))
    buf.add_batch(trans)

    # model and optimizer setup
    model_cfg = cfg.get("model", {})
    state_dim = len(ds.state_cols)
    hidden = tuple(model_cfg.get("hidden", (256, 256)))
    q_net = QNetwork(state_dim, action_dim=1, hidden=hidden).to(device)
    v_net = VNetwork(state_dim, hidden=hidden).to(device)
    pi = GaussianPolicy(state_dim, action_dim=1, hidden=hidden).to(device)

    init_model_weights(q_net)
    init_model_weights(v_net)
    init_model_weights(pi)
    
    # 🔧 Create target Q network for stabilization
    use_target_q = model_cfg.get("use_target_q", False)
    if use_target_q:
        import copy
        q_target = copy.deepcopy(q_net).to(device)
        q_target.eval()
        for param in q_target.parameters():
            param.requires_grad = False
    else:
        q_target = None

    # 🔧 Separate learning rates: V should learn slower to prevent overshooting
    lr = model_cfg.get("lr", 3e-4)
    v_lr_ratio = model_cfg.get("v_lr_ratio", 0.3)  # V lr = 0.3 * Q lr by default
    v_lr = lr * v_lr_ratio
    
    print(f"Learning rates: Q/Pi={lr:.6f}, V={v_lr:.6f} (ratio={v_lr_ratio})")
    
    q_opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
    pi_opt = torch.optim.Adam(pi.parameters(), lr=lr)

    total_steps = cfg.get("train", {}).get("total_steps", 100)
    batch_size = cfg.get("train", {}).get("batch_size", 256)
    log_interval = cfg.get("train", {}).get("log_interval", 10)
    ckpt_interval = cfg.get("train", {}).get("ckpt_interval", 50)
    
    # 🔧 Policy freezing for Q/V pretraining
    policy_warmup_steps = cfg.get("train", {}).get("policy_warmup_steps", 0)
    disable_policy = cfg.get("train", {}).get("disable_policy", False)
    # 🔧 Q freezing for V alignment experiments
    q_warmup_steps = cfg.get("train", {}).get("q_warmup_steps", 0)

    # logging
    workdir = workdir or cfg.get("workdir", f"runs/iql_{int(time.time())}")
    os.makedirs(workdir, exist_ok=True)
    logger = SimpleLogger(log_dir=workdir)

    # training loop
    pos_rate_window = cfg.get("train", {}).get("pos_rate_smooth_window", 5)
    pos_rate_hist = []
    policy_enabled = False
    policy_enable_min_steps = cfg.get("train", {}).get("policy_enable_min_steps", 0)
    policy_enable_patience = cfg.get("train", {}).get("policy_enable_patience", 3)
    policy_enable_pos_ma_min = cfg.get("train", {}).get("policy_enable_pos_ma_min")
    policy_enable_pos_ma_max = cfg.get("train", {}).get("policy_enable_pos_ma_max")
    policy_enable_pos_ma_margin = cfg.get("train", {}).get("policy_enable_pos_ma_margin", 0.05)
    policy_enable_streak = 0
    tau_val = model_cfg.get("tau", 0.7)
    target_pos = 1.0 - tau_val
    if policy_enable_pos_ma_min is None:
        policy_enable_pos_ma_min = max(0.0, target_pos - policy_enable_pos_ma_margin)
    if policy_enable_pos_ma_max is None:
        policy_enable_pos_ma_max = min(1.0, target_pos + policy_enable_pos_ma_margin)
    for step in range(1, total_steps + 1):
        batch = buf.sample(min(batch_size, len(buf)), seed=int(time.time()) % (2 ** 31))
        # move to device
        for k in batch:
            batch[k] = batch[k].to(device)
        
        # 🔧 Determine if policy/Q should train this step
        train_policy = (not disable_policy) and policy_enabled and (step > policy_warmup_steps)
        train_q = (step > q_warmup_steps)

        losses = iql_update_step(batch, q_net, v_net, pi, q_opt, v_opt, pi_opt,
                                 gamma=model_cfg.get("gamma", 0.99),
                                 tau=model_cfg.get("tau", 0.7),
                                 beta=model_cfg.get("beta", 3.0),
                                 weight_clip=model_cfg.get("weight_clip", 1e2),
                                 q_target=q_target,
                                 target_update_rate=model_cfg.get("target_update_rate", 0.005),
                                 reward_scale=model_cfg.get("reward_scale", 1.0),
                                 train_policy=train_policy,
                                 train_q=train_q)

        if step % log_interval == 0 or step == 1:
            q_loss = losses['q_loss'].item()
            v_loss = losses['v_loss'].item()
            pi_loss = losses['pi_loss'].item()
            pos_rate = losses['positive_rate'].item()
            pos_rate_hist.append(pos_rate)
            if pos_rate_window and len(pos_rate_hist) > pos_rate_window:
                pos_rate_hist = pos_rate_hist[-pos_rate_window:]
            pos_rate_ma = sum(pos_rate_hist) / max(1, len(pos_rate_hist))
            mean_adv = losses['mean_adv'].item()
            std_adv = losses['std_adv'].item()
            max_adv = losses['max_adv'].item()
            min_adv = losses['min_adv'].item()
            q25_adv = losses['q25_adv'].item()
            q50_adv = losses['q50_adv'].item()
            q75_adv = losses['q75_adv'].item()
            
            policy_status = "FROZEN" if step <= policy_warmup_steps else "active"
            q_status = "FROZEN" if step <= q_warmup_steps else "active"
            
            # Expectile constraint validation (for tau=0.7, expect ~30% positive)
            target_positive = (1 - model_cfg.get("tau", 0.7)) * 100
            positive_ok = "[OK]" if abs(pos_rate*100 - target_positive) < 10 else "[!!]"
            
            print(f"[step {step}] q_loss={q_loss:.6f} v_loss={v_loss:.6f} pi_loss={pi_loss:.6f}")
            print(f"  pos%={pos_rate*100:.1f}% (ma{pos_rate_window}={pos_rate_ma*100:.1f}%) {positive_ok}(target {target_positive:.0f}%) | "
                  f"Q-V: mu={mean_adv:.2f} sd={std_adv:.2f} [{min_adv:.1f}, {max_adv:.1f}]")
            print(f"  Q-V quantiles: Q25={q25_adv:.2f} Q50={q50_adv:.2f} Q75={q75_adv:.2f} | Q:{q_status} pi:{policy_status}")

            # 🔧 Enable policy only when Positive% moving average is stable
            if (not disable_policy) and (not policy_enabled) and (step > policy_warmup_steps) and (step >= policy_enable_min_steps):
                if policy_enable_pos_ma_min <= pos_rate_ma <= policy_enable_pos_ma_max:
                    policy_enable_streak += 1
                else:
                    policy_enable_streak = 0
                if policy_enable_streak >= policy_enable_patience:
                    policy_enabled = True
                    print(
                        f"Policy enabled: pos%_ma={pos_rate_ma*100:.1f}% within "
                        f"[{policy_enable_pos_ma_min*100:.1f}%, {policy_enable_pos_ma_max*100:.1f}%] "
                        f"for {policy_enable_patience} logs"
                    )
            
            logger.add_scalar("loss/q", q_loss, step)
            logger.add_scalar("loss/v", v_loss, step)
            logger.add_scalar("loss/pi", pi_loss, step)
            logger.add_scalar("metrics/positive_rate", pos_rate, step)
            logger.add_scalar("metrics/positive_rate_ma", pos_rate_ma, step)
            logger.add_scalar("metrics/mean_adv", mean_adv, step)
            logger.add_scalar("metrics/std_adv", std_adv, step)
            logger.add_scalar("metrics/max_adv", max_adv, step)
            logger.add_scalar("metrics/min_adv", min_adv, step)
            logger.add_scalar("metrics/q25_adv", q25_adv, step)
            logger.add_scalar("metrics/q50_adv", q50_adv, step)
            logger.add_scalar("metrics/q75_adv", q75_adv, step)
            logger.add_scalar("metrics/mean_weight", losses['mean_weight'].item(), step)
            logger.add_scalar("metrics/max_weight", losses['max_weight'].item(), step)

        if step % ckpt_interval == 0 or step == total_steps:
            ckpt_path = os.path.join(workdir, f"ckpt_step{step}.pt")
            torch.save({
                "step": step,
                "q_state": q_net.state_dict(),
                "v_state": v_net.state_dict(),
                "pi_state": pi.state_dict(),
                "q_opt": q_opt.state_dict(),
                "v_opt": v_opt.state_dict(),
                "pi_opt": pi_opt.state_dict(),
                "cfg": cfg,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    logger.close()
    return ckpt_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    print("Config loaded:", cfg)
    ckpt = run_training(cfg)
    print("Training finished, last checkpoint:", ckpt)


if __name__ == "__main__":
    main()
