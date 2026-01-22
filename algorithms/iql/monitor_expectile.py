"""实时监控 IQL 训练的 Expectile 约束满足情况"""
import time
import json
import os
import sys

def monitor_training(log_file, tau=0.7, check_interval=5):
    """监控训练日志，验证 Positive Rate 是否符合 expectile 约束
    
    Args:
        log_file: training_log.json 文件路径
        tau: expectile 参数（默认 0.7）
        check_interval: 检查间隔（秒）
    """
    target_positive = (1 - tau) * 100  # 期望的 Positive%
    
    print(f"监控训练进度...")
    print(f"tau={tau} → 目标 Positive% ≈ {target_positive:.0f}%")
    print(f"期望范围: [{target_positive-5:.0f}%, {target_positive+5:.0f}%]")
    print("=" * 70)
    
    last_step = 0
    
    while True:
        try:
            if not os.path.exists(log_file):
                time.sleep(check_interval)
                continue
            
            with open(log_file, 'r') as f:
                metrics = json.load(f)
            
            # 提取最新的 positive_rate
            pos_rates = [m for m in metrics if m['tag'] == 'metrics/positive_rate']
            if not pos_rates:
                time.sleep(check_interval)
                continue
            
            latest = pos_rates[-1]
            step = latest['step']
            pos_rate = latest['value'] * 100
            
            if step > last_step:
                # 检查是否满足约束
                delta = abs(pos_rate - target_positive)
                status = "✓" if delta < 5 else "✗"
                
                # 提取对应步骤的 Q-V 统计
                q25_vals = [m for m in metrics if m['tag'] == 'metrics/q25_adv' and m['step'] == step]
                q50_vals = [m for m in metrics if m['tag'] == 'metrics/q50_adv' and m['step'] == step]
                q75_vals = [m for m in metrics if m['tag'] == 'metrics/q75_adv' and m['step'] == step]
                
                if q25_vals and q50_vals and q75_vals:
                    q25 = q25_vals[0]['value']
                    q50 = q50_vals[0]['value']
                    q75 = q75_vals[0]['value']
                    
                    # 验证分位数是否合理（对于 tau=0.7，Q70 应该 ≈ 0）
                    q70_ok = "✓" if abs(q75) < 3 else "✗"
                    
                    print(f"[Step {step:5d}] Pos%={pos_rate:5.1f}% {status} | "
                          f"Q-V: Q25={q25:6.2f} Q50={q50:6.2f} Q75={q75:6.2f} {q70_ok}")
                else:
                    print(f"[Step {step:5d}] Pos%={pos_rate:5.1f}% {status}")
                
                last_step = step
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception as e:
            print(f"错误: {e}")
            time.sleep(check_interval)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        workdir = sys.argv[1]
    else:
        workdir = "algorithms/iql/runs/exp_fix_expectile"
    
    log_file = os.path.join(workdir, "training_log.json")
    monitor_training(log_file)
