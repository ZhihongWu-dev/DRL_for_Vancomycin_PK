"""查看训练日志（JSON格式）并绘制曲线"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import argparse


def plot_training_log(log_file, save_path=None):
    """读取并绘制JSON格式的训练日志"""
    
    # 读取日志
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    # 按tag分组
    metrics = defaultdict(lambda: {'steps': [], 'values': []})
    for entry in data:
        tag = entry['tag']
        metrics[tag]['steps'].append(entry['step'])
        metrics[tag]['values'].append(entry['value'])
    
    if not metrics:
        print("没有找到训练数据")
        return
    
    # 绘图
    sns.set_style('whitegrid')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, (tag, data) in zip(axes, sorted(metrics.items())):
        ax.plot(data['steps'], data['values'], linewidth=2, color='steelblue')
        ax.set_xlabel('训练步数', fontsize=12)
        ax.set_ylabel('损失值', fontsize=12)
        ax.set_title(tag, fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 显示统计信息
        import numpy as np
        values = np.array(data['values'])
        text = f"最终: {values[-1]:.4f}\n最小: {values.min():.4f}\n均值: {values.mean():.4f}"
        ax.text(0.98, 0.98, text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()
    
    # 打印摘要
    print("\n训练摘要:")
    print("="*60)
    for tag, data in sorted(metrics.items()):
        import numpy as np
        values = np.array(data['values'])
        print(f"{tag}:")
        print(f"  最终值: {values[-1]:.6f}")
        print(f"  最小值: {values.min():.6f} (步数 {data['steps'][values.argmin()]})")
        print(f"  最大值: {values.max():.6f} (步数 {data['steps'][values.argmax()]})")
        print(f"  均值: {values.mean():.6f}")
        print()


def main():
    parser = argparse.ArgumentParser(description='查看训练日志')
    parser.add_argument('--log-file', type=str, required=True,
                       help='训练日志JSON文件路径')
    parser.add_argument('--save', type=str, default=None,
                       help='保存图表的路径')
    
    args = parser.parse_args()
    
    if not Path(args.log_file).exists():
        print(f"错误: 日志文件不存在 - {args.log_file}")
        return
    
    plot_training_log(args.log_file, args.save)


if __name__ == '__main__':
    main()
