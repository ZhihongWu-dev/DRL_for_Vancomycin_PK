"""将训练好的IQL模型导出为单个.pth文件用于测试部署"""
import torch
import argparse
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.iql.models import QNetwork, VNetwork, GaussianPolicy


def export_model(checkpoint_path: str, output_path: str, state_dim: int = 7, 
                 action_dim: int = 1, hidden: list = None):
    """
    将IQL checkpoint导出为整合的单个模型文件
    
    Args:
        checkpoint_path: 训练checkpoint路径
        output_path: 输出.pth文件路径
        state_dim: 状态维度
        action_dim: 动作维度
        hidden: 隐藏层配置
    """
    if hidden is None:
        hidden = [32, 32]
    
    # 加载checkpoint
    print(f"加载checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 创建模型
    q_net = QNetwork(state_dim, action_dim, hidden)
    v_net = VNetwork(state_dim, hidden)
    pi_net = GaussianPolicy(state_dim, action_dim, hidden)
    
    # 加载权重
    q_net.load_state_dict(ckpt['q_state'])
    v_net.load_state_dict(ckpt['v_state'])
    pi_net.load_state_dict(ckpt['pi_state'])
    
    # 设置为评估模式
    q_net.eval()
    v_net.eval()
    pi_net.eval()
    
    # 整合模型和元数据（仅保存state_dict，不保存模型对象）
    export_dict = {
        'q_network': q_net.state_dict(),
        'v_network': v_net.state_dict(),
        'policy_network': pi_net.state_dict(),
        'model_config': {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_dims': hidden,
        },
        'training_info': {
            'step': ckpt.get('step', 'unknown'),
            'source_checkpoint': str(checkpoint_path),
        },
    }
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_dict, output_path)
    
    print(f"\n模型已导出到: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"\n模型配置:")
    print(f"  - 状态维度: {state_dim}")
    print(f"  - 动作维度: {action_dim}")
    print(f"  - 隐藏层: {hidden}")
    print(f"  - 训练步数: {ckpt.get('step', 'unknown')}")
    
    return export_dict


def load_exported_model(model_path: str, device='cpu'):
    """
    加载导出的模型用于推理
    
    Args:
        model_path: 导出的.pth文件路径
        device: 运行设备
        
    Returns:
        (q_net, v_net, pi_net, config)
    """
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # 从状态字典重建模型
    config = checkpoint['model_config']
    q_net = QNetwork(config['state_dim'], config['action_dim'], config['hidden_dims'])
    v_net = VNetwork(config['state_dim'], config['hidden_dims'])
    pi_net = GaussianPolicy(config['state_dim'], config['action_dim'], config['hidden_dims'])
    
    q_net.load_state_dict(checkpoint['q_network'])
    v_net.load_state_dict(checkpoint['v_network'])
    pi_net.load_state_dict(checkpoint['policy_network'])
    
    q_net.to(device)
    v_net.to(device)
    pi_net.to(device)
    
    q_net.eval()
    v_net.eval()
    pi_net.eval()
    
    print("模型加载成功!")
    return q_net, v_net, pi_net, checkpoint['model_config']


def test_exported_model(model_path: str):
    """测试导出的模型是否可以正常加载和推理"""
    import numpy as np
    
    q_net, v_net, pi_net, config = load_exported_model(model_path)
    
    # 创建测试输入
    state_dim = config['state_dim']
    action_dim = config['action_dim']
    
    test_state = torch.randn(1, state_dim)
    test_action = torch.randn(1, action_dim)
    
    print("\n测试推理:")
    with torch.no_grad():
        # 测试Q网络
        q_value = q_net(test_state, test_action)
        print(f"  Q(s,a) = {q_value.item():.4f}")
        
        # 测试V网络
        v_value = v_net(test_state)
        print(f"  V(s) = {v_value.item():.4f}")
        
        # 测试策略网络
        action_mean, action_std = pi_net(test_state)
        print(f"  π(s): mean = {action_mean.item():.4f}, std = {action_std.item():.4f}")
        
        # 采样动作
        action_sample = pi_net.sample(test_state)
        print(f"  采样动作 = {action_sample.item():.4f}")
    
    print("\n✓ 模型测试通过!")


def main():
    parser = argparse.ArgumentParser(description='导出IQL模型')
    parser.add_argument('--checkpoint', type=str, 
                       default='algorithms/iql/runs/exp_conservative/ckpt_step3000.pt',
                       help='训练checkpoint路径')
    parser.add_argument('--output', type=str,
                       default='algorithms/iql/exported_models/iql_model.pth',
                       help='输出模型路径')
    parser.add_argument('--state-dim', type=int, default=7,
                       help='状态维度')
    parser.add_argument('--action-dim', type=int, default=1,
                       help='动作维度')
    parser.add_argument('--hidden', type=int, nargs='+', default=[32, 32],
                       help='隐藏层配置')
    parser.add_argument('--test', action='store_true',
                       help='测试导出的模型')
    
    args = parser.parse_args()
    
    # 导出模型
    export_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden=args.hidden
    )
    
    # 测试模型
    if args.test:
        print("\n" + "="*80)
        test_exported_model(args.output)


if __name__ == '__main__':
    main()
