"""使用导出的IQL模型进行推理测试"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.iql.export_model import load_exported_model


def predict_action(state, q_net, v_net, pi_net, use_greedy=True):
    """
    使用IQL模型预测动作
    
    Args:
        state: 状态张量 shape=(state_dim,)
        q_net: Q网络
        v_net: V网络
        pi_net: 策略网络
        use_greedy: 是否使用贪心策略（最大化Q值）
        
    Returns:
        predicted_action: 预测的动作
    """
    state = torch.FloatTensor(state).unsqueeze(0)  # (1, state_dim)
    
    with torch.no_grad():
        if use_greedy:
            # 贪心策略：在动作空间中搜索最大Q值
            action_range = torch.linspace(-1, 1, 100).unsqueeze(1)  # (100, 1)
            state_batch = state.expand(100, -1)  # (100, state_dim)
            
            q_values = q_net(state_batch, action_range)  # (100, 1)
            best_idx = q_values.argmax()
            action = action_range[best_idx].item()
        else:
            # 随机策略：从策略网络采样
            action = pi_net.sample(state).item()
    
    return action


def evaluate_on_dataset(model_path, data_path, state_cols, use_greedy=True, num_samples=None):
    """
    在数据集上评估模型
    
    Args:
        model_path: 导出的模型路径
        data_path: 数据CSV路径
        state_cols: 状态列名
        use_greedy: 是否使用贪心策略
        num_samples: 评估样本数（None表示全部）
    """
    # 加载模型
    q_net, v_net, pi_net, config = load_exported_model(model_path)
    
    # 加载数据
    df = pd.read_csv(data_path)
    if num_samples:
        df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    print(f"\n评估数据集: {len(df)} 条样本")
    print(f"策略类型: {'贪心策略' if use_greedy else '随机策略'}")
    
    # 预测
    predictions = []
    v_values = []
    
    for idx, row in df.iterrows():
        state = row[state_cols].values.astype(np.float32)
        
        # 预测动作
        action = predict_action(state, q_net, v_net, pi_net, use_greedy)
        predictions.append(action)
        
        # 计算状态价值
        with torch.no_grad():
            v = v_net(torch.FloatTensor(state).unsqueeze(0)).item()
        v_values.append(v)
    
    # 统计
    predictions = np.array(predictions)
    v_values = np.array(v_values)
    
    print(f"\n预测动作统计:")
    print(f"  均值: {predictions.mean():.4f}")
    print(f"  标准差: {predictions.std():.4f}")
    print(f"  最小值: {predictions.min():.4f}")
    print(f"  最大值: {predictions.max():.4f}")
    
    print(f"\n状态价值统计:")
    print(f"  均值: {v_values.mean():.4f}")
    print(f"  标准差: {v_values.std():.4f}")
    print(f"  最小值: {v_values.min():.4f}")
    print(f"  最大值: {v_values.max():.4f}")
    
    return predictions, v_values


def clinical_test_cases():
    """临床测试案例"""
    model_path = 'algorithms/iql/exported_models/iql_model.pth'
    q_net, v_net, pi_net, config = load_exported_model(model_path)
    
    # 定义测试案例（标准化后的值）
    test_cases = [
        {
            'name': '正常患者',
            'state': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 全部为中位数
        },
        {
            'name': '高万古霉素浓度',
            'state': [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # vanco_level高
        },
        {
            'name': '肾功能不全',
            'state': [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # creatinine高
        },
        {
            'name': '感染严重',
            'state': [0.0, 0.0, 3.0, 0.0, 1.5, 0.0, 1.0],  # WBC高、发热、心率快
        },
    ]
    
    print("\n临床测试案例:")
    print("=" * 80)
    
    for case in test_cases:
        state = np.array(case['state'], dtype=np.float32)
        
        # 贪心动作
        greedy_action = predict_action(state, q_net, v_net, pi_net, use_greedy=True)
        
        # 状态价值
        with torch.no_grad():
            v = v_net(torch.FloatTensor(state).unsqueeze(0)).item()
        
        print(f"\n{case['name']}:")
        print(f"  推荐动作（标准化）: {greedy_action:.4f}")
        print(f"  状态价值 V(s): {v:.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='测试IQL模型')
    parser.add_argument('--model', type=str, 
                       default='algorithms/iql/exported_models/iql_model.pth',
                       help='模型路径')
    parser.add_argument('--data', type=str,
                       default='intermediate_data/ready_data.csv',
                       help='测试数据路径')
    parser.add_argument('--greedy', action='store_true', default=True,
                       help='使用贪心策略')
    parser.add_argument('--samples', type=int, default=100,
                       help='测试样本数')
    parser.add_argument('--clinical', action='store_true',
                       help='运行临床测试案例')
    
    args = parser.parse_args()
    
    if args.clinical:
        clinical_test_cases()
    else:
        state_cols = [
            'vanco_level(ug/mL)',
            'creatinine(mg/dL)',
            'wbc(K/uL)',
            'bun(mg/dL)',
            'temperature',
            'sbp',
            'heart_rate'
        ]
        evaluate_on_dataset(args.model, args.data, state_cols, 
                          use_greedy=args.greedy, num_samples=args.samples)
