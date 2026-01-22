"""动作空间处理工具

处理动作的归一化、反归一化和范围限制
"""
import torch
import numpy as np


class ActionNormalizer:
    """处理动作的归一化和反归一化
    
    万古霉素剂量范围：0-2000 mg
    训练时使用标准化后的值，推理时需要转换回实际剂量
    """
    
    def __init__(self, action_min=0.0, action_max=2000.0):
        """
        Args:
            action_min: 最小动作值 (mg)
            action_max: 最大动作值 (mg)
        """
        self.action_min = action_min
        self.action_max = action_max
        
        # 统计量（从训练数据拟合）
        self.mean = None
        self.std = None
    
    def fit(self, actions):
        """从训练数据拟合归一化参数
        
        Args:
            actions: 原始动作数组 (未归一化的mg值)
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        self.mean = float(np.mean(actions))
        self.std = float(np.std(actions))
        if self.std == 0:
            self.std = 1.0
        
        print(f"ActionNormalizer fitted: mean={self.mean:.2f}, std={self.std:.2f}")
    
    def normalize(self, action):
        """归一化动作 (mg -> 标准化值)
        
        Args:
            action: 原始动作值 (mg)
            
        Returns:
            归一化后的动作值
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Must call fit() before normalize()")
        
        if isinstance(action, np.ndarray):
            return (action - self.mean) / self.std
        elif isinstance(action, torch.Tensor):
            return (action - self.mean) / self.std
        else:
            return (action - self.mean) / self.std
    
    def denormalize(self, action_normalized):
        """反归一化动作 (标准化值 -> mg)
        
        Args:
            action_normalized: 归一化的动作值
            
        Returns:
            原始动作值 (mg)
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Must call fit() before denormalize()")
        
        if isinstance(action_normalized, torch.Tensor):
            action = action_normalized * self.std + self.mean
        else:
            action = action_normalized * self.std + self.mean
        
        return action
    
    def clip_to_valid_range(self, action):
        """将动作限制在有效范围内 [0, 2000] mg
        
        Args:
            action: 动作值 (mg)
            
        Returns:
            限制后的动作值
        """
        if isinstance(action, torch.Tensor):
            return torch.clamp(action, self.action_min, self.action_max)
        elif isinstance(action, np.ndarray):
            return np.clip(action, self.action_min, self.action_max)
        else:
            return max(self.action_min, min(action, self.action_max))
    
    def process_policy_output(self, action_normalized):
        """处理策略网络输出：反归一化 + 限制范围
        
        Args:
            action_normalized: 策略网络输出的归一化动作
            
        Returns:
            实际可用的动作值 (mg)，范围 [0, 2000]
        """
        action = self.denormalize(action_normalized)
        action = self.clip_to_valid_range(action)
        return action


def get_normalized_action_range(action_min=0.0, action_max=2000.0, 
                                  data_mean=72.4, data_std=185.0):
    """计算归一化后的动作范围
    
    用于在策略网络中设置合理的输出范围
    
    Args:
        action_min: 最小动作 (mg)
        action_max: 最大动作 (mg)
        data_mean: 数据均值 (从训练集统计)
        data_std: 数据标准差 (从训练集统计)
    
    Returns:
        (norm_min, norm_max): 归一化后的范围
    
    Example:
        >>> norm_min, norm_max = get_normalized_action_range()
        >>> print(f"归一化范围: [{norm_min:.2f}, {norm_max:.2f}]")
        归一化范围: [-0.39, 10.41]
    """
    norm_min = (action_min - data_mean) / data_std
    norm_max = (action_max - data_mean) / data_std
    return norm_min, norm_max


if __name__ == "__main__":
    # 测试
    print("="*60)
    print("动作归一化测试")
    print("="*60)
    
    # 模拟训练数据
    actions_train = np.array([0, 0, 0, 500, 750, 1000, 1500, 0, 0])
    
    normalizer = ActionNormalizer(action_min=0, action_max=2000)
    normalizer.fit(actions_train)
    
    print("\n测试归一化:")
    test_actions = [0, 500, 1000, 1500, 2000]
    for a in test_actions:
        a_norm = normalizer.normalize(a)
        a_denorm = normalizer.denormalize(a_norm)
        print(f"  {a} mg -> {a_norm:.3f} (归一化) -> {a_denorm:.1f} mg (反归一化)")
    
    print("\n测试范围限制:")
    test_values = [-500, 0, 1000, 2500, 3000]
    for v in test_values:
        clipped = normalizer.clip_to_valid_range(v)
        print(f"  {v} mg -> {clipped} mg (限制后)")
    
    print("\n归一化动作范围:")
    norm_min, norm_max = get_normalized_action_range(
        data_mean=normalizer.mean, 
        data_std=normalizer.std
    )
    print(f"  [0, 2000] mg -> [{norm_min:.2f}, {norm_max:.2f}] (归一化)")
    
    print("\n✅ 测试完成")
