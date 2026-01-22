# 动作空间说明

## 定义

- **变量名**: `totalamount_mg`
- **物理含义**: 4小时内万古霉素给药总剂量
- **单位**: mg (毫克)
- **类型**: 连续动作 (Continuous Action)
- **维度**: 1维

## 有效范围

### 物理范围
```
最小值: 0 mg     (不给药)
最大值: 2000 mg  (临床上限)
```

### 数据统计（训练集）
```
实际最小: 0 mg
实际最大: 1500 mg
均值:    72.4 mg
中位数:  0 mg     (超过一半的时间步不给药)
```

## 归一化处理

### 为什么需要归一化？
神经网络训练需要输入/输出在相似的尺度，否则：
- 梯度可能消失或爆炸
- 优化器难以收敛
- 不同特征的重要性失衡

### 归一化公式
```python
# 训练时
action_normalized = (action_mg - mean) / std

# 推理时（反归一化）
action_mg = action_normalized * std + mean

# 然后限制到 [0, 2000]
action_mg = clip(action_mg, 0, 2000)
```

### 归一化后的范围
假设从训练数据统计：mean ≈ 72.4, std ≈ 185

```
0 mg    -> -0.39 (归一化)
500 mg  ->  2.31 (归一化)
1000 mg ->  5.01 (归一化)
2000 mg -> 10.41 (归一化)
```

建议在策略网络中设置输出范围：`[-1, 11]` 或 `[-2, 12]`

## 在IQL中的使用

### 1. 训练阶段

```python
# dataset.py 中归一化
normalizer.fit(actions)  # 拟合均值和标准差
action_norm = normalizer.normalize(action_mg)

# 策略网络输出归一化的动作
mean, std = policy(state)  # mean 和 std 都是归一化空间的

# 损失计算时使用归一化的动作
log_prob = -0.5 * ((action_norm - mean) / std) ** 2 - ...
```

### 2. 推理阶段

```python
from algorithms.iql.action_utils import ActionNormalizer

# 1. 加载归一化器（从训练时保存的统计量）
normalizer = ActionNormalizer(action_min=0, action_max=2000)
normalizer.mean = 72.4  # 从训练集统计
normalizer.std = 185.0

# 2. 策略网络预测（归一化空间）
state_norm = normalizer_state.normalize(state)
action_norm_mean, action_norm_std = policy(state_norm)

# 3. 采样或使用均值
action_norm = action_norm_mean  # 确定性策略
# 或
action_norm = action_norm_mean + torch.randn_like(action_norm_mean) * action_norm_std  # 随机策略

# 4. 转换为实际剂量
action_mg = normalizer.process_policy_output(action_norm)
# 这会自动：反归一化 + 限制到 [0, 2000]

print(f"推荐剂量: {action_mg:.1f} mg")
```

## 策略网络设置

### 添加动作范围约束

```python
# models.py
policy = GaussianPolicy(
    state_dim=7,
    action_dim=1,
    hidden=(256, 256),
    action_range=(-2, 12)  # 归一化空间的合理范围
)
```

这样确保：
1. 网络输出的均值μ在合理范围内
2. 采样的动作也会被clip
3. 反归一化后不会出现极端异常值

## 临床约束

### 安全范围
- **单次最大剂量**: 通常 ≤ 2000 mg
- **每日最大剂量**: 视患者体重和肾功能而定
- **最小有效剂量**: 通常 ≥ 500 mg (如果给药的话)

### 建议
1. **0值处理**: 模型应该能够输出0（不给药）
2. **范围限制**: 推理时强制限制在 [0, 2000] mg
3. **临床验证**: 实际使用前需要医生审核

## 代码示例

### 完整推理流程

```python
import torch
from algorithms.iql.models import GaussianPolicy
from algorithms.iql.action_utils import ActionNormalizer

# 加载模型
policy = GaussianPolicy(state_dim=7, action_dim=1)
policy.load_state_dict(torch.load('model.pth'))
policy.eval()

# 初始化归一化器
action_normalizer = ActionNormalizer(action_min=0, action_max=2000)
action_normalizer.mean = 72.4
action_normalizer.std = 185.0

# 患者状态 (已归一化)
state = torch.tensor([[0.5, -0.3, 0.1, 0.2, -0.1, 0.4, 0.0]])

# 预测
with torch.no_grad():
    mean, std = policy(state)
    action_norm = mean  # 确定性策略
    
    # 转换为实际剂量
    action_mg = action_normalizer.process_policy_output(action_norm)
    
print(f"推荐给药剂量: {action_mg.item():.1f} mg")
```

## 总结

✅ **动作定义**: 0-2000 mg 万古霉素剂量  
✅ **训练**: 使用标准化后的值 (均值0, 标准差1)  
✅ **推理**: 反归一化 + clip到 [0, 2000]  
✅ **工具**: `algorithms/iql/action_utils.py`  
✅ **安全**: 自动限制在临床安全范围内
