# IQL模型导出和测试指南

本目录包含训练好的IQL模型的导出和测试工具。

## 文件说明

- `export_model.py`: 将训练checkpoint导出为单个.pth文件
- `test_model.py`: 使用导出的模型进行推理测试
- `exported_models/iql_model.pth`: 导出的模型文件（22KB）

## 快速开始

### 1. 导出模型

从训练checkpoint导出模型：

```bash
python algorithms/iql/export_model.py \
    --checkpoint algorithms/iql/runs/exp_conservative/ckpt_step3000.pt \
    --output algorithms/iql/exported_models/iql_model.pth \
    --test
```

参数说明：
- `--checkpoint`: 训练checkpoint路径
- `--output`: 输出模型文件路径
- `--state-dim`: 状态维度（默认7）
- `--action-dim`: 动作维度（默认1）
- `--hidden`: 隐藏层配置（默认[32, 32]）
- `--test`: 导出后测试模型

### 2. 测试模型

#### 在数据集上评估

```bash
python algorithms/iql/test_model.py \
    --model algorithms/iql/exported_models/iql_model.pth \
    --data intermediate_data/ready_data.csv \
    --greedy \
    --samples 100
```

#### 运行临床测试案例

```bash
python algorithms/iql/test_model.py --clinical
```

## 在代码中使用模型

### 加载模型

```python
from algorithms.iql.export_model import load_exported_model

# 加载模型
q_net, v_net, pi_net, config = load_exported_model(
    'algorithms/iql/exported_models/iql_model.pth',
    device='cpu'
)
```

### 预测动作

```python
import torch
import numpy as np

# 准备状态（7维）
state = np.array([
    vanco_level,   # 万古霉素浓度
    creatinine,    # 肌酐
    wbc,          # 白细胞计数
    bun,          # 尿素氮
    temperature,   # 体温
    sbp,          # 收缩压
    heart_rate    # 心率
], dtype=np.float32)

state_tensor = torch.FloatTensor(state).unsqueeze(0)

# 贪心策略：搜索最大Q值的动作
with torch.no_grad():
    action_range = torch.linspace(-1, 1, 100).unsqueeze(1)
    state_batch = state_tensor.expand(100, -1)
    q_values = q_net(state_batch, action_range)
    best_action = action_range[q_values.argmax()].item()

# 随机策略：从策略网络采样
with torch.no_grad():
    sampled_action = pi_net.sample(state_tensor).item()

# 计算状态价值
with torch.no_grad():
    v_value = v_net(state_tensor).item()

print(f"贪心动作: {best_action:.4f}")
print(f"采样动作: {sampled_action:.4f}")
print(f"状态价值: {v_value:.4f}")
```

## 模型配置

- **状态维度**: 7（vanco_level, creatinine, wbc, bun, temperature, sbp, heart_rate）
- **动作维度**: 1（万古霉素剂量，标准化后范围[-1, 1]）
- **网络结构**: [32, 32]两层隐藏层
- **训练步数**: 3000
- **源checkpoint**: `exp_conservative/ckpt_step3000.pt`

## 模型文件结构

导出的`.pth`文件包含：

```python
{
    'q_network': {...},           # Q网络权重
    'v_network': {...},           # V网络权重
    'policy_network': {...},      # 策略网络权重
    'model_config': {
        'state_dim': 7,
        'action_dim': 1,
        'hidden_dims': [32, 32]
    },
    'training_info': {
        'step': 3000,
        'source_checkpoint': '...'
    }
}
```

## 测试结果示例

### 临床案例测试

| 案例 | 推荐动作 | 状态价值 |
|------|---------|---------|
| 正常患者 | 0.86 | -0.48 |
| 高万古霉素浓度 | 1.00 | -1.20 |
| 肾功能不全 | 1.00 | -1.58 |
| 感染严重 | 1.00 | -2.32 |

### 数据集评估

在50个样本上的统计：
- 预测动作均值: 1.00（标准化）
- 状态价值均值: -117.79

## 注意事项

1. **PyTorch版本兼容性**: 需要PyTorch 2.6+，使用`weights_only=True`安全加载
2. **动作范围**: 模型输出为标准化动作[-1, 1]，需要反标准化到实际剂量
3. **贪心策略倾向**: 当前模型倾向于推荐最大剂量，建议进一步调整或添加约束
4. **临床验证**: 使用前需要在临床模拟器上充分验证

## 下一步

- [ ] 在临床模拟环境中测试策略
- [ ] 添加安全约束和剂量上限
- [ ] 对比不同checkpoint的性能
- [ ] 可视化决策边界
- [ ] 可解释性分析
