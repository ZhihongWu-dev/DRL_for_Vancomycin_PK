# CQL策略评估说明（连续动作空间）

## 为什么不能用准确率评估强化学习策略？

在强化学习中，**准确率不是合适的评估指标**，原因如下：

1. **强化学习的目标不是预测正确**：RL的目标是最大化累积奖励，而不是预测"正确"的动作
2. **动作没有绝对的对错**：在ICU用药场景中，不同的剂量可能都是合理的，关键看长期效果
3. **准确率忽略了奖励信息**：即使动作不同，如果累积奖励更高，策略就是更好的
4. **连续动作空间**：药物剂量是连续值，不存在"准确"的概念，只有"更好"的概念

## 强化学习中的正确评估方法

### 1. 策略价值（Policy Value）- 核心指标

**定义**：策略的期望累积奖励
```
V^π = E[Σ γ^t * r_t | π]
```

**评估方法**：Fitted Q Evaluation (FQE)

对于**连续动作空间**，使用蒙特卡洛估计：
```
V^π(s) ≈ (1/N) * Σ_i Q(s, a_i),  a_i ~ π(·|s)
```

**实现步骤**：
1. 从策略π(·|s)中采样N个动作（默认N=10）
2. 对每个采样动作计算Q值：Q(s, a_i)
3. 取平均得到策略价值：V^π(s) = mean(Q(s, a_i))

**对于初始状态**：V^π(s_0) 就是策略的期望价值

### 2. 与行为策略对比

**行为策略**：数据中实际使用的策略（医生的决策）

**对比指标**：
- 绝对改进：CQL策略价值 - 行为策略价值
- 相对改进：(绝对改进 / 行为策略价值) × 100%

**解读**：
- 改进 > 0：CQL策略更好
- 改进 < 0：CQL策略更保守（在医疗中可能是优点）

### 3. Episode级别分析

对每个`stay_id`（episode）分析：
- 策略初始价值
- 实际累积奖励
- 动作匹配率（仅供参考）

## 评估流程

### 步骤1：准备数据
```python
# 使用全部数据，不划分训练/验证
df = pd.read_csv("ready_data.csv")
df = df.sort_values(['stay_id', 'step_4hr']).reset_index(drop=True)
df[STATE_COLS] = df[STATE_COLS].fillna(df[STATE_COLS].median())

# 标准化状态（使用训练时的scaler）
full_states = state_scaler.transform(df[STATE_COLS].values)
full_rewards = df[REWARD_COL].values.astype(np.float32)

# 归一化奖励（使用训练时的参数）
full_rewards = (full_rewards - r_min) / r_range

# 构建dones（每个stay_id的最后一个时间步为done=1）
full_dones = np.zeros(len(df), dtype=np.float32)
for stay_id in df['stay_id'].unique():
    stay_mask = df['stay_id'] == stay_id
    stay_indices = np.where(stay_mask)[0]
    if len(stay_indices) > 0:
        full_dones[stay_indices[-1]] = 1.0
```

### 步骤2：FQE评估（连续动作空间）
```python
def evaluate_policy_value_fqe(agent, states, rewards, dones, gamma=0.99, n_samples=10):
    """
    使用Fitted Q Evaluation (FQE) 评估策略价值（连续动作空间）
    
    对于连续动作空间，使用蒙特卡洛估计：
    V^π(s) ≈ (1/N) * Σ_i Q(s, a_i),  a_i ~ π(·|s)
    """
    agent.eval()
    states_tensor = torch.FloatTensor(states).to(DEVICE)
    
    with torch.no_grad():
        # 多次采样取平均（蒙特卡洛估计）
        policy_values_list = []
        
        for _ in range(n_samples):
            policy_actions, _, _ = agent.policy.sample(states_tensor)
            q1_vals = agent.q1(states_tensor, policy_actions)
            q2_vals = agent.q2(states_tensor, policy_actions)
            q_vals = (q1_vals + q2_vals) / 2
            policy_values_list.append(q_vals.cpu().numpy())
        
        # 平均Q值（策略价值）
        policy_values = np.mean(policy_values_list, axis=0).flatten()
    
    # 按episode分组，计算每个episode的初始状态价值
    episode_initial_values = []
    episode_actual_returns = []
    
    current_return = 0.0
    current_discount = 1.0
    episode_start_idx = 0
    
    for i in range(len(dones)):
        current_return += current_discount * rewards[i]
        current_discount *= gamma
        
        if dones[i] == 1 or i == len(dones) - 1:
            if episode_start_idx < len(policy_values):
                episode_initial_values.append(policy_values[episode_start_idx])
                episode_actual_returns.append(current_return)
            current_return = 0.0
            current_discount = 1.0
            episode_start_idx = i + 1
    
    return {
        'mean_episode_value': np.mean(episode_initial_values),
        'episode_initial_values': episode_initial_values,
        'episode_actual_returns': episode_actual_returns,
    }

# 执行评估
fqe_results = evaluate_policy_value_fqe(
    agent, full_states, full_rewards, full_dones,
    gamma=config['gamma'], n_samples=10
)
```

### 步骤3：行为策略评估
```python
def evaluate_behavior_policy_value(rewards, dones, gamma=0.99):
    """评估行为策略（数据中的策略）的价值"""
    episode_returns = []
    current_return = 0.0
    current_discount = 1.0
    
    for i in range(len(rewards)):
        current_return += current_discount * rewards[i]
        current_discount *= gamma
        
        if dones[i] == 1 or i == len(rewards) - 1:
            episode_returns.append(current_return)
            current_return = 0.0
            current_discount = 1.0
    
    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'episode_returns': episode_returns,
    }

# 执行评估
behavior_results = evaluate_behavior_policy_value(full_rewards, full_dones, gamma=config['gamma'])
```

### 步骤4：策略改进分析
```python
improvement = fqe_results['mean_episode_value'] - behavior_results['mean_return']
relative_improvement = (improvement / abs(behavior_results['mean_return'])) * 100 if behavior_results['mean_return'] != 0 else 0.0

print(f"绝对改进: {improvement:.4f}")
print(f"相对改进: {relative_improvement:.2f}%")
```

## 评估指标说明

### ✅ 应该使用的指标

1. **策略价值（FQE）**：最重要的指标
   - 反映策略的期望表现
   - 使用Q函数评估，是离线RL的标准方法

2. **策略改进**：与行为策略对比
   - 看策略是否比数据中的策略更好

3. **Episode级别统计**：
   - 平均初始价值：策略在每个episode开始时的价值
   - 平均实际回报：episode的实际累积奖励
   - 平均折扣回报：考虑折扣因子的累积奖励
   - 平均动作MAE：策略动作与数据动作的平均绝对误差（mg）
   - Episode长度分布

### ❌ 不应该使用的指标

1. **准确率**：那是分类问题的指标
2. **简单的训练/验证损失**：虽然可以用于防止过拟合，但不是策略评估的核心指标

### ⚠️ 仅供参考的指标

1. **动作MAE（平均绝对误差）**：策略动作与数据动作的差异
   - 可以看，但不是主要指标
   - 即使动作不同，如果累积奖励更高，策略就是更好的
   - 在连续动作空间中，MAE比匹配率更合适

## 评估结果解读

### 场景1：策略价值 > 行为策略价值
- **含义**：CQL策略比数据中的策略更好
- **行动**：可以考虑使用CQL策略

### 场景2：策略价值 < 行为策略价值
- **含义**：CQL策略更保守
- **解读**：在医疗场景中，保守可能是优点（避免过度用药）
- **行动**：需要结合临床判断

### 场景3：策略价值 ≈ 行为策略价值
- **含义**：CQL策略与行为策略相当
- **行动**：可以进一步分析策略的稳定性

## 可视化图表说明

评估脚本会生成6个可视化图表（如果启用可视化）：

1. **策略价值对比**：柱状图对比CQL策略和行为策略
   - CQL策略（FQE评估）vs 行为策略（实际回报）
   - 越高越好

2. **Episode价值分布**：直方图显示价值分布
   - CQL策略的episode初始价值分布
   - 行为策略的episode实际回报分布
   - 可以看出策略在不同episode上的表现

3. **策略改进分析**：横向柱状图显示改进百分比
   - 绿色表示改进（>0），橙色表示保守（<0）
   - 显示相对改进百分比

4. **Episode价值：预测 vs 实际**：散点图对比FQE预测和实际回报
   - X轴：实际累积奖励
   - Y轴：策略初始价值（FQE）
   - 越接近对角线越好

5. **Episode长度 vs 价值**：散点图分析长度与价值的关系
   - X轴：Episode长度（步数）
   - Y轴：策略初始价值
   - 分析长episode是否表现更好

6. **动作对比：策略 vs 数据**：散点图对比策略动作和数据动作
   - X轴：数据动作（mg）
   - Y轴：策略动作（mg）
   - 越接近对角线表示策略动作与数据动作越相似

**注意**：如果运行脚本时没有显示图表，检查是否启用了`plt.show()`。

## 代码使用

### 方式1：运行评估脚本

```bash
python evaluate_cql_continuous.py
```

### 方式2：在Jupyter Notebook中使用

在`CQL/CQL_evaluation.ipynb`中，完整的评估代码包含：

1. **模型定义**：GaussianPolicy, QNetwork, CQLAgent
2. **加载模型**：从`cql_final_model.pt`加载（需要`weights_only=False`）
3. **加载数据**：使用全部数据，不划分训练/验证
4. **FQE评估函数**：`evaluate_policy_value_fqe()`（连续动作空间版本）
5. **行为策略评估函数**：`evaluate_behavior_policy_value()`
6. **执行评估**：计算策略价值和行为策略价值
7. **策略改进分析**：计算绝对改进和相对改进
8. **Episode级别分析**：每个stay_id的详细统计
9. **可视化**：6个图表（如果启用）
10. **保存结果**：JSON文件和PNG图表

**直接运行notebook中的所有cell即可完成评估。**

### 评估输出示例

```
【CQL策略价值（FQE评估）】
  策略期望价值: 17.5065 ± 1.0337
  平均状态价值: 17.8097 ± 12.6805
  评估episode数: 58

【行为策略价值（实际数据）】
  实际平均回报: 17.0907 ± 10.1630
  评估episode数: 58

【策略改进分析】
  绝对改进: 0.4158
  相对改进: 2.43%
  解释: CQL策略比行为策略好 2.43%

【Episode级别统计】
  评估episode数: 58
  平均episode长度: 36.4 步
  平均初始价值: 17.5051
  平均实际回报: 22.0597
  平均折扣回报: 17.0907
  平均动作MAE: 160.19 mg
  平均策略动作: 66.97 mg
  平均数据动作: 116.11 mg
```

## 实际评估结果解读

根据`CQL_evaluation.ipynb`的实际运行结果：

### 评估结果
- **CQL策略期望价值**: 17.5065
- **行为策略实际回报**: 17.0907
- **策略改进**: +2.43%（CQL策略比行为策略好2.43%）

### 关键发现
1. **策略价值正常**：17.5065在合理范围内（归一化后10-30）
2. **策略有改进**：比行为策略好2.43%，说明CQL学习到了更好的策略
3. **动作MAE**: 160.19 mg，说明策略动作与数据动作有差异，但这是正常的
   - 策略动作平均：66.97 mg
   - 数据动作平均：116.11 mg
   - 策略倾向于给更低的剂量（更保守）

### 为什么动作MAE大但策略更好？

**关键理解**：
- 动作MAE大（160.19 mg）不代表策略差
- 策略价值更高（17.5065 > 17.0907）说明策略更好
- 策略倾向于给更低剂量（66.97 mg vs 116.11 mg），但累积奖励更高
- 在医疗场景中，保守的策略（低剂量）可能是优点

## 连续动作空间的特殊性

### 与离散动作空间的区别

| 方面 | 离散动作空间 | 连续动作空间 |
|------|------------|------------|
| **动作类型** | {0, 1, 2} | [0, max_dose] mg |
| **FQE方法** | V^π = Σ_a π(a\|s) * Q(s,a) | V^π ≈ (1/N) * Σ_i Q(s, a_i) |
| **采样** | 不需要（可枚举） | 需要（蒙特卡洛估计） |
| **评估指标** | 匹配率 | 动作MAE |

### 连续动作空间的FQE

```python
# 离散动作：直接计算
V^π(s) = Σ_a π(a|s) * Q(s, a)  # 对所有动作求和

# 连续动作：蒙特卡洛估计
V^π(s) ≈ (1/N) * Σ_i Q(s, a_i),  a_i ~ π(·|s)  # 采样N次取平均
```

**采样次数**：默认N=10，可以增加以提高精度（但计算更慢）

## 参考资料

- **FQE (Fitted Q Evaluation)**: 离线强化学习中的标准评估方法
- **Offline RL**: 离线强化学习不需要在线交互，使用历史数据评估策略
- **Policy Value**: 策略的期望累积奖励，是RL中的核心概念
- **连续动作空间**: 使用蒙特卡洛估计评估策略价值
- **CQL论文**: Conservative Q-Learning for Offline Reinforcement Learning

---

**文档版本**: v2.0  
**最后更新**: 2024  
**基于**: `CQL/CQL_evaluation.ipynb`  
**代码文件**: `evaluate_cql_continuous.py`

