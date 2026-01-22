# IQL (Implicit Q-Learning) — Vancomycin给药优化

完整的IQL离线强化学习实现，用于万古霉素(Vancomycin)个性化给药优化。

---

## 📚 目录

1. [核心概念](#-核心概念-iql策略详解)
2. [项目成果](#-项目成果)
3. [代码结构](#-代码结构)
4. [快速开始](#-快速开始)
5. [详细文档](#-详细文档)
6. [优化指南](#-优化指南)

---

## 🧠 核心概念: IQL策略详解

### 什么是IQL？

**Implicit Q-Learning (隐式Q学习)** 是一种**离线强化学习**算法，特别适合从**历史医疗数据**中学习最优策略。

#### 为什么选择IQL？
- ✅ **纯离线学习**: 不需要与真实患者交互，完全从历史数据学习
- ✅ **避免分布偏移**: 通过隐式正则化，避免学习到数据中没有的危险动作
- ✅ **理论保证**: 有收敛性和性能保证

### IQL的三个核心网络

```
┌─────────────────────────────────────────────────────┐
│                   IQL架构                            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  状态 (7维) ──┐                                     │
│  - 万古霉素浓度 │                                   │
│  - 肌酐        │                                     │
│  - 白细胞      │                                     │
│  - 血尿素氮    │                                     │
│  - 体温        │                                     │
│  - 收缩压      │                                     │
│  - 心率        │                                     │
│               ├──→ [Q网络] ──→ Q(s,a) 价值评估      │
│               │                                     │
│               ├──→ [V网络] ──→ V(s) 状态价值        │
│               │                                     │
│               └──→ [策略网络] ──→ π(a|s) 推荐剂量   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

#### 1. Q网络 (动作-价值函数)
**作用**: 评估"在状态s下采取动作a有多好"

```python
Q(s, a) = 预期累积奖励
```

**例子**: 
- Q(浓度10mg/L, 给药600mg) = -50 → 这个剂量可能太小
- Q(浓度10mg/L, 给药1200mg) = -20 → 这个剂量更合适

#### 2. V网络 (状态价值函数)  
**作用**: 评估"状态s本身有多好"

```python
V(s) = τ-expectile of Q(s, a)
```

**Expectile (τ)**: 控制乐观/保守程度
- `τ = 0.7` (当前): V值接近Q的70%分位数（稍微乐观）
- `τ = 0.5`: V值等于Q的中位数（中性）
- `τ = 0.9`: V值接近Q的上分位数（很乐观，可能学到激进策略）

#### 3. 策略网络 (Policy)
**作用**: 直接给出"在状态s下应该做什么"

```python
π(a|s) = 根据优势加权学习最优动作分布
```

**优势加权回归 (AWR)**:
```python
权重 = exp(Advantage(s,a) / β)
Advantage(s,a) = Q(s,a) - V(s)
```

- `β` 小 (如0.5): 权重差异更大 → **更激进**
- `β` 大 (如10.0): 权重差异更小 → **更保守稳定**

### IQL训练流程

```
1. 从历史数据采样批次 (s, a, r, s')
   ↓
2. 更新V网络 (Expectile损失)
   目标: V应该等于Q的τ-expectile
   L_V = expectile_loss(V(s), Q(s,a))
   ↓
3. 更新Q网络 (Bellman误差)
   目标: Q(s,a) = r + γ·V(s')
   L_Q = MSE(Q(s,a), r + γ·V(s'))
   ↓
4. 更新策略网络 (优势加权)
   权重 = exp([Q(s,a) - V(s)] / β)
   L_π = -权重 × log π(a|s)
   ↓
5. 重复直到收敛
```

### 策略诊断提示

如果策略出现过于激进或动作塌缩，优先检查：

1. **Q−V 分布** 是否失衡（`positive_rate` 偏离 $1-\tau$）
2. **优势权重** 是否爆炸（`max_weight` 与 `max(Q−V)`）
3. **奖励尺度** 是否过大/过小（`reward_scale`）
4. **V 学习率** 是否过高（`v_lr_ratio`）

调参建议见 [IQL_DETAILED_GUIDE.md](IQL_DETAILED_GUIDE.md) 的"调参经验"章节。

---

## 🎯 项目成果

### ✅ 核心实现
- **数据处理**: 临床转移样本 + 7维状态特征
- **神经网络**: Q/V/Policy 三网络架构
- **训练系统**: 完整管道 + 检查点 + JSON日志
- **离线评估**: 价值函数与策略质量评估
- **可视化分析**: 交互式策略分析 notebook

### 📊 训练监控指标（推荐）

重点关注 **Q−V 分布** 与策略稳定性：

- `positive_rate ≈ 1 - τ`
- `mean(Q−V)` 接近 0（略正或略负都可）
- `std(Q−V)` 处于中等范围（避免过大/过小）
- `max(Q−V)` 不出现爆炸

训练日志会写入 [runs/<exp_name>/training_log.json](runs/) 并在控制台打印。

---

## 🧩 训练策略与稳定化

为保证离线训练稳定性，当前实现采用以下策略：

1. **Target Q 网络 (EMA)**：缓解Q值震荡与发散
2. **Reward Scale**：缩放奖励以避免Q爆炸
3. **Policy/Q warmup**：先训练Q/V，再启用策略更新
4. **V学习率更低**：通过 `v_lr_ratio` 限制V过冲
5. **优势与权重裁剪**：限制 `Q−V` 和 `exp((Q−V)/β)` 的极值
6. **梯度裁剪**：对Q/V/Policy统一裁剪
7. **分布监控**：持续记录 `mean/std/max(Q−V)` 与 `positive_rate`

这些策略的实现位置：
- 训练步更新与监控: [train_utils.py](train_utils.py)
- 训练主循环与日志: [train_iql.py](train_iql.py)

---

## 📁 代码结构

### 核心模块

```
algorithms/iql/
├── 🔧 核心实现
│   ├── models.py          # Q/V/Policy网络定义
│   ├── losses.py          # Expectile损失和IQL损失函数
│   ├── dataset.py         # 数据加载和ReplayBuffer
│   ├── train_utils.py     # 训练辅助函数（归一化、采样等）
│   └── utils.py           # 通用工具函数
│
├── 🚀 训练和评估
│   ├── train_iql.py       # 主训练脚本（含SimpleLogger）
│   ├── evaluate_iql.py    # 离线评估脚本
│   ├── export_model.py    # 导出模型为.pth格式
│   └── test_model.py      # 模型推理和测试
│
├── 📊 可视化
│   ├── plot_training_log.py  # 绘制训练曲线（从JSON）
│   └── analysis.ipynb        # 交互式分析notebook
│
├── ⚙️ 配置文件 (../configs/)
│   ├── iql_base.yaml          # 基础配置
│   ├── iql_fixed*.yaml        # 修复/稳定化实验
│   ├── iql_stable*.yaml       # 稳定训练配置
│   ├── iql_tuned.yaml         # 调优配置
│   └── iql_v_alignment_test.yaml # V 对齐实验
│
├── 🧪 测试
│   └── tests/
│       ├── test_models.py
│       ├── test_losses.py
│       ├── test_dataset.py
│       └── ... (8个测试文件)
│
├── 📦 训练输出
│   └── runs/
│       └── <exp_name>/
│           ├── ckpt_step*.pt      # 模型检查点
│           ├── training_log.json  # 训练日志
│           └── eval_results.json  # 评估结果
│
└── 📄 文档
   ├── README.md              # 本文档
   ├── IQL_DETAILED_GUIDE.md  # 详细技术指南（含调参经验）
   ├── ACTION_SPACE_README.md # 动作空间说明
   └── MODEL_EXPORT_README.md # 模型导出指南
```

### 关键类和函数

#### `models.py`
```python
class QNetwork(nn.Module)      # Q(s,a) 网络
class VNetwork(nn.Module)      # V(s) 网络  
class GaussianPolicy(nn.Module)  # π(a|s) 网络（高斯策略）
```

#### `losses.py`
```python
def expectile_loss(v, q, tau)            # Expectile回归损失
def iql_update_step(...)                 # 单步IQL更新
```

#### `train_iql.py`
```python
class SimpleLogger              # JSON格式训练日志
def train(config)               # 主训练循环
```

#### `dataset.py`
```python
class OfflineDataset            # 离线数据集加载
class ReplayBuffer              # 经验回放缓冲区
```

---

## 🚀 快速开始

### 1. 训练新模型
```bash
# 选择任一配置文件
python -m algorithms.iql.train_iql --config configs/iql_base.yaml

# 训练会在 runs/<exp_name>/ 下生成:
#   - ckpt_step*.pt: 模型检查点
#   - training_log.json: 训练日志
```

### 2. 查看训练曲线
```bash
python -m algorithms.iql.plot_training_log \
   --log-file runs/<exp_name>/training_log.json

# 生成 training_curves.png
```

### 3. 评估模型
```bash
python -m algorithms.iql.evaluate_iql \
   --checkpoint runs/<exp_name>/ckpt_stepXXXX.pt \
   --config configs/iql_base.yaml \
   --output eval_results.json

# 生成评估指标:
#   - greedy_q: 贪心策略Q值
#   - mc_return: 蒙特卡洛回报
#   - mean_reward: 平均奖励
```

### 4. 导出模型
```bash
python -m algorithms.iql.export_model \
   --checkpoint runs/<exp_name>/ckpt_stepXXXX.pt \
   --output exported_models/iql_model.pth \
   --test

# 生成独立的.pth文件，包含:
#   - 模型权重
#   - 归一化参数
#   - 配置信息
```

### 5. 测试模型推理
```bash
# 临床测试案例
python -m algorithms.iql.test_model --clinical

# 在数据集上评估
python -m algorithms.iql.test_model --samples 100 --mode greedy
```

### 6. 交互式分析
```bash
# 打开 analysis.ipynb
# 可视化:
#   - Q/V值分布
#   - 不同浓度下的推荐剂量
#   - 临床特征敏感性
#   - 策略与行为对比
```

---

## 📖 详细文档

### 完整技术指南
参见 [IQL_DETAILED_GUIDE.md](IQL_DETAILED_GUIDE.md) 了解:
- IQL算法数学推导和原理
- 策略公式详解（状态、动作、网络结构）
- 训练参数详解（8个核心参数+配置演化）
- 实现细节和代码注释
- 超参数影响分析和调参经验
- 结果分析和故障排除指南

### 模型导出和部署
参见 [MODEL_EXPORT_README.md](MODEL_EXPORT_README.md) 了解:
- 如何导出训练好的模型
- 模型推理和测试方法
- 在代码中使用模型

---

## 🎓 优化指南

### 快速改进建议

1. **调整expectile (tau)**
   ```yaml
   tau: 0.5  # 更保守，减少激进行为
   ```

2. **增加beta（AWR温度）**
   ```yaml
   beta: 3.0   # 标准值
   beta: 10.0  # 更平滑的权重
   ```

3. **提高学习率**
   ```yaml
   lr: 1e-4   # 当前3e-5太小
   ```

4. **增加训练步数**
   ```yaml
   total_steps: 10000  # 当前5000
   ```

5. **重新设计奖励函数**
   - 添加正奖励（达到目标浓度）
   - 平衡安全性和有效性
   - 考虑多目标优化

详细参数调优策略请参见 [IQL_DETAILED_GUIDE.md](IQL_DETAILED_GUIDE.md) 的"训练参数详解"和"调参经验"章节

---

## 🧪 测试

运行所有测试:
```bash
pytest algorithms/iql/tests/
```

单独测试模块:
```bash
pytest algorithms/iql/tests/test_models.py
pytest algorithms/iql/tests/test_losses.py
```

---

## 📊 项目指标

- **代码行数**: ~2000行
- **测试覆盖**: 8个测试模块
- **数据量**: 2102个转移
- **训练时间**: ~5分钟 (5000步)
- **模型大小**: 23KB (.pth格式)

---

## 🔗 相关资源

### 论文
- [Offline RL with Implicit Q-Learning (Kostrikov et al., 2021)](https://arxiv.org/abs/2110.06169)
- [Conservative Q-Learning (Kumar et al., 2020)](https://arxiv.org/abs/2006.04779)

### 工具
- PyTorch 2.6+
- NumPy, Pandas
- Matplotlib (可视化)

---

## 📝 更新日志

### 2025-01-13
- ✅ 移除所有TensorBoard依赖
- ✅ 实现SimpleLogger (JSON格式)
- ✅ 完成5000步训练
- ✅ 添加plot_training_log.py
- ✅ 更新模型导出系统
- ✅ 创建IQL_DETAILED_GUIDE.md完整技术文档
- ✅ 重写README.md

### 2025-01-12
- ✅ 完成IQL核心实现
- ✅ 创建测试套件
- ✅ 第一次成功训练

---

## 💬 常见问题

**Q: 为什么所有奖励都是负数？**
A: 这取决于奖励函数设计。当前可能使用了基于误差的惩罚函数。建议查看IQL_DETAILED_GUIDE.md中的参数调优章节。

**Q: 为什么策略总是给最大剂量？**
A: 可能是tau太大（过于乐观）或beta太小（权重过于集中）。尝试tau=0.5, beta=3.0。

**Q: 训练需要多久？**
A: 在CPU上，5000步约5分钟。使用GPU可以更快。

**Q: 如何选择最佳检查点？**
A: 查看eval_results.json中的greedy_q值，越高越好（负数绝对值越小越好）。

---

## 📧 联系方式

项目维护者: [您的名字]
问题反馈: [GitHub Issues链接]
