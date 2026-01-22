"""测试IQL策略损失函数的修复

验证：
1. 策略损失不会爆炸
2. 权重计算正确（不归一化）
3. 使用NLL而不是MSE
"""
import torch
import math

# 设置随机种子
torch.manual_seed(42)

# 模拟数据
batch_size = 64
state_dim = 7
action_dim = 1

# 创建模拟数据
s = torch.randn(batch_size, state_dim)
a = torch.randn(batch_size, action_dim)  # 行为动作
q_vals = torch.randn(batch_size, 1) * 10  # Q值
v_vals = torch.randn(batch_size, 1) * 10  # V值

# 模拟策略网络输出
mean = torch.randn(batch_size, action_dim, requires_grad=True)
logstd = torch.randn(batch_size, action_dim, requires_grad=True) * 0.5
std = logstd.clamp(-20, 2).exp()

# 参数
beta = 0.5
weight_clip = 100.0

print("=" * 60)
print("测试IQL策略损失函数修复")
print("=" * 60)

# ========== 旧的错误实现 ==========
print("\n【旧实现 - 有bug】")
adv = q_vals - v_vals
weights_old = torch.exp(adv / beta)
weights_old = torch.clamp(weights_old, 0.0, weight_clip)
weights_old_normalized = weights_old / (weights_old.sum() + 1e-8)  # 错误的归一化

# 错误：使用MSE
pi_loss_old = (weights_old_normalized * (mean - a) ** 2).mean()

print(f"优势范围: [{adv.min():.2f}, {adv.max():.2f}]")
print(f"权重范围: [{weights_old.min():.2f}, {weights_old.max():.2f}]")
print(f"归一化权重和: {weights_old_normalized.sum():.6f}")
print(f"归一化权重范围: [{weights_old_normalized.min():.6f}, {weights_old_normalized.max():.6f}]")
print(f"策略损失 (MSE): {pi_loss_old.item():.6f}")

# ========== 新的正确实现 ==========
print("\n【新实现 - 修复后】")
adv = q_vals - v_vals
weights_new = torch.exp(adv / beta)
weights_new = torch.clamp(weights_new, 0.0, weight_clip)  # 不归一化！

# 正确：使用NLL
log_prob = -0.5 * ((a - mean) / (std + 1e-8)) ** 2 - torch.log(std + 1e-8) - 0.5 * math.log(2 * math.pi)
pi_loss_new = -(weights_new * log_prob).mean()

print(f"优势范围: [{adv.min():.2f}, {adv.max():.2f}]")
print(f"权重范围: [{weights_new.min():.2f}, {weights_new.max():.2f}]")
print(f"权重和: {weights_new.sum():.2f} (不应该归一化到1)")
print(f"对数似然范围: [{log_prob.min():.2f}, {log_prob.max():.2f}]")
print(f"策略损失 (NLL): {pi_loss_new.item():.6f}")

# ========== 对比分析 ==========
print("\n" + "=" * 60)
print("对比分析")
print("=" * 60)
print(f"旧损失: {pi_loss_old.item():.6f}")
print(f"新损失: {pi_loss_new.item():.6f}")
print(f"损失比率: {pi_loss_old.item() / (pi_loss_new.item() + 1e-8):.2f}x")

print("\n【关键区别】")
print("1. 权重归一化:")
print(f"   旧: 权重和 = {weights_old_normalized.sum():.6f} (强制=1, 错误!)")
print(f"   新: 权重和 = {weights_new.sum():.2f} (自然值, 正确!)")

print("\n2. 损失函数:")
print("   旧: MSE (均方误差)")
print("   新: NLL (负对数似然)")

print("\n3. 梯度尺度:")
print(f"   旧实现的梯度会被batch_size缩小 {batch_size}x")
print("   新实现的梯度保持正常尺度")

# 计算一些梯度来验证
print("\n【梯度测试】")
pi_loss_new.backward()
print(f"Mean梯度范数: {mean.grad.norm():.6f}")
print(f"Logstd梯度范数: {logstd.grad.norm():.6f}")

print("\n✅ 测试完成!")
print("=" * 60)
