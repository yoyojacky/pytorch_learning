"""
第 2 章：自动求导（Autograd）—— 梯度下降的原理
对应赛项知识点：理解概率论、统计学及线性代数在模型中的实现路径；
学习率对训练稳定性的影响（本章首次接触学习率）
"""
import torch  # 导入 PyTorch

# ========== 1. 自动求导基础 ==========
x = torch.tensor(2.0, requires_grad=True)  # 创建一个需要追踪梯度的标量张量
y = x ** 2 + 3 * x + 1  # 定义函数 y = x² + 3x + 1，PyTorch 会自动记录计算图
y.backward()  # 反向传播：计算 y 对 x 的导数 dy/dx = 2x + 3
print("x = 2 时，dy/dx =", x.grad)  # x.grad 保存梯度 -> 2*2+3 = 7

# ========== 2. 用梯度下降拟合 y = 3x + 2（手写训练过程） ==========
torch.manual_seed(42)  # 固定随机种子，保证每次运行结果一致（可复现）

# 生成 100 个样本：y = 3x + 2 + 噪声
X = torch.rand(100, 1) * 10  # 100 个 0~10 之间的随机数作为输入，形状 (100,1)
y_true = 3 * X + 2 + torch.randn(100, 1)  # 真实标签 = 3x+2 加随机噪声

# 初始化待学习的参数 w 和 b
w = torch.randn(1, requires_grad=True)  # 权重，随机初始化，需要梯度
b = torch.zeros(1, requires_grad=True)  # 偏置，初始化为 0，需要梯度

learning_rate = 0.01  # 学习率：每次参数更新的步长（赛项考点：影响训练稳定性）

for epoch in range(200):  # 训练 200 轮（epoch = 遍历一遍全部数据）
    y_pred = w * X + b  # 前向传播：用当前参数做预测
    loss = ((y_pred - y_true) ** 2).mean()  # 均方误差损失 MSE
    loss.backward()  # 反向传播：自动计算 loss 对 w、b 的梯度

    with torch.no_grad():  # 参数更新不需要记录计算图，用 no_grad 关闭追踪
        w -= learning_rate * w.grad  # 梯度下降更新 w：w = w - lr * 梯度
        b -= learning_rate * b.grad  # 梯度下降更新 b

    w.grad.zero_()  # 清空 w 的梯度（PyTorch 默认累积梯度，必须动清零）
    b.grad.zero_()  # 清空 b 的梯度

    if (epoch + 1) % 40 == 0:  # 每 40 轮打印一次
        print(f"第 {epoch+1:3d} 轮 | loss={loss.item():.4f} | w={w.item():.4f} | b={b.item():.4f}")

print(f"\n训练完成：w ≈ {w.item():.3f}（目标 3），b ≈ {b.item():.3f}（目标 2）")
