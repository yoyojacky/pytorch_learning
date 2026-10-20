"""
第 7 章：网络结构优化 —— 对抗过拟合（Dropout / BatchNorm / 早停）
对应赛项知识点：通过网络结构优化实现模型精度与效率平衡
"""
import torch  # 导入 PyTorch
import torch.nn as nn  # 神经网络模块
from torch.utils.data import DataLoader, TensorDataset, random_split  # 数据工具

torch.manual_seed(42)  # 固定随机种子

# ========== 1. 构造一个"容易过拟合"的场景：样本少、特征多 ==========
n = 200  # 只有 200 个样本
X = torch.randn(n, 20)  # 特征多达 20 维
y = ((X[:, 0] + X[:, 1]) > 0).long()  # 标签其实只与前 2 个特征有关，其余 18 维是噪声

train_set, val_set = random_split(TensorDataset(X, y), [160, 40])  # 划分训练/验证
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)  # 训练加载器
val_loader = DataLoader(val_set, batch_size=32)  # 验证加载器

# ========== 2. 定义两个模型：朴素版 vs 优化版 ==========
plain = nn.Sequential(  # 朴素版：又宽又深，没有任何正则化
    nn.Linear(20, 128),  # 20 -> 128
    nn.ReLU(),  # 激活
    nn.Linear(128, 128),  # 128 -> 128
    nn.ReLU(),  # 激活
    nn.Linear(128, 2),  # 输出 2 类
)

optimized = nn.Sequential(  # 优化版：加入三种结构优化手段
    nn.Linear(20, 128),  # 20 -> 128
    nn.BatchNorm1d(128),  # 批归一化：把每层输出拉回标准分布，训练更稳定、更快
    nn.ReLU(),  # 激活
    nn.Dropout(p=0.5),  # Dropout：训练时随机"丢弃"50%神经元，防止过度依赖某些特征
    nn.Linear(128, 64),  # 收窄的隐藏层：128 -> 64（减少参数量 = 效率优化）
    nn.BatchNorm1d(64),  # 批归一化
    nn.ReLU(),  # 激活
    nn.Dropout(p=0.5),  # 再次 Dropout
    nn.Linear(64, 2),  # 输出层
)

# ========== 3. 训练函数（带"早停"机制） ==========
def train_model(model, epochs=100, patience=10):
    """训练模型，并在验证准确率不再提升时提前停止（早停 Early Stopping）"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam 优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    best_val, wait = 0.0, 0  # best_val 记录最佳验证准确率；wait 记录多少轮没进步
    history = []  # 保存每轮 (训练准确率, 验证准确率) 供分析

    for epoch in range(epochs):  # 最多训练 epochs 轮
        model.train()  # 训练模式：Dropout 生效、BatchNorm 用批次统计
        for bx, by in train_loader:  # 逐批次
            loss = criterion(model(bx), by)  # 前向 + 损失
            optimizer.zero_grad()  # 清梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        model.eval()  # 评估模式：Dropout 关闭、BatchNorm 用全局统计
        with torch.no_grad():  # 关闭梯度
            tr = (model(X[:160]).argmax(1) == y[:160]).float().mean().item()  # 训练准确率
            va = (model(X[160:]).argmax(1) == y[160:]).float().mean().item()  # 验证准确率
        history.append((tr, va))  # 记录

        if va > best_val:  # 验证准确率提升
            best_val, wait = va, 0  # 更新最佳值并重置等待计数
        else:  # 没有提升
            wait += 1  # 等待计数加 1
            if wait >= patience:  # 连续 patience 轮没进步
                print(f"  第 {epoch+1} 轮触发早停")  # 提前终止，避免继续过拟合
                break  # 跳出训练循环
    return best_val, history  # 返回最佳验证准确率和历史

# ========== 4. 对比实验 ==========
print("--- 朴素模型 ---")
plain_best, plain_his = train_model(plain)  # 训练朴素模型
print("--- 优化模型 ---")
opt_best, opt_his = train_model(optimized)  # 训练优化模型

print(f"\n朴素模型：训练准确率 {plain_his[-1][0]:.2%} / 验证准确率 {plain_best:.2%}")
print(f"优化模型：训练准确率 {opt_his[-1][0]:.2%} / 验证准确率 {opt_best:.2%}")
print("解读：判断好坏要看验证准确率而不是训练准确率；优化模型的验证准确率更高，")
print("      说明 BatchNorm + Dropout + 早停有效抑制了过拟合，泛化能力更强。")
