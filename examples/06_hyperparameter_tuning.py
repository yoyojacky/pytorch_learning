"""
第 6 章：超参数调优 —— 学习率衰减、批量大小、网格搜索
对应赛项知识点：知道学习率衰减、批量大小选择对训练稳定性的影响；
理解网格搜索、贝叶斯优化等调参方法的原理（Optuna 的思想演示）
"""
import torch  # 导入 PyTorch
import torch.nn as nn  # 神经网络模块
from torch.utils.data import DataLoader, TensorDataset  # 数据集封装与加载

torch.manual_seed(42)  # 固定随机种子

# ========== 1. 准备数据 ==========
n = 800  # 800 个样本
X = torch.randn(n, 6)  # 6 个特征
y = ((X[:, 0] + X[:, 1] * X[:, 2] - X[:, 3]) > 0).long()  # 非线性规则生成标签

dataset = TensorDataset(X, y)  # 打包数据集

# ========== 2. 一个可复用的训练函数（调参实验的基础） ==========
def train_one_run(batch_size, lr, epochs=15, use_scheduler=False):
    """用指定的超参数训练一次，返回最终训练准确率"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 按给定批量构建加载器
    model = nn.Sequential(  # 简单 MLP：6 -> 32 -> 2
        nn.Linear(6, 32),  # 全连接层
        nn.ReLU(),  # 激活
        nn.Linear(32, 2),  # 输出层
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # SGD，学习率为参数 lr
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # StepLR：每过 step_size=5 轮，学习率乘以 gamma=0.5（即衰减一半）—— 赛项考点：学习率衰减

    for epoch in range(epochs):  # 训练指定轮数
        for bx, by in loader:  # 逐批次训练
            loss = nn.CrossEntropyLoss()(model(bx), by)  # 前向 + 交叉熵损失
            optimizer.zero_grad()  # 清梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        if use_scheduler:  # 如果启用学习率衰减
            scheduler.step()  # 每轮结束后让调度器更新学习率
    with torch.no_grad():  # 评估关闭梯度
        acc = (model(X).argmax(dim=1) == y).float().mean().item()  # 计算整体准确率
    return acc  # 返回准确率

# ========== 3. 实验一：对比不同批量大小（batch size） ==========
print("=== 实验一：批量大小对训练的影响 ===")
for bs in [8, 64, 256]:  # 分别尝试小、中、大三种批量
    acc = train_one_run(batch_size=bs, lr=0.1)  # 训练一次
    print(f"batch_size={bs:4d} -> 准确率 {acc:.2%}")  # 小批量噪声大但泛化好，大批量稳定但可能陷入平坦解

# ========== 4. 实验二：学习率衰减 vs 不衰减 ==========
print("\n=== 实验二：学习率衰减的作用 ===")
acc_fixed = train_one_run(batch_size=32, lr=0.5, use_scheduler=False)  # 固定大学习率
acc_decay = train_one_run(batch_size=32, lr=0.5, use_scheduler=True)  # 同样的学习率但逐步衰减
print(f"固定学习率 0.5     -> 准确率 {acc_fixed:.2%}")
print(f"学习率 0.5 + 衰减  -> 准确率 {acc_decay:.2%}")  # 衰减通常更好：前期大步探索，后期小步精调

# ========== 5. 实验三：网格搜索（Grid Search） ==========
print("\n=== 实验三：网格搜索找最优超参数 ===")
results = {}  # 字典保存每组超参数的结果
for bs in [16, 64]:  # 遍历批量候选值
    for lr in [0.01, 0.1, 0.5]:  # 遍历学习率候选值 —— 两层循环穷举所有组合，这就是网格搜索
        acc = train_one_run(batch_size=bs, lr=lr, use_scheduler=True)  # 训练并评估
        results[(bs, lr)] = acc  # 记录结果
        print(f"bs={bs:3d}, lr={lr:.2f} -> {acc:.2%}")

best = max(results, key=results.get)  # 找准确率最高的超参数组合
print(f"\n最优组合: batch_size={best[0]}, lr={best[1]}, 准确率={results[best]:.2%}")
# 提示：贝叶斯优化（如 Optuna）与网格搜索目标相同，但会根据历史结果"聪明地"选择下一组
# 尝试的超参数，用更少的试验找到更优组合。用法：pip install optuna 后定义 objective 函数即可。
