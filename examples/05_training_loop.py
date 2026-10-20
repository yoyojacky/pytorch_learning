"""
第 5 章：完整训练流程 —— 训练/验证/测试 + 模型保存与加载
对应赛项知识点：对复杂的智能系统进行测试和训练并编写报告；
实现模型精度与效率平衡
"""
import torch  # 导入 PyTorch
import torch.nn as nn  # 神经网络模块
from torch.utils.data import DataLoader, TensorDataset, random_split  # random_split 随机划分数据集

torch.manual_seed(42)  # 固定随机种子

# ========== 1. 构造数据并做三路划分：训练/验证/测试 ==========
n = 600  # 600 个样本
X = torch.randn(n, 4)  # 4 个特征
y = ((X[:, 0] * 2 - X[:, 1] + X[:, 2]) > 0).long()  # 由特征线性组合决定的二分类标签

dataset = TensorDataset(X, y)  # 打包成数据集
train_set, val_set, test_set = random_split(dataset, [400, 100, 100])  # 按 400/100/100 随机切分

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)  # 训练加载器：打乱
val_loader = DataLoader(val_set, batch_size=32)  # 验证加载器
test_loader = DataLoader(test_set, batch_size=32)  # 测试加载器

# ========== 2. 定义模型 ==========
class Classifier(nn.Module):  # 定义分类网络
    def __init__(self):  # 构造函数
        super().__init__()  # 调用父类构造
        self.net = nn.Sequential(  # Sequential 把多层按顺序串成一个整体
            nn.Linear(4, 32),  # 输入层 4 -> 隐藏层 32
            nn.ReLU(),  # 非线性激活
            nn.Linear(32, 16),  # 隐藏层 32 -> 16
            nn.ReLU(),  # 非线性激活
            nn.Linear(16, 2),  # 输出层 16 -> 2 个类别
        )

    def forward(self, x):  # 前向传播
        return self.net(x)  # 数据依次流过 Sequential 中各层

model = Classifier()  # 实例化模型
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam 优化器（比 SGD 更常用）

# ========== 3. 评估函数：在任意加载器上计算准确率 ==========
def evaluate(model, loader):  # 定义通用评估函数
    model.eval()  # 切换到评估模式（关闭 Dropout/BN 的训练行为）
    correct, total = 0, 0  # 计数器：预测正确数、总样本数
    with torch.no_grad():  # 评估不需要梯度
        for bx, by in loader:  # 逐批次评估
            pred = model(bx).argmax(dim=1)  # 取得分最高的类别作为预测结果
            correct += (pred == by).sum().item()  # 累加预测正确的数量
            total += len(by)  # 累加样本总数
    return correct / total  # 返回准确率

# ========== 4. 训练 + 每轮验证 ==========
best_val_acc = 0.0  # 记录历史最佳验证准确率
for epoch in range(30):  # 训练 30 轮
    model.train()  # 切换到训练模式（与 eval 对应）
    for bx, by in train_loader:  # 训练循环五步曲
        loss = criterion(model(bx), by)  # 前向 + 算损失
        optimizer.zero_grad()  # 清梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    val_acc = evaluate(model, val_loader)  # 在验证集上评估
    if val_acc > best_val_acc:  # 如果验证准确率创新高
        best_val_acc = val_acc  # 更新最佳记录
        torch.save(model.state_dict(), "best_model.pt")  # 保存当前最优模型的参数
    if (epoch + 1) % 10 == 0:  # 每 10 轮打印
        print(f"第 {epoch+1:2d} 轮 | 验证准确率={val_acc:.2%} | 历史最佳={best_val_acc:.2%}")

# ========== 5. 加载最佳模型并在测试集上出最终报告 ==========
model.load_state_dict(torch.load("best_model.pt"))  # 把保存的最优参数载入模型
test_acc = evaluate(model, test_loader)  # 在从未见过的测试集上做最终评估
print(f"\n===== 最终报告 =====")
print(f"最佳验证准确率: {best_val_acc:.2%}")
print(f"测试集准确率:   {test_acc:.2%}")
