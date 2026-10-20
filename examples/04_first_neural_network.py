"""
第 4 章：第一个神经网络（MLP 二分类）
对应赛项知识点：监督学习的应用场景；根据业务场景选择深度学习算法框架
"""
import torch  # 导入 PyTorch
import torch.nn as nn  # nn 模块提供网络层（Linear、ReLU 等）
from torch.utils.data import DataLoader, TensorDataset  # TensorDataset 快速封装张量数据集

torch.manual_seed(42)  # 固定随机种子

# ========== 1. 构造监督学习数据：两个同心圆二分类 ==========
n = 500  # 总样本数
X = torch.randn(n, 2) * 2  # 500 个二维点
radius = X[:, 0] ** 2 + X[:, 1] ** 2  # 每个点到原点的距离平方
y = (radius > 4).long()  # 距离平方 > 4 的为类别 1，否则为类别 0（非线性边界！）

dataset = TensorDataset(X, y)  # 用 TensorDataset 快速打包特征和标签
loader = DataLoader(dataset, batch_size=32, shuffle=True)  # 每批 32 条并打乱

# ========== 2. 定义神经网络 ==========
class MLP(nn.Module):  # 所有网络都要继承 nn.Module
    def __init__(self):  # 构造函数：定义网络有哪些层
        super().__init__()  # 调用父类构造函数（固定写法）
        self.layer1 = nn.Linear(2, 16)  # 全连接层：输入 2 维 -> 输出 16 维
        self.relu = nn.ReLU()  # 激活函数 ReLU：引入非线性，f(x)=max(0,x)
        self.layer2 = nn.Linear(16, 8)  # 全连接层：16 维 -> 8 维
        self.layer3 = nn.Linear(8, 2)  # 输出层：8 维 -> 2 维（两个类别的得分）

    def forward(self, x):  # forward 定义数据如何流过各层（前向传播）
        x = self.relu(self.layer1(x))  # 第 1 层 + ReLU
        x = self.relu(self.layer2(x))  # 第 2 层 + ReLU
        return self.layer3(x)  # 输出层（交叉熵损失自带 softmax，这里不加激活）

model = MLP()  # 实例化模型
print(model)  # 打印网络结构

# ========== 3. 定义损失函数与优化器 ==========
criterion = nn.CrossEntropyLoss()  # 交叉熵损失：分类任务的标准损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # SGD 优化器，学习率 0.1
# model.parameters() 把模型所有可学习参数交给优化器管理

# ========== 4. 训练循环（PyTorch 五步曲） ==========
for epoch in range(50):  # 训练 50 轮
    total_loss = 0.0  # 累计本轮总损失
    for batch_X, batch_y in loader:  # 逐批次取数据
        pred = model(batch_X)  # 第 1 步：前向传播，得到预测得分 (32, 2)
        loss = criterion(pred, batch_y)  # 第 2 步：计算损失
        optimizer.zero_grad()  # 第 3 步：清空上一轮的梯度
        loss.backward()  # 第 4 步：反向传播计算梯度
        optimizer.step()  # 第 5 步：优化器根据梯度更新参数
        total_loss += loss.item()  # .item() 取出标量值并累加

    if (epoch + 1) % 10 == 0:  # 每 10 轮评估一次
        with torch.no_grad():  # 评估时关闭梯度计算，节省内存
            acc = (model(X).argmax(dim=1) == y).float().mean()  # 预测类别与真实标签比较
        print(f"第 {epoch+1:2d} 轮 | 平均损失={total_loss/len(loader):.4f} | 准确率={acc:.2%}")
