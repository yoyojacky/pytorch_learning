"""
第 9 章：预训练与微调（Fine-tuning）
对应赛项知识点：理解微调（Fine-tuning）技术如何通过领域特定数据调整模型参数，
以适应业务场景需求
"""
import torch  # 导入 PyTorch
import torch.nn as nn  # 神经网络模块
from torch.utils.data import DataLoader, TensorDataset  # 数据工具

torch.manual_seed(42)  # 固定随机种子

# ========== 1. 场景设定 ==========
# "预训练"：模型在大规模通用数据上学会通用特征（这里用合成数据模拟）
# "微调"：把学到的通用能力迁移到具体的业务小数据集上
# 真实业务中：预训练模型是 BERT/ResNet 等，这里用小网络模拟同样的流程

# ========== 2. 第一步：模拟"预训练"—— 在通用数据上训练骨干网络 ==========
class Backbone(nn.Module):  # 骨干网络：负责提取通用特征
    def __init__(self):  # 构造函数
        super().__init__()  # 调用父类构造
        self.features = nn.Sequential(  # 特征提取部分
            nn.Linear(10, 64),  # 10 维输入 -> 64 维特征
            nn.ReLU(),  # 激活
            nn.Linear(64, 64),  # 64 -> 64
            nn.ReLU(),  # 激活
        )
        self.head = nn.Linear(64, 3)  # 预训练任务的分类头：3 个通用类别

    def forward(self, x):  # 前向传播
        feat = self.features(x)  # 提取特征
        return self.head(feat)  # 输出预训练任务的类别得分

pretrained = Backbone()  # 实例化待预训练的模型

# 模拟通用大数据集：1000 个样本，3 个类别
X_pre = torch.randn(1000, 10)  # 1000 个 10 维样本
y_pre = (X_pre[:, 0] * 2 + X_pre[:, 1] > 0).long() + (X_pre[:, 2] > 0).long()  # 3 类标签

pre_loader = DataLoader(TensorDataset(X_pre, y_pre), batch_size=32, shuffle=True)  # 加载器
optimizer = torch.optim.Adam(pretrained.parameters(), lr=0.01)  # 优化器
criterion = nn.CrossEntropyLoss()  # 损失函数

print("--- 阶段一：预训练（通用数据） ---")
for epoch in range(10):  # 预训练 10 轮
    for bx, by in pre_loader:  # 逐批次
        loss = criterion(pretrained(bx), by)  # 前向 + 损失
        optimizer.zero_grad()  # 清梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
print("预训练完成，骨干网络已学会通用特征\n")

# ========== 3. 第二步：微调 —— 迁移到业务小数据 ==========
# 业务场景：只有 80 个样本的 2 分类任务（数据少，从头训练效果差，微调是最佳选择）
X_task = torch.randn(80, 10)  # 80 个业务样本
y_task = (X_task[:, 0] + X_task[:, 2] > 0).long()  # 业务标签（2 类）
task_loader = DataLoader(TensorDataset(X_task, y_task), batch_size=16, shuffle=True)  # 加载器

# 方案 A：冻结骨干，只训练新的分类头（特征提取式微调）
model_a = Backbone()  # 新建模型
model_a.features.load_state_dict(pretrained.features.state_dict())  # 把预训练的骨干参数迁移过来
model_a.head = nn.Linear(64, 2)  # 换成业务任务的 2 分类头（输出维度必须匹配新任务）

for param in model_a.features.parameters():  # 遍历骨干网络的所有参数
    param.requires_grad = False  # 冻结：这些参数在训练中不更新（赛项考点）

optimizer_a = torch.optim.Adam(model_a.head.parameters(), lr=0.01)  # 优化器只管分类头的参数
print("--- 阶段二：微调（冻结骨干，只训练分类头） ---")
for epoch in range(20):  # 微调 20 轮
    for bx, by in task_loader:  # 逐批次
        loss = criterion(model_a(bx), by)  # 前向 + 损失
        optimizer_a.zero_grad()  # 清梯度
        loss.backward()  # 反向传播（被冻结的参数不会产生梯度更新）
        optimizer_a.step()  # 只更新分类头

with torch.no_grad():  # 评估
    acc_a = (model_a(X_task).argmax(1) == y_task).float().mean()  # 方案 A 准确率

# 方案 B：全量微调 —— 用更小的学习率更新所有参数
model_b = Backbone()  # 新建模型
model_b.features.load_state_dict(pretrained.features.state_dict())  # 迁移预训练骨干
model_b.head = nn.Linear(64, 2)  # 换分类头

optimizer_b = torch.optim.Adam([  # 为不同部分设置不同学习率（微调经典技巧）
    {"params": model_b.features.parameters(), "lr": 0.001},  # 骨干：小学习率，避免破坏已学知识
    {"params": model_b.head.parameters(), "lr": 0.01},  # 新分类头：大学习率，快速学习
])
print("--- 阶段二：全量微调（差异化学习率） ---")
for epoch in range(20):  # 训练 20 轮
    for bx, by in task_loader:  # 逐批次
        loss = criterion(model_b(bx), by)  # 前向 + 损失
        optimizer_b.zero_grad()  # 清梯度
        loss.backward()  # 反向传播
        optimizer_b.step()  # 更新所有参数

with torch.no_grad():  # 评估
    acc_b = (model_b(X_task).argmax(1) == y_task).float().mean()  # 方案 B 准确率

print(f"\n冻结骨干微调准确率: {acc_a:.2%}")
print(f"全量微调准确率:     {acc_b:.2%}")
print("结论：数据少时冻结骨干更稳；数据较多时全量微调（小学习率）通常上限更高。")
