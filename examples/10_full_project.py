"""
第 10 章：综合实战 —— 端到端智能训练项目
对应赛项知识点：对复杂的智能系统进行测试和训练并编写报告；
数据清洗 → 标准化 → 训练 → 调优 → 评估报告 全流程
任务：根据 8 个传感器特征预测设备是否故障（二分类）
"""
import torch  # 导入 PyTorch
import torch.nn as nn  # 神经网络模块
from torch.utils.data import DataLoader, TensorDataset, random_split  # 数据工具

torch.manual_seed(42)  # 固定随机种子

# ========== 步骤 1：构造并清洗数据（数据工程） ==========
n = 1200  # 1200 条设备记录
X = torch.randn(n, 8)  # 8 个传感器特征
# 故障与特征的非线性关系（真实业务中的规律往往是隐藏的）
y = ((X[:, 0] ** 2 + X[:, 1] * X[:, 2] - X[:, 3]) > 0.5).long()  # 0=正常, 1=故障

X[0] = float("nan")  # 制造脏数据：缺失值
X[10] = 500.0  # 制造脏数据：异常值

mask = ~torch.isnan(X).any(dim=1)  # 找出含 NaN 的行
mask &= (X.abs().max(dim=1).values < 100)  # 找出含异常值的行
X, y = X[mask], y[mask]  # 剔除脏数据
print(f"步骤 1 数据清洗：剩余 {len(X)} 条有效记录")

# ========== 步骤 2：标准化 ==========
mean, std = X.mean(dim=0), X.std(dim=0)  # 计算各特征的均值和标准差
X = (X - mean) / std  # Z-score 标准化
print("步骤 2 标准化完成：各特征均值≈0，标准差≈1")

# ========== 步骤 3：划分数据集 ==========
dataset = TensorDataset(X, y)  # 打包
train_set, val_set, test_set = random_split(dataset, [800, 200, len(X) - 1000])  # 训练/验证/测试
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)  # 训练加载器
val_loader = DataLoader(val_set, batch_size=64)  # 验证加载器
test_loader = DataLoader(test_set, batch_size=64)  # 测试加载器
print(f"步骤 3 数据集划分：训练 {len(train_set)} / 验证 {len(val_set)} / 测试 {len(test_set)}")

# ========== 步骤 4：构建模型 ==========
model = nn.Sequential(  # 带正则化的 MLP
    nn.Linear(8, 64),  # 输入 8 -> 64
    nn.BatchNorm1d(64),  # 批归一化
    nn.ReLU(),  # 激活
    nn.Dropout(0.3),  # Dropout 防过拟合
    nn.Linear(64, 32),  # 64 -> 32
    nn.BatchNorm1d(32),  # 批归一化
    nn.ReLU(),  # 激活
    nn.Linear(32, 2),  # 输出 2 类
)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam 优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率衰减

# ========== 步骤 5：训练 + 早停 + 保存最佳模型 ==========
def evaluate(loader):  # 评估函数
    model.eval()  # 评估模式
    correct, total = 0, 0  # 计数器
    with torch.no_grad():  # 关闭梯度
        for bx, by in loader:  # 逐批次
            correct += (model(bx).argmax(1) == by).sum().item()  # 累计正确数
            total += len(by)  # 累计总数
    return correct / total  # 返回准确率

best_val, wait, patience = 0.0, 0, 8  # 早停相关变量：最佳验证准确率、等待计数、容忍轮数
print("步骤 5 开始训练：")
for epoch in range(60):  # 最多 60 轮
    model.train()  # 训练模式
    for bx, by in train_loader:  # 训练五步曲
        loss = criterion(model(bx), by)  # 前向 + 损失
        optimizer.zero_grad()  # 清梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    scheduler.step()  # 每轮更新学习率

    val_acc = evaluate(val_loader)  # 验证集评估
    if val_acc > best_val:  # 验证准确率提升
        best_val, wait = val_acc, 0  # 更新记录
        torch.save(model.state_dict(), "device_best.pt")  # 保存最佳模型
    else:  # 未提升
        wait += 1  # 等待计数 +1
        if wait >= patience:  # 连续 8 轮无提升
            print(f"  第 {epoch+1} 轮触发早停")
            break  # 提前停止
    if (epoch + 1) % 10 == 0:  # 定期打印
        print(f"  第 {epoch+1:2d} 轮 | 验证准确率={val_acc:.2%} | 当前学习率={scheduler.get_last_lr()[0]:.4f}")

# ========== 步骤 6：测试并编写评估报告 ==========
model.load_state_dict(torch.load("device_best.pt"))  # 加载最佳模型
test_acc = evaluate(test_loader)  # 测试集准确率

# 计算精确率和召回率（故障检测中，漏报比误报更危险，所以召回率很重要）
model.eval()  # 评估模式
all_pred, all_true = [], []  # 收集所有预测和真实标签
with torch.no_grad():  # 关闭梯度
    for bx, by in test_loader:  # 逐批次
        all_pred.append(model(bx).argmax(1))  # 保存预测
        all_true.append(by)  # 保存真实
pred, true = torch.cat(all_pred), torch.cat(all_true)  # 拼接成完整张量

tp = ((pred == 1) & (true == 1)).sum().item()  # 真正例：预测故障且确实故障
fp = ((pred == 1) & (true == 0)).sum().item()  # 假正例：误报
fn = ((pred == 0) & (true == 1)).sum().item()  # 假负例：漏报
precision = tp / (tp + fp)  # 精确率 = TP/(TP+FP)：报的故障里有多少是真的
recall = tp / (tp + fn)  # 召回率 = TP/(TP+FN)：真实故障里抓到了多少

print("\n========== 评估报告 ==========")
print(f"最佳验证准确率 : {best_val:.2%}")
print(f"测试集准确率   : {test_acc:.2%}")
print(f"精确率         : {precision:.2%}")
print(f"召回率         : {recall:.2%}")
print("==============================")
