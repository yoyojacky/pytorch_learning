"""
第 12 章（进阶实战）：读取 Donkey Car tub 数据，训练 CNN 自动驾驶模型（行为克隆）
对应场景：行为克隆第二步 —— 让神经网络模仿人类驾驶：看图 -> 输出转向和油门

依赖：先运行 examples/11_donkeycar_data.py 生成 data/mytub 数据集
"""
import json  # 读取 JSON 记录
from pathlib import Path  # 遍历 tub 目录
import numpy as np  # 图像数组处理
from PIL import Image  # 读取 jpg 图片
import torch  # 导入 PyTorch
import torch.nn as nn  # 神经网络模块
from torch.utils.data import Dataset, DataLoader, random_split  # 数据工具

torch.manual_seed(42)  # 固定随机种子

# ========== 1. 自定义 Dataset：读取 Donkey Car tub 格式 ==========
class TubDataset(Dataset):  # 继承 Dataset，对应第 3 章的知识
    def __init__(self, tub_dir):
        self.tub_dir = Path(tub_dir)  # tub 目录路径
        # 找到所有 record_*.json，并按序号数值排序（record_2 要排在 record_10 前面）
        self.records = sorted(self.tub_dir.glob("record_*.json"),
                              key=lambda p: int(p.stem.split("_")[1]))

    def __len__(self):  # 数据集大小 = 记录条数
        return len(self.records)

    def __getitem__(self, idx):  # 取出第 idx 条 (图片, 操控) 样本
        with open(self.records[idx]) as f:  # 打开记录文件
            rec = json.load(f)  # 解析 JSON
        img_path = self.tub_dir / rec["cam/image_array"]  # 记录里存着图片文件名
        img = Image.open(img_path).convert("RGB")  # 读图，确保是 RGB 三通道
        img = np.asarray(img, dtype=np.float32) / 255.0  # 转成数组并归一化到 0~1
        img = torch.from_numpy(img).permute(2, 0, 1)  # (高,宽,通道) -> (通道,高,宽)，PyTorch 的图像约定
        label = torch.tensor([rec["user/angle"], rec["user/throttle"]])  # 标签 = [转向, 油门]
        return img, label  # 返回一个训练样本

dataset = TubDataset("data/mytub")  # 实例化数据集
train_set, val_set = random_split(dataset, [300, 60])  # 划分 300 训练 / 60 验证
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)  # 训练加载器
val_loader = DataLoader(val_set, batch_size=32)  # 验证加载器

# ========== 2. CNN 驾驶模型（Donkey Car 默认 pilot 的 PyTorch 版） ==========
# 卷积网络逐层提取"车道线在哪"的视觉特征，最后回归出转向和油门
model = nn.Sequential(
    nn.Conv2d(3, 24, kernel_size=5, stride=2),  # 3通道 -> 24个特征图，步长2下采样
    nn.ReLU(),  # 激活
    nn.Conv2d(24, 32, kernel_size=5, stride=2),  # 24 -> 32
    nn.ReLU(),  # 激活
    nn.Conv2d(32, 64, kernel_size=5, stride=2),  # 32 -> 64
    nn.ReLU(),  # 激活
    nn.Conv2d(64, 64, kernel_size=3),  # 64 -> 64，小卷积核提取细节
    nn.ReLU(),  # 激活
    nn.Flatten(),  # 拉平：64x10x15 = 9600 维向量
    nn.Linear(64 * 10 * 15, 100),  # 全连接：视觉特征 -> 100 维
    nn.ReLU(),  # 激活
    nn.Linear(100, 50),  # 100 -> 50
    nn.ReLU(),  # 激活
    nn.Linear(50, 2),  # 输出 2 个数：[预测转向, 预测油门]
)
criterion = nn.MSELoss()  # 回归任务用均方误差（预测值要逼近真实操控值）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# ========== 3. 训练（还是熟悉的五步曲） ==========
def evaluate(loader):  # 评估函数：返回转向和油门的平均绝对误差
    model.eval()  # 评估模式
    err = 0.0  # 累计误差
    with torch.no_grad():  # 关闭梯度
        for bx, by in loader:  # 逐批次
            err += (model(bx) - by).abs().mean(dim=0).sum().item()  # 分别累计两个输出的绝对误差
    return err / len(loader.dataset)  # 平均到每个样本

for epoch in range(20):  # 训练 20 轮
    model.train()  # 训练模式
    total = 0.0  # 累计损失
    for bx, by in train_loader:  # 逐批次
        loss = criterion(model(bx), by)  # 前向 + MSE 损失
        optimizer.zero_grad()  # 清梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total += loss.item()  # 累计
    if (epoch + 1) % 4 == 0:  # 每 4 轮报告一次
        print(f"第 {epoch+1:2d} 轮 | 训练损失={total/len(train_loader):.4f} | 验证误差(转向+油门)={evaluate(val_loader):.4f}")

# ========== 4. 上车前的"模拟路考"：抽几条记录看预测 ==========
model.eval()  # 评估模式
print("\n===== 模拟路考（看模型预测 vs 人类驾驶） =====")
for i in [0, 20, 40, 59]:  # 从验证集抽 4 条
    img, label = val_set[i]  # 取样本
    with torch.no_grad():  # 关闭梯度
        pred = model(img.unsqueeze(0))[0]  # 加 batch 维度再预测
    print(f"样本 {i:2d}: 转向 真实={label[0]:+.3f} 预测={pred[0]:+.3f} | 油门 真实={label[1]:.3f} 预测={pred[1]:.3f}")

torch.save(model.state_dict(), "donkey_pilot.pt")  # 保存模型，可拷到真车上做推理
print("\n模型已保存到 donkey_pilot.pt —— 真车上加载它即可实现自动驾驶")
