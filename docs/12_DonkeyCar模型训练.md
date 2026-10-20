# 第 12 章：DonkeyCar模型训练

## 本章目标

- 掌握 Donkey Car 行为克隆的第二步：读取 tub 数据（图片 + JSON 记录），训练 CNN 学会"看图 → 输出转向和油门"
- 能独立编写一个自定义 `Dataset` 类（TubDataset），把"图片文件 + 结构化标注"的配对数据喂给 DataLoader
- 理解 CNN 回归模型的设计思想：多层 `Conv2d` 逐层提取视觉特征，`Linear` 回归出连续控制量
- 会用 `MSELoss` + `Adam` 完成一个完整的回归训练流程，并能用"平均绝对误差"评估回归模型
- 理解"模拟路考"的意义：训练后人工抽查几条样本，对比预测与人类驾驶，为模型上车建立信心

## 对应赛项知识点

本章对应"人工智能训练师"职业技能大赛的以下要求：**深度学习模型开发——卷积神经网络搭建与训练**、**回归任务建模——连续控制量预测**，以及**智能系统评估——训练后模型的人工抽查验证**。

比赛中的深度学习模块常要求选手针对一个真实场景完成端到端建模。本章脚本头部 docstring 点明了场景——"行为克隆第二步：让神经网络模仿人类驾驶：看图 -> 输出转向和油门"。这恰好覆盖赛项对 CNN 的三项核心考核：**自定义数据集类**（把非标准格式的 tub 数据接进 PyTorch 数据管线）、**CNN 模型搭建**（Conv2d+ReLU 特征提取 + Flatten+Linear 回归头）、**训练与评估**（MSELoss 回归损失、Adam 优化、验证集误差监控、预测抽查）。掌握了本章，你就掌握了"视觉输入 → 连续输出"这一类赛题的完整套路。

## 核心概念讲解

### tub 数据架构回顾

第 11 章我们已经准备好了 tub 数据集（`data/mytub/`），本章直接复用。回顾一下它的结构：

```
data/mytub/
  ├── 1_cam-image_array_.jpg   ← 摄像头图片
  ├── record_1.json            ← 同一时刻的驾驶记录
  ├── 2_cam-image_array_.jpg
  ├── record_2.json
  └── ...（共 360 对）
```

每条 `record_N.json` 是一条 JSON，长这样：

```json
{"cam/image_array": "1_cam-image_array_.jpg",
 "user/angle": 0.00745,
 "user/throttle": 0.71505,
 "user/mode": "user",
 "timestamp": 1760000001}
```

- `cam/image_array`：记录这条数据对应的图片文件名，是图片与标签的关联键；
- `user/angle`：转向，-1（打死左）~ +1（打死右），0 为直行；
- `user/throttle`：油门，0 ~ 1，越大越快；
- `user/mode` / `timestamp`：数据来源与时间戳，本章训练用不到，但真车排查问题时很重要。

训练的本质就是：让网络看到 `1_cam-image_array_.jpg` 这张图时，输出的两个数尽量接近 `user/angle` 和 `user/throttle`。

### CNN：为什么用卷积网络看图

图片是二维的，车道线、弯道这些关键信息都藏在"局部像素图案"里。全连接网络把图片拉平成一维向量，会彻底丢掉空间结构；而**卷积神经网络（CNN, Convolutional Neural Network）**用一个小窗口（卷积核）在图上滑动，专门捕捉局部特征——边缘、色块、线条方向。

本章的网络是 Donkey Car 官方默认 pilot 的 PyTorch 复刻版，结构分两段：

- **卷积段**（4 层 Conv2d）：逐步把 120×160 的图片压缩成 64 张 10×15 的"特征图"。前面几层看边缘和色块，后面几层看"车道线在哪、往哪弯"这种高级语义。`stride=2` 表示卷积窗口每次跳 2 格，起到下采样（缩小图片、扩大视野）的作用。
- **回归头**（3 层 Linear）：把 64×10×15=9600 个特征值拉平，逐层压缩到 100 → 50 → 2，最后输出 2 个数——预测转向、预测油门。

### 回归：输出连续数值，用 MSE 衡量差距

分类任务输出"是猫还是狗"（离散类别），用交叉熵；而转向和油门是**连续值**——-0.42 和 -0.43 有本质差别（方向盘轻打和重打）。这类任务叫**回归（Regression）**：

- 网络最后一层没有 softmax，直接输出浮点数；
- 损失函数用 **MSELoss（均方误差）**：`(预测-真实)²` 的平均。预测越接近真人操控，损失越小；
- 评估指标用**平均绝对误差（MAE）**：`|预测-真实|` 的平均，直观反映"平均差多少"。

### 训练完为什么要"模拟路考"

损失降得很低不代表模型真能开车——可能是过拟合，也可能是数据有 bug。上车前，人工抽几条验证集样本，把模型预测的转向/油门和人类驾驶员的真实操作并排打印出来对比，是最直接的" sanity check（合理性检查）"。这一步在工程上至关重要：**模型上线前，先用人类能看懂的方式验证它的行为**。

## 完整代码

```python
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
```

## 逐行讲解

### 第 1 段：导入依赖与固定随机种子

```python
import json  # 读取 JSON 记录
from pathlib import Path  # 遍历 tub 目录
import numpy as np  # 图像数组处理
from PIL import Image  # 读取 jpg 图片
import torch  # 导入 PyTorch
import torch.nn as nn  # 神经网络模块
from torch.utils.data import Dataset, DataLoader, random_split  # 数据工具

torch.manual_seed(42)  # 固定随机种子
```

- `import json` —— tub 格式的标注文件 `record_N.json` 是 JSON 文本，读取它必须用标准库 `json`。**注意**：不要用 `eval()` 解析 JSON，既不安全也不规范。
- `from pathlib import Path` —— 面向对象的路径工具。用 `Path` 拼接路径（`tub_dir / filename`）比字符串 `+ "/"` 更健壮，还能直接用 `.glob()` 找文件，是处理数据集目录的标配。
- `import numpy as np` —— PIL 读出的图片对象不能直接被 PyTorch 用，需要先转成 NumPy 数组（像素矩阵），所以 NumPy 是图片预处理的桥梁。
- `from PIL import Image` —— PIL（Pillow）的 `Image` 模块负责打开 jpg 图片文件。这是 Python 图像处理的事实标准。
- `import torch` / `import torch.nn as nn` —— PyTorch 核心与张量计算，`nn` 里是搭建神经网络要用的层（Conv2d、Linear、ReLU 等）和损失函数（MSELoss）。
- `from torch.utils.data import Dataset, DataLoader, random_split` —— 数据三件套：`Dataset` 是自定义数据集的基类，`DataLoader` 负责批量加载和打乱，`random_split` 负责把数据集随机切成训练集和验证集。
- `torch.manual_seed(42)` —— 固定随机种子。这样每次运行代码时，`random_split` 的划分结果、网络参数的随机初始化都完全一致，实验可复现。**比赛时必须固定种子**，否则无法对比不同实验。

### 第 2 段：自定义 TubDataset 类

```python
class TubDataset(Dataset):  # 继承 Dataset，对应第 3 章的知识
    def __init__(self, tub_dir):
        self.tub_dir = Path(tub_dir)  # tub 目录路径
        # 找到所有 record_*.json，并按序号数值排序（record_2 要排在 record_10 前面）
        self.records = sorted(self.tub_dir.glob("record_*.json"),
                              key=lambda p: int(p.stem.split("_")[1]))
```

- `class TubDataset(Dataset)` —— 自定义数据集必须继承 `torch.utils.data.Dataset`，并实现 `__init__`、`__len__`、`__getitem__` 三个方法，这是 PyTorch 数据管线的标准接口。
- `def __init__(self, tub_dir)` —— 构造函数接收 tub 目录路径（如 `"data/mytub"`）。**为什么只存路径不加载数据**：图片有 360 张，全读进内存浪费；只记录"有哪些文件、在哪"，用到哪张再读哪张，这叫惰性加载（lazy loading）。
- `self.tub_dir = Path(tub_dir)` —— 把字符串路径包装成 `Path` 对象，后面拼接图片路径时直接用 `/` 运算符，简洁且跨平台。
- `self.records = sorted(...)` —— 用 `glob("record_*.json")` 找出所有标注文件，然后排序。**排序的坑**：如果按文件名字符串排序，`record_10` 会排在 `record_2` 前面（因为 '1' < '2'），导致顺序错乱。所以 `key=lambda p: int(p.stem.split("_")[1])` 取出文件名里的序号（`p.stem` 是不带扩展名的文件名如 `record_10`，`split("_")[1]` 得到 `"10"`，`int()` 转成数字），按数值排序。这样 `record_2` 正确排在 `record_10` 前。

```python
    def __len__(self):  # 数据集大小 = 记录条数
        return len(self.records)
```

- `__len__` —— 返回数据集大小（即标注文件条数，这里是 360）。`DataLoader` 和 `random_split` 都靠它知道"总共有多少样本"。

```python
    def __getitem__(self, idx):  # 取出第 idx 条 (图片, 操控) 样本
        with open(self.records[idx]) as f:  # 打开记录文件
            rec = json.load(f)  # 解析 JSON
        img_path = self.tub_dir / rec["cam/image_array"]  # 记录里存着图片文件名
        img = Image.open(img_path).convert("RGB")  # 读图，确保是 RGB 三通道
        img = np.asarray(img, dtype=np.float32) / 255.0  # 转成数组并归一化到 0~1
        img = torch.from_numpy(img).permute(2, 0, 1)  # (高,宽,通道) -> (通道,高,宽)，PyTorch 的图像约定
        label = torch.tensor([rec["user/angle"], rec["user/throttle"]])  # 标签 = [转向, 油门]
        return img, label  # 返回一个训练样本
```

- `def __getitem__(self, idx)` —— 数据管线的核心：给定序号 `idx`，返回第 `idx` 条训练样本。`DataLoader` 会反复调用它，每次取一个样本，攒够 `batch_size` 个就拼成一个 batch 送进网络。
- `with open(self.records[idx]) as f:` —— 打开第 `idx` 条 JSON 标注文件。`with` 语句保证文件用完自动关闭，防止文件句柄泄漏。
- `rec = json.load(f)` —— 把 JSON 文本解析成 Python 字典。`rec` 现在长这样：`{"cam/image_array": "...", "user/angle": ..., "user/throttle": ...}`。
- `img_path = self.tub_dir / rec["cam/image_array"]` —— 从记录里取出图片文件名（如 `1_cam-image_array_.jpg`），用 `/` 拼成完整路径（`data/mytub/1_cam-image_array_.jpg`）。**为什么用 JSON 里的字段而不是猜文件名**：这样即使图片命名规则变了，只要 JSON 里的字段对，代码就不用改，更健壮。
- `img = Image.open(img_path).convert("RGB")` —— 用 PIL 打开图片，并强制转成 RGB 三通道。**为什么要 convert**：有些图片可能是灰度图（1 通道）或带透明通道的 RGBA（4 通道），网络第一层固定收 3 通道，不统一会报维度错误。
- `img = np.asarray(img, dtype=np.float32) / 255.0` —— 把 PIL 图片转成 NumPy 数组，并指定数据类型为 `float32`（32 位浮点），再除以 255 把像素值从 0~255 的整数缩放到 0.0~1.0 的小数。**归一化的原因**：神经网络对输入数值范围敏感，0~1 的输入能让训练更稳定、收敛更快。
- `img = torch.from_numpy(img).permute(2, 0, 1)` —— 把 NumPy 数组转成 PyTorch 张量，然后调整维度顺序。PIL/NumPy 的图片是 `(高, 宽, 通道)` 即 `(120, 160, 3)`，但 PyTorch 的卷积层要求 `(通道, 高, 宽)` 即 `(3, 120, 160)`，`permute(2, 0, 1)` 就是把第 2 维（通道）提到最前面。
- `label = torch.tensor([rec["user/angle"], rec["user/throttle"]])` —— 把 JSON 里的转向和油门包成一个 2 元素张量，作为这条样本的标签（监督学习的"标准答案"）。
- `return img, label` —— 返回一个元组 `(图片张量, 标签张量)`，这就是一个完整的训练样本。`DataLoader` 会把多个这样的元组拼成 `(batch_size, 3, 120, 160)` 的图片批和 `(batch_size, 2)` 的标签批。

### 第 3 段：实例化数据集并划分训练/验证集

```python
dataset = TubDataset("data/mytub")  # 实例化数据集
train_set, val_set = random_split(dataset, [300, 60])  # 划分 300 训练 / 60 验证
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)  # 训练加载器
val_loader = DataLoader(val_set, batch_size=32)  # 验证加载器
```

- `dataset = TubDataset("data/mytub")` —— 创建数据集实例。此时只记录了 360 个 JSON 文件的路径，还没读任何图片（惰性加载）。
- `train_set, val_set = random_split(dataset, [300, 60])` —— 把 360 条数据随机切成 300 条训练集 + 60 条验证集。**为什么必须划分**：如果拿训练集评估模型，模型可能是"背答案"，看不出真实水平；留出 60 条模型没见过的数据做验证，才能客观评估泛化能力。因为前面固定了种子 42，每次划分结果都一样。
- `train_loader = DataLoader(train_set, batch_size=32, shuffle=True)` —— 训练数据加载器：`batch_size=32` 表示每次取 32 条样本组成一个批次；`shuffle=True` 表示每轮训练前打乱顺序，防止模型学到样本的排列规律（比如"前面都是左转"这种虚假模式）。
- `val_loader = DataLoader(val_set, batch_size=32)` —— 验证数据加载器。验证时不训练，不用打乱顺序，所以省略 `shuffle`。

### 第 4 段：搭建 CNN 驾驶模型

```python
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
```

- `model = nn.Sequential(...)` —— 用 `Sequential` 把各层按顺序串起来，前一层的输出自动作为后一层的输入，像搭积木一样简单。
- `nn.Conv2d(3, 24, kernel_size=5, stride=2)` —— 第一个卷积层：输入 3 通道（RGB 图片），输出 24 个特征图；`kernel_size=5` 表示用 5×5 的卷积核在图上滑动提取特征；`stride=2` 表示每次跳 2 格，把图片尺寸缩小一半（120×160 → 58×78），既减少计算量又扩大感受野。这一层学会识别边缘、色块等低级特征。
- `nn.ReLU()` —— 激活函数，把负值置 0、正值不变。没有它，多层卷积叠起来等价于单层线性变换，失去表达能力。ReLU 引入非线性，让网络能拟合复杂的"看图→操控"映射。
- `nn.Conv2d(24, 32, kernel_size=5, stride=2)` —— 第二个卷积层：24 → 32 通道，继续用 5×5 核 + 步长 2 下采样（58×78 → 27×37）。特征图变多、尺寸变小，网络开始识别更抽象的图案（比如"这条线是斜的"）。
- `nn.Conv2d(32, 64, kernel_size=5, stride=2)` —— 第三个卷积层：32 → 64 通道（27×37 → 12×17）。至此图片已被压缩 8 倍，但通道数从 3 涨到 64，意味着网络从"看像素"升级到"看语义"——这 64 张特征图里编码了"车道线在哪、往哪弯"等高级信息。
- `nn.Conv2d(64, 64, kernel_size=3)` —— 第四个卷积层：通道数不变（64 → 64），但改用更小的 3×3 核、默认步长 1（不下采样）。小卷积核更擅长捕捉细节纹理，是对前面粗粒度特征的精加工。
- `nn.Flatten()` —— 把 64 张 10×15 的特征图拉平成一维向量：64×10×15 = 9600 个数。卷积段到此结束，接下来交给全连接层做回归。
- `nn.Linear(64 * 10 * 15, 100)` —— 第一个全连接层：把 9600 维的视觉特征压缩到 100 维。这一步是"从特征到决策"的关键一跳——网络要在这 100 个数里总结出"当前路况该怎么开"。
- `nn.Linear(100, 50)` —— 第二个全连接层：100 → 50，进一步压缩、提纯决策信息。
- `nn.Linear(50, 2)` —— 输出层：50 → 2，最终输出 2 个浮点数——预测转向（-1~+1）、预测油门（0~1）。**注意**：这里没有激活函数，直接输出原始数值，因为回归任务就是要预测任意实数。

### 第 5 段：定义损失函数和优化器

```python
criterion = nn.MSELoss()  # 回归任务用均方误差（预测值要逼近真实操控值）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器
```

- `criterion = nn.MSELoss()` —— 均方误差损失函数：计算 `(预测-真实)²` 的平均。预测越准，损失越小。**为什么用 MSE 不用交叉熵**：交叉熵是分类任务用的（预测类别概率分布），而转向/油门是连续数值，必须用回归损失。
- `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)` —— Adam 优化器，负责根据梯度更新网络的所有参数（`model.parameters()`）。`lr=0.001` 是学习率，控制每次更新的步长；太大会震荡不收敛，太小会训练太慢，0.001 是 Adam 的经典默认值。

### 第 6 段：定义评估函数

```python
def evaluate(loader):  # 评估函数：返回转向和油门的平均绝对误差
    model.eval()  # 评估模式
    err = 0.0  # 累计误差
    with torch.no_grad():  # 关闭梯度
        for bx, by in loader:  # 逐批次
            err += (model(bx) - by).abs().mean(dim=0).sum().item()  # 分别累计两个输出的绝对误差
    return err / len(loader.dataset)  # 平均到每个样本
```

- `def evaluate(loader)` —— 自定义评估函数，接收一个数据加载器（训练时传 `val_loader`），返回模型在这批数据上的平均绝对误差。
- `model.eval()` —— 切换到评估模式。虽然本章网络没有 Dropout/BatchNorm 这类"训练/评估行为不同"的层，但养成习惯很重要——**评估前必须 eval()，训练前必须 train()**。
- `err = 0.0` —— 累计误差的计数器，初始为 0。
- `with torch.no_grad():` —— 关闭梯度计算。评估时不需要反向传播，关掉梯度能省内存、加速计算。**忘记加这个会导致内存泄漏**，因为每次前向都会记录计算图。
- `for bx, by in loader:` —— 逐批次取数据：`bx` 是 `(batch_size, 3, 120, 160)` 的图片批，`by` 是 `(batch_size, 2)` 的标签批。
- `err += (model(bx) - by).abs().mean(dim=0).sum().item()` —— 这一行干了四件事：① `model(bx)` 前向预测；② `- by` 算预测与真实的差；③ `.abs()` 取绝对值；④ `.mean(dim=0)` 在 batch 维度求平均，得到转向、油门两个输出各自的平均绝对误差（一个 2 元素张量）；⑤ `.sum()` 把两个误差加起来；⑥ `.item()` 把张量转成 Python 数字，累加到 `err`。
- `return err / len(loader.dataset)` —— 把累计误差除以样本总数，得到"平均每条样本的转向+油门绝对误差之和"。这个值越小，说明模型学得越像人类驾驶员。

### 第 7 段：训练主循环

```python
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
```

- `for epoch in range(20)` —— 训练 20 轮（epoch）。一轮 = 把训练集完整过一遍。20 是经验值：太少学不充分，太多可能过拟合。
- `model.train()` —— 切换到训练模式。与 `model.eval()` 配对，确保网络处于正确的状态。
- `total = 0.0` —— 本轮累计损失的计数器，每轮开始时清零。
- `for bx, by in train_loader` —— 逐批次从训练加载器取数据。由于 `shuffle=True`，每轮的批次顺序都不同。
- `loss = criterion(model(bx), by)` —— 前向传播 + 计算损失：`model(bx)` 让网络预测这批图片的转向/油门，`criterion(...)` 用 MSE 比较预测与真实标签的差距，得到一个标量损失值。
- `optimizer.zero_grad()` —— 清空上一轮的梯度。**必须清零**：PyTorch 默认会累积梯度（方便实现梯度累积等高级技巧），如果不清零，这一轮的梯度会叠加到上一轮上，导致更新方向错误。
- `loss.backward()` —— 反向传播：从损失出发，沿着计算图自动求出每个参数对损失的梯度（偏导数）。这一步是"学习"的数学基础——梯度告诉我们每个参数该往哪个方向调整、调整多少，才能让损失变小。
- `optimizer.step()` —— 优化器根据梯度更新所有参数。Adam 算法会为每个参数自适应调整学习率，比朴素的梯度下降收敛更快、更稳。
- `total += loss.item()` —— 把本批次的损失值累加到 `total`，用于统计这一轮的平均训练损失。`.item()` 把单元素张量转成 Python 数字。
- `if (epoch + 1) % 4 == 0` —— 每 4 轮打印一次（第 4、8、12、16、20 轮）。`epoch` 从 0 开始，所以要 +1 才是人类习惯的轮数。不每轮都打印是为了避免刷屏。
- `print(f"...")` —— 打印三项指标：当前轮数、本轮平均训练损失（`total/len(train_loader)`，即总损失÷批次数）、验证集误差（调用 `evaluate(val_loader)`）。**重点看验证误差**：训练损失低不代表模型好（可能过拟合），验证误差才是真实水平。

### 第 8 段：模拟路考与保存模型

```python
model.eval()  # 评估模式
print("\n===== 模拟路考（看模型预测 vs 人类驾驶） =====")
for i in [0, 20, 40, 59]:  # 从验证集抽 4 条
    img, label = val_set[i]  # 取样本
    with torch.no_grad():  # 关闭梯度
        pred = model(img.unsqueeze(0))[0]  # 加 batch 维度再预测
    print(f"样本 {i:2d}: 转向 真实={label[0]:+.3f} 预测={pred[0]:+.3f} | 油门 真实={label[1]:.3f} 预测={pred[1]:.3f}")

torch.save(model.state_dict(), "donkey_pilot.pt")  # 保存模型，可拷到真车上做推理
print("\n模型已保存到 donkey_pilot.pt —— 真车上加载它即可实现自动驾驶")
```

- `model.eval()` —— 再次切换到评估模式（虽然训练循环里每轮都调了 `train()`，最后一轮结束后模型还在训练模式，这里切回来）。
- `print("\n===== 模拟路考... =====")` —— 打印分隔标题。`\n` 是换行符，让输出更清晰。
- `for i in [0, 20, 40, 59]` —— 从 60 条验证集里抽 4 条（第 0、20、40、59 条），覆盖前、中、后不同位置，抽查更全面。
- `img, label = val_set[i]` —— 取出第 `i` 条验证样本。`img` 是 `(3, 120, 160)` 的图片张量，`label` 是 `(2,)` 的真实标签。
- `with torch.no_grad():` —— 关闭梯度，只前向不反向。
- `pred = model(img.unsqueeze(0))[0]` —— 让模型预测这一条样本。`img.unsqueeze(0)` 在最前面加一个 batch 维度，把 `(3, 120, 160)` 变成 `(1, 3, 120, 160)`，因为模型要求输入必须是批次（哪怕只有 1 条）；`[0]` 再把这个批次里唯一的一条预测结果取出来，得到 `(2,)` 的预测张量。
- `print(f"样本 {i:2d}: ...")` —— 打印这条样本的真实值与预测值，并排对比。`{label[0]:+.3f}` 的 `+` 表示正数也显示符号（+0.123），`.3f` 保留 3 位小数，方便看清微小差异。
- `torch.save(model.state_dict(), "donkey_pilot.pt")` —— 保存模型。`model.state_dict()` 是一个字典，包含所有层的参数（卷积核权重、全连接层权重等）；`torch.save` 把它序列化存到 `donkey_pilot.pt` 文件。**为什么只存参数不存整个模型**：`state_dict` 更灵活、更轻量，加载时先搭一个同样的空模型再 `load_state_dict` 灌入参数即可，不依赖类的定义。
- `print("模型已保存到 donkey_pilot.pt —— ...")` —— 提示保存成功，并说明下一步：这个文件可以拷到树莓派小车上，真车的推理代码加载它就能实现自动驾驶。

## 运行结果与解读

运行 `python examples/12_donkeycar_train.py` 后，输出如下（真实运行结果）：

```text
第  4 轮 | 训练损失=0.1606 | 验证误差(转向+油门)=0.0194
第  8 轮 | 训练损失=0.1545 | 验证误差(转向+油门)=0.0196
第 12 轮 | 训练损失=0.1556 | 验证误差(转向+油门)=0.0195
第 16 轮 | 训练损失=0.0139 | 验证误差(转向+油门)=0.0054
第 20 轮 | 训练损失=0.0009 | 验证误差(转向+油门)=0.0013

===== 模拟路考（看模型预测 vs 人类驾驶） =====
样本  0: 转向 真实=-0.669 预测=-0.708 | 油门 真实=0.424 预测=0.423
样本 20: 转向 真实=-0.232 预测=-0.221 | 油门 真实=0.588 预测=0.576
样本 40: 转向 真实=+0.735 预测=+0.701 | 油门 真实=0.401 预测=0.379
样本 59: 转向 真实=+0.330 预测=+0.323 | 油门 真实=0.585 预测=0.553

模型已保存到 donkey_pilot.pt —— 真车上加载它即可实现自动驾驶
```

**逐项解读：**

1. **训练损失与验证误差的变化** —— 前 12 轮损失在 0.15 附近徘徊，下降缓慢，这是 CNN 常见的"热身期"：网络在摸索基本的视觉特征（边缘、色块）。第 16 轮突然降到 0.0139，第 20 轮降到 0.0009，说明网络在第 12~16 轮之间"顿悟"了——找到了"车道线位置 → 转向/油门"的映射规律，损失断崖式下跌。这种"先平台后骤降"的曲线是深度学习的典型现象。

2. **验证误差始终低于训练损失** —— 第 4 轮训练损失 0.1606，验证误差只有 0.0194；第 20 轮训练损失 0.0009，验证误差 0.0013。**注意**：这两个指标单位不同——训练损失是 MSE（平方误差），验证误差是 MAE（绝对误差），不能直接比大小。但两者同步下降、最终都很小，说明模型没有过拟合（训练集、验证集表现一致）。

3. **模拟路考的 4 条样本** —— 每条都显示了"人类真实操控 vs 模型预测"的对比：
   - **样本 0**：人类左转 -0.669，模型预测 -0.708，误差 0.039，方向正确、幅度略大；油门几乎完美（0.424 vs 0.423）。
   - **样本 20**：人类轻微左转 -0.232，模型预测 -0.221，误差仅 0.011；油门误差 0.012，非常精准。
   - **样本 40**：人类右转 +0.735（大幅度），模型预测 +0.701，误差 0.034，方向正确、幅度略保守；油门误差 0.022。
   - **样本 59**：人类右转 +0.330，模型预测 +0.323，误差仅 0.007；油门误差 0.032，也很接近。

   **总体判断**：4 条样本的转向方向全部正确，幅度误差都在 0.04 以内（转向范围是 -1~+1，误差不到 2%）；油门误差也都在 0.04 以内（范围 0~1，误差不到 4%）。这个精度已经足够让小车沿赛道稳定行驶——模型通过了"模拟路考"，可以上车了。

4. **模型保存成功** —— 最后一行提示 `donkey_pilot.pt` 已生成。这个文件包含了训练好的全部参数，可以直接拷到树莓派上，用 Donkey Car 的推理代码加载，实现真正的自动驾驶。

## 小结

- **tub 数据格式**是 Donkey Car 的核心：图片文件与 JSON 记录通过文件名序号配对，JSON 里的 `cam/image_array` 字段是图片与标签的关联键。
- **自定义 Dataset** 要实现 `__init__`、`__len__`、`__getitem__` 三个方法，`__getitem__` 里完成"读图 → 预处理 → 读标签 → 返回样本"的完整流程。
- **CNN 回归模型**的结构是"卷积段提特征 + 全连接段做回归"，最后一层输出连续数值、不加激活函数；损失函数用 MSELoss，评估指标用平均绝对误差。
- **训练五步曲**（前向 → 清梯度 → 反向 → 更新 → 累计）是 PyTorch 的标准范式，本章与前面章节的唯一区别是损失函数从交叉熵换成了 MSE。
- **模拟路考**是模型上线前的必要检查：抽几条样本人工对比预测与真实，确认方向正确、误差可接受，再考虑部署。

## 动手练习

1. **调整网络深度**：在第三个卷积层后再加一层 `nn.Conv2d(64, 128, kernel_size=3)` 和 `nn.ReLU()`，观察训练损失是否下降更快、验证误差是否更小。注意：加了层后，Flatten 后的维度会变，需要根据新的特征图尺寸调整第一个 Linear 层的输入维度。

2. **尝试不同的学习率**：把 `lr=0.001` 改成 `lr=0.01` 或 `lr=0.0001`，对比训练 20 轮后的验证误差。学习率过大会看到损失震荡甚至发散（越来越大），过小会看到下降缓慢，找出最适合这个任务的学习率。

3. **增加数据增强**：在 `__getitem__` 里，读图后随机对图片做水平翻转（`img = np.fliplr(img).copy()`），同时把标签里的转向取反（`label[0] = -label[0]`），这样能把数据集"免费扩大一倍"。重新训练后对比验证误差，体会数据增强对提升模型泛化能力的作用。
