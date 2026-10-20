"""
第 3 章：数据工程 —— Dataset 与 DataLoader
对应赛项知识点：构建结构化数据集，完成数据清洗和增强处理；
数据清洗、标准化及增强技术，确保数据质量符合模型输入要求
"""
import torch  # 导入 PyTorch
from torch.utils.data import Dataset, DataLoader  # Dataset 定义数据集，DataLoader 批量加载

torch.manual_seed(42)  # 固定随机种子保证可复现

# ========== 1. 模拟一份"脏数据"（数据工程的真实场景） ==========
raw_X = torch.randn(120, 4)  # 120 个样本，每个 4 个特征
raw_y = (raw_X[:, 0] + raw_X[:, 1] > 0).long()  # 标签：前两个特征之和>0 为 1，否则为 0

raw_X[5] = float("nan")  # 故意制造第 5 个样本为缺失值 NaN（脏数据）
raw_X[20] = 999.0  # 故意制造第 20 个样本为异常值 999（脏数据）

# ========== 2. 数据清洗：剔除含 NaN 或异常值的样本 ==========
mask = ~torch.isnan(raw_X).any(dim=1)  # dim=1 按行检查，any 表示该行任一特征为 NaN 就标记
mask &= (raw_X.abs().max(dim=1).values < 100)  # 再叠加条件：该行所有特征绝对值都要 < 100
X_clean = raw_X[mask]  # 布尔索引：只保留干净样本
y_clean = raw_y[mask]  # 标签同步过滤（特征和标签必须一一对应）
print(f"清洗前 {len(raw_X)} 条 -> 清洗后 {len(X_clean)} 条")  # 120 -> 118

# ========== 3. 标准化（Z-score）：均值 0、标准差 1 ==========
mean = X_clean.mean(dim=0)  # dim=0 按列求均值 -> 每个特征的均值
std = X_clean.std(dim=0)  # 每个特征的标准差
X_std = (X_clean - mean) / std  # 标准化公式：(x - μ) / σ
print("标准化后各特征均值≈0:", X_std.mean(dim=0).round())  # 验证结果

# ========== 4. 自定义 Dataset（PyTorch 数据的标准封装方式） ==========
class MyDataset(Dataset):  # 继承 Dataset 类，必须实现两个方法
    def __init__(self, X, y, augment=False):  # 初始化：传入特征、标签、是否做数据增强
        self.X = X  # 保存特征张量
        self.y = y  # 保存标签张量
        self.augment = augment  # 记录是否启用数据增强

    def __len__(self):  # 必须实现：返回数据集总样本数
        return len(self.X)

    def __getitem__(self, idx):  # 必须实现：根据索引 idx 取出一个样本
        x = self.X[idx]  # 取第 idx 个特征
        if self.augment:  # 训练时启用数据增强
            x = x + torch.randn_like(x) * 0.05  # 加 5% 高斯噪声，模拟数据增强
        return x, self.y[idx]  # 返回 (特征, 标签) 对

# ========== 5. 划分训练集/测试集 ==========
n_train = 100  # 前 100 条做训练，剩余做测试
train_set = MyDataset(X_std[:n_train], y_clean[:n_train], augment=True)  # 训练集启用增强
test_set = MyDataset(X_std[n_train:], y_clean[n_train:], augment=False)  # 测试集不增强！

# ========== 6. DataLoader：自动分批、打乱 ==========
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)  # 每批 16 条，每轮打乱顺序
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)  # 测试集无需打乱

for batch_X, batch_y in train_loader:  # 遍历 DataLoader，一次取一个批次
    print("一个批次的特征形状:", batch_X.shape)  # torch.Size([16, 4])
    print("一个批次的标签形状:", batch_y.shape)  # torch.Size([16])
    break  # 只看第一个批次就退出循环
