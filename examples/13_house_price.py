"""
第 13 章：生活场景实战 —— 二手房房价预测（回归任务端到端案例）
对应赛项知识点：构建结构化数据集；监督学习的应用场景；
对智能系统进行测试和训练并编写报告
任务：根据面积、房龄、楼层、距地铁距离、卧室数 5 个特征预测房屋总价（连续值）
流程：业务分析 → 数据探索(EDA) → 划分与标准化 → 线性基线 → 神经网络
      → 测试集终评 → 误差分析 → 模型保存与上线推理
"""
import torch  # 导入 PyTorch
import torch.nn as nn  # 神经网络模块
from torch.utils.data import DataLoader, TensorDataset  # 数据工具

torch.manual_seed(42)  # 固定随机种子，保证结果可复现

# ========== 步骤 1：业务分析 + 构造数据 ==========
# 业务问题：中介如何根据一套房的基本信息快速估算合理价格？
# 问题翻译：输入 5 个特征、输出一个连续数值 —— 典型的"回归"任务
n = 2000  # 模拟 2000 条房源记录

area  = torch.rand(n) * 120 + 40            # 面积：40~160 ㎡，均匀分布
age   = torch.rand(n) * 30                  # 房龄：0~30 年
floor = torch.randint(1, 34, (n,)).float()  # 楼层：1~33 层的整数
dist  = torch.rand(n) * 2.9 + 0.1           # 距地铁：0.1~3.0 公里
rooms = torch.randint(1, 6, (n,)).float()   # 卧室数：1~5 间
X = torch.stack([area, age, floor, dist, rooms], dim=1)  # 拼成特征矩阵 (2000, 5)：每行一套房

# 人为设定的"市场定价规律"（万元）——现实中它藏在数据里，模型的任务就是从数据中学会它
premium = (area - 120).clamp(min=0) * 1.5   # 超过 120㎡ 的大户型额外溢价（这是非线性项！）
y = (area * 4.5        # 面积是价格主力：每平方米约 4.5 万
     - age * 2.0       # 房龄折旧：每老一年约贬 2 万
     + floor * 0.3     # 楼层高一点略加分
     - dist * 20.0     # 离地铁越远越便宜：每公里减 20 万
     + rooms * 5.0     # 每多一间卧室加 5 万
     + premium         # 大户型溢价
     + 120)            # 基础价 120 万
y = y + torch.randn(n) * 15  # 叠加标准差 15 万的市场噪声（同小区同房型成交价也不同）

feature_names = ["面积", "房龄", "楼层", "距地铁", "卧室数"]  # 特征名，后面打印报告用
print(f"步骤 1 数据构造完成：X={tuple(X.shape)}（{n} 套房 × 5 个特征），y={tuple(y.shape)}")

# ========== 步骤 2：探索性数据分析（EDA）——先看数据再建模 ==========
print("\n步骤 2 各特征统计：")
for i, name in enumerate(feature_names):  # 逐特征检查取值范围是否合理
    col = X[:, i]  # 取出第 i 列（所有房源的该特征）
    print(f"  {name}: 最小 {col.min():.2f} | 最大 {col.max():.2f} | 均值 {col.mean():.2f} | 标准差 {col.std():.2f}")
print(f"  房价: 最小 {y.min():.2f} | 最大 {y.max():.2f} | 均值 {y.mean():.2f} | 标准差 {y.std():.2f}")

print("步骤 2 各特征与房价的相关系数（+ 正相关 / - 负相关 / 接近 0 弱线性相关）：")
for i, name in enumerate(feature_names):
    corr = torch.corrcoef(torch.stack([X[:, i], y]))[0, 1]  # 皮尔逊相关系数，取值 [-1, 1]
    print(f"  {name}: {corr:+.4f}")

# ========== 步骤 3：划分数据集 + 标准化（统计量只用训练集，防数据泄漏） ==========
idx = torch.randperm(n)  # 把 0~1999 的序号随机打乱
train_idx, val_idx, test_idx = idx[:1400], idx[1400:1700], idx[1700:]  # 70% 训练 / 15% 验证 / 15% 测试
X_train, y_train = X[train_idx], y[train_idx]  # 用同一组序号同时索引 X 和 y，保证一一对应
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

x_mean, x_std = X_train.mean(dim=0), X_train.std(dim=0)  # 关键：只在训练集上算，验证/测试集套用
y_mean, y_std = y_train.mean(), y_train.std()            # 标签也要缩放（房价几百万，直接回归 loss 巨大）
X_train = (X_train - x_mean) / x_std  # 训练集 Z-score 标准化
X_val   = (X_val   - x_mean) / x_std  # 验证集套用【训练集】的统计量
X_test  = (X_test  - x_mean) / x_std  # 测试集同理
y_train_s = (y_train - y_mean) / y_std  # 标签标准化（_s 表示 scaled）
y_val_s   = (y_val   - y_mean) / y_std
# y_test 故意不标准化：最终评估必须在"万元"的业务尺度上进行

train_loader = DataLoader(TensorDataset(X_train, y_train_s), batch_size=64, shuffle=True)  # 训练：打乱
val_loader   = DataLoader(TensorDataset(X_val,   y_val_s),   batch_size=256)               # 验证：不打乱
print(f"\n步骤 3 划分+标准化完成：训练 {len(X_train)} / 验证 {len(X_val)} / 测试 {len(X_test)}")
print(f"  训练集特征均值（应≈0）: {X_train.mean(dim=0)}")
print(f"  验证集特征均值（不恰好为 0，恰是没有数据泄漏的证明）: {X_val.mean(dim=0)}")

# ========== 步骤 4：通用训练函数（带最佳参数存档）+ 训练两个模型 ==========
criterion = nn.MSELoss()  # 回归任务的标准损失：均方误差（分类用交叉熵，回归用 MSE）

def train_model(model, epochs=300):
    """通用训练循环：验证创新高时把参数复制存档，结束后恢复到最佳那一轮"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器绑定模型参数，函数内新建
    best_loss, best_state = float("inf"), None  # 回归比"谁 loss 小"，初值设无穷大
    for epoch in range(epochs):
        model.train()  # 训练模式
        for bx, by in train_loader:  # 训练五步曲
            loss = criterion(model(bx).squeeze(1), by)  # squeeze(1)：(64,1)->(64,)，回归经典形状坑
            optimizer.zero_grad()  # 清梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        model.eval()  # 评估模式
        with torch.no_grad():  # 验证集按样本数加权求整体 MSE（最后一批可能不满）
            val_loss = sum(criterion(model(bx).squeeze(1), by).item() * len(bx)
                           for bx, by in val_loader) / len(X_val)
        if val_loss < best_loss:  # 验证表现创新高
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}  # 复制存档，必须 clone
    model.load_state_dict(best_state)  # 训练结束恢复到最佳那一轮，而不是停在最后一轮
    return best_loss

torch.manual_seed(42)  # 训练前重置种子：建模型→训练的随机序列完全可控、可复现
linear_model = nn.Linear(5, 1)  # 基线：线性回归 = 没有隐藏层的迷你神经网络（仅 6 个参数）
linear_best = train_model(linear_model)
print(f"\n步骤 4 线性回归：参数量 {sum(p.numel() for p in linear_model.parameters())} | 最佳验证loss={linear_best:.4f}")

torch.manual_seed(42)  # 同样重置，保证 MLP 的训练也可复现
mlp = nn.Sequential(  # 神经网络：每个 ReLU 是一个"折点"，能拟合 120㎡ 溢价这种非线性转折
    nn.Linear(5, 32),   # 输入 5 -> 隐藏层 32
    nn.ReLU(),          # 非线性激活（没有它，多层线性叠起来仍是一条直线）
    nn.Linear(32, 16),  # 32 -> 16
    nn.ReLU(),          # 非线性激活
    nn.Linear(16, 1),   # 16 -> 1 个价格
)
mlp_best = train_model(mlp)
print(f"步骤 4 MLP：     参数量 {sum(p.numel() for p in mlp.parameters())} | 最佳验证loss={mlp_best:.4f}")
print(f"         理论下限：{(15.0 / y_std) ** 2:.4f}（噪声决定的极限，任何模型都突破不了）")

# ========== 步骤 5：测试集终评（MAE / RMSE / R²） ==========
def predict(model, X_data):
    """推理 + 逆变换回万元尺度（指标必须用业务尺度报告）"""
    model.eval()  # 评估模式
    with torch.no_grad():  # 关闭梯度，省内存提速
        return model(X_data).squeeze(1) * y_std + y_mean  # 标准化尺度 → 万元

def evaluate_regression(name, pred, true):
    """回归三大指标：MAE 平均估错多少万；RMSE 严惩大误差（恒≥MAE）；R² 解释了多少价格波动"""
    mae  = (pred - true).abs().mean()                                          # 平均绝对误差
    rmse = ((pred - true) ** 2).mean().sqrt()                                  # 均方根误差：先平方再平均最后开方
    r2   = 1 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum()  # 决定系数，1=完美 0=猜均价
    print(f"  {name}: MAE={mae:.1f} 万 | RMSE={rmse:.1f} 万 | R²={r2:.4f}")

print("\n步骤 5 测试集最终报告（300 套从未参与任何训练决策的房子）：")
evaluate_regression("线性回归", predict(linear_model, X_test), y_test)
evaluate_regression("MLP     ", predict(mlp, X_test),          y_test)

# ========== 步骤 6：误差分析 —— 模型在哪些房子上失手？ ==========
pred_mlp = predict(mlp, X_test)  # 测试集预测价（万元）
errors = pred_mlp - y_test       # 每套房的带符号误差（正=估高了，负=估低了）

print("\n步骤 6 分价位段误差分析：")
for lo, hi in [(0, 350), (350, 500), (500, 650), (650, 1000)]:
    m = (y_test >= lo) & (y_test < hi)  # 布尔掩码挑出该价位段（逐元素与用 &，不能写 and）
    if m.any():
        mae_b = errors[m].abs().mean()                        # 绝对误差（万元）
        rel_b = (errors[m].abs() / y_test[m]).mean() * 100    # 相对误差（占房价的 %）
        print(f"  {lo}~{hi} 万: {m.sum().item():3d} 套 | 平均误差 {mae_b:.1f} 万 | 相对误差 {rel_b:.1f}%")

print("步骤 6 误差最大的 5 套房：")
top5 = errors.abs().argsort(descending=True)[:5]  # argsort 返回的是下标，按绝对误差从大到小取前 5
for i in top5:
    area_i = X_test[i][0] * x_std[0] + x_mean[0]  # 面积从标准化尺度逆变换回平米
    print(f"  第 {i.item():3d} 套: 面积 {area_i:.0f}㎡ | 真实 {y_test[i]:.1f} 万 | 预测 {pred_mlp[i]:.1f} 万 | 误差 {errors[i]:+.1f} 万")

# ========== 步骤 7：保存模型（含预处理统计量）+ 模拟上线推理 ==========
torch.save({
    "model_state": mlp.state_dict(),   # 最佳模型参数
    "x_mean": x_mean, "x_std": x_std,  # 特征标准化统计量（上线后新数据必须做同一套变换！）
    "y_mean": y_mean, "y_std": y_std,  # 标签统计量（预测完要逆变换回万元）
}, "house_pricing_model.pt")  # 一个文件打包交付
print("\n步骤 7 模型已保存到 house_pricing_model.pt（参数 + 预处理统计量打包）")

ckpt = torch.load("house_pricing_model.pt")  # 模拟另一台机器上的上线程序：读回整个包
model_online = nn.Sequential(  # state_dict 只存参数不存结构，先搭一个结构相同的"空壳"
    nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1),
)
model_online.load_state_dict(ckpt["model_state"])  # 灌入保存的参数
model_online.eval()  # 上线永远是评估模式

new_house = torch.tensor([[89.0, 5.0, 12.0, 0.8, 3.0]])  # 一套新房：89㎡/5年房龄/12楼/0.8km/3室
x_new = (new_house - ckpt["x_mean"]) / ckpt["x_std"]     # 用保存的训练集统计量做标准化
with torch.no_grad():
    price = model_online(x_new).squeeze() * ckpt["y_std"] + ckpt["y_mean"]  # 预测 + 逆变换回万元
print(f"  上线估价：89㎡/5年房龄/12楼/距地铁0.8km/3室 → {price:.1f} 万"
      f"（对照公式价 {89*4.5 - 5*2 + 12*0.3 - 0.8*20 + 3*5 + 120:.1f} 万，89㎡ 不到 120 无溢价）")
