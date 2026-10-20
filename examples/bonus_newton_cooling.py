"""
附加案例：牛顿冷却定律 —— Autograd 解决现实中的参数反演问题
场景：测得一杯热水 30 分钟内的温度记录（含测量误差），
用自动求导反推出物理参数，预测"什么时候降到 60°C 可以喝"。
"""
import torch  # 导入 PyTorch

torch.manual_seed(0)  # 固定随机种子

# ========== 1. 现实背景 ==========
# 牛顿冷却定律：物体温度 T 随时间 t 的变化为  T(t) = T环境 + A · e^(-k·t)
# T环境=环境温度，A=初始温差，k=冷却系数（杯子保温能力）
# 现实中这三个参数都无法直接测量，只能从温度记录中"反演"出来

# 模拟真实测量：真实参数是 25°C 室温、初始温差 70°C、k=0.08（假装我们不知道）
t = torch.arange(0, 31, 2.0)  # 每 2 分钟测一次，共 16 个测量点
T_measured = 25 + 70 * torch.exp(-0.08 * t) + torch.randn(len(t)) * 0.8  # 加 0.8°C 测量噪声

# ========== 2. 用 Autograd 反演参数 ==========
# 为什么必须用梯度下降？因为 k 藏在指数 e^(-k·t) 里，模型对 k 是非线性的，
# 不像 y=wx+b 那样有现成的求解公式 —— 这正是 Autograd 的用武之地
T_env = torch.tensor(20.0, requires_grad=True)  # 估计值：环境温度，随便猜个 20 起步
A = torch.tensor(50.0, requires_grad=True)  # 估计值：初始温差，猜 50
k = torch.tensor(0.05, requires_grad=True)  # 估计值：冷却系数，猜 0.05

optimizer = torch.optim.Adam([T_env, A, k], lr=0.1)  # Adam 自动处理三个参数梯度量级差异大的问题
# （k 只有 0.0x 量级，T_env 是几十，梯度差上千倍，手动设统一学习率很难调）

for epoch in range(3000):  # 迭代 3000 次
    T_pred = T_env + A * torch.exp(-k * t)  # 用当前估计的参数算预测温度曲线
    loss = ((T_pred - T_measured) ** 2).mean()  # 预测曲线与实测数据的均方误差
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # Autograd：一次调用同时算出 loss 对三个参数的梯度
    optimizer.step()  # 沿梯度反方向更新三个参数

# ========== 3. 看结果：参数反演得准不准 ==========
print("===== 参数反演结果 =====")
print(f"环境温度 T_env: 学到 {T_env.item():.2f}°C（真实 25°C）")
print(f"初始温差 A:     学到 {A.item():.2f}°C（真实 70°C）")
print(f"冷却系数 k:     学到 {k.item():.4f}（真实 0.08）")
print(f"最终 loss: {loss.item():.4f}（≈ 噪声方差 0.8²=0.64 即为完美）")

# ========== 4. 解决实际问题：几分钟后降到 60°C？ ==========
# 用学到的公式反解：60 = T_env + A·e^(-k·t)  →  t = -ln((60-T_env)/A) / k
t_drink = -torch.log((60 - T_env) / A) / k  # 代入学到的参数
t_true = -torch.log(torch.tensor((60 - 25) / 70)) / 0.08  # 用真实参数算的标准答案
print("\n===== 实际问题：什么时候能喝 =====")
print(f"模型预测: {t_drink.item():.2f} 分钟后降到 60°C")
print(f"标准答案: {t_true.item():.2f} 分钟")
