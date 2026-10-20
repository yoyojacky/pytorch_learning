"""
第 1 章：张量（Tensor）基础
对应赛项知识点：PyTorch 工具链基础、线性代数在模型中的实现路径
"""
import torch  # 导入 PyTorch 库，约定俗成命名为 torch

# ========== 1. 创建张量 ==========
a = torch.tensor([1.0, 2.0, 3.0])  # 从 Python 列表创建一维浮点张量
b = torch.zeros(2, 3)  # 创建 2 行 3 列的全 0 张量
c = torch.ones(3, 2)  # 创建 3 行 2 列的全 1 张量
d = torch.randn(2, 4)  # 创建 2 行 4 列的随机张量（标准正态分布）
e = torch.arange(0, 10, 2)  # 创建等差序列 [0,2,4,6,8]，类似 range

print("a =", a)  # 打印张量 a 的内容
print("b 的形状:", b.shape)  # shape 属性查看张量形状 -> torch.Size([2, 3])
print("d 的数据类型:", d.dtype)  # dtype 属性查看元素类型 -> torch.float32
print("d 所在设备:", d.device)  # device 属性查看张量在 CPU 还是 GPU -> cpu

# ========== 2. 张量运算（线性代数基础） ==========
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2x2 矩阵
y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])  # 另一个 2x2 矩阵

add = x + y  # 对应位置元素相加（逐元素加法）
mul = x * y  # 对应位置元素相乘（逐元素乘法，不是矩阵乘法！）
matmul = x @ y  # 矩阵乘法（线性代数中的点积），等价于 torch.matmul(x, y)

print("\n逐元素加法:\n", add)
print("逐元素乘法:\n", mul)
print("矩阵乘法:\n", matmul)  # 结果 [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]

# ========== 3. 广播机制 ==========
v = torch.tensor([10.0, 20.0])  # 形状 (2,)
broadcast = x + v  # v 自动"复制"成两行再相加，这就是广播
print("\n广播加法:\n", broadcast)  # [[11,22],[13,24]]

# ========== 4. 形状变换 ==========
t = torch.arange(12)  # 0~11 的一维张量，形状 (12,)
t2 = t.reshape(3, 4)  # 重塑为 3 行 4 列
t3 = t2.flatten()  # 展平回一维
t4 = t2.transpose(0, 1)  # 转置：交换第 0 维和第 1 维，形状变为 (4, 3)

print("\nreshape 后:\n", t2)
print("转置后形状:", t4.shape)

# ========== 5. 索引与切片 ==========
print("\n第 0 行:", t2[0])  # 取第 0 行 -> tensor([0,1,2,3])
print("第 1 列:", t2[:, 1])  # 取所有行的第 1 列 -> tensor([1,5,9])
print("前两行后两列:\n", t2[:2, 2:])  # 切片：行 0-1，列 2-3

# ========== 6. 与 NumPy 互转（数据工程中常见） ==========
import numpy as np  # 导入 NumPy
arr = np.array([[1, 2], [3, 4]], dtype=np.float32)  # NumPy 数组
tensor_from_np = torch.from_numpy(arr)  # NumPy -> 张量（共享内存）
back_to_np = tensor_from_np.numpy()  # 张量 -> NumPy
print("\nNumPy 转张量:\n", tensor_from_np)
