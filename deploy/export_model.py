"""
部署第 1 步（在开发机上运行）：把训练好的模型导出为 TorchScript 单文件
为什么要 TorchScript：donkey_pilot.pt 里只存了权重数字，加载时必须先有模型结构定义；
TorchScript 把"结构 + 权重"打包成一个文件，小车上不需要任何模型定义代码即可加载。
"""
import torch  # 导入 PyTorch
import torch.nn as nn  # 神经网络模块

# ========== 1. 重建与训练时完全相同的模型结构 ==========
model = nn.Sequential(  # 必须与 examples/12_donkeycar_train.py 中一致
    nn.Conv2d(3, 24, kernel_size=5, stride=2),  # 3通道 -> 24特征图
    nn.ReLU(),  # 激活
    nn.Conv2d(24, 32, kernel_size=5, stride=2),  # 24 -> 32
    nn.ReLU(),  # 激活
    nn.Conv2d(32, 64, kernel_size=5, stride=2),  # 32 -> 64
    nn.ReLU(),  # 激活
    nn.Conv2d(64, 64, kernel_size=3),  # 64 -> 64
    nn.ReLU(),  # 激活
    nn.Flatten(),  # 拉平为 64*10*15=9600 维
    nn.Linear(64 * 10 * 15, 100),  # 全连接
    nn.ReLU(),  # 激活
    nn.Linear(100, 50),  # 100 -> 50
    nn.ReLU(),  # 激活
    nn.Linear(50, 2),  # 输出 [转向, 油门]
)
model.load_state_dict(torch.load("donkey_pilot.pt"))  # 装入训练好的权重
model.eval()  # 切换到推理模式

# ========== 2. 导出 TorchScript ==========
scripted = torch.jit.script(model)  # 把模型编译成 TorchScript（结构+权重打包）
scripted.save("donkey_pilot_scripted.pt")  # 保存为单文件

# ========== 3. 验证：导出前后输出必须一致 ==========
dummy = torch.rand(1, 3, 120, 160)  # 造一张假图片做测试输入
with torch.no_grad():  # 关闭梯度
    out_before = model(dummy)  # 导出前的输出
    loaded = torch.jit.load("donkey_pilot_scripted.pt")  # 重新加载导出的文件
    out_after = loaded(dummy)  # 导出后的输出
diff = (out_before - out_after).abs().max().item()  # 两次输出的最大差异
print(f"导出前输出: {out_before[0].tolist()}")
print(f"导出后输出: {out_after[0].tolist()}")
print(f"最大差异: {diff:.2e} —— {'✅ 一致，可以部署' if diff < 1e-6 else '❌ 不一致，需排查'}")
