"""
部署第 2 步（在小车上运行）：加载 TorchScript 模型进行实时驾驶推理
这个文件不需要知道模型的任何结构定义 —— 一切都在 donkey_pilot_scripted.pt 里

两种用法：
  A. 作为 Donkey Car 的自定义 part 接入 manage.py（推荐，见 deploy/README.md）
  B. 直接运行本文件做加载自测：  python3 car_pilot.py
"""
import numpy as np  # 摄像头画面在 Donkey Car 里是 numpy 数组
import torch  # 小车上只需 CPU 版 torch

IMG_H, IMG_W = 120, 160  # 必须和训练时的图像尺寸一致（120x160）


class Pilot:
    """驾驶模型 part：输入一帧画面，输出 (转向, 油门)，与 Donkey Car part 规范兼容"""

    def __init__(self, model_path="donkey_pilot_scripted.pt"):
        self.model = torch.jit.load(model_path)  # 加载 TorchScript 模型（单文件，无需结构定义）
        self.model.eval()  # 推理模式

    def run(self, img_arr):  # Donkey Car 约定：part 的输入方法叫 run
        """img_arr: 摄像头画面，numpy 数组 (120,160,3)，uint8，RGB —— 就是 tub 里 cam/image_array
        返回: (angle, throttle)，与 tub 里 user/angle、user/throttle 同含义同量程"""
        img = img_arr.astype(np.float32) / 255.0  # 归一化到 0~1（与训练预处理一致）
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # HWC->CHW，再加 batch 维度
        with torch.no_grad():  # 推理不需要梯度
            out = self.model(img)[0]  # 前向推理，取 batch 中第 0 条
        angle = float(out[0])  # 预测转向 [-1, 1]
        throttle = float(out[1])  # 预测油门 [0, 1]
        return angle, throttle

    def shutdown(self):  # Donkey Car 约定：part 可提供 shutdown 做清理
        pass  # 本模型无需清理


if __name__ == "__main__":  # 直接运行本文件 = 加载自测
    pilot = Pilot()  # 加载模型
    test_img = np.random.randint(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)  # 造一帧假画面
    angle, throttle = pilot.run(test_img)  # 推理一次
    print(f"模型加载成功！测试推理: 转向={angle:+.3f}, 油门={throttle:.3f}")

    import time  # 测一下单帧推理耗时（决定小车能达到的驾驶帧率）
    t0 = time.time()  # 计时起点
    for _ in range(20):  # 连续推理 20 帧
        pilot.run(test_img)
    fps = 20 / (time.time() - t0)  # 计算帧率
    print(f"推理速度: {fps:.1f} 帧/秒（≥10 帧/秒即可流畅驾驶）")
