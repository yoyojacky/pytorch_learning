"""
第 11 章（进阶实战）：Donkey Car 数据架构 —— 采集（图片, 转向, 油门）
对应场景：自动驾驶小车的行为克隆第一步 —— 数据采集

Donkey Car 的数据架构（tub 格式）：
  data/tub/
    ├── 1_cam-image_array_.jpg   ← 摄像头图片
    ├── record_1.json            ← 与该图片同一时刻的驾驶数据（转向/油门等）
    ├── 2_cam-image_array_.jpg
    ├── record_2.json
    └── ...
每条 record 是一条 JSON，记录"看到这张图时，人/自动驾驶的操控是什么"：
  {"cam/image_array": "1_cam-image_array_.jpg",
   "user/angle": -0.42,      ← 转向，-1=最左，+1=最右
   "user/throttle": 0.55,    ← 油门，0~1
   "user/mode": "user",      ← 数据来源：user=人工遥控
   "timestamp": 1760000000}
图片与记录通过文件名里的序号一一对应 —— 这就是"图片+标注"的配对结构。

真车上：图片来自摄像头，angle/throttle 来自游戏手柄。
本脚本用程序合成"道路画面"来模拟一次遥控驾驶，生成的数据格式与真车完全一致，
将来有真车数据时直接替换 tub 目录即可无缝衔接第 12 章的训练。
"""
import json  # 写 JSON 记录文件
from pathlib import Path  # 目录处理
import numpy as np  # 生成图像数组
from PIL import Image  # 把数组保存成 jpg 图片

np.random.seed(42)  # 固定随机种子

TUB_DIR = Path("data/mytub")  # tub 数据目录（Donkey Car 的约定叫法）
TUB_DIR.mkdir(parents=True, exist_ok=True)  # 递归创建目录
IMG_H, IMG_W = 120, 160  # 图像尺寸 120x160 —— Donkey Car 的默认分辨率

def make_frame(angle):  # 根据当前转向角合成一帧"道路画面"
    """angle=-1 时车道线在最左，angle=+1 时在最右（模拟弯道中看到的画面）"""
    img = np.full((IMG_H, IMG_W, 3), 60, dtype=np.uint8)  # 深灰色路面打底
    img[:40] = 100  # 画面上方 1/3 是较亮的"远处/天空"
    line_x = int(IMG_W / 2 + angle * 60)  # 车道线横向位置随 angle 左右移动
    img[40:, line_x - 2: line_x + 2] = 255  # 下半部画一条白色车道线（宽 4 像素）
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)  # 模拟摄像头噪点
    return np.clip(img + noise, 0, 255).astype(np.uint8)  # 叠加噪点并限制在 0~255

# ========== 模拟一次驾驶：录制 360 帧 ==========
N = 360  # 录 360 条记录（真车上约等于 30 秒 × 12 帧/秒）
for i in range(1, N + 1):  # 序号从 1 开始
    angle = float(np.sin(i / 40) * 0.8 + np.random.uniform(-0.05, 0.05))
    # 转向角：正弦波模拟"S 形过弯"，幅度 ±0.8，再叠一点人手的抖动噪声
    angle = float(np.clip(angle, -1.0, 1.0))  # 转向物理范围限制在 [-1, 1]
    throttle = float(0.7 - 0.4 * abs(angle) + np.random.uniform(-0.02, 0.02))
    # 油门：弯道（|angle| 大）就收油，直道全油 —— 符合真实驾驶习惯

    img_name = f"{i}_cam-image_array_.jpg"  # Donkey Car 的图片命名约定
    Image.fromarray(make_frame(angle)).save(TUB_DIR / img_name)  # 保存 jpg 图片

    record = {  # 组装 Donkey Car 标准格式的记录
        "cam/image_array": img_name,  # 图片文件名（关联键）
        "user/angle": angle,  # 此刻的转向值
        "user/throttle": throttle,  # 此刻的油门值
        "user/mode": "user",  # 数据来源：人工驾驶
        "timestamp": 1760000000 + i,  # 时间戳
    }
    with open(TUB_DIR / f"record_{i}.json", "w") as f:  # 记录文件名与图片序号一致
        json.dump(record, f)  # 写入 JSON 文件

print(f"采集完成：{TUB_DIR}/ 下共 {N} 条记录（{N} 张 jpg + {N} 个 json）")
print(f"示例记录: {json.dumps({'cam/image_array': '1_cam-image_array_.jpg', 'user/angle': round(angle,3), 'user/throttle': round(throttle,3)}, ensure_ascii=False)}")
