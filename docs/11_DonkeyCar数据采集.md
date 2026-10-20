# 第 11 章：DonkeyCar数据采集

## 本章目标

- 了解 Donkey Car（开源树莓派自动驾驶小车）项目的基本概念：行为克隆（Behavior Cloning）+ 端到端神经网络
- 掌握 Donkey Car 的 tub 数据架构：图片文件与 JSON 记录文件通过文件名序号一一配对
- 学会用 NumPy 程序合成"道路画面"、用 PIL 保存 jpg、用 json 模块写标注记录，完整模拟一次数据采集
- 理解转向（angle，-1~+1）与油门（throttle，0~1）这两个回归标签的物理含义和取值约束
- 为第 12 章的 TubDataset + CNN 训练准备好与真车完全同构的数据目录

## 对应赛项知识点

本章对应"人工智能训练师"职业技能大赛的以下要求：**数据工程——数据采集与数据标注规范**，以及**智能系统设计——自动驾驶场景的感知与决策数据链路设计**。

比赛中的数据工程模块常要求选手：明确"模型需要什么数据 → 数据从哪来 → 以什么格式存 → 图片与标注如何关联"这一完整链路。本章脚本头部 docstring 里写得很清楚——tub 格式下每张图片 `N_cam-image_array_.jpg` 配一条 `record_N.json`，JSON 里记录"看到这张图时人的操控是什么"（`user/angle` 转向、`user/throttle` 油门、`user/mode` 数据来源、`timestamp` 时间戳）。这正是行为克隆任务的标注规范：**输入是图像，标签是人的驾驶行为**。掌握这种"图片+结构化标注"的配对思想，就掌握了视觉类监督学习数据采集的通用套路。

## 核心概念讲解

### Donkey Car 是什么

Donkey Car 是一个开源的自动驾驶遥控车项目：给普通遥控车装上树莓派和摄像头，让它通过学习人类的驾驶行为实现自动沿赛道行驶。它的技术路线是经典的**行为克隆（Behavior Cloning）**：

1. 人用手柄遥控小车跑赛道，同时录制"每一刻摄像头看到的画面 + 手柄的操作"；
2. 用这批数据训练一个神经网络，让网络学会"看到这样的画面，就该给出这样的转向和油门"；
3. 把车交给网络开——摄像头画面实时送入网络，网络输出的转向/油门直接控制电机。

行为克隆本质上就是监督学习，只是"老师"不是人工逐张打标，而是人开车时的操作被天然地记录成了标签。这也是它最迷人的地方：**采集过程就是标注过程**。

### tub 数据架构

Donkey Car 把一次录制的数据放在一个叫 tub（澡盆，约定俗成的叫法）的目录里：

```
data/tub/
  ├── 1_cam-image_array_.jpg   ← 摄像头图片
  ├── record_1.json            ← 同一时刻的驾驶数据
  ├── 2_cam-image_array_.jpg
  ├── record_2.json
  └── ...
```

关键设计是**文件名序号配对**：图片叫 `N_cam-image_array_.jpg`，记录叫 `record_N.json`，序号 N 相同即为同一时刻的一对（输入，标签）。JSON 记录内部再存图片文件名 `cam/image_array` 作为关联键，形成双重冗余——即使文件名规则变了，靠 JSON 内的字段也能找回对应图片。这种"文件序号 + 内容字段"的配对方式，在自建数据集时非常值得借鉴。

一条标准记录长这样：

```json
{"cam/image_array": "1_cam-image_array_.jpg",
 "user/angle": -0.42,
 "user/throttle": 0.55,
 "user/mode": "user",
 "timestamp": 1760000000}
```

- `user/angle`：转向，**-1 表示打死向左，+1 表示打死向右**，0 为直行。键名前缀 `user/` 表示这条操控来自人工遥控（`pilot/` 前缀则来自自动驾驶）。
- `user/throttle`：油门，**0~1**，越大越快。
- `timestamp`：时间戳，用于按时间排序、排查丢帧。

### 为什么转向和油门是回归而不是分类

转向角是连续值：-0.02 和 -0.03 只有细微差别，把它们硬切成"左/中/右"三类会损失大量信息（方向盘打死和轻打都被归进"左"）。因此这类控制量预测是**回归任务**——网络输出连续数值，训练时用 MSELoss（均方误差）衡量预测与真人操控的差距。这与前面章节分类任务"argmax 取类别、交叉熵算损失"的范式不同，第 12 章会完整展示回归式训练。

### 程序合成画面：为什么模拟

真车上，图片来自摄像头、angle/throttle 来自游戏手柄。但学习阶段手里没有车，本章用程序合成"道路画面"来模拟一次遥控驾驶：画深灰路面、远处天空、一条随转向左右摆动的白色车道线，再叠噪点。**生成的数据格式与真车完全一致**，将来拿到真车数据，只需替换 tub 目录，第 12 章的训练代码无需任何改动——这就是"架构先行、数据可换"的工程思想。

## 完整代码

```python
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
```

## 逐行讲解

### 模块导入与随机种子

```python
import json  # 写 JSON 记录文件
from pathlib import Path  # 目录处理
import numpy as np  # 生成图像数组
from PIL import Image  # 把数组保存成 jpg 图片

np.random.seed(42)  # 固定随机种子
```

- `import json` —— 导入 Python 标准库的 JSON 模块，负责把 Python 字典序列化成 JSON 文本写入 `record_N.json`。JSON 是 Donkey Car 记录文件的格式，也是数据交换领域最通用的文本格式：可读、跨语言、无需安装第三方库。
- `from pathlib import Path` —— 导入面向对象的路径处理类。相比传统的 `os.path`，`Path` 可以用 `/` 运算符直接拼接路径（如 `TUB_DIR / img_name`），代码更直观，还自带 `mkdir` 等目录操作方法。
- `import numpy as np` —— NumPy 负责生成和操作图像数组。一张图片在程序里就是一个三维数组 `(高, 宽, 通道)`，合成道路画面本质上是"往数组的特定位置填数值"。
- `from PIL import Image` —— PIL（Pillow）是 Python 最常用的图像库，`Image.fromarray()` 能把 NumPy 数组变成图片对象，再 `.save()` 存成 jpg。NumPy 管"数值"，PIL 管"图片文件"，两者配合完成落地。
- `np.random.seed(42)` —— 固定随机种子。本脚本有两处随机：车道画面的噪点、转向/油门上的抖动噪声。固定种子后每次运行生成的数据和打印结果完全一致，方便复现和写报告；42 是社区惯用的"魔法数字"，换任何整数都行，关键是固定。

### 常量与目录准备

```python
TUB_DIR = Path("data/mytub")  # tub 数据目录（Donkey Car 的约定叫法）
TUB_DIR.mkdir(parents=True, exist_ok=True)  # 递归创建目录
IMG_H, IMG_W = 120, 160  # 图像尺寸 120x160 —— Donkey Car 的默认分辨率
```

- `TUB_DIR = Path("data/mytub")` —— 定义 tub 数据目录为当前目录下的 `data/mytub`（Donkey Car 项目里习惯叫 `data/tub`，这里用 `mytub` 以示"自己录的这一趟"）。此时只是创建了一个路径对象，磁盘上还没有这个目录。
- `TUB_DIR.mkdir(parents=True, exist_ok=True)` —— 真正创建目录。两个参数都很重要：`parents=True` 表示父目录 `data/` 不存在时也一并创建（递归创建）；`exist_ok=True` 表示目录已存在时不报错——没有它，第二次运行脚本就会抛 `FileExistsError`。
- `IMG_H, IMG_W = 120, 160` —— 图像尺寸：高 120 像素、宽 160 像素，这是 Donkey Car 的默认分辨率。选这么小是有讲究的：小车端用树莓派推理，算力有限，图片越小，CNN 前向计算越快；同时 120×160 已足以容纳车道线等关键信息。一行里用多重赋值同时定义两个常量，是 Python 的惯用写法。

### make_frame：合成一帧道路画面

```python
def make_frame(angle):  # 根据当前转向角合成一帧"道路画面"
    """angle=-1 时车道线在最左，angle=+1 时在最右（模拟弯道中看到的画面）"""
    img = np.full((IMG_H, IMG_W, 3), 60, dtype=np.uint8)  # 深灰色路面打底
    img[:40] = 100  # 画面上方 1/3 是较亮的"远处/天空"
    line_x = int(IMG_W / 2 + angle * 60)  # 车道线横向位置随 angle 左右移动
    img[40:, line_x - 2: line_x + 2] = 255  # 下半部画一条白色车道线（宽 4 像素）
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)  # 模拟摄像头噪点
    return np.clip(img + noise, 0, 255).astype(np.uint8)  # 叠加噪点并限制在 0~255
```

- `def make_frame(angle):` —— 定义画面合成函数，入参 `angle` 是当前转向角。核心思想：**画面内容由 angle 唯一决定**——这正是行为克隆能成立的前提（画面里必须包含足以推断转向的信息，网络才有东西可学）。
- docstring 说明映射规则：angle=-1 车道线在最左、+1 在最右。想象在弯道上：需要向右打方向时，车道线在视野里偏左；需要向左打时，车道线偏右。函数用这个简化的对应关系模拟"过弯中看到的画面"。
- `img = np.full((IMG_H, IMG_W, 3), 60, dtype=np.uint8)` —— 创建一个 `(120, 160, 3)` 的数组，全部填 60（深灰色）作为路面底色。`dtype=np.uint8` 指定无符号 8 位整数，取值 0~255，这是图像数据的标准类型；3 个通道对应 RGB 三原色，三个通道数值相等时呈现灰色。
- `img[:40] = 100` —— 把数组前 40 行（画面上方 1/3）整体赋为 100（比路面亮一点的灰），模拟远处的路面/天空，让画面有简单的纵深感。`img[:40]` 是 NumPy 切片：行取 0~39，列和通道全取。
- `line_x = int(IMG_W / 2 + angle * 60)` —— 计算车道线的横向位置：`IMG_W / 2`（80）是画面中心，`angle * 60` 让车道线随转向左右偏移，angle=+1 时偏到 140（接近最右），angle=-1 时偏到 20（接近最左）。`int()` 转成整数，因为像素索引必须是整数。
- `img[40:, line_x - 2: line_x + 2] = 255` —— 在画面下半部（第 40 行到底）、`line_x` 左右各 2 列的窄条区域填 255（纯白），画出一条宽 4 像素的白色车道线。车道线只画在下半部，模拟"近处看得清车道线、远处看不清"的透视效果。
- `noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)` —— 生成与图像同形状的随机噪声数组，每个像素每个通道取 0~19 的随机整数，模拟真实摄像头不可避免的低幅噪点。**注意**：没有噪点的"完美画面"会让网络学得太容易，加一点噪声能让数据更接近真实、模型更鲁棒。
- `return np.clip(img + noise, 0, 255).astype(np.uint8)` —— 噪声叠加到画面上（`img + noise`），用 `np.clip` 把越界的像素值截断回 [0, 255]（比如白线 255+15 会被压回 255），最后 `.astype(np.uint8)` 确保数据类型正确——因为加法运算可能让 NumPy 自动提升类型，而 PIL 保存 jpg 要求 uint8。

### 模拟驾驶主循环：生成转向与油门

```python
N = 360  # 录 360 条记录（真车上约等于 30 秒 × 12 帧/秒）
for i in range(1, N + 1):  # 序号从 1 开始
    angle = float(np.sin(i / 40) * 0.8 + np.random.uniform(-0.05, 0.05))
    # 转向角：正弦波模拟"S 形过弯"，幅度 ±0.8，再叠一点人手的抖动噪声
    angle = float(np.clip(angle, -1.0, 1.0))  # 转向物理范围限制在 [-1, 1]
    throttle = float(0.7 - 0.4 * abs(angle) + np.random.uniform(-0.02, 0.02))
    # 油门：弯道（|angle| 大）就收油，直道全油 —— 符合真实驾驶习惯
```

- `N = 360` —— 共录制 360 条记录。真车上 Donkey Car 默认录制帧率约 12 帧/秒，360 帧约等于连续开 30 秒——一趟不算长的赛道练习。
- `for i in range(1, N + 1):` —— 循环变量 `i` 从 1 到 360。**从 1 开始而非 Python 惯例的 0**，是因为 Donkey Car 的文件序号约定从 1 起：`1_cam-image_array_.jpg`、`record_1.json`。数据格式与官方生态对齐，将来用官方工具（如 tub 查看器）打开才不会错位。
- `angle = float(np.sin(i / 40) * 0.8 + np.random.uniform(-0.05, 0.05))` —— 生成第 i 帧的转向角，分两部分：`np.sin(i / 40) * 0.8` 用正弦波模拟"S 形过弯"——i 从 1 到 360，`i/40` 从 0 到 9，正弦值在 -1~1 之间摆动约 1.4 个周期，乘 0.8 把转向幅度限制在 ±0.8（不打满，符合平稳驾驶）；`np.random.uniform(-0.05, 0.05)` 叠加 ±0.05 的均匀分布随机抖动，模拟真人打方向盘时手不可能完全平滑。`float()` 把 NumPy 标量转成 Python 原生 float——后面 `json.dump` 对原生类型的序列化最稳妥。
- `angle = float(np.clip(angle, -1.0, 1.0))` —— 把转向角截断到物理合法范围 [-1, 1]。当前参数下 sin 波加抖动最多到 0.85，这行其实不会触发，但它是**防御性编程**：万一以后调大波幅或抖动，标签也不会越界。标签越界是数据采集的大忌——模型学到一个不可能执行的操控，真车执行时可能被舵机限位"硬切"，造成行为偏差。
- `throttle = float(0.7 - 0.4 * abs(angle) + np.random.uniform(-0.02, 0.02))` —— 生成油门，规则是"弯道收油、直道全油"：基础油门 0.7，转向越猛（`|angle|` 越大）减得越多，angle=0 直道时约 0.7、angle=±0.8 大弯时降到约 0.38，再叠 ±0.02 的小抖动。这完全符合真人开车习惯——入弯减速、出弯加速。**注意这个设计的深意**：throttle 与 |angle| 强相关，网络将来既能从画面预测 angle，也能间接学到"弯道该慢"的油门策略。

### 保存图片

```python
    img_name = f"{i}_cam-image_array_.jpg"  # Donkey Car 的图片命名约定
    Image.fromarray(make_frame(angle)).save(TUB_DIR / img_name)  # 保存 jpg 图片
```

- `img_name = f"{i}_cam-image_array_.jpg"` —— 按 Donkey Car 官方命名约定生成图片文件名，如 `1_cam-image_array_.jpg`。f-string 把帧序号嵌进文件名，文件名里的 `cam-image_array` 表示"cam 摄像头节点的 image_array 图像数组输出"，是 Donkey Car 数据流中的标准部件名。
- `Image.fromarray(make_frame(angle)).save(TUB_DIR / img_name)` —— 一行做了三件事：`make_frame(angle)` 用当前转向角合成画面数组；`Image.fromarray(...)` 把 `(120,160,3)` 的 uint8 数组转成 PIL 图片对象；`.save(TUB_DIR / img_name)` 保存到 tub 目录，PIL 根据扩展名 `.jpg` 自动选用 JPEG 编码。**注意**：这里传入的 angle 必须是标签 angle（裁剪后的值），保证"图片内容"与"记录的标签"严格一致——画面和标签错位是行为克隆数据最隐蔽的致命伤。

### 组装并写入 JSON 记录

```python
    record = {  # 组装 Donkey Car 标准格式的记录
        "cam/image_array": img_name,  # 图片文件名（关联键）
        "user/angle": angle,  # 此刻的转向值
        "user/throttle": throttle,  # 此刻的油门值
        "user/mode": "user",  # 数据来源：人工驾驶
        "timestamp": 1760000000 + i,  # 时间戳
    }
    with open(TUB_DIR / f"record_{i}.json", "w") as f:  # 记录文件名与图片序号一致
        json.dump(record, f)  # 写入 JSON 文件
```

- `record = {...}` —— 按 Donkey Car 标准格式组装一条记录（Python 字典）。字典的键名设计体现了 Donkey Car 的"部件/字段"命名风格：`cam/image_array` 表示来自 cam 部件的图像字段，`user/angle` 表示来自 user（人工）的转向字段。
- `"cam/image_array": img_name` —— 记录该帧对应的图片文件名，这是 JSON 与图片的**内容级关联键**（除了文件名序号配对外，记录内部也存了一份指向）。第 12 章的 TubDataset 正是读这个字段来定位图片的。
- `"user/angle": angle` —— 此刻的转向标签，范围 [-1, 1]。这是第 12 章训练的第一个回归目标。
- `"user/throttle": throttle` —— 此刻的油门标签，范围约 [0, 1]。第二个回归目标。
- `"user/mode": "user"` —— 标记本条数据的来源是人工驾驶。真车上还可能混入 `pilot`（自动驾驶）产生的数据，训练时通常只取 `user` 部分——让模型学人，而不是学另一个模型的（可能有错的）驾驶。这里虽然全是模拟数据，但字段照真车格式填全。
- `"timestamp": 1760000000 + i` —— 时间戳，用一个基准时刻加帧序号模拟单调递增的真实时间。真车上这是 `time.time()` 的实际值，可用于按时间排序、检测丢帧、计算行驶时长。
- `with open(TUB_DIR / f"record_{i}.json", "w") as f:` —— 以文本写模式打开 `record_N.json`，记录文件的序号 N 与图片文件名中的序号**严格一致**，形成配对。`with` 语句保证写完后文件被正确关闭，即使中途出错也不会留下半个文件句柄。
- `json.dump(record, f)` —— 把字典序列化为 JSON 文本写入文件。生成的每条记录都是单行 JSON，如 `{"cam/image_array": "1_cam-image_array_.jpg", "user/angle": ..., ...}`。一帧一文件（而非所有记录塞进一个大文件）是 Donkey Car 的设计：单条损坏不连累全趟数据，追加录制也不需要重写旧文件。

### 收尾打印

```python
print(f"采集完成：{TUB_DIR}/ 下共 {N} 条记录（{N} 张 jpg + {N} 个 json）")
print(f"示例记录: {json.dumps({'cam/image_array': '1_cam-image_array_.jpg', 'user/angle': round(angle,3), 'user/throttle': round(throttle,3)}, ensure_ascii=False)}")
```

- 第一行 `print` —— 汇报采集结果：数据目录、记录总数、文件构成（N 张 jpg + N 个 json）。采集脚本结束时打印"落盘了多少条"是好习惯——如果实际文件数与预期不符，第一时间就能发现。
- 第二行 `print` —— 打印一条示例记录，供人眼检查格式。`json.dumps(...)` 与 `json.dump` 的区别是不带 `s` 的写入文件、带 `s` 的返回字符串；`round(angle, 3)` 把数值保留 3 位小数让输出整洁；`ensure_ascii=False` 允许直接输出非 ASCII 字符（本例全英文无影响，但养成习惯后打印中文字段不会变成 `\uXXXX` 转义）。**注意**：这里打印的 angle/throttle 是循环结束后残留的**最后一帧**（i=360）的值，而非第 1 帧的——文件名 `1_cam-image_array_.jpg` 只是用作格式示例。这一点在读输出时要有数。

## 运行结果与解读

本程序的真实运行输出如下：

```text
采集完成：data/mytub/ 下共 360 条记录（360 张 jpg + 360 个 json）
示例记录: {"cam/image_array": "1_cam-image_array_.jpg", "user/angle": 0.359, "user/throttle": 0.537}
```

逐项解读：

- **"采集完成：data/mytub/ 下共 360 条记录（360 张 jpg + 360 个 json）"** —— 与脚本设定 `N = 360` 完全一致，说明 360 次循环全部成功落盘：每次循环产出 1 张 jpg + 1 个 json，共 720 个文件。可以用 `ls data/mytub | wc -l` 验证文件数应为 720。
- **示例记录 `{"cam/image_array": "1_cam-image_array_.jpg", "user/angle": 0.359, "user/throttle": 0.537}`** —— JSON 结构符合 Donkey Car 标准格式：有关联键 `cam/image_array`、两个回归标签 `user/angle` 和 `user/throttle`。
- **`"user/angle": 0.359`** —— 注意这是循环残留的**最后一帧**（i=360）的转向值：`sin(360/40) = sin(9) ≈ 0.412`，乘 0.8 得约 0.330，再叠加一点 ±0.05 的抖动，落在 0.36 附近完全合理。0.359 表示此刻需要向右中等幅度打方向（正值 = 向右）。
- **`"user/throttle": 0.537`** —— 用公式验证：`0.7 - 0.4 × |0.359| ≈ 0.7 - 0.144 = 0.556`，叠 ±0.02 抖动后得到 0.537，吻合。转向幅度中等（|angle|≈0.36），油门从直道的 0.7 收到 0.54 左右，"弯道收油"的策略在数据里如实体现。
- **固定种子的可复现性** —— 由于 `np.random.seed(42)`，任何人在任何机器上运行本脚本，输出的示例记录都精确等于上面这一行（0.359 / 0.537）。这为写报告、做对比实验提供了稳定基准。

至此，`data/mytub/` 目录下已有 360 组"图片 + 标签"配对数据，第 12 章将编写 TubDataset 读取它们，训练一个 CNN 直接从画面预测转向和油门。

## 小结

- 行为克隆 = 用"人操作时的画面 + 操作记录"做监督学习；采集过程即标注过程，标签天然免费。
- Donkey Car 的 tub 架构用文件名序号配对图片与 JSON 记录（`N_cam-image_array_.jpg` ↔ `record_N.json`），JSON 内部再以 `cam/image_array` 字段冗余关联，简单且鲁棒。
- 转向（-1~+1）与油门（0~1）是连续控制量，属于回归标签；采集时必须保证标签在物理合法范围内（`np.clip` 防御性截断），且画面与标签严格同步。
- 合成数据的价值在于**格式与真车完全一致**：`np.full` 打底、切片画车道线、加噪点、`Image.fromarray().save()` 存 jpg，将来真车数据只需替换目录即可无缝衔接训练。
- 采集脚本收尾要打印落盘数量与示例记录，便于第一时间发现"少帧、错格式、标签越界"等问题；固定随机种子保证全流程可复现。

## 动手练习

1. **改变赛道形状**：把转向生成公式中的 `np.sin(i / 40) * 0.8` 改成 `np.sin(i / 20) * 0.9`（弯更急、打方向更猛），重新运行，观察输出的示例 angle 如何变化；再打开 `data/mytub` 里第 100 帧的图片，看看车道线位置与旧数据有什么不同。
2. **给油门加物理约束**：现在 throttle 没有 `clip` 保护。先尝试把基础油门从 0.7 改为 1.2，观察生成的 json 里出现的越界油门值；然后仿照 angle 的写法给 throttle 加一行 `np.clip(throttle, 0.0, 1.0)`，验证所有记录的油门都落在 [0, 1]。
3. **增加"停车帧"**：在循环末尾（i 接近 360 时，例如 `if i > 340:`）让 throttle 强制线性降到 0（模拟冲线刹车），并保持 angle 不变。检查最后 20 条 json 记录，确认油门平滑归零而转向仍在波动——体会"采集脚本可以编程制造特定驾驶场景"这一思路（真实数据集中也常需要人工补充这类特殊样本）。
