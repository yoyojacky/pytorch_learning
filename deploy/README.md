# 部署指南：把训练好的模型装到 Donkey Car 小车上

本目录是一个**自包含部署包**，把整个文件夹拷上小车即可用。

```
deploy/
├── README.md                    ← 本文件
├── export_model.py              ← 开发机上运行：权重 -> TorchScript 单文件
├── car_pilot.py                 ← 小车上运行：加载模型 + 实时推理（含 Donkey Car part 接口）
└── donkey_pilot_scripted.pt     ← 导出的模型（结构+权重打包，无需模型定义代码）
```

## 一、为什么要导出 TorchScript

训练脚本保存的 `donkey_pilot.pt` 只存了权重数字，加载它必须先重建模型结构代码。
TorchScript 把「结构 + 权重」打包成一个文件，小车端只需要 `torch.jit.load()` 一行就能用。

每次重新训练后，在开发机上重新导出一次：

```bash
.venv/bin/python deploy/export_model.py   # 生成 deploy/ 同级后拷入 deploy/donkey_pilot_scripted.pt
```

## 二、小车端环境搭建（树莓派）

```bash
# 1. 安装依赖（Donkey Car 官方要求 python3.9+；torch 装 CPU 版即可）
python3 -m venv ~/pilot_env
~/pilot_env/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/pilot_env/bin/pip install numpy

# 2. 如果还没装 Donkey Car 框架（用它的摄像头/电机驱动和驾驶循环）
~/pilot_env/bin/pip install donkeycar
```

## 三、把部署包拷上小车

在开发机上执行（把 `donkeypi` 换成你小车的 IP 或主机名）：

```bash
scp -r deploy pi@donkeypi:~/mycar/
```

## 四、小车端自测（先不通电、不上赛道）

```bash
cd ~/mycar/deploy
~/pilot_env/bin/python car_pilot.py
# 期望输出：模型加载成功！... 推理速度: xx 帧/秒
```

## 五、接入 Donkey Car 驾驶循环

编辑小车上 `~/mycar/manage.py`，在 Vehicle 组装处加入我们的 part：

```python
from deploy.car_pilot import Pilot   # 导入我们的 part

# 在 V = dk.vehicle.Vehicle() 之后，找到添加摄像头之后的位置加入：
pilot = Pilot(model_path="deploy/donkey_pilot_scripted.pt")
V.add(pilot,
      inputs=['cam/image_array'],        # 输入：摄像头画面
      outputs=['pilot/angle', 'pilot/throttle'],  # 输出：预测转向/油门
      run_condition='run_pilot')         # 只在自动驾驶模式下运行
```

然后把驾驶模式切到 `local`（全自动驾驶）：

```bash
python manage.py drive --model deploy/donkey_pilot_scripted.pt
# 手柄/网页遥控器上切到 Local Pilot 模式，小车即由模型驾驶
```

## 六、重要：当前模型还上不了真实赛道！

本项目第 11 章生成的是**合成画面**，模型只认识"程序画的假车道线"。
要真正开起来，必须用真车采集数据再训练——这正是当初严格遵循 Donkey Car
tub 格式的原因，全流程无缝衔接：

```bash
# 1. 小车上人工遥控采集（Donkey Car 官方流程，数据自动存成 tub 格式）
python manage.py drive --js        # 用手柄开 10~20 圈

# 2. 把 tub 数据拷回开发机
scp -r pi@donkeypi:~/mycar/data/tub_* data/

# 3. 用第 12 章脚本直接训练（TubDataset 兼容真车 tub 格式，无需改代码）
.venv/bin/python examples/12_donkeycar_train.py   # 把 tub 目录名改成真车数据目录

# 4. 重新导出 + 拷回小车
.venv/bin/python deploy/export_model.py
scp deploy/donkey_pilot_scripted.pt pi@donkeypi:~/mycar/deploy/
```

## 七、常见问题

| 现象 | 排查 |
| --- | --- |
| `pip install torch` 卡住/失败 | 换镜像：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch` |
| 推理帧率太低 | 确认装的是 CPU 版 torch；检查画面尺寸是否 160x120（大了会慢） |
| 小车乱开 | 几乎肯定是用了合成数据训练的模型——按第六节用真车数据重训练 |
| 预测值方向和实际相反 | 采集时转向方向与电机接反，把输出取负或重新采集 |
