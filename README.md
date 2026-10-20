# pytorch_learning —— PyTorch 学习手册（人工智能训练师备赛版）

依据《2026 年上海市专项职业技能大赛"人工智能训练师"项目技术文件》的知识点编写的
PyTorch 循序渐进入门教程。共 **6 个阶段、10 个章节**，每章包含：

- 一个可独立运行的小案例（`examples/` 下的 `.py` 脚本）
- 对应的 Jupyter 练习笔记本（`notebooks/` 下的 `.ipynb`）
- 逐行讲解代码的 Markdown 手册（`docs/` 目录）
- 真实运行结果存档与解读（`docs/outputs/` 目录）

全部案例使用 **CPU 版 PyTorch** 即可运行，**无需 GPU**。

## 一、环境要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Windows 10+ / Linux / macOS 均可 |
| Python | 3.9 及以上（竞赛现场环境为 3.9.13，本教程在 3.13 上验证通过） |
| GPU | 不需要，纯 CPU 运行 |
| 磁盘空间 | 约 500 MB（虚拟环境 + PyTorch CPU 版） |

## 二、安装配置方法

在项目根目录执行：

```bash
# 1. 创建虚拟环境
python3 -m venv .venv

# 2. 安装 PyTorch（CPU 版）与 numpy
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy

# 3.（可选）安装 Jupyter，用于浏览器交互式练习
.venv/bin/pip install jupyter
```

Windows 下将 `.venv/bin/` 换成 `.venv\Scripts\`。
国内网络较慢时，Jupyter 可加镜像参数：
`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple jupyter`

## 三、启动方法

**方式 1：直接运行案例脚本**

```bash
.venv/bin/python examples/01_tensor_basics.py
```

**方式 2：启动 Jupyter，在浏览器中练习（推荐）**

```bash
.venv/bin/jupyter notebook
```

启动后终端会打印带 token 的链接（形如 `http://127.0.0.1:8888/?token=...`），
用浏览器打开，进入 `notebooks/` 目录，双击任一 `.ipynb` 开始练习。
单击选中单元格，按 `Shift + Enter` 运行当前格。

## 四、测试方法

**1. 验证单个案例能正常运行：**

```bash
.venv/bin/python examples/01_tensor_basics.py
# 输出应与 docs/outputs/01_output.txt 中存档的结果一致
```

**2. 一键回归测试全部 10 个案例：**

```bash
for f in examples/*.py; do .venv/bin/python "$f" > /dev/null 2>&1 && echo "PASS $f" || echo "FAIL $f"; done
```

**3. 验证 Jupyter 笔记本可正常执行：**

```bash
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/01_tensor_basics.ipynb
```

## 五、学习路线（6 阶段 10 章）

```
阶段一  PyTorch 基础
  第 1 章  张量基础              docs/01_张量基础.md
  第 2 章  自动求导              docs/02_自动求导.md
阶段二  数据工程
  第 3 章  数据集与数据加载      docs/03_数据集与数据加载.md
阶段三  模型开发
  第 4 章  第一个神经网络        docs/04_第一个神经网络.md
  第 5 章  完整训练流程          docs/05_完整训练流程.md
阶段四  训练调优
  第 6 章  超参数调优            docs/06_超参数调优.md
  第 7 章  网络结构优化          docs/07_网络结构优化.md
阶段五  Transformer 与微调
  第 8 章  自注意力与Transformer docs/08_自注意力与Transformer.md
  第 9 章  预训练与微调          docs/09_预训练与微调.md
阶段六  综合实战
  第 10 章 综合项目实战          docs/10_综合项目实战.md
```

建议按章节顺序学习，每章约 30~60 分钟：先运行案例确认输出正确，
再对照手册"逐行讲解"读透代码，最后完成章末"动手练习"。

## 六、与赛项知识点的对应关系

| 赛项技术文件要求 | 对应章节 |
| --- | --- |
| 熟练使用 PyTorch 工具链；线性代数在模型中的实现路径 | 第 1、2 章 |
| 构建结构化数据集，完成数据清洗和增强处理；数据清洗、标准化及增强技术 | 第 3 章 |
| 监督学习的应用场景；选择深度学习算法框架 | 第 4 章 |
| 对智能系统进行测试和训练并编写报告 | 第 5、10 章 |
| 学习率衰减、批量大小对训练稳定性的影响；网格搜索、贝叶斯优化（Optuna） | 第 6 章 |
| 通过超参数调整、网络结构优化实现模型精度与效率平衡 | 第 7 章 |
| 预训练模型（Transformer、BERT）的架构与自监督学习机制 | 第 8 章 |
| 微调（Fine-tuning）技术：通过领域特定数据调整模型参数 | 第 9 章 |
| 数据工程 → 训练 → 调优 → 评估报告 全流程 | 第 10 章 |

## 七、项目结构

```
pytorch_learning/
├── README.md                  ← 本文件
├── make_notebooks.py          ← 由 examples/ 重新生成 notebooks/ 的工具脚本
├── examples/                  ← 10 个可独立运行的案例脚本
│   ├── 01_tensor_basics.py        张量创建、运算、广播、变形、索引
│   ├── 02_autograd.py             自动求导与手写梯度下降
│   ├── 03_dataset_dataloader.py   数据清洗、标准化、Dataset/DataLoader
│   ├── 04_first_neural_network.py MLP 二分类与训练五步曲
│   ├── 05_training_loop.py        训练/验证/测试划分与模型保存加载
│   ├── 06_hyperparameter_tuning.py 批量大小、学习率衰减、网格搜索
│   ├── 07_network_optimization.py Dropout / BatchNorm / 早停
│   ├── 08_transformer.py          手写注意力 + nn.TransformerEncoder
│   ├── 09_finetuning.py           冻结骨干微调 vs 全量微调
│   └── 10_full_project.py         端到端设备故障预测综合项目
├── notebooks/                 ← 与案例对应的 Jupyter 练习笔记本
└── docs/                      ← 逐行讲解手册（Markdown）
    ├── 01_张量基础.md  ……  10_综合项目实战.md
    └── outputs/               ← 各案例的真实运行输出存档
```

## 八、学习建议

1. **先跑通再理解**：每章先把代码原样运行一次，确认输出与手册一致，再回头读讲解。
2. **逐行过一遍**：手册的"逐行讲解"是核心，务必逐行对照代码读完。
3. **动手改代码**：完成每章末尾的练习——在 Jupyter 里调参、改结构、观察现象，
   是备赛调参的核心训练。
4. **串联全流程**：学完第 10 章后，尝试把前 9 章的技巧都用到综合项目里，
   例如给第 10 章的模型加上第 6 章的网格搜索。
