# 第 8 章：自注意力机制与 Transformer

## 本章目标

- 理解自注意力（Self-Attention）的基本思想：让每个词"看到"序列里的其他词，并按重要程度融合信息。
- 能手写缩放点积注意力（Scaled Dot-Product Attention），弄清 Q、K、V 三个角色和公式 `softmax(QK^T / √d) V` 中每一步的张量形状变化。
- 掌握 PyTorch 自带的 `nn.TransformerEncoderLayer` / `nn.TransformerEncoder` 的用法，包括 `d_model`、`nhead`（多头注意力）、`dim_feedforward`、`batch_first` 等关键参数。
- 理解为什么 Transformer 需要位置编码（Positional Encoding），以及"词嵌入 + 位置编码"的标准做法。
- 能用 Transformer 编码器完成一个完整的序列分类任务（构造数据 → 定义模型 → 训练 → 评估），并知道 BERT 等预训练模型与本章代码的关系。

## 对应赛项知识点

本章对应"人工智能训练师"赛项要求中的：**知道预训练模型（如 Transformer、BERT）的架构与自监督学习机制**。

具体来说，赛项要求选手了解：

- Transformer 的整体架构：它完全建立在自注意力机制之上，摒弃了循环结构，可以并行处理整个序列；
- 多头注意力（Multi-Head Attention）的作用：多组"注意力"并行工作，分别关注序列中不同类型的关联模式；
- BERT 的本质：就是本章这种 Transformer 编码器（Encoder）的多层堆叠，先在海量无标注文本上做**自监督学习**（比如掩码语言模型——随机遮住一些词让模型预测），再针对具体任务微调。微调的内容将在第 9 章展开。

## 核心概念讲解

### 什么是自注意力

想象你读一句话："小明把苹果给了小红，因为**它**很甜。"要理解"它"指代什么，你的大脑会自动把注意力放到"苹果"上，而不是"小明"或"小红"上。**自注意力机制**就是让模型学会这种能力：处理序列中每个位置时，都去看一眼序列里所有其他位置，并根据"相关程度"决定从每个位置吸收多少信息。

"自"（Self）的意思是：Query、Key、Value 三者都来自同一个序列，是自己对自己做注意力，而不是对别的序列。

### Q、K、V 三个角色

自注意力里每个词都会生成三个向量，可以类比"查字典"：

- **Query（查询，Q）**：我在找什么。相当于你在字典检索框里输入的关键词。
- **Key（键，K）**：我有什么特征，可以被什么样的查询找到。相当于字典里每个词条的索引。
- **Value（值，V）**：我实际提供的信息内容。相当于词条对应的正文。

计算过程是：用我的 Query 去和每个词的 Key 做匹配（点积衡量相似度），得到一组分数；分数经过 softmax 归一化成权重；再用这组权重对所有词的 Value 加权求和，得到的新向量就融合了全序列的信息。

### 缩放点积注意力公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

逐步拆解（假设有 3 个词，每个词 4 维，即本章代码中的设置）：

1. `QK^T`：Q 的形状是 (3, 4)，K 转置后是 (4, 3)，相乘得到 (3, 3) 的**得分矩阵**。第 i 行第 j 列表示"第 i 个词对第 j 个词的关注程度"。
2. 除以 `√d_k`（这里 √4 = 2）：维度越大，点积的数值倾向于越大，softmax 输入太大会导致梯度极小（进入饱和区），训练困难。除以 √d_k 把得分的方差拉回合理范围，这叫**缩放**（Scaling），是 Transformer 论文里的关键细节。
3. `softmax`：对得分矩阵的每一行做 softmax，使每行元素都非负且和为 1，得到**注意力权重矩阵**。第 i 行就是第 i 个词分配给 3 个词的注意力比例。
4. 乘 `V`：权重矩阵 (3, 3) 乘 V (3, 4)，得到 (3, 4) 的输出。每个词的新表示 = 所有词的 Value 按注意力权重加权求和。

### 多头注意力（赛项考点）

一组 Q/K/V 只能从一种角度衡量词与词的关系。**多头注意力**（Multi-Head Attention）把向量切成几份（本章是 4 个头），每个头独立做一遍注意力，各自学习不同的关联模式——比如一个头关注"数量关系"，另一个头关注"位置邻近关系"——最后把各头结果拼接起来。这类似用多个评委从不同维度打分，比单个评委更可靠。

### 为什么需要位置编码

自注意力对输入是"集合式"的：把序列打乱顺序，注意力计算本身的结果不变（它不知道哪个词在前哪个词在后）。但语言和很多任务里顺序至关重要。解决办法是给每个词向量**加上一个代表其位置的向量**——位置编码（Positional Encoding）。本章使用**可学习的位置编码**：把位置向量当作模型参数，让训练自动学出来。BERT 采用的也是这种方式。

### Transformer 编码器层的内部结构

`nn.TransformerEncoderLayer` 一层内部包含两大块：

1. **多头自注意力子层**：就是上面讲的机制，外面包着残差连接（Residual Connection，即"输入 + 子层输出"，防止深层网络梯度消失）和层归一化（LayerNorm）。
2. **前馈神经网络子层**（Feed-Forward Network）：对序列中每个位置独立地做"升维 → 激活 → 降维"（本章从 32 维升到 64 维再降回 32 维），同样包着残差连接和层归一化。

把若干层这样的编码器堆起来（本章堆 2 层），就是 `nn.TransformerEncoder`。**BERT 本质上就是几十层这种编码器的堆叠**（BERT-base 是 12 层）。

### 本章的任务

我们造一个简单的序列分类任务：输入长度为 10 的 0/1 序列，判断其中 1 的个数是否超过一半（超过 5 个则为类别 1）。流程是：

```
0/1 序列 → Embedding 嵌入成 32 维向量 → 加位置编码 → 2 层 Transformer 编码器
→ 对 10 个位置取平均（池化）→ 线性层输出 2 类得分 → 交叉熵损失训练
```

这个任务对人来说数一数就行，但模型必须自己学会"把所有位置的信息汇总起来"——这正是自注意力擅长的事。

## 完整代码

```python
"""
第 8 章：自注意力机制与 Transformer
对应赛项知识点：知道预训练模型（如 Transformer、BERT）的架构与自监督学习机制
"""
import torch  # 导入 PyTorch
import torch.nn as nn  # 神经网络模块
import math  # 用于开平方（缩放因子）

torch.manual_seed(42)  # 固定随机种子

# ========== 1. 手写缩放点积注意力（Transformer 的核心） ==========
def scaled_dot_product_attention(Q, K, V):
    """注意力公式：Attention(Q,K,V) = softmax(QK^T / √d) V"""
    d_k = Q.size(-1)  # 取 Key 向量的维度 d_k
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # Q·K转置 算相似度得分，再除以 √d 防止数值过大
    weights = torch.softmax(scores, dim=-1)  # softmax 把得分变成和为 1 的注意力权重
    return weights @ V  # 用权重对 Value 加权求和，得到注意力输出

# 演示：3 个词，每个词 4 维向量
Q = torch.randn(3, 4)  # Query：我在找什么
K = torch.randn(3, 4)  # Key：我有什么特征
V = torch.randn(3, 4)  # Value：我实际提供的信息
out = scaled_dot_product_attention(Q, K, V)  # 计算注意力
print("手写注意力输出形状:", out.shape)  # (3, 4)：每个词都融合了其他词的信息

# ========== 2. 用 nn.TransformerEncoder 做序列分类 ==========
# 任务：输入长度为 10 的整数序列（元素为 0/1），判断 1 的个数是否超过一半
n = 1000  # 1000 条序列
seq_len = 10  # 每条序列长度 10
X = torch.randint(0, 2, (n, seq_len))  # 随机生成 0/1 序列，形状 (1000, 10)
y = (X.sum(dim=1) > seq_len // 2).long()  # 标签：1 的个数 > 5 则为类别 1

class TransformerClassifier(nn.Module):  # 定义 Transformer 分类器
    def __init__(self, d_model=32):  # d_model 是词嵌入维度
        super().__init__()  # 调用父类构造
        self.embedding = nn.Embedding(2, d_model)  # 嵌入层：把离散数字 0/1 映射成 32 维向量
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)  # 位置编码（可学习）
        # Transformer 本身不感知顺序，必须加位置信息（BERT 也用可学习位置编码）
        encoder_layer = nn.TransformerEncoderLayer(  # 定义一层 Transformer 编码器
            d_model=d_model,  # 输入/输出维度
            nhead=4,  # 多头注意力的头数：4 个头并行关注不同模式（赛项考点）
            dim_feedforward=64,  # 前馈网络隐藏层维度
            batch_first=True,  # 输入形状为 (batch, 序列长度, 维度)，更直观
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)  # 堆叠 2 层编码器
        self.fc = nn.Linear(d_model, 2)  # 分类头：32 维 -> 2 个类别

    def forward(self, x):  # 前向传播
        x = self.embedding(x) + self.pos  # 词嵌入 + 位置编码，形状 (batch, 10, 32)
        x = self.encoder(x)  # 通过 Transformer 编码器，形状不变
        x = x.mean(dim=1)  # 对序列维度取平均：把 10 个位置的信息聚合成一个向量
        return self.fc(x)  # 分类头输出 2 类得分

# ========== 3. 训练 ==========
model = TransformerClassifier()  # 实例化模型
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam，学习率 0.001（Transformer 常用较小学习率）

from torch.utils.data import DataLoader, TensorDataset  # 数据加载工具
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)  # 构建加载器

for epoch in range(10):  # 训练 10 轮
    for bx, by in loader:  # 逐批次
        loss = criterion(model(bx), by)  # 前向 + 损失
        optimizer.zero_grad()  # 清梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    with torch.no_grad():  # 评估
        acc = (model(X).argmax(1) == y).float().mean()  # 整体准确率
    print(f"第 {epoch+1:2d} 轮 | 损失={loss.item():.4f} | 准确率={acc:.2%}")

# BERT 的本质就是这种 Transformer 编码器的堆叠，先在大量无标注文本上自监督预训练
# （如掩码语言模型：随机遮住一些词让模型预测），再针对具体任务微调 —— 见第 9 章。
```

## 逐行讲解

### 模块文档字符串与导入

```python
"""
第 8 章：自注意力机制与 Transformer
对应赛项知识点：知道预训练模型（如 Transformer、BERT）的架构与自监督学习机制
"""
```

- 三引号包裹的是模块级 docstring（文档字符串），说明本章主题和它对应的赛项知识点：了解 Transformer、BERT 这类预训练模型的架构，以及自监督学习机制。

```python
import torch  # 导入 PyTorch
import torch.nn as nn  # 神经网络模块
import math  # 用于开平方（缩放因子）
```

- `import torch` —— 导入 PyTorch 主包，张量创建、矩阵运算、随机种子等都靠它。
- `import torch.nn as nn` —— 导入神经网络模块并起别名 `nn`。后面用到的 `nn.Module`（所有模型的基类）、`nn.Embedding`、`nn.TransformerEncoderLayer`、`nn.Linear`、`nn.CrossEntropyLoss` 都来自这里。起别名是 PyTorch 社区约定俗成的写法。
- `import math` —— 导入 Python 标准数学库。本章用它计算 √d_k，即注意力公式里的缩放因子。注意这里不需要 `math` 之外的高级功能，用 `d_k ** 0.5` 也能达到同样效果。

```python
torch.manual_seed(42)  # 固定随机种子
```

- `torch.manual_seed(42)` —— 把 PyTorch 的随机数生成器种子固定为 42。本章里 `torch.randn`（随机向量）、`torch.randint`（随机 0/1 序列）、嵌入层和位置编码的初始化都涉及随机数；固定种子后每次运行结果完全一致，方便复现和调试。注意：它只影响 PyTorch 的随机数，如果还用了 Python 自带 `random` 或 NumPy，需要分别再设种子。

### 第一部分：手写缩放点积注意力

```python
def scaled_dot_product_attention(Q, K, V):
    """注意力公式：Attention(Q,K,V) = softmax(QK^T / √d) V"""
```

- `def scaled_dot_product_attention(Q, K, V):` —— 定义一个名为"缩放点积注意力"的函数，接收三个参数：Q（Query，查询）、K（Key，键）、V（Value，值）。在自注意力场景下三者都来自同一个序列；本演示中三者形状都是 (3, 4)：3 个词，每个词 4 维向量。
- 函数体内的 docstring 写出核心公式，方便对照代码阅读。

```python
    d_k = Q.size(-1)  # 取 Key 向量的维度 d_k
```

- `d_k = Q.size(-1)` —— `Q.size(-1)` 取 Q 最后一个维度的大小，即每个词向量的维度，这里是 4。公式中缩放因子是 √d_k，其中 d_k 严格说是 Key 向量的维度；在自注意力里 Q 和 K 维度相同，所以取 Q 的最后一维即可。`size(-1)` 用负数索引取"最后一维"，这种写法对二维 (3, 4) 和带 batch 的三维 (batch, 3, 4) 都成立，更具通用性。

```python
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # Q·K转置 算相似度得分，再除以 √d 防止数值过大
```

这行做了两件事，是整章最核心的计算：

- `K.transpose(-2, -1)` —— 交换 K 的最后两个维度，(3, 4) 变成 (4, 3)。矩阵乘法要求左边矩阵的列数等于右边矩阵的行数，转置后才能与 Q 相乘。用 `-2, -1` 而不是 `0, 1` 同样是为了兼容带 batch 维的三维张量。
- `Q @ K.transpose(-2, -1)` —— `@` 是 Python 的矩阵乘法运算符，等价于 `torch.matmul`。(3, 4) @ (4, 3) 得到 (3, 3) 的**得分矩阵**：第 i 行第 j 列 = 第 i 个词的 Query 与第 j 个词的 Key 的点积。点积越大，说明两个向量方向越一致，即"词 i 越应该关注词 j"。
- `/ math.sqrt(d_k)` —— 整体除以 √4 = 2。为什么要除？词向量维度 d_k 越大，点积结果的绝对值就越大（大致随 √d_k 增长）；而 softmax 函数在输入很大时会进入"饱和区"——输出极度接近 0 或 1，梯度几乎为 0，模型训不动。除以 √d_k 把得分的尺度标准化，保证 softmax 工作在梯度良好的区间。这就是"缩放（Scaled）"一词的由来，是 Transformer 原论文的关键设计。

```python
    weights = torch.softmax(scores, dim=-1)  # softmax 把得分变成和为 1 的注意力权重
```

- `torch.softmax(scores, dim=-1)` —— 对得分矩阵沿最后一维（即每一行）做 softmax。softmax 会把任意实数变成非负且和为 1 的数，所以结果 (3, 3) 矩阵的每一行是一个概率分布：第 i 行表示"第 i 个词应该把注意力按什么比例分配给词 1、2、3"。`dim=-1` 一定要写对：如果误写成 `dim=0`，就变成"对每个词位置跨 query 归一化"，含义完全错了。

```python
    return weights @ V  # 用权重对 Value 加权求和，得到注意力输出
```

- `weights @ V` —— 权重矩阵 (3, 3) 乘 Value 矩阵 (3, 4)，得到 (3, 4) 的输出。输出第 i 行 = 所有词的 Value 按第 i 行注意力权重加权求和。也就是说，每个词的新表示都"揉进"了全序列其他词的信息，关注谁多就吸收谁的信息多。这就是注意力的最终目的。

### 第一部分的演示代码

```python
# 演示：3 个词，每个词 4 维向量
Q = torch.randn(3, 4)  # Query：我在找什么
K = torch.randn(3, 4)  # Key：我有什么特征
V = torch.randn(3, 4)  # Value：我实际提供的信息
```

- `torch.randn(3, 4)` —— 生成形状为 (3, 4) 的张量，元素服从标准正态分布（均值 0、方差 1）。可以理解为"3 个词，每个词用 4 个数字来描述"。
- Q、K、V 分别扮演三个角色：Q 是"我想找什么"，K 是"我有什么可被查到的特征"，V 是"我实际贡献的内容"。真实模型里这三个矩阵不是独立的随机数，而是输入分别乘以三个可学习矩阵 W_Q、W_K、W_V 得到的；这里为了演示公式本身，直接用随机数代替。

```python
out = scaled_dot_product_attention(Q, K, V)  # 计算注意力
print("手写注意力输出形状:", out.shape)  # (3, 4)：每个词都融合了其他词的信息
```

- `out = scaled_dot_product_attention(Q, K, V)` —— 调用刚才手写的函数，走完"相似度打分 → 缩放 → softmax → 加权求和"全流程。
- `print(..., out.shape)` —— 打印输出的形状。输出应是 (3, 4)：3 个词还在、每个词还是 4 维，但每个向量已经是融合过上下文信息的新表示。**注意力不改变形状，改变的是内容**——这是理解 Transformer"层可以无限堆叠"的关键。

### 第二部分：准备序列分类数据

```python
# 任务：输入长度为 10 的整数序列（元素为 0/1），判断 1 的个数是否超过一半
n = 1000  # 1000 条序列
seq_len = 10  # 每条序列长度 10
```

- `n = 1000` —— 数据集共 1000 条样本（序列）。把数据规模写成变量而不是散落在代码里的"魔法数字"，方便日后调整。
- `seq_len = 10` —— 每条序列固定长 10，即相当于"10 个词"。同理写成变量，后面的位置编码、标签计算都引用它。

```python
X = torch.randint(0, 2, (n, seq_len))  # 随机生成 0/1 序列，形状 (1000, 10)
```

- `torch.randint(0, 2, (n, seq_len))` —— 生成整数随机张量：取值范围 [0, 2)（即只有 0 和 1 两种），形状 (1000, 10)。相当于 1000 条"句子"，每条由 10 个"词"组成，词汇表里只有 0、1 两个词。`randint` 返回的是 `int64` 整型张量——这一点很重要，因为后面的 `nn.Embedding` 要求输入必须是整型索引。

```python
y = (X.sum(dim=1) > seq_len // 2).long()  # 标签：1 的个数 > 5 则为类别 1
```

这行连写了三步，逐步看：

- `X.sum(dim=1)` —— 沿维度 1（序列维度）求和，形状从 (1000, 10) 变成 (1000,)，每个数是"这条序列里 1 的个数"。
- `> seq_len // 2` —— `seq_len // 2` 是整数除法，10 // 2 = 5。比较得到布尔张量 (1000,)：1 的个数**严格大于** 5 为 `True`（类别 1），否则为 `False`（类别 0）。注意是 `>` 而不是 `>=`，即恰好 5 个 1 算类别 0。
- `.long()` —— 把布尔张量转成 int64 整型（True→1，False→0）。这是必须的：`nn.CrossEntropyLoss` 要求标签是 int64 的类别编号。

### 第二部分：定义 Transformer 分类模型

```python
class TransformerClassifier(nn.Module):  # 定义 Transformer 分类器
```

- 定义模型类 `TransformerClassifier`，继承 `nn.Module`。PyTorch 里所有自定义模型都要继承这个基类，才能获得参数自动注册、`model.parameters()`、训练/评估模式切换等能力。

```python
    def __init__(self, d_model=32):  # d_model 是词嵌入维度
        super().__init__()  # 调用父类构造
```

- `def __init__(self, d_model=32)` —— 构造函数，参数 `d_model` 默认 32。`d_model` 是 Transformer 的"主干道宽度"：词嵌入维度、编码器的输入输出维度都是它。Transformer 论文用 512，BERT-base 用 768，本章任务简单，32 足够。给默认值意味着调用 `TransformerClassifier()` 不传参也能工作。
- `super().__init__()` —— 调用父类 `nn.Module` 的构造函数。**这一行绝不能漏**：漏了的话，后面赋值的 `self.embedding`、`self.encoder` 等子模块不会被注册，优化器拿不到参数，模型什么也学不到。

```python
        self.embedding = nn.Embedding(2, d_model)  # 嵌入层：把离散数字 0/1 映射成 32 维向量
```

- `nn.Embedding(2, d_model)` —— 创建嵌入层。第一个参数 2 是"词汇表大小"（只有 0、1 两个词），第二个参数是嵌入维度 32。它本质是一张 (2, 32) 的查找表：输入 0 就查第 0 行，输入 1 就查第 1 行。为什么要嵌入？单个数字 0/1 携带的信息太少，映射成 32 维向量后，模型有了足够的表达能力去刻画每个"词"。这张表的初始值是随机的，随训练不断更新。

```python
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)  # 位置编码（可学习）
        # Transformer 本身不感知顺序，必须加位置信息（BERT 也用可学习位置编码）
```

- `torch.randn(1, seq_len, d_model) * 0.1` —— 生成形状 (1, 10, 32) 的随机张量并缩小到 1/10 量级。形状的含义：第 1 维留 1 是为了广播（batch 里每条序列共用同一份位置编码）；第 2 维 10 对应 10 个位置；第 3 维 32 与词向量维度一致，这样才能相加。乘以 0.1 让初始位置信息比较弱，不至于一上来就盖过词嵌入本身的信息。
- `nn.Parameter(...)` —— 把这个张量包装成**模型参数**。这样它会被纳入 `model.parameters()`，被优化器更新——位置编码不是人工设计的固定值，而是让模型自己学出来。BERT 用的正是这种可学习位置编码。
- 注释点明了动机：自注意力对序列顺序不敏感（把词打乱，注意力计算结果不变），必须额外注入位置信息。另一种常见方案是正弦/余弦函数编码（Transformer 原论文的做法），无需学习。

```python
        encoder_layer = nn.TransformerEncoderLayer(  # 定义一层 Transformer 编码器
            d_model=d_model,  # 输入/输出维度
            nhead=4,  # 多头注意力的头数：4 个头并行关注不同模式（赛项考点）
            dim_feedforward=64,  # 前馈网络隐藏层维度
            batch_first=True,  # 输入形状为 (batch, 序列长度, 维度)，更直观
        )
```

这里创建一个完整的 Transformer 编码器层，逐参数看：

- `d_model=d_model` —— 编码器的输入和输出维度都是 32。编码器内部是"残差连接"结构（输出 = 输入 + 子层结果），要求输入输出同维，所以整条主路自始至终是 32 维。
- `nhead=4` —— 多头注意力的头数，即把 32 维切成 4 份，每份 8 维独立做一遍注意力，再拼接回来。4 个头可以并行学习不同的关联模式。**硬性约束：`d_model` 必须能被 `nhead` 整除**（32 ÷ 4 = 8），否则报错。这是赛项常考的概念点。
- `dim_feedforward=64` —— 编码器内部前馈网络的隐藏层维度。注意力子层之后，每个位置的向量会独立经过"32 → 64 → 32"的两层全连接网络（中间有激活函数），增强非线性表达能力。常见做法是设为 d_model 的 2~4 倍。
- `batch_first=True` —— 约定输入张量形状为 (batch, 序列长度, 维度)，即 (32, 10, 32)。**这是个高频坑**：该参数默认是 `False`（历史原因），默认形状是 (序列长度, batch, 维度)，如果不设置，形状对不上就会报错或静默出错。显式写 `True` 让形状更直观，与大多数人的直觉一致。

```python
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)  # 堆叠 2 层编码器
```

- `nn.TransformerEncoder(encoder_layer, num_layers=2)` —— 把刚才定义的一层当作"模板"，复制堆叠成 2 层。注意 2 层的参数各自独立（不是共享同一组权重），前一层喂给后一层。层数越多模型越深、能力越强但越难训练；BERT-base 是 12 层，本章 2 层即可。

```python
        self.fc = nn.Linear(d_model, 2)  # 分类头：32 维 -> 2 个类别
```

- `nn.Linear(d_model, 2)` —— 线性层（全连接层），把 32 维向量映射成 2 个数，即两个类别的得分（logits，未归一化的原始分数）。这一步俗称"分类头"：主干网络（Transformer）负责提取特征，分类头负责把特征翻译成类别判断。

### 第二部分：前向传播

```python
    def forward(self, x):  # 前向传播
```

- 定义 `forward` 方法，描述数据如何从输入流到输出。调用 `model(bx)` 时 PyTorch 会自动执行这里定义的逻辑（同时自动挂接反向传播所需的计算图，所以不要直接调 `model.forward(bx)`）。输入 `x` 的形状是 (batch, 10) 的整数张量。

```python
        x = self.embedding(x) + self.pos  # 词嵌入 + 位置编码，形状 (batch, 10, 32)
```

- `self.embedding(x)` —— 查嵌入表：输入 (batch, 10) 的整数索引，输出 (batch, 10, 32)，每个词变成了 32 维向量。
- `+ self.pos` —— 加上位置编码 (1, 10, 32)。形状不匹配怎么办？靠**广播**（broadcasting）：第 1 维的 1 会自动扩展到 batch 大小，每条序列加上同一份位置向量。这一步之后，向量里既有"这个词是什么"（词嵌入），也有"这个词在第几位"（位置信息），形状保持 (batch, 10, 32)。

```python
        x = self.encoder(x)  # 通过 Transformer 编码器，形状不变
```

- `self.encoder(x)` —— 数据依次流过 2 层编码器，每层做"多头自注意力 + 前馈网络"。10 个位置的向量互相交换信息：每个位置都看过整条序列后更新自己。输出形状仍是 (batch, 10, 32)——编码器只改变内容不改变形状，这也是可以任意堆叠层数的原因。

```python
        x = x.mean(dim=1)  # 对序列维度取平均：把 10 个位置的信息聚合成一个向量
```

- `x.mean(dim=1)` —— 沿维度 1（序列维度，长度为 10）求平均，形状从 (batch, 10, 32) 变成 (batch, 32)。分类任务需要一个向量代表整条序列，这一步叫**平均池化**（mean pooling）。由于每个位置已经通过注意力融合了全局信息，取平均就能汇总整句。另一种常见做法是像 BERT 那样在句首加一个特殊的 `[CLS]` 标记，直接取它的输出做分类。

```python
        return self.fc(x)  # 分类头输出 2 类得分
```

- `self.fc(x)` —— (batch, 32) 经过线性层变成 (batch, 2)，即每条序列得到 2 个类别的得分。注意这里**不需要手动加 softmax**：`nn.CrossEntropyLoss` 内部已经包含了 softmax 计算，直接收原始得分即可。

### 第三部分：训练

```python
model = TransformerClassifier()  # 实例化模型
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam，学习率 0.001（Transformer 常用较小学习率）
```

- `model = TransformerClassifier()` —— 用默认配置（d_model=32、4 头、2 层）实例化模型。此时所有参数（嵌入表、位置编码、编码器权重、分类头权重）被随机初始化。
- `criterion = nn.CrossEntropyLoss()` —— 交叉熵损失函数，分类任务的标准选择。它先对模型输出做 softmax 得到各类别概率，再取正确类别概率的负对数作为损失：正确类别的概率越接近 1，损失越小。再次提醒：它内部含 softmax，模型输出原始得分即可。
- `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)` —— Adam 优化器，它会为每个参数维护自适应的学习率，比裸 SGD 收敛更快更稳。`model.parameters()` 把模型所有注册参数交给优化器。学习率 0.001 是 Transformer 类模型的常用量级；学习率太大会让注意力权重的训练剧烈震荡甚至发散。

```python
from torch.utils.data import DataLoader, TensorDataset  # 数据加载工具
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)  # 构建加载器
```

- `from torch.utils.data import DataLoader, TensorDataset` —— 导入数据工具。写在文件中间虽然能运行，但规范做法是集中放在文件开头。
- `TensorDataset(X, y)` —— 把特征张量 X 和标签张量 y 打包成一个数据集，要求两者第 0 维一致（都是 1000），取第 i 条样本就是 `(X[i], y[i])`。
- `DataLoader(..., batch_size=32, shuffle=True)` —— 把数据集包装成可迭代的分批加载器：每次吐出 32 条样本（最后一批不足 32 则为 24 条），共 ⌈1000/32⌉ = 32 批；`shuffle=True` 让每轮（epoch）开始前打乱顺序，避免模型记住样本排列的伪规律。这里再解释两个术语：**batch（批次）**指每次喂给模型的一小撮样本；**epoch（轮次）**指把整个数据集完整过一遍。

```python
for epoch in range(10):  # 训练 10 轮
    for bx, by in loader:  # 逐批次
```

- `for epoch in range(10)` —— 外层循环：完整遍历数据集 10 次，即训练 10 个 epoch。
- `for bx, by in loader:` —— 内层循环：每次从加载器取一个批次，`bx` 是 (32, 10) 的输入序列，`by` 是 (32,) 的标签。

```python
        loss = criterion(model(bx), by)  # 前向 + 损失
        optimizer.zero_grad()  # 清梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
```

这是 PyTorch 训练四步曲，顺序和作用必须烂熟于心：

- `loss = criterion(model(bx), by)` —— `model(bx)` 前向传播得到 (32, 2) 的类别得分，`criterion` 对比得分与真实标签 `by`，算出本批次的交叉熵损失（一个标量）。
- `optimizer.zero_grad()` —— 把所有参数的梯度清零。**漏掉这行是最常见的新手错误**：PyTorch 默认把梯度**累加**到 `.grad` 上，不清零的话本批次的梯度会叠加到上一批次的残留上，更新方向就乱了。
- `loss.backward()` —— 反向传播：从损失出发，沿计算图自动求出每个参数的梯度（链式法则），存到各参数的 `.grad` 属性里。对 Transformer 这种复杂模型，手写梯度几乎不可能，自动微分是 PyTorch 的核心价值。
- `optimizer.step()` —— 优化器读取各参数的梯度，按 Adam 算法更新参数，模型向损失更小的方向挪一步。

```python
    with torch.no_grad():  # 评估
        acc = (model(X).argmax(1) == y).float().mean()  # 整体准确率
```

- `with torch.no_grad():` —— 上下文管理器：块内所有计算不记录计算图、不计算梯度。评估时不需要反向传播，关掉它可以省内存、提速度。
- `acc = (...)` —— 在**全部 1000 条数据**上算准确率，拆成四步：① `model(X)` 得到 (1000, 2) 的得分；② `.argmax(1)` 沿维度 1 取最大得分的下标，得到 (1000,) 的预测类别；③ `== y` 与真实标签逐元素比较，得到布尔张量；④ `.float().mean()` 转成浮点数（True→1.0）再取平均，即预测正确的比例。注意这是在训练集上的准确率，只能反映"学没学会"，不能反映"能不能泛化"；严谨做法是单独留一部分数据做测试集（前面章节已介绍过）。
- 小细节：这里是在内层训练循环**之外**（与内层 `for` 同级），所以每轮 epoch 结束才评估一次。

```python
    print(f"第 {epoch+1:2d} 轮 | 损失={loss.item():.4f} | 准确率={acc:.2%}")
```

- 用 f-string 格式化打印训练进度。`{epoch+1:2d}`：epoch 从 0 开始，显示时加 1，`:2d` 让个位数轮次前面补空格对齐；`{loss.item():.4f}`：`.item()` 把单元素张量取出为 Python 数字，`.4f` 保留 4 位小数；`{acc:.2%}`：以百分比显示并保留 2 位小数（如 `100.00%`）。注意一个细节：这里打印的 `loss` 是本 epoch **最后一个批次**的损失，不是整轮平均值，所以轮与轮之间可能有小幅波动，观察大趋势即可。

```python
# BERT 的本质就是这种 Transformer 编码器的堆叠，先在大量无标注文本上自监督预训练
# （如掩码语言模型：随机遮住一些词让模型预测），再针对具体任务微调 —— 见第 9 章。
```

- 结尾注释把本章与赛项知识点串起来：BERT 的架构 = 更多层、更宽的 Transformer 编码器堆叠（结构与本章完全一致，只是规模大得多）；**自监督学习**指不需要人工标注，从数据本身构造训练信号——掩码语言模型（MLM）就是随机遮住句子里的一些词，让模型根据上下文猜原词，用这种方式在海量无标注文本上**预训练**；之后再针对具体任务做**微调**（fine-tuning），这是下一章的主题。

## 运行结果与解读

```text
手写注意力输出形状: torch.Size([3, 4])
第  1 轮 | 损失=0.1672 | 准确率=100.00%
第  2 轮 | 损失=0.0079 | 准确率=100.00%
第  3 轮 | 损失=0.0026 | 准确率=100.00%
第  4 轮 | 损失=0.0012 | 准确率=100.00%
第  5 轮 | 损失=0.0012 | 准确率=100.00%
第  6 轮 | 损失=0.0016 | 准确率=100.00%
第  7 轮 | 损失=0.0009 | 准确率=100.00%
第  8 轮 | 损失=0.0009 | 准确率=100.00%
第  9 轮 | 损失=0.0010 | 准确率=100.00%
第 10 轮 | 损失=0.0006 | 准确率=100.00%
```

逐项解读：

- `手写注意力输出形状: torch.Size([3, 4])` —— 手写注意力函数的输出是 (3, 4)：3 个词、每个词 4 维，与输入形状一致。这验证了注意力的特性：**改变内容，不改变形状**。如果你得到的是 (3, 3) 或其他形状，说明矩阵乘法或转置写错了，请回到"逐行讲解"第一部分核对。
- `第 1 轮 | 损失=0.1672 | 准确率=100.00%` —— 仅训练一轮，准确率就达到 100%。因为"1 的个数是否过半"这个规律非常简单，而随机初始化的嵌入和注意力已具备足够表达能力，模型很快就学会了"汇总所有位置的 1"。
- `第 2~3 轮` —— 损失从 0.1672 迅速降到 0.0079、0.0026。准确率已到顶之后损失仍在下降，这是正常现象：交叉熵不满足于"分对"，还会继续**推高正确类别的置信度**（让正确类别概率更接近 1）。
- `第 4~10 轮` —— 损失降到 0.001 上下并小幅波动（如第 6 轮 0.0016 略高于第 5 轮）。波动的原因是打印的是每轮最后一个批次的损失而非整轮均值，且每轮 shuffle 后批次组合不同——只要整体处于低位、准确率保持 100%，就说明训练完全收敛了。
- 验收标准：看到输出形状为 `[3, 4]`，且损失单调下降的大趋势、准确率 100.00%，就说明你成功用 Transformer 编码器完成了序列分类，可以进入下一章了。

## 小结

- 自注意力的核心是公式 `softmax(QK^T / √d_k) V`：Q 与 K 点积算相似度，除以 √d_k 防止数值过大，softmax 归一化成权重，再对 V 加权求和。输入输出形状相同，内容已被上下文增强。
- 多头注意力（`nhead`）把向量切成多份并行做注意力，从多个角度捕捉词间关系；注意 `d_model` 必须能被 `nhead` 整除。
- Transformer 本身不感知顺序，必须注入位置编码；本章用 `nn.Parameter` 实现可学习位置编码，与 BERT 的做法一致。
- `nn.TransformerEncoderLayer` + `nn.TransformerEncoder` 是 PyTorch 现成的编码器积木；记得设 `batch_first=True` 让输入形状为 (batch, 序列长度, 维度)。
- BERT = 多层 Transformer 编码器堆叠 + 无标注数据上的自监督预训练（掩码语言模型）+ 下游任务微调；微调将在第 9 章介绍。

## 动手练习

1. **去掉位置编码**：把 `forward` 里的 `x = self.embedding(x) + self.pos` 改成 `x = self.embedding(x)`，重新运行。这个任务对顺序不敏感，准确率可能仍然 100%——请进一步思考：如果任务改成"判断序列是否为 1111100000 这种前 5 位全 1 的模式"，去掉位置编码后还能学会吗？为什么？
2. **调整多头数**：把 `nhead=4` 依次改成 1、2、8，对比收敛速度（第 1 轮的损失值）。再把 `d_model` 改成 30 保持 `nhead=4`，观察会报什么错，体会"d_model 必须被 nhead 整除"这条约束。
3. **加深模型**：把 `nn.TransformerEncoder(encoder_layer, num_layers=2)` 的 `num_layers` 改成 1 和 4，观察损失下降速度和最终损失的变化。再试试把学习率从 0.001 改成 0.01，看看训练是否还稳定。
