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
