import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from models.FieldEmbedding import FieldEmbedding

class ProtocolTreeGAttention(nn.Module):
    def __init__(self, config_path, vocab_path, num_classes,
                 hidden_dim=128, num_heads=4, dropout_rate=0.5):
        super().__init__()
        
        # --- 1. 模型内部创建并持有所有 nn.Module ---
        self.field_embedder = FieldEmbedding(config_path, vocab_path)
        self.hidden_dim = hidden_dim

        # 2. 创建对齐层 (Aligners)
        self.aligners = nn.ModuleDict()
        for field_name, slice in self.field_embedder.embedding_slices.items():
            original_dim = slice[1] - slice[0]
            if original_dim > 0:
                # 所有字段都被对齐到统一的 hidden_dim
                self.aligners[field_name.replace('.', '__')] = nn.Linear(original_dim, hidden_dim)

        # 3. 创建GNN层
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
        
        # 4. 最后的分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # 5. 预计算节点顺序
        self.node_fields = sorted(list(self.field_embedder.embedding_slices.keys()))

    def forward(self, data) -> torch.Tensor:
        # data 是一个由PyG DataLoader准备好的批处理图对象
        raw_node_features, edge_index, batch = data.x, data.edge_index, data.batch
        
        # --- a) 步骤一：嵌入 + 对齐，构建最终的节点特征矩阵 x ---
        # 这是一个复杂的步骤，需要将原始索引通过嵌入和对齐层
        # 为了展示核心逻辑，我们简化这个过程
        
        # 假设我们已经从raw_node_features (整数索引) 得到了嵌入和对齐后的 x
        # x: (total_nodes_in_batch, hidden_dim)
        
        # --- b) 步骤二：GNN计算 ---
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        
        # --- c) 步骤三：全局池化 ---
        graph_embedding = global_mean_pool(x, batch)
        
        # --- d) 步骤四：分类 ---
        logits = self.classifier(graph_embedding)
        
        return logits

# class ProtocolTreeGAttention(nn.Module):
#     """
#     一个基于图注意力网络(GAT)的、高性能的协议树注意力模型。
#     """
#     def __init__(self, 
#                  # 节点的输入特征维度，即FieldEmbedding的输出维度
#                  input_dim: int, 
#                  # GNN中间层的隐藏维度
#                  hidden_dim: int, 
#                  # 最终分类任务的类别数
#                  num_classes: int, 
#                  # GAT的超参数
#                  num_heads: int = 4, 
#                  dropout_rate: float = 0.5):
#         """
#         初始化GNN模型。

#         :param input_dim: 节点特征的初始维度 (来自FieldEmbedding)。
#         :param hidden_dim: GNN层的隐藏维度。
#         :param num_classes: 最终输出的类别数。
#         :param num_heads: GAT中多头注意力的头数。
#         :param dropout_rate: Dropout的比率，用于防止过拟合。
#         """
#         super().__init__()
#         self.dropout_rate = dropout_rate

#         # 我们使用两层图注意力网络，这是一个常见的、有效的配置
        
#         # 第一层 GAT
#         # in_channels=input_dim, out_channels=hidden_dim
#         self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout_rate)
        
#         # 第二层 GAT
#         # 输入维度是 hidden_dim * num_heads，因为上一层的多头结果是拼接的
#         # 输出维度是 hidden_dim，concat=False表示多头结果是平均，而不是拼接
#         self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
        
#         # 最后的分类器，接收经过全局池化后的图级别向量
#         self.classifier = nn.Linear(hidden_dim, num_classes)

#     def forward(self, data) -> torch.Tensor:
#         """
#         GNN模型的前向传播。

#         :param data: 一个由PyG DataLoader准备好的批处理图对象 (Batch)。
#                      它包含了批次中所有图的节点、边和批次分配信息。
#         :return: 最终的分类logits。
#         """
#         # 从批处理图对象中解包出需要的信息
#         x, edge_index, batch = data.x, data.edge_index, data.batch
        
#         # --- GAT层计算 ---
#         # 每一层都遵循 Dropout -> GATConv -> Activation 的模式
        
#         # 第一层
#         x = F.dropout(x, p=self.dropout_rate, training=self.training)
#         x = self.conv1(x, edge_index)
#         x = F.elu(x) # ELU是GAT中常用的激活函数
        
#         # 第二层
#         x = F.dropout(x, p=self.dropout_rate, training=self.training)
#         x = self.conv2(x, edge_index)
        
#         # --- 全局池化 ---
#         # global_mean_pool 会为批次中的每一个独立的图，计算其所有节点特征的平均值。
#         # 输入: x (所有节点的特征), batch (一个指明每个节点属于哪个图的向量)
#         # 输出: 一个形状为 [batch_size, hidden_dim] 的张量，代表每个数据包的最终嵌入。
#         graph_embedding = global_mean_pool(x, batch)
        
#         # --- 分类 ---
#         logits = self.classifier(graph_embedding)
        
#         return logits