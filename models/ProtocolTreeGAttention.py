import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from models.FieldEmbedding import FieldEmbedding
from collections import defaultdict
from typing import Dict, List

class ProtocolTreeGAttention(nn.Module):
    def __init__(self, config_path: str, vocab_path: str, num_classes: int, node_fields_list: List[str],
                 hidden_dim: int = 128, num_heads: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        
        self.field_embedder = FieldEmbedding(config_path, vocab_path)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.node_fields = node_fields_list # 接收从Dataset传来的、当前Block的节点列表
        
        # --- 1. 创建“融合”的对齐层 (Fused Aligners) ---
        self.aligners = nn.ModuleDict()
        # 扫描所有可能的字段，按其嵌入维度创建唯一的对齐层
        unique_dims = set(s[1] - s[0] for s in self.field_embedder.embedding_slices.values())
        for dim in unique_dims:
            if dim > 0:
                self.aligners[f'aligner_from_{dim}'] = nn.Linear(dim, hidden_dim)

        # --- 2. 创建GNN层和分类器 ---
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _align_fused(self, embedded_vectors: Dict[str, torch.Tensor], device: torch.device):
        """
        一个辅助函数，用于执行高效的“融合对齐”操作。
        """
        aligned_vectors = {}
        
        # a) 按维度对输入向量进行分组
        vectors_by_dim = defaultdict(list)
        names_by_dim = defaultdict(list)
        for name, vec in embedded_vectors.items():
            dim = vec.shape[-1]
            # 只有当存在对应维度的对齐层时，才处理
            if f'aligner_from_{dim}' in self.aligners:
                vectors_by_dim[dim].append(vec)
                names_by_dim[dim].append(name)
        
        # b) 对每个维度组执行一次批处理对齐
        for dim, vecs in vectors_by_dim.items():
            # (num_vecs_in_group, batch_size, dim) -> (batch_size, num_vecs_in_group, dim)
            stacked_vecs = torch.stack(vecs, dim=1)
            
            aligner = self.aligners[f'aligner_from_{dim}']
            # 一次性完成对齐
            aligned_stacked = aligner(stacked_vecs)
            
            # c) 将结果“散射”回字典
            for i, name in enumerate(names_by_dim[dim]):
                aligned_vectors[name] = aligned_stacked[:, i, :]
        
        return aligned_vectors

    def forward(self, data) -> torch.Tensor:
        # data 是一个由PyG DataLoader准备好的批处理图对象
        
        # --- 步骤一：初始嵌入 ---
        batch_dict = {key: val for key, val in data if key not in ['edge_index', 'y', 'num_nodes', 'batch', 'ptr']}
        embedded_vectors = self.field_embedder(batch_dict)
        
        # ==================== 核心修改点：向量化的对齐与组装 ====================
        
        # --- 步骤二：融合对齐 ---
        aligned_vectors = self._align_fused(embedded_vectors, data.edge_index.device)
        
        # --- 步骤三：并行组装节点特征矩阵 x ---
        node_feature_list = []
        # 按照Dataset中定义的节点顺序进行组装
        for field_name in self.node_fields:
            # 使用 .get() 方法，如果字段不存在于对齐后的字典中（例如抽象节点），则返回None
            vec = aligned_vectors.get(field_name)
            if vec is not None:
                node_feature_list.append(vec)
            else:
                # 为抽象节点或缺失的真实节点创建零向量占位符
                zero_vec = torch.zeros(data.num_graphs, self.hidden_dim, device=data.edge_index.device)
                node_feature_list.append(zero_vec)

        # (batch_size, num_nodes, hidden_dim)
        stacked_x = torch.stack(node_feature_list, dim=1)
        # (batch_size * num_nodes, hidden_dim)
        x = stacked_x.view(-1, self.hidden_dim)
        
        # ========================================================================

        # --- 步骤四：GNN计算 ---
        edge_index, batch_idx = data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        
        # --- 步骤五：全局池化和分类 ---
        graph_embedding = global_mean_pool(x, batch_idx)
        logits = self.classifier(graph_embedding)
        
        return logits

# class ProtocolTreeGAttention(nn.Module):
#     def __init__(self, config_path, vocab_path, num_classes, node_field_list,
#                  hidden_dim=128, num_heads=4, dropout_rate=0.3):
#         super().__init__()
        
#         # --- 1. 模型内部创建并持有所有 nn.Module ---
#         self.field_embedder = FieldEmbedding(config_path, vocab_path)
#         self.hidden_dim = hidden_dim
#         self.dropout_rate = dropout_rate
#         self.node_fields = node_field_list
#         # 2. 创建对齐层 (Aligners)
#         self.aligners = nn.ModuleDict()
#         for field_name, slice in self.field_embedder.embedding_slices.items():
#             original_dim = slice[1] - slice[0]
#             if original_dim > 0:
#                 # 所有字段都被对齐到统一的 hidden_dim
#                 self.aligners[field_name.replace('.', '__')] = nn.Linear(original_dim, hidden_dim)

#         # 3. 创建GNN层
#         self.conv1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout_rate)
#         self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
        
#         # 4. 最后的分类器
#         self.classifier = nn.Linear(hidden_dim, num_classes)
        
#         # 5. 预计算节点顺序
#         # self.node_fields = sorted(list(self.field_embedder.embedding_slices.keys()))

#     def forward(self, data) -> torch.Tensor: 
#         # data 是一个由PyG DataLoader准备好的批处理图对象
#         # 它现在包含了 data.eth_dst, data.ip_src 等【整数索引张量】属性
        
#         # --- a) 步骤一：初始嵌入 ---
#         # 将批处理图对象的所有属性转换为一个字典，以便送入嵌入器
#         batch_dict = {key: val for key, val in data if key not in ['edge_index', 'y', 'num_nodes', 'batch', 'ptr']}
#         embedded_vectors = self.field_embedder(batch_dict)
        
#         # --- b) 步骤二：对齐 & 构建节点特征矩阵 x ---
#         aligned_vectors_list = []
#         # 按照预设的节点顺序，确保x的行序是固定的
#         for field_name in self.node_fields:
#             # 检查这个字段是否存在于当前批次的数据中
#             if field_name in embedded_vectors:
#                 vec = embedded_vectors[field_name]
#                 aligner = self.aligners[field_name.replace('.', '__')]
#                 aligned_vectors_list.append(aligner(vec))
#             else:
#                 # 如果字段不存在，则为批次中的每个图添加一个零向量占位符
#                 zero_vec = torch.zeros(data.num_graphs, self.hidden_dim, device=data.edge_index.device)
#                 aligned_vectors_list.append(zero_vec)

#         # (batch_size, num_nodes, hidden_dim)
#         stacked_x = torch.stack(aligned_vectors_list, dim=1)
#         # (batch_size * num_nodes, hidden_dim)
#         x = stacked_x.view(-1, self.hidden_dim)

#         # --- c) 步骤三：GNN计算 ---
#         edge_index, batch_idx = data.edge_index, data.batch
#         x = F.dropout(x, p=self.dropout_rate, training=self.training)
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)
#         x = self.conv2(x, edge_index)
        
#         # --- d) 步骤四：全局池化和分类 ---
#         graph_embedding = global_mean_pool(x, batch_idx)
#         logits = self.classifier(graph_embedding)
        
#         return logits
        # # data.x 的形状是 [total_nodes, 1]，包含了所有节点的【整数值】
        # # data.batch 的形状是 [total_nodes]，指明每个节点属于哪个图
        # x_indices, edge_index, batch_idx = data.x, data.edge_index, data.batch
        
        # # ==================== 核心修改点：在模型内部构建节点特征矩阵 ====================
        
        # # 1. 准备一个空的节点特征矩阵 x
        # #    它的总行数与PyG为我们准备的batch_idx向量的长度完全一致
        # x_features = torch.zeros(data.num_nodes, self.hidden_dim, device=edge_index.device)
        
        # # 2. 这个循环只遍历固定的、少量的字段类型，而不是整个批次
        # #    它的效率远高于之前的方案
        # for i, field_name in enumerate(self.node_fields):
        #     # a) 找到所有属于当前字段类型的节点
        #     #    data.ptr 是一个指针，data.ptr[j] 指向第j个图的起始节点索引
        #     #    第i个节点在每个图中的全局索引是 data.ptr[:-1] + i
        #     node_mask = (torch.arange(data.num_nodes, device=edge_index.device) % len(self.node_fields)) == i
            
        #     # b) 获取这些节点的整数值
        #     indices_for_field = x_indices[node_mask]
            
        #     # c) 嵌入和对齐
        #     layer_key = self.field_embedder.field_to_key_map.get(field_name)
        #     aligner_key = field_name.replace('.', '__')

        #     if layer_key and aligner_key and layer_key in self.field_embedder.embedding_layers:
        #         layer = self.field_embedder.embedding_layers[layer_key]
        #         aligner = self.aligners[aligner_key]

        #         embedded_vecs = layer(indices_for_field.squeeze(1))
        #         aligned_vecs = aligner(embedded_vecs)

        #         # d) 将对齐后的向量“散射”回x_features的正确位置
        #         x_features.masked_scatter_(node_mask.unsqueeze(1), aligned_vecs)
        
        # # ==================== GNN计算与分类 (无变化) ====================
        # x = F.dropout(x_features, p=self.dropout_rate, training=self.training)
        # x = self.conv1(x, edge_index)
        # x = F.elu(x)
        # x = self.conv2(x, edge_index)
        
        # graph_embedding = global_mean_pool(x, batch_idx)
        # logits = self.classifier(graph_embedding)
        
        # return logits
