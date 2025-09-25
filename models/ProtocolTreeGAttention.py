import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from models.FieldEmbedding import FieldEmbedding
from collections import defaultdict
from typing import Dict, List

class ProtocolTreeGAttention(nn.Module):
    def __init__(self, #config_path: str, vocab_path: str, 
                num_classes: int, node_fields_list: List[str], field_embedder: FieldEmbedding, 
                hidden_dim: int = 128, num_heads: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        
        # self.field_embedder = FieldEmbedding(config_path, vocab_path)
        self.field_embedder = field_embedder
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
        # self.classifier = nn.Linear(hidden_dim, num_classes)
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(), # 可能造成了梯度消失
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(hidden_dim // 2, num_classes)
        # )
        self.classifier = nn.Sequential(
            # 第一个线性层
            nn.Linear(hidden_dim, hidden_dim // 2),
            # “稳定器”：批量归一化
            # nn.BatchNorm1d(hidden_dim),
            # “防死亡”激活函数
            nn.LeakyReLU(),
            # Dropout层
            nn.Dropout(p=dropout_rate),
            # 第二个线性层，输出最终的logits
            nn.Linear(hidden_dim // 2, num_classes)
        )

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

    # def forward(self, data) -> torch.Tensor:
    def forward_features(self, data) -> torch.Tensor:
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
        return graph_embedding
        # logits = self.classifier(graph_embedding)
        
        # return logits
    def forward(self, data) -> torch.Tensor: 
        graph_embedding = self.forward_features(data) 
        logits = self.classifier(graph_embedding)
        return logits

